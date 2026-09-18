"""SDB reader: MDBR container, Block-A table walk, compact ints, material resolver.

Provenance (survey 03 §7; notes/FORMATS/sdb.md):
  (A) framing — renderer mcp_Mdb_LoadSdbFile +0x13B1C0, mcp_Mdb_ParseBlocks +0xECC60 (table order and shapes),
      compact reader +0xDB550, program blobs +0xFF9D0; resolver joins — ResolveTextureBindings +0x131D40,
      ApplyParameterOverrides +0xD89F0 (value-type widths), preset editor export +0x44D70 (0xAA record layout).
  (B) DyingLightExplorer containers.py:Sdb / materials.py:MaterialDatabase (independent implementation here).
  (C) 27 of 39 tables are opaque; their raw records are exposed but never interpreted.

Container
    MDBR outer (16 B): char[4] 'MDBR' | u32 version 0x23062801 | u32 size_a | u32 size_b     (16 + A + B == file size)
    MDB inner (36 B at 16): char[4] 'MDB ' | u32 version | 28 opaque bytes (C)
             version 0x24012001 = DLTB layout; 0x23062801 = DL2 layout (LAYOUT_DL2: 0x62 is 43 B, 0x28 is 25 B,
             one vectors-2 table fewer — byte-exact corpus fit, notes/FORMATS/sdb.md §9)
    Block A: 27 tables, then 3 program groups (kind 0, 2, 3: compact count, count × u32 blob length; blobs live in
             Block B consecutively), then 12 tables — in the fixed order of TABLE_ORDER.
    Compact int: b = first byte; tag = b & 3; width = (1, 2, 4, 1)[tag]; value = little-endian(width bytes) >> 2.

Table shapes (keys are the renderer owner's u32-word indices):
    flat W        compact count; count × W bytes
    vectors W     compact count; per record: compact n, n × W bytes (record span excludes the prefix)
    pair_u16      compact count; per record: 4 bytes, compact n, n × 2 bytes
    u16_records8  compact count; per record: 2 bytes, compact n, n × 8 bytes
    seven         compact count; per record: 7 × (compact n, n × 2 bytes)
    trailing      compact count; per record: 23 bytes; compact gA × {8 bytes, compact n, n bytes};
                  compact gB × {4 bytes, compact n, n × 4 bytes}
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..errors import FormatError


class SdbError(FormatError):
    pass


OUTER_MAGIC = b"MDBR"
OUTER_VERSION = 0x23062801
INNER_MAGIC = b"MDB "
INNER_VERSION = 0x24012001

# (key, shape, width) in native stream order; the program groups sit between the two halves.
TABLE_ORDER_A = [
    (0x0C, "flat", 22), (0x10, "vectors", 2), (0x14, "vectors", 4), (0x18, "vectors", 4),
    (0x1C, "vectors", 8), (0x20, "vectors", 4), (0x24, "vectors", 2), (0x9A, "flat", 25),
    (0x9E, "pair_u16", 0), (0xA2, "vectors", 4), (0x62, "flat", 44), (0x76, "pair_u16", 0),
    (0x6A, "u16_records8", 0), (0x72, "vectors", 4), (0x28, "flat", 27), (0x30, "vectors", 4),
    (0x34, "vectors", 4), (0x3C, "vectors", 2), (0x38, "vectors", 2), (0x40, "vectors", 2),
    (0x48, "vectors", 2), (0x44, "vectors", 2), (0x2C, "seven", 0), (0x8E, "vectors", 2),
    (0x92, "flat", 4), (0x7E, "flat", 32), (0x86, "flat", 8),
]
PROGRAM_KINDS = (0, 2, 3)
TABLE_ORDER_B = [
    (0xC2, "vectors", 8), (0xC6, "flat", 8), (0xCA, "flat", 10), (0xCE, "vectors", 4),
    (0xD2, "vectors", 1), (0xD6, "flat", 8), (0xAE, "vectors", 1), (0xB2, "vectors", 1),
    (0xB6, "vectors", 1), (0xBA, "vectors", 1), (0xBE, "flat", 4), (0xAA, "trailing", 0),
]
TABLE_ORDER = TABLE_ORDER_A + TABLE_ORDER_B

INNER_VERSION_DL2 = 0x23062801


@dataclass(frozen=True)
class Layout:
    """Table stream layout selected by the inner MDB version word (never by the selected game)."""
    name: str
    inner_version: int
    order_a: tuple
    order_b: tuple

    @property
    def order(self) -> tuple:
        return self.order_a + self.order_b


def _dl2_order_a() -> tuple:
    # DL2 (inner 0x23062801), byte-exact on runtime_dx11/dx12.sdb and the DevTools sample (2026-09-16):
    # 0x62 pass records are 43 bytes (DLTB 44), 0x28 stage descriptors 25 (DLTB 27), and the 0x40..0x2C group has
    # one vectors-2 table fewer. Which DLTB key the missing table corresponds to is not known natively (C); it is
    # listed here as 0x48 being absent.
    out = []
    for key, shape, width in TABLE_ORDER_A:
        if key == 0x48:
            continue
        width = {0x62: 43, 0x28: 25}.get(key, width)
        out.append((key, shape, width))
    return tuple(out)


LAYOUT_DLTB = Layout("dltb", INNER_VERSION, tuple(TABLE_ORDER_A), tuple(TABLE_ORDER_B))
LAYOUT_DL2 = Layout("dl2", INNER_VERSION_DL2, _dl2_order_a(), tuple(TABLE_ORDER_B))
LAYOUTS = {l.inner_version: l for l in (LAYOUT_DLTB, LAYOUT_DL2)}

TABLE_MEANING = {
    0x20: "destination override descriptors (+0xD89F0)", 0x9A: "shader records", 0xA2: "texture binding arrays (u16 param id, u16 default AE index)",
    0x62: "render pass records", 0x28: "stage descriptors (+2 u16 program blob index)", 0xC2: "selector arrays (u64; high 16 = 0x9A index)",
    0xCA: "routes: u16 material(B2), program(C2), tokens(B6), slots(CE), values(D2)", 0xCE: "slot descriptors (u32: pid low16 | D2 offset bits16..30 | flag bit31)",
    0xD2: "value blobs", 0xAE: "shared names (textures + parameter identifiers)", 0xB2: "material names", 0xB6: "token strings 'preset;token;…'",
    0xBA: "expression/annotation/preset-name strings", 0xBE: "(u16 id-or-hash, u16 BA index)", 0xAA: "preset records",
}
STRING_TABLES = (0xAE, 0xB2, 0xB6, 0xBA)

VALUE_TYPES = ("bool", "int", "float", "vec2", "vec3", "vec4", "matrix", "string")
VALUE_FORMATS = {0: "<?", 1: "<i", 2: "<f", 3: "<2f", 4: "<3f", 5: "<4f", 6: "<16f", 7: "<H"}
RUNTIME_STRING_BASE = 0xFBFF

_WIDTH = (1, 2, 4, 1)


class _Cursor:
    __slots__ = ("data", "pos", "end", "tags")

    def __init__(self, data, start: int, end: int):
        self.data = data
        self.pos = start
        self.end = end
        self.tags = [0, 0, 0, 0]

    def skip(self, n: int) -> int:
        if n < 0 or self.pos + n > self.end:
            raise SdbError(f"SDB stream: need {n} bytes at 0x{self.pos:X}, block ends at 0x{self.end:X}")
        p = self.pos
        self.pos += n
        return p

    def compact(self) -> int:
        d = self.data
        p = self.pos
        if p >= self.end:
            raise SdbError(f"SDB stream: compact int past block end at 0x{p:X}")
        b = d[p]
        tag = b & 3
        w = _WIDTH[tag]
        if p + w > self.end:
            raise SdbError(f"SDB stream: truncated compact int at 0x{p:X}")
        self.tags[tag] += 1
        self.pos = p + w
        if w == 1:
            return b >> 2
        return int.from_bytes(d[p : p + w], "little") >> 2

    def count(self, minimum: int) -> int:
        n = self.compact()
        if minimum and n > (self.end - self.pos) // minimum:
            raise SdbError(f"SDB stream: count {n} cannot fit at 0x{self.pos:X}")
        return n


@dataclass
class Table:
    key: int
    shape: str
    width: int
    start: int
    end: int
    spans: list = field(default_factory=list)      # (offset, size) per record, absolute file offsets

    @property
    def count(self) -> int:
        return len(self.spans)

    def to_json(self) -> dict:
        return {"key": f"0x{self.key:02X}", "shape": self.shape, "width": self.width, "count": self.count,
                "bytes": self.end - self.start, "meaning": TABLE_MEANING.get(self.key, "opaque (C)")}


class Sdb:
    """Read-only SDB. Use `Sdb.open(path)` as a context manager; records are memoryview slices of the mmap."""

    def __init__(self, data, path: Path | str = "<memory>", *, mm=None, fh=None):
        self.path = Path(path)
        self.data = data
        self._mm = mm
        self._fh = fh
        self.tables: dict[int, Table] = {}
        self.programs: dict[int, list[tuple[int, int]]] = {}
        self._parse()
        self._material_index: dict[str, list[int]] | None = None
        self._routes_by_material: dict[int, list[int]] | None = None
        self._presets_by_name: dict[str, list[int]] | None = None
        self.preset = lru_cache(maxsize=256)(self._preset)

    # ---- lifecycle -------------------------------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> "Sdb":
        path = Path(path)
        fh = open(path, "rb")
        try:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            fh.close()
            raise
        try:
            return cls(mm, path, mm=mm, fh=fh)
        except Exception:
            mm.close()
            fh.close()
            raise

    def close(self) -> None:
        self.preset.cache_clear()
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Sdb":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- framing ---------------------------------------------------------------------------------------------

    def _parse(self) -> None:
        d = self.data
        n = len(d)
        if n < 52:
            raise SdbError(f"{self.path}: shorter than the MDBR + MDB headers")
        magic, version, size_a, size_b = struct.unpack_from("<4sIII", d, 0)
        if magic != OUTER_MAGIC or version != OUTER_VERSION:
            raise SdbError(f"{self.path}: expected MDBR version 0x{OUTER_VERSION:08X}, got {magic!r} 0x{version:08X}")
        if 16 + size_a + size_b != n:
            raise SdbError(f"{self.path}: 16 + A({size_a}) + B({size_b}) != file size {n}")
        self.size_a, self.size_b = size_a, size_b
        imagic, iversion = struct.unpack_from("<4sI", d, 16)
        layout = LAYOUTS.get(iversion) if imagic == INNER_MAGIC else None
        if layout is None:
            known = ", ".join(f"0x{v:08X}" for v in LAYOUTS)
            raise SdbError(f"{self.path}: expected inner MDB version in ({known}), got {imagic!r} 0x{iversion:08X}")
        self.layout = layout
        self.inner_version = iversion
        self.inner_opaque = bytes(d[24:52])
        a = _Cursor(d, 52, 16 + size_a)
        b = _Cursor(d, 16 + size_a, n)
        for key, shape, width in layout.order_a:
            self._table(a, key, shape, width)
        for kind in PROGRAM_KINDS:
            cnt = a.count(4)
            spans = []
            for _ in range(cnt):
                length = struct.unpack_from("<I", d, a.skip(4))[0]
                spans.append((b.skip(length), length))
            self.programs[kind] = spans
        for key, shape, width in layout.order_b:
            self._table(a, key, shape, width)
        if a.pos != a.end or b.pos != b.end:
            raise SdbError(f"{self.path}: unconsumed bytes: A={a.end - a.pos} B={b.end - b.pos}")
        self.compact_tags = a.tags
        # route references range-checked once
        for i in range(self.tables[0xCA].count):
            m, p, t, s, v = self.route_row(i)
            for val, key in ((m, 0xB2), (p, 0xC2), (t, 0xB6), (s, 0xCE), (v, 0xD2)):
                if val >= self.tables[key].count:
                    raise SdbError(f"{self.path}: route {i} references 0x{key:02X}[{val}] of {self.tables[key].count}")

    def _table(self, a: _Cursor, key: int, shape: str, width: int) -> None:
        d = self.data
        start = a.pos
        minimum = width if shape == "flat" else {"vectors": 1, "pair_u16": 5, "u16_records8": 3, "seven": 7, "trailing": 25}[shape]
        cnt = a.count(minimum)
        spans = []
        if shape == "flat":
            base = a.skip(cnt * width)
            spans = [(base + i * width, width) for i in range(cnt)]
        elif shape == "vectors":
            for _ in range(cnt):
                ln = a.count(width) * width
                spans.append((a.skip(ln), ln))
        elif shape == "pair_u16":
            for _ in range(cnt):
                r = a.pos
                a.skip(4)
                a.skip(a.count(2) * 2)
                spans.append((r, a.pos - r))
        elif shape == "u16_records8":
            for _ in range(cnt):
                r = a.pos
                a.skip(2)
                a.skip(a.count(8) * 8)
                spans.append((r, a.pos - r))
        elif shape == "seven":
            for _ in range(cnt):
                r = a.pos
                for _ in range(7):
                    a.skip(a.count(2) * 2)
                spans.append((r, a.pos - r))
        elif shape == "trailing":
            for _ in range(cnt):
                r = a.pos
                a.skip(23)
                for _ in range(a.count(9)):
                    a.skip(8)
                    a.skip(a.count(1))
                for _ in range(a.count(5)):
                    a.skip(4)
                    a.skip(a.count(4) * 4)
                spans.append((r, a.pos - r))
        else:  # pragma: no cover
            raise SdbError(f"unknown table shape {shape}")
        self.tables[key] = Table(key, shape, width, start, a.pos, spans)

    # ---- raw access ------------------------------------------------------------------------------------------

    def table(self, key: int) -> Table:
        try:
            return self.tables[key]
        except KeyError:
            raise SdbError(f"unknown SDB table 0x{key:02X}") from None

    def record(self, key: int, index: int):
        t = self.table(key)
        if not 0 <= index < t.count:
            raise SdbError(f"0x{key:02X}[{index}] out of range (count {t.count})")
        off, size = t.spans[index]
        return memoryview(self.data)[off : off + size]

    def record_hex(self, key: int, index: int) -> str:
        return bytes(self.record(key, index)).hex()

    def text(self, key: int, index: int) -> str:
        if key not in STRING_TABLES:
            raise SdbError(f"0x{key:02X} is not a string table")
        return bytes(self.record(key, index)).decode("utf-8", "surrogateescape")

    def program(self, kind: int, index: int):
        spans = self.programs.get(kind)
        if spans is None or not 0 <= index < len(spans):
            raise SdbError(f"program kind {kind} index {index} out of range")
        off, size = spans[index]
        return memoryview(self.data)[off : off + size]

    def route_row(self, index: int) -> tuple[int, int, int, int, int]:
        """(material B2, program C2, tokens B6, slots CE, values D2) — the on-disk order (native reorders)."""
        off, _ = self.tables[0xCA].spans[index]
        return struct.unpack_from("<5H", self.data, off)

    # ---- indices ---------------------------------------------------------------------------------------------

    def materials(self) -> list[str]:
        return [self.text(0xB2, i) for i in range(self.tables[0xB2].count)]

    def _build_indices(self) -> None:
        if self._material_index is not None:
            return
        idx: dict[str, list[int]] = {}
        for i in range(self.tables[0xB2].count):
            idx.setdefault(self.text(0xB2, i).casefold(), []).append(i)
        self._material_index = idx
        rbm: dict[int, list[int]] = {}
        for r in range(self.tables[0xCA].count):
            rbm.setdefault(self.route_row(r)[0], []).append(r)
        self._routes_by_material = rbm
        pbn: dict[str, list[int]] = {}
        for i, (off, _) in enumerate(self.tables[0xAA].spans):
            name_id = struct.unpack_from("<I", self.data, off + 8)[0]
            pbn.setdefault(self.text(0xBA, name_id), []).append(i)
        self._presets_by_name = pbn

    def find_material(self, name: str) -> list[int]:
        """All B2 indices whose name matches (case-insensitive; a missing `.mat` suffix is tolerated)."""
        self._build_indices()
        key = name.casefold()
        return list(self._material_index.get(key) or self._material_index.get(key + ".mat") or [])

    def material_index(self, name: str | int) -> int:
        if isinstance(name, int):
            if not 0 <= name < self.tables[0xB2].count:
                raise SdbError(f"material index {name} out of range")
            return name
        hits = self.find_material(name)
        if len(hits) != 1:
            raise SdbError(f"{name!r}: {len(hits)} material matches in {self.path.name}")
        return hits[0]

    def routes_of(self, material_index: int) -> list[int]:
        self._build_indices()
        return list(self._routes_by_material.get(material_index, []))

    def presets_named(self, name: str) -> list[int]:
        self._build_indices()
        return list(self._presets_by_name.get(name, []))

    # ---- presets (0xAA) ---------------------------------------------------------------------------------------

    def _preset(self, index: int) -> dict:
        raw = bytes(self.record(0xAA, index))
        c = _Cursor(raw, 0, len(raw))
        key, name_id, flags = struct.unpack_from("<QII", raw, c.skip(16))
        masks = list(raw[c.skip(7) : c.pos])
        params = []
        for _ in range(c.count(9)):
            pid, expr, ann, type_flags = struct.unpack_from("<4H", raw, c.skip(8))
            ln = c.count(1)
            default = raw[c.skip(ln) : c.pos]
            params.append({"id": pid, "name": self.text(0xAE, pid), "expression": self.text(0xBA, expr),
                           "annotation": self.text(0xBA, ann), "type": type_flags & 0x7FFF, "type_flags": type_flags,
                           "type_name": VALUE_TYPES[type_flags & 0x7FFF] if (type_flags & 0x7FFF) < 8 else f"unknown_{type_flags & 0x7FFF}",
                           "default_hex": default.hex()})
        groups = []
        for _ in range(c.count(5)):
            gkey = struct.unpack_from("<I", raw, c.skip(4))[0]
            n = c.count(4)
            vals = list(struct.unpack_from(f"<{n}I", raw, c.skip(4 * n)))
            groups.append({"key": gkey, "values": vals})
        return {"index": index, "name": self.text(0xBA, name_id), "key": key, "flags": flags, "masks": masks,
                "parameters": params, "groups_b": groups, "complete": c.pos == len(raw)}

    # ---- resolver --------------------------------------------------------------------------------------------

    def string_value(self, v: int) -> dict:
        if v >= RUNTIME_STRING_BASE:
            return {"string_index": v, "runtime_index": v - RUNTIME_STRING_BASE, "name": None}
        return {"string_index": v, "name": self.text(0xAE, v)}

    def route(self, route_index: int) -> dict:
        d = self.data
        m, prog, tok, slots_i, values_i = self.route_row(route_index)
        tokens = self.text(0xB6, tok)
        preset_name = tokens.split(";", 1)[0]
        preset_ids = self.presets_named(preset_name)
        decl = {}
        if len(preset_ids) == 1:
            decl = {p["id"]: p for p in self.preset(preset_ids[0])["parameters"]}
        blob = bytes(self.record(0xD2, values_i))
        params = []
        by_id: dict[int, dict] = {}
        for (packed,) in struct.iter_unpack("<I", self.record(0xCE, slots_i)):
            pid = packed & 0xFFFF
            offset = (packed >> 16) & 0x7FFF
            dcl = decl.get(pid)
            p = {"id": pid, "name": self.text(0xAE, pid), "offset": offset, "flag_bit31": bool(packed & 0x80000000),
                 "raw": packed, "declared": dcl is not None}
            if dcl is not None:
                kind = dcl["type"]
                p["type"] = dcl["type_name"]
                fmt = VALUE_FORMATS.get(kind)
                if fmt is not None:
                    size = struct.calcsize(fmt)
                    if offset + size > len(blob):
                        raise SdbError(f"route {route_index}: parameter {pid} value at {offset}+{size} exceeds D2 blob {len(blob)}")
                    val = struct.unpack_from(fmt, blob, offset)
                    p["value"] = val[0] if len(val) == 1 else list(val)
                    p["value_hex"] = blob[offset : offset + size].hex()
                    if kind == 7:
                        p["string"] = self.string_value(val[0])
                        p["value"] = p["string"]["name"]
            params.append(p)
            by_id.setdefault(pid, p)
        variants = []
        for (sel,) in struct.iter_unpack("<Q", self.record(0xC2, prog)):
            shader = sel >> 48
            rec = self.record(0x9A, shader)
            tex_arr = struct.unpack_from("<H", rec, 0x12)[0]
            bindings = []
            if tex_arr:
                for b, (pid, default) in enumerate(struct.iter_unpack("<HH", self.record(0xA2, tex_arr))):
                    slot = by_id.get(pid) if pid else None
                    if slot is not None:
                        off = slot["offset"]
                        if off + 2 > len(blob):
                            raise SdbError(f"route {route_index}: texture index for pid {pid} exceeds D2 blob")
                        value = struct.unpack_from("<H", blob, off)[0]
                        source = "override"
                    else:
                        value = default
                        source = "shader_default"
                    sv = self.string_value(value)
                    bindings.append({"binding": b, "param_id": pid, "param": self.text(0xAE, pid) if pid else None,
                                     "texture": sv["name"], "string_index": value, "source": source,
                                     **({"runtime_index": sv["runtime_index"]} if "runtime_index" in sv else {})})
            variants.append({"selector": sel, "selector_hex": f"0x{sel:016X}", "shader": shader, "texture_array": tex_arr,
                             "render_pass_count": rec[0x17] & 15, "texture_bindings": bindings})
        return {"index": route_index, "material": m, "program": prog, "tokens_index": tok, "slots": slots_i, "values": values_i,
                "tokens": tokens, "preset": preset_name, "preset_indices": preset_ids, "parameters": params, "variants": variants}

    def material(self, name: str | int) -> dict:
        idx = self.material_index(name)
        routes = [self.route(r) for r in self.routes_of(idx)]
        variants = [v for r in routes for v in r["variants"]]
        return {"index": idx, "name": self.text(0xB2, idx), "database": self.path.name, "routes": routes,
                "non_rendering": bool(variants) and all(v["render_pass_count"] == 0 for v in variants)}

    def textures_for_material(self, name: str | int) -> set[str]:
        out = set()
        for r in self.material(name)["routes"]:
            for v in r["variants"]:
                for b in v["texture_bindings"]:
                    if b["texture"]:
                        out.add(b["texture"])
        return out

    # ---- summaries -------------------------------------------------------------------------------------------

    def stats(self) -> dict:
        return {
            "path": str(self.path), "size": len(self.data), "block_a": self.size_a, "block_b": self.size_b,
            "inner_opaque_hex": self.inner_opaque.hex(), "compact_tags": self.compact_tags,
            "tables": [self.tables[k].to_json() for k, _, _ in self.layout.order],
            "programs": {str(k): {"count": len(v), "bytes": sum(n for _, n in v)} for k, v in self.programs.items()},
            "materials": self.tables[0xB2].count, "routes": self.tables[0xCA].count, "presets": self.tables[0xAA].count,
        }

    def validate_all(self, progress=None) -> dict:
        """Resolve every material; return totals comparable with the old validation (348,164 parameters,
        1,854,984 / 1,854,986 bindings, 0 runtime-name uses)."""
        c = {"materials": 0, "routes": 0, "parameters": 0, "parameters_declared": 0, "parameters_typed": 0,
             "bindings": 0, "bindings_override": 0, "bindings_shader_default": 0, "runtime_name_uses": 0,
             "materials_without_route": 0, "routes_preset_missing": 0, "routes_preset_ambiguous": 0, "errors": 0,
             "materials_non_rendering": 0, "texture_names": 0}
        errors = []
        names = set()
        n = self.tables[0xB2].count
        for i in range(n):
            try:
                m = self.material(i)
            except SdbError as exc:
                c["errors"] += 1
                if len(errors) < 20:
                    errors.append({"material": i, "error": str(exc)})
                continue
            c["materials"] += 1
            if not m["routes"]:
                c["materials_without_route"] += 1
            c["materials_non_rendering"] += m["non_rendering"]
            for r in m["routes"]:
                c["routes"] += 1
                if not r["preset_indices"]:
                    c["routes_preset_missing"] += 1
                elif len(r["preset_indices"]) > 1:
                    c["routes_preset_ambiguous"] += 1
                for p in r["parameters"]:
                    c["parameters"] += 1
                    c["parameters_declared"] += p["declared"]
                    c["parameters_typed"] += ("value" in p)
                    if p.get("string") and p["string"].get("runtime_index") is not None:
                        c["runtime_name_uses"] += 1
                for v in r["variants"]:
                    for b in v["texture_bindings"]:
                        c["bindings"] += 1
                        c["bindings_override"] += (b["source"] == "override")
                        c["bindings_shader_default"] += (b["source"] == "shader_default")
                        if "runtime_index" in b:
                            c["runtime_name_uses"] += 1
                        if b["texture"]:
                            names.add(b["texture"])
            if progress and i % 2000 == 0:
                progress(f"{i}/{n}")
        c["texture_names"] = len(names)
        return {"path": str(self.path), "totals": c, "errors": errors}
