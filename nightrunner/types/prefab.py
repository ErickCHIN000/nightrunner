"""Prefab (0x61 image + 0x62 fixups): ClassReader graph inspection.

Provenance (see notes/FORMATS/types.md, section "0x61/0x62 Prefab"):

* (A) exactly one logical `Prefabs` per pack in 23 packs; storage flags 0x80 (method 0, version 8).
* (L → A by census 2026-09-15, `run_census()`): part 0x62 = 8-byte prefix (two u32 whose sum equals the low 31 bits
  of the fixups objectCount) + the standard ClassReader stream (types/classreader.py); every pack uses the secondary
  image; the stream parses to the exact end and re-serialises byte-identically.
* (L → A by census) secondary-image "string pool" objects (class word 0xB1000000): +0x00 u64 0, +0x08 tagged pointer
  (slot kind 0x0E) → the characters at +0x18 of the same object, +0x10 u32 length, +0x14 u32 capacity. A primary
  slot of kind 0x09 (second pass + secondary target) points at such an object; its text is the string.
* (L, partially verified by census — counts reported, not assumed) class 0xC000000F "CEntity": +0x40 f32[3]
  m_Translate, +0x4C f32[3] m_Rotate, +0x58 f32[3] m_Scale, +0x68 m_Name, +0x70 m_PrefabName, +0x78 m_PresetNames
  (';'-separated), +0x80 m_Configurations — each of the four at +0x68..+0x80 a kind-0x09 slot. Class 0xC0000042 is a
  type descriptor whose +0x08 carries the type name (inline chars or a kind-0x0E pointer). Everything else about
  class layouts is (C); there is no basis for a writer.
"""

from __future__ import annotations

import math
import struct
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import FormatError
from . import classreader as cr
from .raw import counter_json, strings

PREFIX = struct.Struct("<II")
CLASS_STRING_POOL = 0xB1000000
CLASS_ENTITY = 0xC000000F
CLASS_DESCRIPTOR = 0xC0000042
CLASS_FIELD_DESCRIPTOR = 0xC0000043


class PrefabGraph:
    def __init__(self, image, fixups_blob):
        self.image = memoryview(image)
        self.blob = bytes(fixups_blob)
        if len(self.blob) < 8:
            raise FormatError("prefab fixups shorter than the 8-byte prefix")
        self.prefix = PREFIX.unpack_from(self.blob, 0)
        self.fx = cr.parse(self.blob, offset=8)
        self.secondary = memoryview(self.fx.secondary or b"")
        self.slots_primary = {}
        self.slots_secondary = {}
        for o, k in zip(self.fx.slot_offsets, self.fx.slot_kinds):
            (self.slots_secondary if k & 4 else self.slots_primary)[o] = k

    # ---- pointers ---------------------------------------------------------------------------------------------

    def _space(self, secondary: bool):
        return self.secondary if secondary else self.image

    def pointer(self, off: int, secondary: bool) -> tuple[int, int, int] | None:
        """(kind, target offset, tag) of the slot at *off* in the given image, or None when no slot is there
        or the stored value is null."""
        kind = (self.slots_secondary if secondary else self.slots_primary).get(off)
        if kind is None:
            return None
        buf = self._space(secondary)
        if off + 8 > len(buf):
            return None
        s = struct.unpack_from("<Q", buf, off)[0]
        if s == 0:
            return None
        if kind & 2:
            return kind, (s & 0xFFFFFFFFFFFF) - 1, s >> 48
        return kind, s - 1, 0

    def pool_string(self, obj_off: int) -> str | None:
        """Text of the string-pool object at *obj_off* in the secondary image (None if it is not one)."""
        sec = self.secondary
        if obj_off + 0x18 > len(sec):
            return None
        p = self.pointer(obj_off + 8, True)
        if p is None:
            return None
        kind, tgt, _ = p
        if not (kind & 8) or tgt != obj_off + 0x18:
            return None
        length = struct.unpack_from("<I", sec, obj_off + 0x10)[0]
        if tgt + length + 1 > len(sec) or sec[tgt + length] != 0:
            return None
        raw = bytes(sec[tgt : tgt + length])
        return raw.decode("utf-8", "surrogateescape")

    def string_via_slot(self, off: int, secondary: bool) -> str | None:
        """Resolve a slot to text: kind 0x09 (→ string-pool object) or kind 0x0E/0x02/0x00 (→ chars)."""
        p = self.pointer(off, secondary)
        if p is None:
            return None
        kind, tgt, _ = p
        target_secondary = bool(kind & 8)
        if kind & 1:
            return self.pool_string(tgt) if target_secondary else None
        return cr.cstring_at(self._space(target_secondary), tgt, 4096)

    # ---- verified object views ----------------------------------------------------------------------------------

    def string_pool(self) -> tuple[list[tuple[int, str]], int]:
        """All class-0xB1000000 records that resolve as string-pool objects, plus the count that do not."""
        out = []
        bad = 0
        for r in self.fx.records:
            if r.class_word != CLASS_STRING_POOL or not r.secondary:
                continue
            s = self.pool_string(r.offset)
            if s is None:
                bad += 1
            else:
                out.append((r.offset, s))
        return out, bad

    def entities(self) -> list[dict]:
        img = self.image
        out = []
        for r in self.fx.records:
            if r.class_word != CLASS_ENTITY or r.secondary:
                continue
            base = r.offset
            e = {"offset": base, "count": r.count}
            if base + 0x88 <= len(img):
                f = struct.unpack_from("<9f", img, base + 0x40)
                e["translate"] = [round(x, 4) for x in f[0:3]]
                e["rotate"] = [round(x, 4) for x in f[3:6]]
                e["scale"] = [round(x, 4) for x in f[6:9]]
                e["floats_finite"] = all(math.isfinite(x) for x in f)
                for name, fo in (("name", 0x68), ("prefab_name", 0x70), ("preset_names", 0x78), ("configurations", 0x80)):
                    k = self.slots_primary.get(base + fo)
                    e[name] = self.string_via_slot(base + fo, False) if k is not None else None
                    e[name + "_slot_kind"] = k
                e["parent_slot"] = self.slots_primary.get(base + 0x18)
            out.append(e)
        return out

    def descriptors(self) -> list[dict]:
        sec = self.secondary
        out = []
        for r in self.fx.records:
            if r.class_word != CLASS_DESCRIPTOR or not r.secondary:
                continue
            base = r.offset
            name = self.string_via_slot(base + 8, True)
            how = "slot"
            if name is None and base + 8 < len(sec):
                name = cr.cstring_at(sec, base + 8, 64)
                how = "inline" if name else None
            out.append({"offset": base, "type_name": name, "via": how})
        return out


def parse(parts: list[tuple[int, bytes]]) -> PrefabGraph:
    types = [t for t, _ in parts]
    if types != [0x61, 0x62] or any(d is None for _, d in parts):
        raise FormatError(f"prefab: unexpected part shape {[f'0x{t:02X}' for t in types]}")
    return PrefabGraph(parts[0][1], parts[1][1])


def dump(parts: list[tuple[int, bytes]], *, max_strings: int = 60, max_entities: int = 40) -> dict:
    out = {"kind": "prefab", "shape": [f"0x{t:02X}" for t, _ in parts], "sizes": [len(d) if d is not None else None for _, d in parts]}
    try:
        g = parse(parts)
    except FormatError as exc:
        out["error"] = str(exc)
        return out
    fx = g.fx
    pool, pool_bad = g.string_pool()
    ents = g.entities()
    descs = g.descriptors()
    ent_ok = sum(1 for e in ents if e.get("name") is not None and e.get("prefab_name") is not None)
    out.update({
        "prefix": list(g.prefix), "prefix_sum_eq_object_count": sum(g.prefix) == fx.object_count,
        "fixups": fx.summary(), "image_check": cr.check_against_image(fx, g.image),
        "reserialise_identical": cr.serialise(fx, prefix=g.blob[:8]) == g.blob,
        "string_pool": {"count": len(pool), "unresolved": pool_bad, "sample": [s for _, s in pool[:max_strings]]},
        "descriptors": {"count": len(descs), "type_names": counter_json(Counter(d["type_name"] for d in descs if d["type_name"]), limit=60),
                        "unresolved": sum(1 for d in descs if not d["type_name"])},
        "entities_0xC000000F": {"count": len(ents), "name_and_prefab_resolved": ent_ok,
                                "scale_is_unit": sum(1 for e in ents if e.get("scale") == [1.0, 1.0, 1.0]),
                                "sample": [{k: e.get(k) for k in ("offset", "name", "prefab_name", "preset_names", "translate", "rotate", "scale")}
                                           for e in ents[:max_entities]]},
        "primary_strings_sample": strings(g.image, limit=40),
    })
    return out


def _find_prefab_packs(assets: Path) -> list[Path]:
    out = []
    for p in sorted(assets.rglob("*.rpack")):
        if "custom_rpacks" in p.parts or p.name.lower().startswith("assets_"):
            continue
        try:
            with Pack.open(p) as pk:
                if 0x61 in pk.type_histogram():
                    out.append(p)
        except Exception:  # noqa: BLE001
            continue
    return out


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    c = Counter()
    classes = Counter()
    kinds = Counter()
    type_names = Counter()
    per_pack = []
    packs = _find_prefab_packs(assets)
    for n, p in enumerate(packs):
        if limit and n >= limit:
            break
        if progress:
            progress(f"prefab census {p.name}")
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x61):
                parts = [(pk.part_type(i), bytes(pk.read_part(i))) for i in res.part_indices]
                c["resources"] += 1
                try:
                    g = parse(parts)
                except FormatError as exc:
                    c["parse_error"] += 1
                    per_pack.append({"pack": p.name, "error": str(exc)})
                    continue
                fx = g.fx
                pool, pool_bad = g.string_pool()
                ents = g.entities()
                descs = g.descriptors()
                rec = {
                    "pack": p.name, "name": res.name, "image": len(parts[0][1]), "fixups": len(parts[1][1]),
                    "prefix": list(g.prefix), "object_count": fx.object_count, "table_count": fx.table_count,
                    "relocations": len(fx.slot_offsets), "secondary": len(g.secondary),
                    "string_pool": len(pool), "string_pool_unresolved": pool_bad,
                    "entities": len(ents), "descriptors": len(descs),
                }
                ok_prefix = sum(g.prefix) == fx.object_count
                # stream_end is an absolute position in the blob (the parse started at offset 8, and the secondary
                # image is aligned relative to the part start — see tests/test_types.py::test_prefab_graph), so a
                # stream that fills the part ends at len(blob). The 2026-09-15 census report counted 0/23 here only
                # because this line compared against len(blob) - 8 (QA fix; reserialise_identical was 23/23).
                ok_end = fx.stream_end == len(g.blob)
                ok_ser = cr.serialise(fx, prefix=g.blob[:8]) == g.blob
                ok_size = fx.data_size <= len(parts[0][1])
                chk = cr.check_against_image(fx, g.image)
                c["prefix_sum_eq_object_count"] += ok_prefix
                c["stream_parsed_to_exact_end"] += ok_end
                c["reserialise_identical"] += ok_ser
                c["data_size_le_image"] += ok_size
                c["data_size_eq_image"] += (fx.data_size == len(parts[0][1]))
                c["has_secondary"] += fx.has_secondary
                c["no_bounds_problems"] += (not chk["problems"])
                c["objects_total"] += fx.object_count
                c["records_total"] += fx.table_count
                c["relocations_total"] += len(fx.slot_offsets)
                c["string_pool_objects"] += len(pool)
                c["string_pool_unresolved"] += pool_bad
                c["entities_0xC000000F"] += len(ents)
                c["entities_name_resolved"] += sum(1 for e in ents if e.get("name") is not None)
                c["entities_prefab_name_resolved"] += sum(1 for e in ents if e.get("prefab_name") is not None)
                c["entities_preset_names_resolved"] += sum(1 for e in ents if e.get("preset_names") is not None)
                c["entities_configurations_resolved"] += sum(1 for e in ents if e.get("configurations") is not None)
                c["entities_slots_68_70_78_80_kind9"] += sum(1 for e in ents if all(e.get(k + "_slot_kind") == 9 for k in ("name", "prefab_name", "preset_names", "configurations")))
                c["entities_floats_finite"] += sum(1 for e in ents if e.get("floats_finite"))
                c["entities_scale_unit"] += sum(1 for e in ents if e.get("scale") == [1.0, 1.0, 1.0])
                c["entities_parent_slot_kind0"] += sum(1 for e in ents if e.get("parent_slot") == 0)
                c["descriptors_0xC0000042"] += len(descs)
                c["descriptors_type_name_resolved"] += sum(1 for d in descs if d["type_name"])
                c["descriptors_inline"] += sum(1 for d in descs if d["via"] == "inline")
                c["descriptors_via_slot"] += sum(1 for d in descs if d["via"] == "slot")
                classes.update(f"0x{r.class_word:08X}" for r in fx.records)
                kinds.update(fx.slot_kinds)
                type_names.update(d["type_name"] for d in descs if d["type_name"])
                per_pack.append(rec)
    return {"family": "prefab", "type": "0x61", "packs": len(packs), "totals": dict(sorted(c.items())),
            "class_histogram": counter_json(classes, limit=80), "class_distinct": len(classes),
            "slot_kinds": {f"0x{k:02X}": v for k, v in sorted(kinds.items())},
            "descriptor_type_names": counter_json(type_names, limit=80), "descriptor_type_names_distinct": len(type_names),
            "per_pack": per_pack}
