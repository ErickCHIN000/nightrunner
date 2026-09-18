"""Worker-side helpers for the SDB tab: list data, preset/texture joins, raw record dumps and usage scans.

Everything except the two small QObject job classes at the bottom is plain Python and safe to call from a
`TaskRunner` worker. The long scans (every catalog mesh, `Sdb.validate_all`) run on their own daemon threads so
they never occupy the shared thread pool the other tabs rely on for previews and searches.
"""
from __future__ import annotations

import struct
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..sdb.reader import VALUE_FORMATS, VALUE_TYPES, Sdb

MESH_TYPE = 0x10
TEXTURE_TYPE = 0x20
MAX_MESH_IMAGE = 256 << 20          # an image part larger than this is not a real mesh; skip without decoding


# ---- list data ----------------------------------------------------------------------------------------------------
@dataclass
class SdbLists:
    """Everything the left-hand lists need, computed once per opened SDB."""
    names: list[str]
    lower: list[str]
    by_name: dict[str, list[int]]                    # casefold name (also without ".mat") -> material indices
    material_preset: list[str] = field(default_factory=list)
    presets: list[dict] = field(default_factory=list)          # {index, name, params}
    preset_materials: dict[str, list[int]] = field(default_factory=dict)
    preset_lower: list[str] = field(default_factory=list)


def load_names(sdb: Sdb) -> SdbLists:
    """Material names and the case-insensitive name map (fast: ~30 ms for 26k materials)."""
    names = sdb.materials()
    lower = [n.casefold() for n in names]
    by_name: dict[str, list[int]] = {}
    for i, n in enumerate(lower):
        by_name.setdefault(n, []).append(i)
        if n.endswith(".mat"):
            by_name.setdefault(n[:-4], []).append(i)
    return SdbLists(names, lower, by_name)


def load_presets(sdb: Sdb, lists: SdbLists) -> SdbLists:
    """Fill the per-material preset name, the preset list and the preset -> materials join (from route tokens)."""
    n = len(lists.names)
    mat_preset = [""] * n
    seen = [False] * n
    tok_cache: dict[int, str] = {}
    by_preset: dict[str, list[int]] = {}
    for r in range(sdb.tables[0xCA].count):
        m, _, tok, _, _ = sdb.route_row(r)
        p = tok_cache.get(tok)
        if p is None:
            p = tok_cache[tok] = sdb.text(0xB6, tok).split(";", 1)[0]
        if m < n and not seen[m]:
            seen[m] = True
            mat_preset[m] = p
            by_preset.setdefault(p, []).append(m)
    presets = []
    for i, (off, _) in enumerate(sdb.tables[0xAA].spans):
        name_id = struct.unpack_from("<I", sdb.data, off + 8)[0]
        name = sdb.text(0xBA, name_id)
        try:
            nparams = len(sdb.preset(i)["parameters"])
        except Exception:  # noqa: BLE001
            nparams = -1
        presets.append({"index": i, "name": name, "params": nparams, "materials": len(by_preset.get(name, []))})
    lists.material_preset = mat_preset
    lists.presets = presets
    lists.preset_materials = by_preset
    lists.preset_lower = [p["name"].casefold() for p in presets]
    return lists


def filter_rows(lower: list[str], text: str, index_prefix: bool = True) -> np.ndarray:
    """Row indices whose lower-case name contains every word of *text*. `#123` matches index 123."""
    words = [w for w in text.casefold().split() if w]
    if not words:
        return np.arange(len(lower), dtype=np.int64)
    if index_prefix and len(words) == 1 and words[0].startswith("#") and words[0][1:].isdigit():
        i = int(words[0][1:])
        return np.asarray([i] if i < len(lower) else [], dtype=np.int64)
    w0, rest = words[0], words[1:]
    return np.asarray([i for i, s in enumerate(lower) if w0 in s and all(w in s for w in rest)], dtype=np.int64)


# ---- value helpers ------------------------------------------------------------------------------------------------
def format_value(v) -> str:
    """Compact display of a decoded parameter value."""
    if v is None:
        return ""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    if isinstance(v, (list, tuple)):
        return "(" + ", ".join(format_value(x) for x in v) + ")"
    if isinstance(v, dict):
        return str(v.get("name") if v.get("name") is not None else v)
    return str(v)


def decode_default(sdb: Sdb, type_id: int, default_hex: str):
    """A preset default decoded with the parameter's value type; None when the bytes do not fit the type."""
    fmt = VALUE_FORMATS.get(type_id)
    if fmt is None:
        return None
    raw = bytes.fromhex(default_hex)
    if len(raw) != struct.calcsize(fmt):
        return None
    val = struct.unpack(fmt, raw)
    if type_id == 7:
        sv = sdb.string_value(val[0])
        return sv["name"] if sv.get("name") is not None else f"<runtime {sv['runtime_index']}>"
    return val[0] if len(val) == 1 else list(val)


def type_name(type_id: int) -> str:
    return VALUE_TYPES[type_id] if 0 <= type_id < len(VALUE_TYPES) else f"unknown_{type_id}"


def raw_records(sdb: Sdb, m: dict) -> list[dict]:
    """Hex of every record a resolved material references (same set as `nr sdb material --raw`)."""
    refs = {"0xB2": [m["index"]], "0xCA": [], "0xC2": [], "0xB6": [], "0xCE": [], "0xD2": [], "0xAA": [], "0x9A": [],
            "0xA2": []}
    for r in m["routes"]:
        refs["0xCA"].append(r["index"])
        refs["0xC2"].append(r["program"])
        refs["0xB6"].append(r["tokens_index"])
        refs["0xCE"].append(r["slots"])
        refs["0xD2"].append(r["values"])
        refs["0xAA"].extend(r["preset_indices"])
        for v in r["variants"]:
            refs["0x9A"].append(v["shader"])
            if v["texture_array"]:
                refs["0xA2"].append(v["texture_array"])
    out = []
    for t, idxs in refs.items():
        for i in sorted(set(idxs)):
            try:
                hx = sdb.record_hex(int(t, 16), i)
            except Exception as exc:  # noqa: BLE001
                hx = f"<{exc}>"
            out.append({"table": t, "index": i, "hex": hx})
    return out


def hexdump(hx: str, width: int = 16) -> str:
    """Offset + hex + ASCII lines for a hex string."""
    data = bytes.fromhex(hx) if hx and not hx.startswith("<") else b""
    if not data:
        return "  " + (hx or "(empty record)")
    lines = []
    for o in range(0, len(data), width):
        chunk = data[o:o + width]
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"  {o:06X}  {chunk.hex(' '):<{width * 3}} {asc}")
    return "\n".join(lines)


def texture_providers(catalog, names) -> dict[str, list[int]]:
    """texture name -> catalog gids (type 0x20), for every name. First call builds the catalog name table."""
    out = {}
    for n in names:
        try:
            out[n] = catalog.lookup(n, TEXTURE_TYPE) if n else []
        except Exception:  # noqa: BLE001
            out[n] = []
    return out


def material_detail(sdb: Sdb, catalog, index: int) -> dict:
    """Resolved material + preset declarations + raw records + catalog providers (one worker job)."""
    m = sdb.material(index)
    decls: dict[int, dict] = {}
    presets: list[dict] = []
    for r in m["routes"]:
        for pi in r["preset_indices"]:
            p = sdb.preset(pi)
            presets.append({"index": pi, "name": p["name"]})
            for d in p["parameters"]:
                dd = dict(d)
                dd["default"] = decode_default(sdb, d["type"], d["default_hex"])
                decls.setdefault(d["id"], dd)
    names = sorted({b["texture"] for r in m["routes"] for v in r["variants"] for b in v["texture_bindings"]
                    if b["texture"]})
    return {"material": m, "decls": decls, "presets": presets, "raw": raw_records(sdb, m),
            "providers": texture_providers(catalog, names), "catalog_ready": bool(getattr(catalog, "is_ready", False))}


def preset_detail(sdb: Sdb, index: int) -> dict:
    p = dict(sdb.preset(index))
    params = []
    for d in p["parameters"]:
        dd = dict(d)
        dd["default"] = decode_default(sdb, d["type"], d["default_hex"])
        params.append(dd)
    p["parameters"] = params
    return p


# ---- model references ---------------------------------------------------------------------------------------------
def build_model_refs(paks) -> dict:
    """Every material reference in every .model document, keyed by casefold material name.

    Two kinds of rows: ``embedded`` — the name appears in ``materialsData`` (a submesh material embedded in the
    mesh that the model remaps); ``resource`` — the name is a ``materialsResources`` entry (the base material the
    model applies, with its ``rttiValues`` overrides)."""
    refs: dict[str, list[dict]] = {}
    errors: list[str] = []
    t0 = time.perf_counter()
    models = paks.models()
    for rec in models:
        try:
            doc = paks.load_model(rec)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{rec['name']}: {exc}")
            continue
        base = {"model": rec["name"], "pak": rec["pak"].name, "overridden_by": list(rec.get("overridden_by") or [])}
        slots = doc.get("slots") if isinstance(doc, dict) else None
        for s in slots if isinstance(slots, list) else []:
            if not isinstance(s, dict):
                continue
            res = (s.get("meshResources") or {}).get("resources") if isinstance(s.get("meshResources"), dict) else None
            for r in res if isinstance(res, list) else []:
                if not isinstance(r, dict):
                    continue
                mds = [md for md in r.get("materialsData") or [] if isinstance(md, dict)]
                embedded_of = {md.get("number"): md.get("name") for md in mds}
                common = dict(base, slot=s.get("name"), mesh=r.get("name"), mesh_selected=bool(r.get("selected")))
                for md in mds:
                    nm = md.get("name")
                    if isinstance(nm, str) and nm:
                        refs.setdefault(nm.casefold(), []).append(dict(
                            common, kind="embedded", number=md.get("number"), embedded=nm, selected=None,
                            rtti=[], alternatives=None))
                for g in r.get("materialsResources") or []:
                    if not isinstance(g, dict):
                        continue
                    alts = [x for x in g.get("resources") or [] if isinstance(x, dict)]
                    for x in alts:
                        nm = x.get("name")
                        if isinstance(nm, str) and nm:
                            rtti = [v for v in x.get("rttiValues") or [] if isinstance(v, dict)]
                            refs.setdefault(nm.casefold(), []).append(dict(
                                common, kind="resource", number=g.get("number"), embedded=embedded_of.get(g.get("number")),
                                selected=bool(x.get("selected")), rtti=rtti, alternatives=len(alts)))
    return {"refs": refs, "models": len(models), "errors": errors, "seconds": time.perf_counter() - t0}


def rtti_text(values: list[dict]) -> str:
    parts = []
    for v in values:
        val = next((v[k] for k in v if k.startswith("val_")), None)
        parts.append(f"{v.get('name')}={format_value(val)}")
    return "; ".join(parts)


# ---- mesh material scan -------------------------------------------------------------------------------------------
def _fallback_mesh_materials(pack, logical_index: int) -> list[str]:
    """Material names in a mesh's material table (image + fixups only; vertex data is not touched)."""
    from ..mesh.decode import decode_parts
    res = pack.resource(logical_index)
    img = res.read_part_by_type(0x10)
    fx = res.read_part_by_type(0x11)
    if img is None or fx is None or len(img) < 16 or len(img) > MAX_MESH_IMAGE:
        return []
    model = decode_parts(res.name, bytes(img), bytes(fx))
    return [m.name_str for m in model.materials]


def mesh_materials_fn():
    """`gui.meshdata.mesh_materials` when that module exists, else the private decoder above."""
    try:
        from .meshdata import mesh_materials  # type: ignore[attr-defined]
        return mesh_materials
    except Exception:  # noqa: BLE001 - absent or broken: fall back
        return _fallback_mesh_materials


class MeshMaterialScanner(QObject):
    """Background scan: embedded material name -> mesh gids, over every indexed catalog pack.

    Incremental by pack (packs indexed after a scan are picked up by the next `start()`); cached in memory for the
    session; cancellable. Garbage meshes (decode errors) are counted and skipped."""
    progress = Signal(int, int)          # meshes done, meshes total
    finished = Signal(bool)              # True when every pack in the snapshot was scanned

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.catalog = catalog
        self._lock = threading.Lock()
        self.index: dict[str, list[int]] = {}
        self.scanned: set[int] = set()
        self.failed = 0                  # meshes without readable material names
        self.meshes = 0
        self.seconds = 0.0
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def pending_packs(self) -> list:
        return [e for e in list(self.catalog.packs) if e.pack is not None and e.id not in self.scanned
                and e.type_counts.get(MESH_TYPE)]

    def complete(self) -> bool:
        return not self.running() and not self.pending_packs() and bool(self.scanned or self.catalog.is_ready)

    def start(self) -> bool:
        if self.running():
            return False
        packs = self.pending_packs()
        if not packs:
            self.finished.emit(True)
            return False
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(packs,), daemon=True, name="sdb-mesh-scan")
        self._thread.start()
        return True

    def cancel(self) -> None:
        self._cancel.set()

    def wait(self, timeout: float = 60.0) -> bool:
        t = self._thread
        if t is not None:
            t.join(timeout)
        return not self.running()

    def lookup(self, material: str) -> list[int]:
        with self._lock:
            return list(self.index.get(material.casefold(), []))

    def _run(self, packs) -> None:
        fn = mesh_materials_fn()
        total = sum(e.type_counts.get(MESH_TYPE, 0) for e in packs)
        done = 0
        t0 = time.perf_counter()
        last = 0.0
        ok = True
        for e in packs:
            if self._cancel.is_set():
                ok = False
                break
            pk = e.pack
            local: dict[str, list[int]] = {}
            failed = 0
            try:
                idxs = [i for i, lg in enumerate(pk.logicals) if lg.type == MESH_TYPE]
            except Exception:  # noqa: BLE001 - pack closed underneath us
                idxs = []
            for i in idxs:
                if self._cancel.is_set():
                    break
                try:
                    names = fn(pk, i)
                except Exception:  # noqa: BLE001 - garbage / unsupported mesh
                    names = []
                if not names:
                    failed += 1                 # unreadable, or no material table
                gid = e.base + i
                for n in {x.casefold() for x in names if x}:
                    local.setdefault(n, []).append(gid)
                done += 1
                now = time.perf_counter()
                if now - last > 0.1:
                    last = now
                    self.progress.emit(done, total)
            if self._cancel.is_set():
                ok = False
                break
            with self._lock:
                for k, v in local.items():
                    self.index.setdefault(k, []).extend(v)
                self.scanned.add(e.id)
                self.failed += failed
                self.meshes += len(idxs)
        self.seconds += time.perf_counter() - t0
        self.progress.emit(done, total)
        self.finished.emit(ok)


# ---- validate job -------------------------------------------------------------------------------------------------
class _Cancelled(Exception):
    pass


class ValidateJob(QObject):
    """`Sdb.validate_all` on a daemon thread with progress and cancel (cancel raises from the progress hook)."""
    progress = Signal(str)
    done = Signal(object)                # result dict, or None when cancelled
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, sdb: Sdb) -> None:
        if self.running():
            return
        self._cancel.clear()
        self._thread = threading.Thread(target=self._run, args=(sdb,), daemon=True, name="sdb-validate")
        self._thread.start()

    def cancel(self) -> None:
        self._cancel.set()

    def _hook(self, msg: str) -> None:
        if self._cancel.is_set():
            raise _Cancelled()
        self.progress.emit(msg)

    def _run(self, sdb: Sdb) -> None:
        t0 = time.perf_counter()
        try:
            res = sdb.validate_all(progress=self._hook)
            res["seconds"] = time.perf_counter() - t0
            self.done.emit(res)
        except _Cancelled:
            self.done.emit(None)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(f"{type(exc).__name__}: {exc}")
