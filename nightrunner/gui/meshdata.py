"""Mesh geometry for the GUI viewers: decode a type-0x10 resource into per-submesh numpy buffers + a summary dict.

Pure data (numpy + nightrunner.mesh), no Qt: every function here is safe to call from a `TaskRunner` worker.

    geoms, info = load_mesh_geometry(pack, logical_index, lods=[0])
    names = mesh_materials(pack, logical_index)            # material table only, no vertex decode

Coordinates are the native engine ones (the same numbers `nr mesh export` writes to the Cast). Each submesh is
compacted to the vertices its indices use, so `len(g.positions)` equals the Cast mesh's vertex count and
`len(g.indices)` equals the submesh's stored index count.

"LOD" is the position of a geometry entry inside its class-6 array (multi-entry arrays are LOD chains,
notes/FORMATS/mesh.md); `lods=[0]` therefore keeps the most detailed entry of every part.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..container.rp6l import Pack

PART_NAMES = {0x10: "image", 0x11: "fixups", 0x12: "skin (_SKIN_ variants)", 0xF0: "vertex", 0xF1: "index",
              0xF3: "cloth"}
PART_FILES = {0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin",
              0xF3: "cloth.bin"}
FORMAT_NAMES = {0: "0 (16 B static, half)", 3: "3 (32 B static)", 6: "6 (40 B skinned)", 8: "8 (80 B skinned+ext)"}


@dataclass
class SubmeshGeom:
    """One drawable (geometry entry, submesh) with its own compacted vertex set."""
    key: str                            # unique within the mesh: "e<entry>/s<submesh>"
    entry: int                          # global geometry entry index
    submesh: int
    material: str                       # material name embedded in the mesh (class-11 table)
    positions: np.ndarray               # (N,3) float32, native coordinates
    indices: np.ndarray                 # (M,) uint32 triangle list into positions
    normals: np.ndarray | None          # (N,3) float32
    uv: np.ndarray | None               # (N,2) float32 (uv0)
    lod: int = 0                        # element position inside the class-6 array (0 = most detailed)
    material_slot: int = 0
    vertex_ids: np.ndarray | None = None    # (N,) index of each vertex inside the entry's vertex window


def _pack_of(pack) -> Pack:
    """Accept a `Pack` or anything with a `.pack` attribute (catalog PackEntry)."""
    pk = getattr(pack, "pack", None)
    return pk if isinstance(pk, Pack) else pack


def _compact(idx: np.ndarray, nv: int) -> tuple[np.ndarray, np.ndarray]:
    """(used vertex ids ascending, remapped uint32 indices) — same order as `np.unique(return_inverse)`."""
    mask = np.zeros(nv, dtype=bool)
    mask[idx] = True
    used = np.flatnonzero(mask)
    lut = np.cumsum(mask, dtype=np.int64) - 1
    return used, lut[idx].astype(np.uint32)


def _bounds(p: np.ndarray) -> list[list[float]] | None:
    if p is None or not len(p):
        return None
    finite = p[np.isfinite(p).all(axis=1)]
    if not len(finite):
        return None
    return [[float(x) for x in finite.min(axis=0)], [float(x) for x in finite.max(axis=0)]]


def part_list(pk: Pack, index: int) -> list[dict]:
    """Raw part table of one logical resource (no payload reads)."""
    res = pk.resource(index)
    out = []
    for k, i in enumerate(res.part_indices):
        t = pk.part_type(i)
        p = pk.physicals[i]
        try:
            direct = pk.part_is_direct(i)
        except Exception:  # noqa: BLE001
            direct = False
        out.append({"ordinal": k, "physical": i, "type": t, "type_name": PART_NAMES.get(t, f"part_{t:02X}"),
                    "size": int(p.size), "offset": int(pk.part_offset(i)), "direct": direct})
    return out


def decode_variants(skin: bytes | None, model=None) -> tuple[dict | None, str | None]:
    """(decoded dict, error) for part 0x12 via nightrunner.mesh.variants (optional module)."""
    if skin is None:
        return None, None
    try:
        from ..mesh import variants  # type: ignore[attr-defined]
    except Exception as exc:  # noqa: BLE001 - module is written by another component; absence is fine
        return None, f"variants decoder unavailable ({type(exc).__name__}: {exc})"
    try:
        return variants.decode(bytes(skin), model), None
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def skin_strings(data: bytes, min_len: int = 3) -> list[str]:
    """Printable ASCII runs of a raw part (fallback view of the variants table)."""
    return [m.decode("ascii") for m in re.findall(rb"[\x20-\x7e]{%d,}" % min_len, bytes(data))]


def load_mesh_geometry(pack, logical_index: int, lods: list[int] | None = None
                       ) -> tuple[list[SubmeshGeom], dict[str, Any]]:
    """Decode mesh *logical_index* of *pack* (a `Pack` or catalog `PackEntry`).

    Returns (geometry, info). Never raises for bad data: a resource that fails to decode yields ([], info) with
    ``info["error"]`` set. *lods* filters by LOD level (element position in the class-6 array); None = all.
    info keys: name, index, layout ("dltb" / "dl2", detected from the image), error, errors, warnings, parts, entities, entries, lod_count, formats, materials,
    bones, vertices, triangles, bounds, skinned, skin_size, cloth_size, variants, variants_error, skin_raw,
    decode_ms.
    """
    from ..mesh.decode import decode_resource
    t0 = time.perf_counter()
    pk = _pack_of(pack)
    info: dict[str, Any] = {"name": "", "index": int(logical_index), "error": None, "errors": [], "warnings": [],
                            "parts": [], "entities": 0, "entries": [], "lod_count": 0, "formats": [],
                            "materials": [], "bones": [], "vertices": 0, "triangles": 0, "bounds": None,
                            "skinned": False, "skin_size": None, "cloth_size": None, "variants": None,
                            "variants_error": None, "skin_raw": None, "decode_ms": 0.0}
    try:
        res = pk.resource(int(logical_index))
        info["name"] = res.name
        info["parts"] = part_list(pk, int(logical_index))
        if res.type != 0x10:
            raise ValueError(f"resource type 0x{res.type:02X} is not a mesh (0x10)")
        model = decode_resource(res)
    except Exception as exc:  # noqa: BLE001 - garbage data must come back as an error, not a crash
        info["error"] = f"{type(exc).__name__}: {exc}"
        info["decode_ms"] = (time.perf_counter() - t0) * 1000
        return [], info

    info["warnings"] = list(model.warnings)
    info["layout"] = model.layout
    info["entities"] = len(model.entities)
    info["embedded_name"] = model.embedded_name.decode("utf-8", "replace")
    info["materials"] = [m.name_str for m in model.materials]
    info["formats"] = sorted({e.format for e in model.geometry_entries})
    info["skinned"] = model.skinned
    info["skin_size"] = None if model.skin_raw is None else len(model.skin_raw)
    info["cloth_size"] = None if model.cloth_raw is None else len(model.cloth_raw)
    info["skin_raw"] = model.skin_raw
    info["bones"] = [{"index": en.index, "name": en.name.decode("utf-8", "replace"), "parent": int(en.parent),
                      "type": int(en.type), "flags": int(en.flags), "geometry_count": int(en.geometry_count),
                      "position": [float(x) for x in en.local[:, 3]]} for en in model.entities]
    info["variants"], info["variants_error"] = decode_variants(model.skin_raw, model)

    wanted = None if lods is None else {int(x) for x in lods}
    geoms: list[SubmeshGeom] = []
    lo = np.full(3, np.inf)
    hi = np.full(3, -np.inf)
    max_lod = -1
    for e in model.geometry_entries:
        max_lod = max(max_lod, e.element)
        owner = e.owner_entity
        erec: dict[str, Any] = {
            "index": e.index, "lod": e.element, "array_record": e.array_record, "owner_entity": owner,
            "owner_name": model.entities[owner].name.decode("utf-8", "replace") if owner is not None else None,
            "format": e.format, "vertex_count": e.vertex_count, "index_count": e.index_count,
            "decoded": e.vertices is not None, "loaded": wanted is None or e.element in wanted, "submeshes": []}
        info["entries"].append(erec)
        info["vertices"] += e.vertex_count
        v = e.vertices
        for s in e.submeshes:
            srec: dict[str, Any] = {"index": s.index, "key": f"e{e.index}/s{s.index}",
                                    "material": model.material_name(s.material_slot),
                                    "material_slot": s.material_slot, "indices": s.index_count,
                                    "triangles": s.triangle_count, "vertices": 0, "bones": len(s.palette),
                                    "bounds": None, "error": None}
            erec["submeshes"].append(srec)
            info["triangles"] += s.triangle_count
            if v is None:
                srec["error"] = f"vertex format {e.format} not decodable / no vertex buffer"
                continue
            if s.index_count == 0:
                srec["error"] = "empty submesh"
                continue
            try:
                idx = np.asarray(s.indices, dtype=np.int64)
                if int(idx.max()) >= v.count:
                    raise ValueError(f"index {int(idx.max())} >= vertex count {v.count}")
                used, faces = _compact(idx, v.count)
                if len(faces) % 3:
                    faces = faces[:len(faces) - len(faces) % 3]
                pos = np.ascontiguousarray(v.positions[used], dtype=np.float32)
            except Exception as exc:  # noqa: BLE001
                srec["error"] = f"{type(exc).__name__}: {exc}"
                info["errors"].append(f"{srec['key']}: {srec['error']}")
                continue
            srec["vertices"] = int(len(used))
            srec["bounds"] = _bounds(pos)
            if srec["bounds"]:
                lo = np.minimum(lo, srec["bounds"][0])
                hi = np.maximum(hi, srec["bounds"][1])
            if wanted is not None and e.element not in wanted:
                continue
            geoms.append(SubmeshGeom(
                key=srec["key"], entry=e.index, submesh=s.index, material=srec["material"], positions=pos,
                indices=faces, normals=np.ascontiguousarray(v.normals[used], dtype=np.float32),
                uv=np.ascontiguousarray(v.uv0[used], dtype=np.float32), lod=e.element,
                material_slot=s.material_slot, vertex_ids=used.astype(np.uint32)))
    info["lod_count"] = max_lod + 1
    if np.isfinite(lo).all():
        info["bounds"] = [lo.tolist(), hi.tolist()]
    info["decode_ms"] = (time.perf_counter() - t0) * 1000
    return geoms, info


def mesh_materials(pack, logical_index: int) -> list[str]:
    """Material names of a mesh's class-11 table (image + fixups parse only). [] when unreadable."""
    from ..classreader.fixups import Fixups
    from ..classreader.graph import MeshGraph
    from ..classreader.image import Image
    try:
        res = _pack_of(pack).resource(int(logical_index))
        if res.type != 0x10:
            return []
        img_b = res.read_part_by_type(0x10)
        fx_b = res.read_part_by_type(0x11)
        if img_b is None or fx_b is None:
            return []
        fx = Fixups.parse(bytes(fx_b))
        g = MeshGraph(Image(bytes(img_b), fx))
        out = []
        for mv in g.materials():
            nm = mv.name
            out.append("" if nm is None else nm.decode("utf-8", "surrogateescape"))
        return out
    except Exception:  # noqa: BLE001
        return []


# ---- export --------------------------------------------------------------------------------------------------

_CAST_LOCK = threading.Lock()
_CAST_HASH_BASE = 0x534E495752545250      # castlib's initial castHashBase
_BAD_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')


def safe_name(name: str) -> str:
    """A file-system safe version of a resource name (slashes become folders, Windows-invalid chars '_')."""
    s = _BAD_CHARS.sub("_", name.replace("\\", "/")).strip()
    parts = [p.strip(" .") or "_" for p in s.split("/") if p not in ("", ".", "..")]
    return "/".join(parts) or "_"


def export_cast_files(pack, logical_index: int, out: Path, sidecar: bool = True) -> dict:
    """Write `<out>` (Cast) and `<out stem>.mesh.json` exactly as `nr mesh export` does. Returns the report."""
    from ..cast.export import export_cast
    from ..mesh.decode import decode_resource
    from ..mesh.sidecar import build_sidecar
    from ..util.jsonio import dump_json
    pk = _pack_of(pack)
    r = pk.resource(int(logical_index))
    m = decode_resource(r)
    src = {"pack": str(pk.path), "index": r.index, "name": r.name}
    out = Path(out)
    from ..cast import castlib
    with _CAST_LOCK:
        # castlib numbers nodes from a process-global counter; restart it so the file is byte-identical to a fresh
        # `nr mesh export` (and to a re-export). Hashes only need to be unique within one file.
        saved = castlib.castHashBase
        castlib.castHashBase = _CAST_HASH_BASE
        try:
            rep = export_cast(m, out)
        finally:
            castlib.castHashBase = max(saved, castlib.castHashBase)
    result = rep.to_json()
    if sidecar:
        side = build_sidecar(m, cast={"file": out.name, "nodes": rep.nodes, "skipped": rep.skipped,
                                      "meshes": [{"name": x.name, "entry": x.entry, "submesh": x.submesh}
                                                 for x in rep.meshes]},
                             source=src)
        sp = out.with_name(out.stem + ".mesh.json")
        dump_json(side, sp)
        result["sidecar"] = str(sp)
    result["warnings"] = list(m.warnings)
    return result


def export_raw_parts(pack, logical_index: int, out_dir: Path) -> list[Path]:
    """Write every part of the resource as `<type name>.bin` into *out_dir*."""
    pk = _pack_of(pack)
    r = pk.resource(int(logical_index))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    seen: dict[str, int] = {}
    for k, i in enumerate(r.part_indices):
        t = pk.part_type(i)
        fn = PART_FILES.get(t, f"part_{t:02X}.bin")
        if fn in seen:
            fn = f"{Path(fn).stem}_{k}.bin"
        seen[fn] = k
        dest = out_dir / fn
        tmp = dest.with_name(dest.name + ".partial")
        data = pk.read_part(i)
        try:
            tmp.write_bytes(data)
            tmp.replace(dest)
        finally:
            if isinstance(data, memoryview):
                data.release()
            tmp.unlink(missing_ok=True)
        written.append(dest)
    return written
