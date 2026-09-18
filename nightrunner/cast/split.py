"""Split an edited single-file model Cast back into per-mesh `model.cast` files that `nr build` accepts.

Input: the Cast written by the Models tab "Single Cast" export (usually re-exported from Blender with the official
Cast add-on) and its `<model>.cast.json` report. Output, per source mesh, a folder shaped like an `.rpx` mesh
resource: `model.cast`, `mesh.json`, the raw parts — plus `split_report.json`.

For every source mesh:

1. The mesh is decoded again from the pack recorded in the report, and its **original** per-mesh Cast is built
   (`cast/export.build_cast`) — the template: skeleton = that mesh's own entities, every LOD, every bp_* property.
2. The merged skeleton of the export is rebuilt (`cast/assemble.py`, same order; checked against the report) and the
   edited meshes are mapped back by name (`<slot>.<mesh>.e<entry>.s<sub>`, Blender `.NNN` suffixes tolerated).
3. **Un-rebind**: bones are matched by name to the mesh's entities; each vertex is moved back from the merged rest
   pose with the inverse of `Σ w · G_skel[b] · inv_bind_part[b]` (the exporter applied the forward matrix).
   Normals use the inverse of the same 3×3.
4. **Snap**: an edited vertex within `tol` of an original vertex of that submesh (and within `uv_tol` of its UV0)
   takes the original record's exact position / normal / tangent / UVs, and its original weights when they agree
   to 1/255 (the Blender round trip averages normals over faces and reorders weight lanes; snapping keeps
   untouched areas byte-identical). Moved vertices keep the edited values; their tangent frame comes from the
   nearest original vertex re-orthogonalised to the new normal.
5. The template's LOD-0 submesh nodes get the edited buffers. When every vertex snapped to a distinct original
   id, `bp_vertex_id` is written (in-place encode); otherwise it is dropped and the encoder recovers ids by exact
   match (snapped vertices are exact).
6. The result is checked with the mesh encoder (`mesh.codec.rebuild_from_files`) and the outcome is reported.

Refused (named in the report, other meshes still written): a weight on a bone the source mesh does not have
(bone add is unsupported by the encoder), more than 4 influences after pruning, a mesh that maps to no source.
Bone transform edits are ignored: the per-mesh skeleton is always the native one.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..container.rp6l import Pack
from ..mesh.decode import decode_resource
from ..mesh.sidecar import build_sidecar
from ..util.jsonio import dump_json
from . import castlib
from .assemble import MergedSkeleton, rebind_matrices
from .export import MAX_INFLUENCES, build_cast, cast_text
from ..util.schema import matches as schema_matches

FORMAT = "nightrunner.model_cast/1"
PART_FILES = {0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin",
              0xF3: "cloth.bin"}
_SUFFIX = re.compile(r"\.\d{3}$")
MOVE_RADIUS = 0.5          # a moved vertex is matched to a same-UV original at most this far away (engine units)


class SplitError(Exception):
    pass


def _vals(node, key):
    p = node.properties.get(key)
    return None if p is None else list(p.values)


def _arr(v, width, dtype=np.float64):
    return None if v is None else np.asarray(v, dtype=dtype).reshape(-1, width)


def _edited_meshes(cast) -> tuple[list[str], list[dict]]:
    root = cast.Roots()[0]
    models = [c for c in root.ChildrenOfType(castlib.Model)]
    if not models:
        raise SplitError("the Cast has no Model node")
    bones: list[str] = []
    meshes: list[dict] = []
    for mdl in models:
        sk = mdl.Skeleton()
        base = len(bones)
        if sk is not None:
            bones += [b.Name() or "" for b in sk.Bones()]
        mats = {x.Hash(): x.Name() for x in mdl.Materials()}
        for m in mdl.Meshes():
            n = len(m.VertexPositionBuffer() or []) // 3
            layers = m.UVLayerCount() or 0
            mi = m.MaximumWeightInfluence() or 0
            wb = _vals(m, "wb")
            wv = _vals(m, "wv")
            meshes.append({
                "name": m.Name() or "",
                "positions": _arr(m.VertexPositionBuffer(), 3),
                "normals": _arr(m.VertexNormalBuffer(), 3),
                "uv0": _arr(m.VertexUVLayerBuffer(0), 2) if layers > 0 else None,
                "uv1": _arr(m.VertexUVLayerBuffer(1), 2) if layers > 1 else None,
                "faces": np.asarray(m.FaceBuffer() or [], dtype=np.int64),
                "bones": None if not (mi and wb) else np.asarray(wb, dtype=np.int64).reshape(n, mi) + base,
                "weights": None if not (mi and wv) else np.asarray(wv, dtype=np.float64).reshape(n, mi),
                "material": mats.get((_vals(m, "m") or [None])[0]),
            })
    return bones, meshes


def _top4(bones: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the 4 largest influences per vertex, renormalised. Returns (bones, weights, vertices pruned)."""
    n, k = weights.shape
    if k < MAX_INFLUENCES:
        bones = np.pad(bones, ((0, 0), (0, MAX_INFLUENCES - k)))
        weights = np.pad(weights, ((0, 0), (0, MAX_INFLUENCES - k)))
        k = MAX_INFLUENCES
    order = np.argsort(-weights, axis=1, kind="stable")[:, :MAX_INFLUENCES]
    pruned = int(((weights > 0).sum(axis=1) > MAX_INFLUENCES).sum())
    b = np.take_along_axis(bones, order, axis=1)
    w = np.take_along_axis(weights, order, axis=1)
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 0, w / np.maximum(s, 1e-12), w)
    return b, w, pruned


def _node_buffers(mesh_node) -> dict[str, Any]:
    g = mesh_node.properties
    n = len(g["vp"].values) // 3
    mi = g["mi"].values[0] if "mi" in g else 0
    out = {"vp": np.asarray(g["vp"].values, dtype=np.float64).reshape(n, 3),
           "vn": np.asarray(g["vn"].values, dtype=np.float64).reshape(n, 3),
           "vt": np.asarray(g["vt"].values, dtype=np.float64).reshape(n, 3) if "vt" in g else None,
           "u0": np.asarray(g["u0"].values, dtype=np.float64).reshape(n, 2),
           "u1": np.asarray(g["u1"].values, dtype=np.float64).reshape(n, 2) if "u1" in g else None,
           "wb": np.asarray(g["wb"].values, dtype=np.int64).reshape(n, mi) if mi and "wb" in g else None,
           "wv": np.asarray(g["wv"].values, dtype=np.float64).reshape(n, mi) if mi and "wv" in g else None,
           "ids": np.asarray(g["bp_vertex_id"].values, dtype=np.int64) if "bp_vertex_id" in g else None,
           "sign": np.asarray(g["bp_tangent_sign"].values, dtype=np.float64) if "bp_tangent_sign" in g else None}
    return out


def _set(node, key, kind, values) -> None:
    node.CreateProperty(key, kind).values = list(values)


def _drop(node, key) -> None:
    node.properties.pop(key, None)


def _merge_ems(name: str, ems: list[dict]) -> dict:
    """Several edited objects -> one submesh (concatenated; weight lanes padded to the widest)."""
    if len(ems) == 1:
        return dict(ems[0], name=name)
    out: dict[str, Any] = {"name": name, "material": ems[0].get("material")}
    for k in ("positions", "normals", "uv0", "uv1"):
        parts = [e.get(k) for e in ems]
        if any(p is None for p in parts):
            if k in ("uv0", "positions"):
                raise SplitError(f"{name}: an object has no {k}")
            out[k] = None
        else:
            out[k] = np.concatenate(parts)
    faces, base = [], 0
    for e in ems:
        faces.append(np.asarray(e["faces"], dtype=np.int64) + base)
        base += len(e["positions"])
    out["faces"] = np.concatenate(faces)
    if all(e.get("weights") is None for e in ems):
        out["bones"] = out["weights"] = None
    else:
        if any(e.get("weights") is None for e in ems):
            raise SplitError(f"{name}: some objects are skinned, some are not")
        k = max(e["weights"].shape[1] for e in ems)
        out["bones"] = np.concatenate([np.pad(e["bones"], ((0, 0), (0, k - e["bones"].shape[1]))) for e in ems])
        out["weights"] = np.concatenate([np.pad(e["weights"], ((0, 0), (0, k - e["weights"].shape[1]))) for e in ems])
    return out


def _double_sided(em: dict) -> dict:
    """Append a back side: every vertex again with the normal negated, every triangle again reversed."""
    n = len(em["positions"])
    out = dict(em)
    out["positions"] = np.concatenate([em["positions"], em["positions"]])
    if em.get("normals") is not None:
        out["normals"] = np.concatenate([em["normals"], -em["normals"]])
    for k in ("uv0", "uv1", "bones", "weights"):
        if em.get(k) is not None:
            out[k] = np.concatenate([em[k], em[k]])
    f = np.asarray(em["faces"], dtype=np.int64).reshape(-1, 3)
    out["faces"] = np.concatenate([f, f[:, ::-1] + n]).reshape(-1)
    return out


def _hidden_em(name: str, ebones: list[str], ents: dict[str, int]) -> dict:
    """One vertex + one degenerate triangle: the submesh draws nothing (a submesh cannot be removed)."""
    bone = next((i for i, b in enumerate(ebones) if b.lower() in ents), None)
    return {"name": name, "positions": np.zeros((1, 3)), "normals": np.array([[0.0, 1.0, 0.0]]),
            "uv0": np.zeros((1, 2)), "uv1": None, "faces": np.zeros(3, dtype=np.int64),
            "bones": None if bone is None else np.array([[bone]], dtype=np.int64),
            "weights": None if bone is None else np.ones((1, 1)), "material": None}


def _lod_twins(model, entry: int, submesh: int) -> list[tuple[int, int]]:
    """(entry, submesh) of the other LOD levels of the same entity drawing the same material slot."""
    ges = model.geometry_entries
    if not 0 <= entry < len(ges) or not 0 <= submesh < len(ges[entry].submeshes):
        return []
    ge = ges[entry]
    slot = ge.submeshes[submesh].material_slot
    out = []
    for g in ges:
        if g.index == entry or g.owner_entity != ge.owner_entity or g.element == ge.element:
            continue
        for sm in g.submeshes:
            if sm.material_slot == slot:
                out.append((g.index, sm.index))
                break
    return out


def split_model_cast(edited: Path | str, report: Path | str | dict, out_dir: Path | str, *, tol: float = 1e-4,
                     uv_tol: float = 1e-4, check: bool = True, pack_paths: dict[str, str] | None = None,
                     rpx: Path | None = None, assign: dict[str, str] | None = None,
                     hide: list[str] | None = None, lods: bool = True,
                     double_sided: list[str] | None = None) -> dict:
    """Split *edited* (a Cast) using the export *report* (`<model>.cast.json` path or dict) into *out_dir*.
    *pack_paths* maps a recorded pack path to its current location (when the packs moved). With *rpx* (an extracted
    `.rpx` tree) each applied, encoder-checked model.cast is also copied into the tree's mesh folder of the same
    logical name (the tree's own file is kept once as `model.cast.orig`), ready for `nr build`."""
    rep = report if isinstance(report, dict) else json.loads(Path(report).read_text(encoding="utf-8"))
    if not schema_matches(rep.get("format"), FORMAT):
        raise SplitError("the report is not a nightrunner single-Cast export report (format "
                         f"{rep.get('format')!r}); re-export the model with the current explorer")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    remap = pack_paths or {}
    packs: dict[str, Pack] = {}

    def pack(path: str) -> Pack:
        p = remap.get(path, path)
        if p not in packs:
            packs[p] = Pack.open(p)
        return packs[p]

    result: dict[str, Any] = {"edited": str(edited), "model": rep.get("model"), "meshes": [], "problems": [],
                              "unmatched_meshes": []}
    try:
        # ---- merged skeleton, rebuilt exactly as exported ------------------------------------------------------
        skel = MergedSkeleton()
        sk = rep.get("skeleton") or {}
        models = []
        for pr in rep["parts"]:
            models.append(decode_resource(pack(pr["pack_path"]).resource(int(pr["index"]))))
        if sk.get("pack_path") and not sk.get("from_part"):
            skel.add_model(decode_resource(pack(sk["pack_path"]).resource(int(sk["index"]))), sk.get("name") or "")
        elif sk.get("from_part"):
            first = next(i for i, pr in enumerate(rep["parts"])
                         if pr["pack_path"] == sk["pack_path"] and int(pr["index"]) == int(sk["index"]))
            skel.add_model(models[first], rep["parts"][first]["mesh"])
        emaps = [skel.map_model(m, pr["mesh"]) for m, pr in zip(models, rep["parts"])]
        if [cast_text(n) for n in skel.names] != rep.get("bones"):
            raise SplitError("the merged skeleton no longer matches the export (packs changed since the export?)")

        from .gltf import load_scene
        edited_cast = load_scene(edited)
        ebones, emeshes = _edited_meshes(edited_cast)
        assign = dict(assign or {})
        hide_set = set(hide or [])
        two = set(double_sided or [])
        for i, m in enumerate(emeshes):
            if m["name"] in two or _SUFFIX.sub("", m["name"]) in two:
                emeshes[i] = _double_sided(m)
        by_name = {}
        grouped: dict[str, list[dict]] = {}
        for m in emeshes:
            key = m["name"] if m["name"] in assign else _SUFFIX.sub("", m["name"])
            if key in assign:
                m["_used"] = True
                if assign[key]:
                    grouped.setdefault(assign[key], []).append(m)
                continue
            by_name.setdefault(m["name"], m)
        for m in emeshes:
            if not m.get("_used"):
                by_name.setdefault(_SUFFIX.sub("", m["name"]), m)
                if _SUFFIX.sub("", m["name"]) in hide_set:
                    m["_used"] = True          # the original object of a hidden submesh
        replaced = set()
        for target, ems in grouped.items():
            if target in hide_set:
                raise SplitError(f"{target}: both assigned and hidden")
            merged = _merge_ems(target, ems)
            merged["material"] = None     # a replacement keeps the target's game material (textures via rttiValues)
            by_name[target] = merged
            replaced.add(target)
        known = {x["name"] for x in rep["mesh_map"]}
        bad = sorted((set(grouped) | hide_set) - known)
        if bad:
            raise SplitError("unknown target submesh: " + ", ".join(bad[:5]))

        for pi, (model, pr) in enumerate(zip(models, rep["parts"])):
            entry = {"mesh": pr["mesh"], "logical_name": model.name, "dir": None, "submeshes": [],
                     "problems": [], "check": None}
            result["meshes"].append(entry)
            ents = {e.name.decode("utf-8", "replace").lower(): e.index for e in model.entities}
            R, _ = rebind_matrices(model, skel, emaps[pi])
            template, crep = build_cast(model)
            tmesh = {x.Name(): x for x in template.Roots()[0].ChildOfType(castlib.Model).Meshes()}
            ok_any = False
            jobs = []
            for mm in (x for x in rep["mesh_map"] if x["part"] == pi):
                sub = {"name": mm["name"], "entry": mm["entry"], "submesh": mm["submesh"]}
                entry["submeshes"].append(sub)
                if mm["name"] in hide_set:
                    em, whole = _hidden_em(mm["name"], ebones, ents), True
                    sub["hidden"] = True
                else:
                    em, whole = by_name.get(mm["name"]), mm["name"] in replaced
                if em is None:
                    sub["status"] = "not in the edited Cast (kept unchanged)"
                    continue
                em["_used"] = True
                if whole:
                    sub["replaced"] = not sub.get("hidden")
                jobs.append((sub, em, mm))
                if whole and lods:
                    for e2, s2 in _lod_twins(model, int(mm["entry"]), int(mm["submesh"])):
                        name2 = f"{mm['name']}>lod e{e2}.s{s2}"
                        sub2 = {"name": name2, "entry": e2, "submesh": s2, "lod_of": mm["name"],
                                "hidden": sub.get("hidden", False)}
                        entry["submeshes"].append(sub2)
                        jobs.append((sub2, dict(em, name=name2), dict(mm, entry=e2, submesh=s2)))
            for sub, em, mm in jobs:
                node = tmesh.get(f"{cast_text(model.name)}.e{mm['entry']}.s{mm['submesh']}")
                if node is None:
                    sub["status"] = "no template mesh (skipped at export)"
                    continue
                try:
                    stats = _apply(node, em, ebones, ents, R, model, mm, tol, uv_tol,
                                   template_materials=template.Roots()[0].ChildOfType(castlib.Model))
                    sub.update(stats)
                    sub["status"] = "applied"
                    ok_any = True
                except SplitError as exc:
                    sub["status"] = f"REFUSED: {exc}"
                    entry["problems"].append(f"{mm['name']}: {exc}")
            d = out_dir / re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", pr["mesh"].removesuffix(".msh")).strip(" .")
            d.mkdir(parents=True, exist_ok=True)
            entry["dir"] = str(d)
            template.save(str(d / "model.cast"))
            res = pack(pr["pack_path"]).resource(int(pr["index"]))
            side = build_sidecar(model, cast={"file": "model.cast", "nodes": crep.nodes, "skipped": crep.skipped,
                                              "meshes": [{"name": x.name, "entry": x.entry, "submesh": x.submesh}
                                                         for x in crep.meshes]},
                                 source={"pack": str(pack(pr["pack_path"]).path), "index": res.index,
                                         "name": res.name})
            dump_json(side, d / "mesh.json")
            parts = {}
            for i in res.part_indices:
                t = res.pack.part_type(i)
                if t in PART_FILES:
                    data = bytes(res.pack.read_part(i))
                    (d / PART_FILES[t]).write_bytes(data)
                    parts[t] = data
            if check and ok_any:
                try:
                    from ..mesh.codec import rebuild_from_files
                    rb = rebuild_from_files(d / "model.cast", d / "mesh.json", parts, res.name_raw,
                                            ignore_bone_changes=True)
                    entry["check"] = {"ok": True, "entries": [x.get("path") for x in rb.report.get("entries", [])],
                                      "warnings": rb.report.get("warnings", [])[:20],
                                      "unchanged": (rb.image == parts.get(0x10) and rb.vertex == parts.get(0xF0)
                                                    and rb.index == parts.get(0xF1))}
                except Exception as exc:  # noqa: BLE001
                    entry["check"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                    entry["problems"].append(f"encoder check failed: {exc}")
        if rpx is not None:
            _install_into_rpx(Path(rpx), result)
        result["unmatched_meshes"] = [m["name"] for m in emeshes if not m.get("_used")]
        if result["unmatched_meshes"]:
            result["problems"].append("objects with no target (assign or skip them): "
                                      + ", ".join(result["unmatched_meshes"][:10]))
    finally:
        for p in packs.values():
            p.close()
    dump_json(result, out_dir / "split_report.json")
    return result


def _apply(node, em: dict, ebones: list[str], ents: dict[str, int], R: np.ndarray, model, mm: dict,
           tol: float, uv_tol: float, template_materials=None) -> dict:
    orig = _node_buffers(node)
    if em["uv0"] is None or em["normals"] is None or len(em["faces"]) == 0:
        raise SplitError("mesh has no UVs / normals / faces")
    em = _drop_loose(em)
    pos = em["positions"]
    n = len(pos)
    ge = model.geometry_entries[mm["entry"]]
    skinned = ge.vertices is not None and ge.vertices.skinned
    stats: dict[str, Any] = {"vertices": n, "original_vertices": len(orig["vp"]), "faces": len(em["faces"]) // 3,
                             "loose_dropped": em.get("loose_dropped", 0)}
    em_mat = _SUFFIX.sub("", em.get("material") or "")
    if em_mat and template_materials is not None and em_mat != mm.get("material"):
        # a different material than the one exported (single-Cast names may carry '#<override hash>')
        _assign_material(node, em_mat.split("#", 1)[0], template_materials, stats)

    nrm = em["normals"]
    bones = weights = None
    if skinned:
        if em["weights"] is None:
            raise SplitError("skinned mesh lost its weights")
        wb, wv, pruned = _top4(em["bones"], em["weights"])
        stats["pruned_influences"] = pruned
        part_bones = np.zeros_like(wb)
        active = wv > 0
        for b in np.unique(wb[active]):
            name = ebones[int(b)].lower() if 0 <= int(b) < len(ebones) else None
            if name is None or name not in ents:
                raise SplitError(f"weights on bone {ebones[int(b)] if name else b!r} that "
                                 f"{model.name!r} does not have")
            part_bones[wb == b] = ents[name]
        part_bones = np.where(active, part_bones, 0)
        blend = np.einsum("nk,nkij->nij", wv, R[part_bones])
        inv = np.linalg.inv(blend)
        pos = np.einsum("nij,nj->ni", inv[:, :3, :3], pos) + inv[:, :3, 3]
        nrm = np.einsum("nij,nj->ni", inv[:, :3, :3], nrm)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        bones, weights = part_bones, wv
    uv0 = em["uv0"]
    # ---- snap -----------------------------------------------------------------------------------------------
    snap = np.full(n, -1, dtype=np.int64)
    n_orig = len(orig["vp"])
    ou0 = orig["u0"]
    if n == n_orig:
        # Blender keeps vertex order when it had no UV seam to split (the Cast already carries split vertices)
        same = ((np.abs(orig["vp"] - pos).max(axis=1) <= tol) & (_uv_dist(ou0, uv0) <= uv_tol)
                & (np.abs(orig["vn"] - nrm).max(axis=1) <= 1e-2))
        snap[same] = np.flatnonzero(same)
    claimed = np.zeros(n_orig, dtype=bool)
    claimed[snap[snap >= 0]] = True
    for t in sorted({min(tol, 1e-6), tol}):
        # tight pass first: a near neighbour must not steal an original from its exact (float-noise) match
        rest = np.flatnonzero(snap < 0)
        if not len(rest):
            break
        cell = max(t * 4, 1e-9)
        grid: dict[tuple, list[int]] = {}
        for j in np.flatnonzero(~claimed):
            grid.setdefault(tuple(np.floor(orig["vp"][j] / cell).astype(np.int64)), []).append(int(j))
        keys = np.floor(pos / cell).astype(np.int64)
        offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
        for i in rest:
            k = keys[i]
            best, bd = -1, np.inf
            for dx, dy, dz in offsets:
                for j in grid.get((k[0] + dx, k[1] + dy, k[2] + dz), ()):
                    if claimed[j]:
                        continue
                    d = float(np.abs(orig["vp"][j] - pos[i]).max())
                    du = float(_uv_dist(ou0[j:j + 1], uv0[i:i + 1])[0])
                    if d <= t and du <= uv_tol:
                        score = d + du * 10 + float(np.abs(orig["vn"][j] - nrm[i]).sum())
                        if score < bd:
                            best, bd = j, score
            if best >= 0:
                snap[i] = best
                claimed[best] = True
    dup = np.full(n, -1, dtype=np.int64)            # extra copies of an original vertex (Blender normal splits)
    left = np.flatnonzero(snap < 0)
    if len(left):
        cell = max(tol * 4, 1e-9)
        grid2: dict[tuple, list[int]] = {}
        for j, k in enumerate(map(tuple, np.floor(orig["vp"] / cell).astype(np.int64))):
            grid2.setdefault(k, []).append(j)
        for i in left:
            k = np.floor(pos[i] / cell).astype(np.int64)
            for j in grid2.get(tuple(k), ()):
                if float(np.abs(orig["vp"][j] - pos[i]).max()) <= tol and \
                        float(_uv_dist(ou0[j:j + 1], uv0[i:i + 1])[0]) <= uv_tol:
                    dup[i] = j
                    break
    # moved vertices: UVs survive a move, so an unclaimed original with the same UV0 (nearest in space) is taken
    uvmatch = np.full(n, -1, dtype=np.int64)
    left = np.flatnonzero((snap < 0) & (dup < 0))
    if len(left):
        free = np.flatnonzero(~claimed)
        if len(free):
            q = max(uv_tol, 1e-9)
            buckets: dict[tuple, list[int]] = {}
            fu = np.nan_to_num(ou0[free], nan=9e9, posinf=9e9, neginf=-9e9)
            for j, k in zip(free.tolist(), map(tuple, np.round(fu / q).astype(np.int64))):
                buckets.setdefault(k, []).append(j)
            for i in left:
                cand = [j for j in buckets.get(tuple(np.round(uv0[i] / q).astype(np.int64)), ()) if not claimed[j]]
                if not cand:
                    continue
                c = np.asarray(cand)
                dist = np.linalg.norm(orig["vp"][c] - pos[i], axis=1)
                j = int(c[int(np.argmin(dist))])
                if float(dist.min()) <= MOVE_RADIUS:
                    uvmatch[i] = j
                    claimed[j] = True
    faces = em["faces"].reshape(-1, 3)
    if (dup >= 0).any():
        # copies of an already-claimed original (Blender split them by normal): fold them into the claimer
        owner = {int(j): i for i, j in enumerate(snap.tolist()) if j >= 0}
        fold = np.arange(n)
        for i in np.flatnonzero(dup >= 0):
            if int(dup[i]) in owner:
                fold[i] = owner[int(dup[i])]
    else:
        fold = np.arange(n)
    rest2 = np.flatnonzero((snap < 0) & (uvmatch < 0) & (dup < 0))
    if len(rest2):
        # split copies of a moved vertex: same edited position + UV as a vertex that was identified
        ided = np.flatnonzero((snap >= 0) | (uvmatch >= 0))
        cell = max(tol * 4, 1e-9)
        g3: dict[tuple, list[int]] = {}
        for i2, k in zip(ided.tolist(), map(tuple, np.floor(pos[ided] / cell).astype(np.int64))):
            g3.setdefault(k, []).append(i2)
        for i in rest2:
            for i2 in g3.get(tuple(np.floor(pos[i] / cell).astype(np.int64)), ()):
                if float(np.abs(pos[i2] - pos[i]).max()) <= tol and float(np.abs(uv0[i2] - uv0[i]).max()) <= uv_tol:
                    fold[i] = i2
                    break
    if (fold != np.arange(n)).any():
        drop = fold != np.arange(n)
        if drop.any():
            keep = ~drop
            lut = np.cumsum(keep) - 1
            faces = lut[fold[faces]]
            pos, nrm, uv0 = pos[keep], nrm[keep], uv0[keep]
            if skinned:
                bones, weights = bones[keep], weights[keep]
            if em.get("uv1") is not None and len(em["uv1"]) == n:
                em = dict(em, uv1=em["uv1"][keep])
            snap, dup, uvmatch = snap[keep], dup[keep], uvmatch[keep]
            em = dict(em, faces=faces.reshape(-1))
            stats["folded_duplicates"] = int(drop.sum())
            n = int(keep.sum())
            stats["vertices"] = n
    exact = snap >= 0
    if exact.all() and not (stats.get("folded_duplicates") is None and n == n_orig
                            and _same_triangles(snap[faces], np.asarray(node.FaceBuffer(), dtype=np.int64).reshape(-1, 3))):
        kept = _unchanged_geometry(node, orig, snap, faces, em, bones, weights, skinned, stats)
        if kept:
            return stats
    ofaces = np.asarray(node.FaceBuffer(), dtype=np.int64).reshape(-1, 3)
    if n == n_orig and _same_triangles(faces, ofaces):
        ident = np.arange(n, dtype=np.int64)          # same order, same triangles: vertex i is original i
    else:
        ident = _complete_by_topology(np.where(snap >= 0, snap, uvmatch), faces, ofaces)
    has = ident >= 0
    reorder = (n == n_orig and bool(has.all()) and len(np.unique(ident)) == n
               and _same_triangles(ident[faces], ofaces))
    if not reorder and n == n_orig and bool(has.all()) and len(faces) == len(ofaces):
        # exact snaps may have picked the wrong one of several identical originals: re-pick through the triangles
        alt = _resolve_by_triangles(ident, faces, ofaces, orig)
        if alt is not None:
            ident, reorder = alt, True
    near = ident.copy()
    if (~has).any():                                  # nearest original (by position) for new vertices
        for i in np.flatnonzero(~has):
            near[i] = int(np.argmin(np.linalg.norm(orig["vp"] - pos[i], axis=1)))
    src = near
    exact_or_dup = exact | (~has & (dup >= 0))
    src = np.where(~has & (dup >= 0), dup, src)
    out_pos = np.where(exact_or_dup[:, None], orig["vp"][src], pos)
    out_nrm = np.where(exact_or_dup[:, None], orig["vn"][src], nrm)
    out_uv0 = np.where(exact_or_dup[:, None], orig["u0"][src], uv0)
    out_uv1 = None
    if orig["u1"] is not None:
        eu1 = em["uv1"] if em["uv1"] is not None and len(em["uv1"]) == n else None
        out_uv1 = np.where(exact_or_dup[:, None], orig["u1"][src], eu1 if eu1 is not None else orig["u1"][src])
    out_tan = None
    if orig["vt"] is not None:
        t = orig["vt"][src].copy()
        moved = ~exact_or_dup
        if moved.any():
            t[moved] -= out_nrm[moved] * np.sum(t[moved] * out_nrm[moved], axis=1, keepdims=True)
            t[moved] /= np.maximum(np.linalg.norm(t[moved], axis=1, keepdims=True), 1e-12)
        out_tan = t
    if skinned:
        ob, ow = orig["wb"], orig["wv"]
        same = np.zeros(n, dtype=bool)
        if ob is not None and has.any():
            for i in np.flatnonzero(has):
                a_ = {int(b_): float(w) for b_, w in zip(ob[src[i]], ow[src[i]]) if w > 0}
                c_ = {int(b_): float(w) for b_, w in zip(bones[i], weights[i]) if w > 0}
                same[i] = a_.keys() == c_.keys() and all(abs(a_[k] - c_[k]) <= 0.6 / 255 for k in a_)
            bones = np.where(same[:, None], ob[src], bones)
            weights = np.where(same[:, None], ow[src], weights)
        stats["weights_kept"] = int(same.sum())
    stats["snapped"] = int(exact.sum())
    stats["moved"] = int((has & ~exact).sum())
    stats["new"] = int((~has).sum())
    stats["duplicates"] = int((~has & (dup >= 0)).sum())
    stats["in_place"] = reorder
    ids = None
    if reorder:                                       # same vertices and triangles: back to the template order
        perm = np.empty(n, dtype=np.int64)
        perm[ident] = np.arange(n)
        out_pos, out_nrm, out_uv0 = out_pos[perm], out_nrm[perm], out_uv0[perm]
        out_uv1 = None if out_uv1 is None else out_uv1[perm]
        out_tan = None if out_tan is None else out_tan[perm]
        if skinned:
            bones, weights = bones[perm], weights[perm]
        src = np.arange(n)
        ids = orig["ids"]
        em = dict(em, faces=None)
    # ---- write into the template node ------------------------------------------------------------------------
    node.SetVertexPositionBuffer(out_pos.astype(np.float32).tolist())
    node.SetVertexNormalBuffer(out_nrm.astype(np.float32).tolist())
    if out_tan is not None:
        node.SetVertexTangentBuffer(out_tan.astype(np.float32).tolist())
    node.SetVertexUVLayerBuffer(0, out_uv0.astype(np.float32).tolist())
    if out_uv1 is not None:
        node.SetVertexUVLayerBuffer(1, out_uv1.astype(np.float32).tolist())
    if em["faces"] is not None:
        node.SetFaceBuffer(em["faces"].astype(np.int64).tolist())
    if skinned:
        node.SetMaximumWeightInfluence(MAX_INFLUENCES)
        node.SetVertexWeightBoneBuffer(bones.reshape(-1).astype(np.int64).tolist())
        node.SetVertexWeightValueBuffer(weights.reshape(-1).astype(np.float64).tolist())
    if orig["sign"] is not None:
        _set(node, "bp_tangent_sign", "f", orig["sign"][src].astype(np.float64).tolist())
    if ids is not None:
        _set(node, "bp_vertex_id", "i", [int(x) for x in ids])
    else:
        _drop(node, "bp_vertex_id")
    return stats


def _assign_material(node, name: str, mdl, stats: dict) -> None:
    """Point the template mesh at material *name* (existing by exact / case-insensitive name, else a new one)."""
    mats = list(mdl.Materials())
    cur = next((m for m in mats if m.Hash() == (_vals(node, "m") or [None])[0]), None)
    if cur is not None and (cur.Name() or "") == name:
        return
    base = re.sub(r"\.\d{3}$", "", name)
    hit = next((m for m in mats if (m.Name() or "") in (name, base)), None) or \
        next((m for m in mats if (m.Name() or "").lower() in (name.lower(), base.lower())), None)
    if hit is None:
        hit = mdl.CreateMaterial()
        hit.SetName(base)
        hit.SetType("pbr")
    node.SetMaterial(hit.Hash())
    stats["material"] = hit.Name()


def normalize_mesh_scene(edited, model, *, tol: float = 1e-4, uv_tol: float = 1e-4) -> tuple["castlib.Cast", list[str]]:
    """Per-mesh edit (e.g. a glTF round trip through Blender) → the mesh's own template Cast with the edited
    geometry applied the same way the model splitter does (identity rebind: the file is in the mesh's bind space).
    Meshes of the template the edit does not contain are removed (submesh / LOD removal is then judged by the
    encoder); edited meshes the template does not know are appended unchanged (new submeshes)."""
    template, _ = build_cast(model)
    tm = template.Roots()[0].ChildOfType(castlib.Model)
    ebones, emeshes = _edited_meshes(edited)
    ents = {e.name.decode("utf-8", "replace").lower(): e.index for e in model.entities}
    R = np.tile(np.eye(4), (max(len(model.entities), 1), 1, 1))
    by_name = {}
    for m in emeshes:
        by_name.setdefault(_SUFFIX.sub("", m["name"]), m)
    notes: list[str] = []
    for node in list(tm.Meshes()):
        em = by_name.pop(node.Name(), None)
        if em is None:
            tm.childNodes.remove(node)
            notes.append(f"{node.Name()}: not in the edit (removed)")
            continue
        props = node.properties
        mm = {"entry": int(props["bp_entry"].values[0]), "name": node.Name()}
        try:
            st = _apply(node, em, ebones, ents, R, model, mm, tol, uv_tol, template_materials=tm)
            notes.append(f"{node.Name()}: snapped {st['snapped']}, moved {st['moved']}, new {st['new']}"
                         f"{', duplicates ' + str(st['duplicates']) if st.get('duplicates') else ''}")
        except SplitError as exc:
            raise SplitError(f"{node.Name()}: {exc}") from exc
    for name, em in by_name.items():
        notes.append(f"{name}: not in the original mesh — kept as a new submesh")
        _append_mesh(tm, em, ebones, ents)
    return template, notes


def _append_mesh(tm, em: dict, ebones: list[str], ents: dict[str, int]) -> None:
    node = tm.CreateMesh()
    node.SetName(em["name"])
    node.SetVertexPositionBuffer(em["positions"].astype(np.float32).tolist())
    if em["normals"] is not None:
        node.SetVertexNormalBuffer(em["normals"].astype(np.float32).tolist())
    layers = [em[k] for k in ("uv0", "uv1") if em.get(k) is not None]
    for i, uv in enumerate(layers):
        node.SetVertexUVLayerBuffer(i, uv.astype(np.float32).tolist())
    node.SetUVLayerCount(len(layers))
    node.SetFaceBuffer(em["faces"].astype(np.int64).tolist())
    if em["weights"] is not None:
        wb, wv, _ = _top4(em["bones"], em["weights"])
        mapped = np.array([[ents.get(ebones[int(b)].lower(), -1) if 0 <= int(b) < len(ebones) else -1 for b in row]
                           for row in wb], dtype=np.int64)
        if ((mapped < 0) & (wv > 0)).any():
            raise SplitError(f"{em['name']}: weights on bones the mesh does not have")
        node.SetMaximumWeightInfluence(MAX_INFLUENCES)
        node.SetVertexWeightBoneBuffer(np.where(wv > 0, mapped, 0).reshape(-1).tolist())
        node.SetVertexWeightValueBuffer(wv.reshape(-1).tolist())
    if em.get("material"):
        _assign_material(node, em["material"], tm, {})


def _install_into_rpx(tree: Path, result: dict) -> None:
    import shutil
    spec = json.loads((tree / "pack.json").read_text(encoding="utf-8"))
    by_name = {}
    for r in spec.get("resources", []):
        if int(str(r.get("type", "0")), 0) == 0x10:
            by_name.setdefault(r["name"], r["dir"])
    for m in result["meshes"]:
        if m["problems"] or not (m.get("check") or {}).get("ok", not m["submeshes"]):
            m["rpx"] = "not installed (problems)"
            continue
        if not any(s.get("status") == "applied" for s in m["submeshes"]):
            continue
        d = by_name.get(m["logical_name"])
        if d is None:
            m["rpx"] = f"no mesh named {m['logical_name']!r} in {tree}"
            continue
        dst = tree / d / "model.cast"
        if dst.exists() and not dst.with_name("model.cast.orig").exists():
            shutil.copy2(dst, dst.with_name("model.cast.orig"))
        shutil.copy2(Path(m["dir"]) / "model.cast", dst)
        m["rpx"] = str(dst)


def _unchanged_geometry(node, orig, snap, faces, em, bones, weights, skinned, stats) -> bool:
    """Every edited vertex sits exactly on an original and the triangles are the original ones up to vertices that
    share position + UV (Blender's glTF exporter merges such vertices and splits others by normal): the template
    node is kept as it is — only weight edits are carried onto every original of the edited vertex's group."""
    ofaces = np.asarray(node.FaceBuffer(), dtype=np.int64).reshape(-1, 3)
    if len(faces) != len(ofaces):
        return False
    key = np.ascontiguousarray(np.column_stack([orig["vp"], np.nan_to_num(orig["u0"], nan=9e9, posinf=9e9,
                                                                             neginf=-9e9)]))
    _, group = np.unique(key, axis=0, return_inverse=True)
    group = group.reshape(-1)

    def canon(rows):
        r = np.sort(rows, axis=1)          # orientation-free multiset of group triples is enough here
        return r[np.lexsort(r.T[::-1])]

    if not np.array_equal(canon(group[snap[faces]]), canon(group[ofaces])):
        return False
    stats["in_place"] = True
    stats["kept_template"] = True
    if skinned and orig["wb"] is not None:
        wb, wv = orig["wb"].copy(), orig["wv"].copy()
        changed = 0
        members: dict[int, list[int]] = {}
        for j, g in enumerate(group.tolist()):
            members.setdefault(g, []).append(j)
        for i in range(len(snap)):
            j = int(snap[i])
            a = {int(b): float(w) for b, w in zip(wb[j], wv[j]) if w > 0}
            c = {int(b): float(w) for b, w in zip(bones[i], weights[i]) if w > 0}
            if a.keys() == c.keys() and all(abs(a[k] - c[k]) <= 0.6 / 255 for k in a):
                continue
            for m in members[int(group[j])]:
                wb[m], wv[m] = bones[i], weights[i]
                changed += 1
        stats["weights_kept"] = len(snap) - changed
        if changed:
            node.SetVertexWeightBoneBuffer(wb.reshape(-1).astype(np.int64).tolist())
            node.SetVertexWeightValueBuffer(wv.reshape(-1).astype(np.float64).tolist())
    stats["snapped"], stats["moved"], stats["new"] = len(snap), 0, 0
    return True


def _resolve_by_triangles(ident: np.ndarray, faces: np.ndarray, ofaces: np.ndarray, orig: dict) -> np.ndarray | None:
    """Vertices with identical position + UV0 are interchangeable for snapping; choose among them so every edited
    triangle is an original triangle. Returns the new identity or None when no consistent choice exists."""
    key = np.ascontiguousarray(np.column_stack([orig["vp"], np.nan_to_num(orig["u0"], nan=9e9, posinf=9e9,
                                                                             neginf=-9e9)]))
    _, group = np.unique(key.view([("", key.dtype)] * key.shape[1]), return_inverse=True)
    group = group.reshape(-1)

    def canon(tri):
        k = min(range(3), key=lambda r: (group[tri[r]], r))
        return tuple(tri[(k + i) % 3] for i in range(3))

    pool: dict[tuple, list[tuple]] = {}
    for t in ofaces.tolist():
        c = canon(t)
        pool.setdefault(tuple(group[v] for v in c), []).append(c)
    out = np.full(len(ident), -1, dtype=np.int64)
    used = np.zeros(len(orig["vp"]), dtype=bool)
    for t in faces.tolist():
        g = [int(group[ident[v]]) for v in t]
        k = min(range(3), key=lambda r: (g[r], r))
        nt = [t[(k + i) % 3] for i in range(3)]
        cands = pool.get(tuple(g[(k + i) % 3] for i in range(3)))
        if not cands:
            return None
        pick = None
        for ci, c in enumerate(cands):
            if all(out[nv] in (-1, ov) and (out[nv] == ov or not used[ov]) for nv, ov in zip(nt, c)):
                pick = ci
                break
        if pick is None:
            return None
        c = cands.pop(pick)
        for nv, ov in zip(nt, c):
            out[nv] = ov
            used[ov] = True
    if (out < 0).any() or len(np.unique(out)) != len(out):
        return None
    return out


def _uv_dist(orig_uv: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Max per-row UV difference, ignoring components that are non-finite in the original (shipped meshes carry
    inf / NaN UVs; glTF tools write 0 there)."""
    fin = np.isfinite(orig_uv)
    d = np.where(fin, np.abs(np.where(fin, orig_uv, 0.0) - uv), 0.0)
    return d.max(axis=1) if d.ndim == 2 else d


def _drop_loose(em: dict) -> dict:
    """Remove vertices no triangle uses (Blender keeps them after face deletion; their UV is meaningless)."""
    n = len(em["positions"])
    used = np.zeros(n, dtype=bool)
    used[em["faces"]] = True
    if used.all():
        return em
    lut = np.cumsum(used) - 1
    out = dict(em)
    for k in ("positions", "normals", "uv0", "uv1", "bones", "weights"):
        if out.get(k) is not None and len(out[k]) == n:
            out[k] = out[k][used]
    out["faces"] = lut[em["faces"]]
    out["loose_dropped"] = int((~used).sum())
    return out


def _complete_by_topology(snap: np.ndarray, faces: np.ndarray, ofaces: np.ndarray) -> np.ndarray:
    """Extend exact position matches through the triangles: a new triangle with two identified corners whose
    directed edge exists in the original takes the original third corner (moved vertices keep their identity)."""
    ident = snap.copy()
    edge: dict[tuple[int, int], int] = {}
    for a, b, c in ofaces.tolist():
        edge[(a, b)] = c
        edge[(b, c)] = a
        edge[(c, a)] = b
    changed = True
    while changed:
        changed = False
        for tri in faces.tolist():
            ids = [int(ident[v]) for v in tri]
            if ids.count(-1) != 1:
                continue
            k = ids.index(-1)
            a, b = ids[(k + 1) % 3], ids[(k + 2) % 3]
            c = edge.get((a, b))
            if c is not None:
                ident[tri[k]] = c
                changed = True
    # an original id claimed twice is ambiguous: drop the non-exact claims
    vals, counts = np.unique(ident[ident >= 0], return_counts=True)
    dup = set(vals[counts > 1].tolist())
    if dup:
        for i in np.flatnonzero(ident >= 0):
            if int(ident[i]) in dup and snap[i] < 0:
                ident[i] = -1
    return ident


def _same_triangles(a: np.ndarray, b: np.ndarray) -> bool:
    """Same triangle list up to the rotation of each triangle (order of triangles must match too)."""
    if a.shape != b.shape:
        return False
    def canon(t):
        r = np.argmin(t, axis=1)
        return np.stack([np.take_along_axis(t, ((r + k) % 3)[:, None], axis=1)[:, 0] for k in range(3)], axis=1)
    ca, cb = canon(a), canon(b)
    if np.array_equal(ca, cb):
        return True
    return np.array_equal(ca[np.lexsort(ca.T[::-1])], cb[np.lexsort(cb.T[::-1])])
