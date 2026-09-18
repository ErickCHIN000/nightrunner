"""One-file export of a resolved `.model`: a single Cast holding the merged skeleton, every chosen mesh (LOD 0)
skinned to it, and materials whose file slots point at PNGs written next to the Cast.

    <out>/<model>.cast
    <out>/textures/<material>_albedo.png      material-aware preview (eyes / hair alpha / opacity, gui/matpreview)
    <out>/textures/<texture>_nrm.png          normal map (Z rebuilt for two-channel BC5 maps)
    <out>/<model>.cast.json                   what went where: skeleton merge, rebinds, materials, problems

Skeleton. The preset skeleton (`preset.skeletonName`) gives the bone list and rest pose. Every part mesh carries
its own entity table (with stored inverse binds); its bones are matched to the skeleton **by name**. Bones the
skeleton lacks are appended (parent matched by name, rest pose from the part). Without a preset skeleton the first
skinned part provides it.

Rebind. The engine skins a part vertex with `pose[b] · inv_bind_part[b]` (the part's stored inverse bind), so at
the skeleton's rest pose the vertex sits at `Σ w · G_skel[b] · inv_bind_part[b] · v`. The exporter bakes exactly
that, so the merged meshes line up on the one armature in Blender. Per-part residuals are reported in the JSON.
Unskinned parts are written as-is and fully weighted to their owner bone.

Scope: an authoring/viewing file (Blender Cast importer: one armature, one object per submesh, material image
slots). Materials carry bp_* properties (base material, recipe, alpha mode/cutoff, every original texture name);
import back to native packs goes through the per-mesh Casts of the split export, not this file.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..cast import castlib
from ..cast.export import MAX_INFLUENCES, SOFTWARE, cast_text, polar_rotation, quaternion_xyzw
from ..cast.assemble import MergedSkeleton as _Skeleton
from ..cast.assemble import rebind_matrices as _rebind_matrices
from ..mesh.decode import decode_resource

MESH_TYPE = 0x10
SKIP_PREFIXES = ("auto_shadow_caster", "shadowcaster", "shadow_caster", "null")
ALBEDO_DIM = 2048
NORMAL_DIM = 2048
_CAST_LOCK = threading.Lock()


def _safe(name: str, limit: int = 60) -> str:
    s = "".join(c if c.isalnum() or c in "._-" else "_" for c in (name or "_"))
    return s[:limit] or "_"


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(n, 1e-12)


# ---- textures ----------------------------------------------------------------------------------------------------

def _normal_png(ctx, name: str, dest: Path) -> str | None:
    from .texpreview import decode_texture
    from .matpreview import array_to_qimage, qimage_to_array
    gids = ctx.catalog.lookup(name, 0x20)
    if not gids:
        return "not in any loaded pack"
    e, i = ctx.catalog.split(gids[0])
    img, info = decode_texture(e.pack, i, mip=None, max_dim=NORMAL_DIM)
    if img.isNull():
        return info.get("error") or "undecodable"
    a = qimage_to_array(img)
    if float(a[:, :, 2].std()) < 1e-3:          # two-channel (BC5) map: rebuild Z
        x, y = a[:, :, 0] * 2 - 1, a[:, :, 1] * 2 - 1
        a[:, :, 2] = (np.sqrt(np.clip(1 - x * x - y * y, 0, 1)) + 1) / 2
    a[:, :, 3] = 1.0
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not array_to_qimage(a).save(str(dest), "PNG"):
        return "PNG write failed"
    return None


# ---- main ----------------------------------------------------------------------------------------------------------

def export_model_cast(ctx, resolution: dict, out_dir: Path, stem: str, *, textures: bool = True,
                      progress: Callable[[str], Any] | None = None, cancel: threading.Event | None = None,
                      formats: tuple[str, ...] = ("cast",)) -> dict:
    """Write the single-file scene for *resolution* (modelresolve.resolve_model output) into *out_dir*, once per
    requested format: "cast", "glb" (textures embedded) and/or "gltf" (+ .bin, textures referenced)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"cast": None, "model": resolution.get("name"), "variant": resolution.get("variant"),
                              "skeleton": None, "parts": [], "materials": {}, "problems": [], "cancelled": False}

    def tick(text):
        if progress:
            progress(text)
        return bool(cancel and cancel.is_set())

    skel = _Skeleton()
    sk = resolution.get("skeleton") or {}
    if sk.get("gids"):
        try:
            e, i = ctx.catalog.split(sk["gids"][0])
            skel.add_model(decode_resource(e.pack.resource(i)), sk.get("name") or "skeleton")
            report["skeleton"] = {"name": sk.get("name"), "pack": e.label, "bones": len(skel.names),
                                  "pack_path": str(e.path), "index": int(i)}
        except Exception as exc:  # noqa: BLE001
            report["problems"].append(f"skeleton {sk.get('name')}: {type(exc).__name__}: {exc}")
    elif sk.get("name"):
        report["problems"].append(f"skeleton {sk['name']} not in any loaded pack; parts provide the bones")

    # ---- decode parts ------------------------------------------------------------------------------------------
    parts = []
    for slot in resolution.get("slots") or []:
        for m in slot.get("meshes") or []:
            if not m.get("chosen"):
                continue
            if not m.get("gids"):
                report["problems"].append(f"{slot.get('name')}: mesh {m.get('name')} missing")
                continue
            if tick(f"decode {m['name']}"):
                report["cancelled"] = True
                return report
            try:
                e, i = ctx.catalog.split(m["gids"][0])
                model = decode_resource(e.pack.resource(i))
            except Exception as exc:  # noqa: BLE001
                report["problems"].append(f"{m['name']}: {type(exc).__name__}: {exc}")
                continue
            parts.append((slot, m, model, e.label, str(e.path), int(i)))
    if not parts:
        report["problems"].append("no mesh of this model is in the loaded packs")
        return report
    if not skel.names:
        first = next((p for p in parts if p[2].skinned), parts[0])
        skel.add_model(first[2], first[1]["name"])
        report["skeleton"] = {"name": first[1]["name"], "pack": first[3], "bones": len(skel.names),
                              "pack_path": first[4], "index": first[5], "from_part": True,
                              "note": "no preset skeleton: taken from the first skinned part"}

    # ---- geometry --------------------------------------------------------------------------------------------
    meshes = []            # (name, material key, pos, nrm, tan, uv0, faces, bones, weights, props)
    materials: dict[str, dict] = {}
    mesh_map = []
    for pi, (slot, m, model, label, ppath, pidx) in enumerate(parts):
        emap = skel.map_model(model, m["name"])
        rebind, worst = _rebind_matrices(model, skel, emap)
        subs = {(s.get("entry"), s.get("submesh")): s for s in m.get("submeshes") or []}
        prec = {"slot": slot.get("name"), "mesh": m["name"], "pack": label, "pack_path": ppath, "index": pidx,
                "logical_name": model.name, "rebind_max_delta": round(worst, 6), "submeshes": 0, "skipped": []}
        mstem = _safe(m["name"].removesuffix(".msh"), 40)
        for ge in model.geometry_entries:
            if ge.element != 0:
                continue
            v = ge.vertices
            for s in ge.submeshes:
                key = f"e{ge.index}.s{s.index}"
                sub = subs.get((ge.index, s.index)) or {}
                base = sub.get("base_material") or model.material_name(s.material_slot)
                if (base or "").casefold().startswith(SKIP_PREFIXES):
                    prec["skipped"].append(f"{key}: {base}")
                    continue
                if v is None or s.index_count == 0:
                    prec["skipped"].append(f"{key}: no decodable vertices")
                    continue
                idx = np.asarray(s.indices, dtype=np.int64)
                used, faces = np.unique(idx, return_inverse=True)
                pos = v.positions[used].astype(np.float64)
                nrm = v.normals[used].astype(np.float64)
                tan = v.tangents[used].astype(np.float64)
                if v.skinned and len(s.palette):
                    joints = v.joints[used].astype(np.int64)
                    w = v.weights[used].astype(np.float64)
                    pal = s.palette.astype(np.int64)
                    active = w > 0
                    ents = np.where(active, pal[np.minimum(joints, len(pal) - 1)], 0)
                    mats = rebind[ents]                                  # (n, 4, 4, 4)
                    blend = np.einsum("nk,nkij->nij", w / np.maximum(w.sum(1, keepdims=True), 1e-12), mats)
                    pos = np.einsum("nij,nj->ni", blend[:, :3, :3], pos) + blend[:, :3, 3]
                    nrm = _normalize(np.einsum("nij,nj->ni", blend[:, :3, :3], nrm))
                    tan = _normalize(np.einsum("nij,nj->ni", blend[:, :3, :3], tan))
                    bones = np.where(active, emap[ents], 0)
                else:
                    owner = ge.owner_entity if ge.owner_entity is not None else 0
                    bones = np.zeros((len(used), MAX_INFLUENCES), dtype=np.int64)
                    bones[:, 0] = emap[owner] if len(emap) else 0
                    w = np.zeros((len(used), MAX_INFLUENCES))
                    w[:, 0] = 1.0
                override = {t["param"]: t["texture"] for t in sub.get("textures") or []
                            if t.get("param") and t.get("texture") and t.get("source") == "model override"}
                mkey = base if not override else f"{base}#" + hashlib.sha1(
                    json.dumps(sorted(override.items())).encode()).hexdigest()[:6]
                materials.setdefault(mkey, {"base": base, "override": override, "textures": sub.get("textures") or []})
                name = f"{_safe(slot.get('name') or 'slot', 20)}.{mstem}.{key}"
                props = {"bp_slot": slot.get("name") or "", "bp_source_mesh": m["name"], "bp_entry": ge.index,
                         "bp_submesh": s.index, "bp_material_slot": s.material_slot,
                         "bp_vertex_id": used.astype(np.int64)}
                mesh_map.append({"name": name, "part": pi, "entry": ge.index, "submesh": s.index, "material": mkey})
                meshes.append((name, mkey, pos, nrm, tan, v.uv0[used], v.uv1[used] if v.uv1 is not None else None,
                               faces.astype(np.uint32), bones, w, props))
                prec["submeshes"] += 1
        report["parts"].append(prec)

    # ---- materials / textures ----------------------------------------------------------------------------------
    if textures:
        from . import matpreview                       # Qt (QImage) only needed for textures
        fetch = matpreview.TextureFetcher(ctx.catalog, max_dim=ALBEDO_DIM)
    mat_files: dict[str, dict] = {}
    written_normals: dict[str, str | None] = {}
    for mkey, md in materials.items():
        if tick(f"material {md['base']}"):
            report["cancelled"] = True
            return report
        rec: dict[str, Any] = {"base": md["base"], "override": md["override"], "albedo": None, "normal": None,
                               "recipe": None, "alpha_mode": "opaque", "cutoff": 0.5, "warnings": [],
                               "all_textures": sorted({t["texture"] for t in md["textures"] if t.get("texture")})}
        if textures:
            try:
                sp = matpreview.preview_for_material(ctx, md["base"], fetch, md["override"] or None)
                rec.update(recipe=sp.recipe, alpha_mode=sp.alpha_mode, cutoff=sp.cutoff, warnings=list(sp.warnings))
                if sp.image is not None:
                    img = sp.image
                    if sp.alpha_mode == "mask":        # hard alpha: Blender's default blend then looks right
                        a = matpreview.qimage_to_array(img)
                        a[:, :, 3] = (a[:, :, 3] >= sp.cutoff).astype(np.float32)
                        img = matpreview.array_to_qimage(a)
                    rel = f"textures/{_safe(mkey.replace('.mat', '').replace('#', '_'), 80)}_albedo.png"
                    (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)
                    if img.save(str(out_dir / rel), "PNG"):
                        rec["albedo"] = rel
                info = ctx.sdb.material_info(md["base"])
                if info:
                    _t, _v, tex, _w = matpreview.material_facts(info, md["override"] or None)
                    nname = tex.get("nrm_0_tex")
                    if nname and not nname.lower().startswith("default_"):
                        rel = f"textures/{_safe(nname.rsplit('.', 1)[0], 80)}.png"
                        if nname not in written_normals:
                            written_normals[nname] = _normal_png(ctx, nname, out_dir / rel)
                        if written_normals[nname] is None:
                            rec["normal"] = rel
                        else:
                            rec["warnings"].append(f"normal {nname}: {written_normals[nname]}")
            except Exception as exc:  # noqa: BLE001
                rec["warnings"].append(f"{type(exc).__name__}: {exc}")
        mat_files[mkey] = rec
    report["materials"] = mat_files

    # ---- write the Cast ----------------------------------------------------------------------------------------
    if tick("writing cast"):
        report["cancelled"] = True
        return report
    path = out_dir / f"{_safe(stem, 80)}.cast"
    with _CAST_LOCK:
        cast = castlib.Cast()
        root = cast.CreateRoot()
        meta = root.CreateMetadata()
        meta.SetUpAxis("y")
        meta.SetSoftware(SOFTWARE)
        mdl = root.CreateModel()
        mdl.SetName(cast_text(stem))
        mdl.CreateProperty("bp_model", "s").values = [resolution.get("name") or stem]
        mdl.CreateProperty("bp_variant", "s").values = [resolution.get("variant") or "Default"]
        sk_node = mdl.CreateSkeleton()
        for i, name in enumerate(skel.names):
            g = skel.globals[i]
            p = skel.parents[i]
            local = g if p < 0 else np.linalg.inv(skel.globals[p]) @ g
            b = sk_node.CreateBone()
            b.SetName(cast_text(name))
            b.SetParentIndex(int(p))
            b.SetSegmentScaleCompensate(False)
            b.SetLocalPosition([float(x) for x in local[:3, 3]])
            b.SetLocalRotation(quaternion_xyzw(polar_rotation(local[:3, :3])))
            b.SetWorldPosition([float(x) for x in g[:3, 3]])
            b.SetWorldRotation(quaternion_xyzw(polar_rotation(g[:3, :3])))
            b.SetScale((1.0, 1.0, 1.0))
            b.CreateProperty("bp_source", "s").values = [skel.source[i]]
        mat_nodes = {}
        for mkey, rec in mat_files.items():
            mn = mdl.CreateMaterial()
            mn.SetName(cast_text(mkey))
            mn.SetType("pbr")
            for slot_name in ("albedo", "normal"):
                if rec.get(slot_name):
                    f = mn.CreateFile()
                    f.SetPath(rec[slot_name])
                    mn.SetSlot(slot_name, f.Hash())
            mn.CreateProperty("bp_material", "s").values = [cast_text(rec["base"])]
            mn.CreateProperty("bp_alpha_mode", "s").values = [rec["alpha_mode"]]
            mn.CreateProperty("bp_alpha_cutoff", "f").values = [float(rec["cutoff"])]
            mn.CreateProperty("bp_recipe", "s").values = [rec["recipe"] or ""]
            mn.CreateProperty("bp_textures", "s").values = [json.dumps(rec["all_textures"])]
            mat_nodes[mkey] = mn
        for (name, mkey, pos, nrm, tan, uv0, uv1, faces, bones, w, props) in meshes:
            me = mdl.CreateMesh()
            me.SetName(cast_text(name))
            me.SetMaterial(mat_nodes[mkey].Hash())
            me.SetUVLayerCount(2 if uv1 is not None else 1)
            me.SetColorLayerCount(0)
            me.SetVertexPositionBuffer(pos.astype(np.float32).tolist())
            me.SetVertexNormalBuffer(nrm.astype(np.float32).tolist())
            me.SetVertexTangentBuffer(tan.astype(np.float32).tolist())
            me.SetVertexUVLayerBuffer(0, np.asarray(uv0, dtype=np.float32).tolist())
            if uv1 is not None:
                me.SetVertexUVLayerBuffer(1, np.asarray(uv1, dtype=np.float32).tolist())
            me.SetFaceBuffer(faces.tolist())
            me.SetMaximumWeightInfluence(MAX_INFLUENCES)
            me.SetSkinningMethod("linear")
            me.SetVertexWeightBoneBuffer(np.asarray(bones, dtype=np.int64).reshape(-1).tolist())
            me.SetVertexWeightValueBuffer(np.asarray(w, dtype=np.float64).reshape(-1).tolist())
            for k, val in props.items():
                if isinstance(val, str):
                    me.CreateProperty(k, "s").values = [cast_text(val)]
                elif isinstance(val, np.ndarray):
                    me.CreateProperty(k, "i").values = [int(x) for x in val]
                else:
                    me.CreateProperty(k, "i").values = [int(val)]
        from ..cast.gltf import save_scene
        written = {}
        for fmt in formats:
            target = path.with_suffix("." + fmt)
            tmp = target.with_name(target.name + ".partial")
            if fmt == "cast":
                cast.save(str(tmp))
                tmp.replace(target)
            else:
                save_scene(cast, target, base_dir=out_dir)
            written[fmt] = str(target)
    report["files"] = written
    report["cast"] = written.get("cast") or next(iter(written.values()))
    report["skeleton"] = dict(report["skeleton"] or {}, merged_bones=len(skel.names),
                              added=[cast_text(n) for n, s in zip(skel.names, skel.source)
                                     if s != (sk.get("name") or "skeleton")][:200])
    report["meshes"] = len(meshes)
    report["mesh_map"] = mesh_map
    report["bones"] = [cast_text(n) for n in skel.names]
    report["format"] = "nightrunner.model_cast/1"
    report["blender_script"] = BLENDER_SCRIPT_NAME if "cast" in formats else None
    # the report is named after the model (<stem>.cast.json) whichever formats were written
    (out_dir / f"{path.name}.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    if "cast" in formats:           # glTF carries alpha natively; only the Cast import needs the helper
        (out_dir / BLENDER_SCRIPT_NAME).write_text(BLENDER_SCRIPT, encoding="utf-8")
    return report


BLENDER_SCRIPT_NAME = "nightrunner_blender_materials.py"
BLENDER_SCRIPT = '''"""nightrunner: finish the materials of an imported model Cast in Blender.

The Cast importer links the albedo PNG to Base Color but ignores its alpha. Run this once after
File > Import > Cast (Scripting tab > Open > this file > Run Script). It reads every *.cast.json next to it and,
for materials whose alpha mode is "mask" (hair cut-outs, already hard 0/1 alpha) or "blend" (eye shadow, wet eye,
forearm hair), connects the albedo alpha to the shader and sets the render method. Safe to run twice.
"""
import json
import os

import bpy

HERE = os.path.dirname(os.path.abspath(bpy.context.space_data.text.filepath)) \\
    if getattr(bpy.context, "space_data", None) and getattr(bpy.context.space_data, "text", None) \\
    and bpy.context.space_data.text.filepath else os.path.dirname(os.path.abspath(__file__))


def fix(mat, mode):
    if not mat.use_nodes:
        return False
    nodes = mat.node_tree.nodes
    bsdf = next((n for n in nodes if n.type == "BSDF_PRINCIPLED"), None)
    if bsdf is None:
        return False
    src = None
    for link in mat.node_tree.links:
        if link.to_node == bsdf and link.to_socket.name == "Base Color" and link.from_node.type == "TEX_IMAGE":
            src = link.from_node
    if src is None:
        return False
    if not any(l.to_socket == bsdf.inputs["Alpha"] for l in mat.node_tree.links):
        mat.node_tree.links.new(bsdf.inputs["Alpha"], src.outputs["Alpha"])
    if hasattr(mat, "surface_render_method"):          # Blender 4.2+
        mat.surface_render_method = "BLENDED" if mode == "blend" else "DITHERED"
    if hasattr(mat, "blend_method"):
        try:
            mat.blend_method = "BLEND" if mode == "blend" else "CLIP"
        except TypeError:
            pass
    if hasattr(mat, "use_backface_culling"):
        mat.use_backface_culling = False
    return True


done = 0
for fn in os.listdir(HERE):
    if not fn.endswith(".cast.json"):
        continue
    with open(os.path.join(HERE, fn), encoding="utf-8") as fh:
        rep = json.load(fh)
    for name, rec in (rep.get("materials") or {}).items():
        mode = rec.get("alpha_mode")
        mat = bpy.data.materials.get(name)
        if mat is not None and mode in ("mask", "blend") and fix(mat, mode):
            done += 1
print(f"nightrunner: alpha set on {done} material(s)")
'''
