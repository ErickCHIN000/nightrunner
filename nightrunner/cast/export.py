"""Model → Cast scene (notes/FORMATS/cast-mapping.md; ARCHITECTURE §5).

    Root
      Metadata          up 'y', software 'nightrunner <version>'   — native engine coordinates, unchanged
      Model (name)
        Skeleton        Bone per entity, class-4 order: name, parent index, local pos/rot from the 3×4 local matrix,
                        world pos/rot from the composed globals, scale (1,1,1). Rotations are the polar (SVD)
                        rotation of the 3×3 part — native frames may carry scale/shear that Cast bones cannot;
                        the exact 3×4s live in mesh.json. Custom: bp_entity, bp_flags, bp_type, bp_geometry_count.
        Material        one per referenced material slot: Name '<x>.mat', Type 'pbr' (no file links in phase 1)
        Mesh            one per (geometry entry, submesh) named '<mesh>.e<entry>.s<sub>': vp/vn/vt, u0 (+u1 for
                        formats 3/6/8), f (indices compacted to the submesh's used vertices, native winding),
                        wb/wv with mi = 4 for skinned formats (bone = global entity index via the submesh palette,
                        weight = raw/255, inactive lanes bone 0 weight 0). Custom bp_* properties carry the
                        vertex format, entry/submesh indices, material slot, the entry's opaque words, per-vertex
                        original ids (index into the entry's vertex window) and the tangent handedness (±1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .. import __version__
from ..errors import FormatError
from ..mesh.model import Model
from . import castlib

SOFTWARE = f"nightrunner {__version__}"
MAX_INFLUENCES = 4


@dataclass
class MeshMapping:
    name: str
    entry: int
    submesh: int
    vertex_ids: np.ndarray       # original vertex index within the entry window, per Cast vertex
    vertex_count: int
    face_count: int


@dataclass
class ExportReport:
    path: str
    nodes: dict = field(default_factory=dict)
    meshes: list[MeshMapping] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    bone_frame_residual: float = 0.0     # max |R_polar − M3x3| over bones (0 = pure rotations)

    def to_json(self) -> dict:
        return {"path": self.path, "nodes": self.nodes, "skipped": self.skipped,
                "bone_frame_residual": self.bone_frame_residual,
                "meshes": [{"name": m.name, "entry": m.entry, "submesh": m.submesh, "vertices": m.vertex_count,
                            "faces": m.face_count} for m in self.meshes]}


def _prop(node, name: str, kind: str, values) -> None:
    node.CreateProperty(name, kind).values = list(values)


def cast_text(s: str | bytes) -> str:
    """Cast strings are UTF-8: a name with bytes that are not valid UTF-8 (e.g. common_meshes #2572 entity
    'dlc_ft_ce_a_hatch_1x_a_anm\x8c') is written with U+FFFD replacement; the exact bytes stay in mesh.json
    (name_hex) and the importer compares against the same replacement form."""
    raw = s if isinstance(s, bytes) else s.encode("utf-8", "surrogateescape")
    return raw.decode("utf-8", "replace")


def polar_rotation(m: np.ndarray) -> np.ndarray:
    """Closest proper rotation to a 3×3 (SVD polar factor, det forced positive)."""
    u, _, vt = np.linalg.svd(m)
    r = u @ vt
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1
        r = u @ vt
    return r


def quaternion_xyzw(r: np.ndarray) -> list[float]:
    """Proper rotation matrix → (x, y, z, w), w ≥ 0 (symmetric-eigenvector method, robust for any rotation)."""
    k = np.array([
        [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[2, 1] - r[1, 2]],
        [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[0, 2] - r[2, 0]],
        [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[1, 0] - r[0, 1]],
        [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1], r[0, 0] + r[1, 1] + r[2, 2]],
    ]) / 3.0
    _, vectors = np.linalg.eigh(k)
    q = vectors[:, -1]
    if q[3] < 0:
        q = -q
    return [float(x) for x in q]


def build_cast(model: Model) -> tuple[castlib.Cast, ExportReport]:
    cast = castlib.Cast()
    root = cast.CreateRoot()
    meta = root.CreateMetadata()
    meta.SetUpAxis("y")
    meta.SetSoftware(SOFTWARE)
    mdl = root.CreateModel()
    mdl.SetName(cast_text(model.name))
    rep = ExportReport(path="")
    counts = {"root": 1, "meta": 1, "modl": 1, "skel": 0, "bone": 0, "matl": 0, "mesh": 0}

    # ---- skeleton -----------------------------------------------------------------------------------------
    globals_ = model.entity_globals()
    skel = mdl.CreateSkeleton()
    counts["skel"] = 1
    residual = 0.0
    for ent in model.entities:
        bone = skel.CreateBone()
        bone.SetName(cast_text(ent.name))
        bone.SetParentIndex(int(ent.parent))
        bone.SetSegmentScaleCompensate(False)
        local = ent.local.astype(np.float64)
        rl = polar_rotation(local[:, :3])
        residual = max(residual, float(np.abs(rl - local[:, :3]).max()))
        bone.SetLocalPosition([float(x) for x in local[:, 3]])
        bone.SetLocalRotation(quaternion_xyzw(rl))
        bone.SetScale((1.0, 1.0, 1.0))
        g = globals_[ent.index]
        bone.SetWorldPosition([float(x) for x in g[:3, 3]])
        bone.SetWorldRotation(quaternion_xyzw(polar_rotation(g[:3, :3])))
        _prop(bone, "bp_entity", "i", [ent.index])
        _prop(bone, "bp_flags", "i", [ent.flags])
        _prop(bone, "bp_type", "i", [ent.type])
        _prop(bone, "bp_geometry_count", "i", [ent.geometry_count])
        counts["bone"] += 1
    rep.bone_frame_residual = residual

    # ---- materials ----------------------------------------------------------------------------------------
    used_slots = sorted({s.material_slot for e in model.geometry_entries for s in e.submeshes})
    mat_nodes: dict[int, castlib.Material] = {}
    for slot in used_slots:
        m = mdl.CreateMaterial()
        m.SetName(cast_text(model.material_name(slot)))
        m.SetType("pbr")
        _prop(m, "bp_material_slot", "i", [slot])
        mat_nodes[slot] = m
        counts["matl"] += 1

    # ---- meshes -------------------------------------------------------------------------------------------
    for e in model.geometry_entries:
        v = e.vertices
        for s in e.submeshes:
            name = f"{cast_text(model.name)}.e{e.index}.s{s.index}"
            if v is None:
                rep.skipped.append(f"{name}: vertex format {e.format} not decodable / no vertex buffer")
                continue
            if s.index_count == 0:
                rep.skipped.append(f"{name}: empty submesh (0 indices)")
                continue
            idx = s.indices.astype(np.uint32)
            if idx.size and int(idx.max()) >= v.count:
                raise FormatError(f"{name}: index {int(idx.max())} >= vertex count {v.count}")
            used, remap = np.unique(idx, return_inverse=True)
            faces = remap.astype(np.uint32).reshape(-1)
            mesh = mdl.CreateMesh()
            mesh.SetName(name)
            mesh.SetMaterial(mat_nodes[s.material_slot].Hash())
            mesh.SetUVLayerCount(2 if v.uv1 is not None else 1)
            mesh.SetColorLayerCount(0)
            mesh.SetVertexPositionBuffer(v.positions[used].tolist())
            mesh.SetVertexNormalBuffer(v.normals[used].tolist())
            mesh.SetVertexTangentBuffer(v.tangents[used].tolist())
            mesh.SetVertexUVLayerBuffer(0, v.uv0[used].tolist())
            if v.uv1 is not None:
                mesh.SetVertexUVLayerBuffer(1, v.uv1[used].tolist())
            mesh.SetFaceBuffer(faces.tolist())
            if v.skinned:
                joints = v.joints[used].astype(np.int64)
                weights = v.weights[used]
                pal = s.palette.astype(np.int64)
                active = weights > 0
                if pal.size == 0:
                    if active.any():
                        raise FormatError(f"{name}: skinned vertices with an empty palette")
                    bones = np.zeros_like(joints)
                else:
                    if int(joints[active].max(initial=0)) >= pal.size:
                        raise FormatError(f"{name}: joint index outside the {pal.size}-entry palette")
                    bones = np.where(active, pal[np.minimum(joints, pal.size - 1)], 0)
                mesh.SetMaximumWeightInfluence(MAX_INFLUENCES)
                mesh.SetSkinningMethod("linear")
                mesh.SetVertexWeightBoneBuffer(bones.reshape(-1).tolist())
                mesh.SetVertexWeightValueBuffer(weights.reshape(-1).astype(np.float64).tolist())
            _prop(mesh, "bp_format", "b", [e.format])
            _prop(mesh, "bp_entry", "i", [e.index])
            _prop(mesh, "bp_submesh", "i", [s.index])
            _prop(mesh, "bp_material_slot", "i", [s.material_slot])
            _prop(mesh, "bp_owner_entity", "i", [-1 if e.owner_entity is None else e.owner_entity])
            _prop(mesh, "bp_vertex_base", "i", [e.vertex_base])
            _prop(mesh, "bp_vertex_count", "i", [e.vertex_count])
            _prop(mesh, "bp_index_base", "i", [s.index_base])
            _prop(mesh, "bp_raw_00", "s", [e.raw_00.hex()])
            _prop(mesh, "bp_raw_12", "i", [e.raw_12])
            _prop(mesh, "bp_raw_14", "i", [e.raw_14])
            _prop(mesh, "bp_raw_17", "i", [e.raw_17])
            _prop(mesh, "bp_raw_34", "s", [e.raw_34.hex()])
            _prop(mesh, "bp_vertex_id", "i", used.tolist())
            _prop(mesh, "bp_tangent_sign", "f", v.tangent_sign[used].astype(np.float64).tolist())
            counts["mesh"] += 1
            rep.meshes.append(MeshMapping(name, e.index, s.index, used.astype(np.uint32), int(used.size), int(faces.size // 3)))
    rep.nodes = counts
    return cast, rep


def export_cast(model: Model, path: Path | str) -> ExportReport:
    """Write the model as Cast, or as glTF / GLB when *path* ends in .gltf / .glb (cast/gltf.py)."""
    path = Path(path)
    cast, rep = build_cast(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() in (".gltf", ".glb"):
        from .gltf import save_scene
        save_scene(cast, path)
    else:
        tmp = path.with_name(path.name + ".tmp")
        cast.save(str(tmp))
        tmp.replace(path)
    rep.path = str(path)
    return rep


def load_cast(path: Path | str) -> castlib.Cast:
    return castlib.Cast.load(str(path))


def cast_node_counts(cast: castlib.Cast) -> dict[str, int]:
    """Node identifier census of a loaded Cast (fourcc strings)."""
    out: dict[str, int] = {}

    def walk(n):
        key = n.identifier.to_bytes(4, "little").decode("ascii", "replace")
        out[key] = out.get(key, 0) + 1
        for c in n.childNodes:
            walk(c)

    for r in cast.Roots():
        walk(r)
    return out
