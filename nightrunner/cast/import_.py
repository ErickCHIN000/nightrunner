"""Cast scene (+ mesh.json sidecar) → resolved edit description (notes/FORMATS/cast-mapping.md §import).

Two readers feed the phase-2 encoder (nightrunner.mesh.rebuild):

    scene = read_cast(path)                 parse only: bones, materials, meshes, custom bp_* properties
    res   = resolve_import(scene, sidecar)  match every Cast mesh to a (geometry entry, submesh) of the sidecar,
                                            every bone to an entity, every material to a slot (or a NEW name),
                                            and normalise the vertex attributes into numpy arrays

Accepted inputs (A: read from the official Blender plugin sources, github.com/dtzxporter/cast master, 2026-09-15):
  * a Cast written by nightrunner (bp_* properties present, names `<mesh>.e<entry>.s<sub>`);
  * the same Cast after a round trip through the official Blender plugin: custom properties dropped, no tangent
    buffer (the exporter writes none), UVs flipped twice (V → 1−V on import and export), faces rotated twice
    (import (a,b,c)→(b,c,a), export loop[2],loop[0],loop[1] → (a,b,c)), vertices split on UV seams, up axis /
    scale from the export dialog (defaults 'y' / 1.0; the importer applies NO axis conversion), object names
    possibly suffixed `.001` by Blender. Coordinates are therefore always taken as native engine space; a
    Metadata up axis other than 'y' only produces a warning.

Matching order per mesh: `bp_entry`/`bp_submesh` when present → the `.e<n>.s<m>` suffix of the mesh name
(optionally followed by Blender's `.NNN`) → refusal naming the mesh. Bones: identical name list → identity;
otherwise a unique-name permutation (or `bp_entity`) → mapped; anything else is refused (adding/removing bones
is not supported). Materials: `bp_material_slot` when it still names the same material → the slot; else exact
name → slot; else case-insensitive name → slot (warning); else a NEW material name (the encoder appends it to
the class-11 table when capacity allows, or refuses).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..errors import BuildError, FormatError, UnsupportedError
from ..mesh.vertex import is_skinned, tangents_from_uv
from . import castlib
from .export import polar_rotation, quaternion_xyzw
from ..util.schema import matches as schema_matches

MESH_NAME_RE = re.compile(r"\.e(\d+)\.s(\d+)(?:\.\d{3})?$")
MAX_INFLUENCES = 4
BONE_POS_TOL = 1e-4        # bone transform equality (Cast stores f32; polar rotation of the native 3×4)
BONE_ROT_TOL = 1e-3


# --------------------------------------------------------------------------------------------------------------
# parsed scene
# --------------------------------------------------------------------------------------------------------------

@dataclass
class ImportedBone:
    index: int
    name: str
    parent: int
    local_pos: np.ndarray | None
    local_rot: np.ndarray | None        # xyzw
    scale: np.ndarray | None
    entity: int | None                  # bp_entity when present


@dataclass
class ImportedMesh:
    name: str
    node_index: int
    positions: np.ndarray               # N×3 f32
    normals: np.ndarray | None          # N×3 f32
    tangents: np.ndarray | None         # N×3 f32 (None: not in the Cast → derived from UVs on resolve)
    tangent_sign: np.ndarray | None     # N int8 (None: derived from UVs on resolve)
    uv0: np.ndarray | None              # N×2 f32
    uv1: np.ndarray | None
    faces: np.ndarray                   # M×3 u32 (Cast vertex indices)
    weights: np.ndarray | None          # N×K f32 (K = MaximumWeightInfluence)
    bones: np.ndarray | None            # N×K int64 (Cast bone indices)
    material_hash: int | None
    props: dict = field(default_factory=dict)   # bp_* custom properties (raw value lists)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    def prop(self, name: str, default=None):
        v = self.props.get(name)
        return default if v is None else v


@dataclass
class CastScene:
    path: str
    software: str | None
    up_axis: str | None
    model_name: str | None
    bones: list[ImportedBone]
    materials: dict[int, str]           # node hash → material name
    material_slots: dict[int, int]      # node hash → bp_material_slot when present
    meshes: list[ImportedMesh]
    warnings: list[str] = field(default_factory=list)


def _vals(node, key: str):
    p = node.properties.get(key)
    return None if p is None else p.values


def _arr(values, cols: int, dtype) -> np.ndarray | None:
    if values is None:
        return None
    a = np.asarray(values, dtype=dtype)
    if cols > 1:
        if a.size % cols:
            raise FormatError(f"buffer of {a.size} values is not a multiple of {cols}")
        a = a.reshape(-1, cols)
    return np.ascontiguousarray(a)


def read_cast(path, cast: "castlib.Cast | None" = None) -> CastScene:
    """Parse a .cast / .gltf / .glb file (or an in-memory *cast*, *path* then only labels it) into plain arrays.
    Raises FormatError for a malformed scene."""
    path = Path(path)
    if cast is None:
        try:
            from .gltf import load_scene
            cast = load_scene(path)
        except (ValueError, OSError, UnicodeDecodeError, EOFError, Exception) as exc:  # noqa: BLE001
            raise FormatError(f"{path}: cannot read scene: {exc}") from exc
    roots = cast.Roots()
    if not roots:
        raise FormatError(f"{path}: no root node")
    warnings: list[str] = []
    root = roots[0]
    if len(roots) > 1:
        warnings.append(f"{len(roots)} root nodes; only the first is read")
    meta = root.ChildOfType(castlib.Metadata)
    software = meta.Software() if meta else None
    up = meta.UpAxis() if meta else None
    if up is not None and up.lower() != "y":
        warnings.append(f"Metadata up axis is {up!r}: coordinates are still read as native engine space (the official "
                        "Blender plugin applies no axis conversion on import or export)")
    models = root.ChildrenOfType(castlib.Model)
    if not models:
        raise FormatError(f"{path}: no Model node")
    if len(models) > 1:
        warnings.append(f"{len(models)} Model nodes; only the first is read")
    mdl = models[0]

    bones: list[ImportedBone] = []
    skel = mdl.Skeleton()
    if skel is not None:
        for i, b in enumerate(skel.Bones()):
            ent = _vals(b, "bp_entity")
            bones.append(ImportedBone(
                index=i, name=b.Name() or "", parent=int(b.ParentIndex()),
                local_pos=_arr(b.LocalPosition(), 1, np.float64), local_rot=_arr(b.LocalRotation(), 1, np.float64),
                scale=_arr(b.Scale(), 1, np.float64), entity=None if not ent else int(ent[0])))

    materials: dict[int, str] = {}
    material_slots: dict[int, int] = {}
    for m in mdl.Materials():
        materials[m.Hash()] = m.Name() or ""
        s = _vals(m, "bp_material_slot")
        if s:
            material_slots[m.Hash()] = int(s[0])

    meshes: list[ImportedMesh] = []
    for k, m in enumerate(mdl.Meshes()):
        name = m.Name() or f"mesh_{k}"
        pos = _arr(_vals(m, "vp"), 3, np.float32)
        if pos is None:
            raise FormatError(f"{path}: mesh {name!r} has no vertex position buffer")
        n = len(pos)
        faces = _arr(_vals(m, "f"), 3, np.uint32)
        if faces is None:
            faces = np.zeros((0, 3), dtype=np.uint32)
        if faces.size and int(faces.max()) >= n:
            raise FormatError(f"{path}: mesh {name!r}: face index {int(faces.max())} >= {n} vertices")
        mi = int(m.MaximumWeightInfluence() or 0)
        wb = wv = None
        if mi > 0:
            wb = _arr(_vals(m, "wb"), mi, np.int64)
            wv = _arr(_vals(m, "wv"), mi, np.float32)
            if wb is None or wv is None or len(wb) != n or len(wv) != n:
                raise FormatError(f"{path}: mesh {name!r}: weight buffers do not match {n} vertices × {mi} influences")
        props = {}
        for pname, p in m.properties.items():
            if pname.startswith("bp_"):
                props[pname] = list(p.values)
        sign = props.get("bp_tangent_sign")
        sign_arr = None
        if sign is not None:
            if len(sign) != n:
                warnings.append(f"mesh {name!r}: bp_tangent_sign has {len(sign)} values for {n} vertices; ignored")
            else:
                sign_arr = np.where(np.asarray(sign, dtype=np.float64) < 0, -1, 1).astype(np.int8)
        mesh = ImportedMesh(
            name=name, node_index=k, positions=pos,
            normals=_arr(_vals(m, "vn"), 3, np.float32), tangents=_arr(_vals(m, "vt"), 3, np.float32),
            tangent_sign=sign_arr,
            uv0=_arr(_vals(m, "u0"), 2, np.float32), uv1=_arr(_vals(m, "u1"), 2, np.float32),
            faces=faces, weights=wv, bones=wb, material_hash=(_vals(m, "m") or [None])[0], props=props)
        for label, a in (("normals", mesh.normals), ("tangents", mesh.tangents), ("uv0", mesh.uv0), ("uv1", mesh.uv1)):
            if a is not None and len(a) != n:
                raise FormatError(f"{path}: mesh {name!r}: {label} has {len(a)} rows for {n} vertices")
        meshes.append(mesh)
    return CastScene(path=str(path), software=software, up_axis=up, model_name=mdl.Name(), bones=bones,
                     materials=materials, material_slots=material_slots, meshes=meshes, warnings=warnings)


# --------------------------------------------------------------------------------------------------------------
# resolution against the sidecar
# --------------------------------------------------------------------------------------------------------------

@dataclass
class ResolvedMesh:
    src: ImportedMesh
    entry: int
    submesh: int
    matched_by: str                     # "bp" | "name"
    format: int
    material_name: str
    material_slot: int | None           # None ⇒ new material (encoder appends or refuses)
    material_matched_by: str            # "bp" | "exact" | "fold" | "new"
    positions: np.ndarray               # N×3 f32
    normals: np.ndarray                 # N×3 f32
    tangents: np.ndarray                # N×3 f32
    tangent_sign: np.ndarray            # N int8
    uv0: np.ndarray                     # N×2 f32
    uv1: np.ndarray | None              # N×2 f32 (None when the Cast has no second layer)
    faces: np.ndarray                   # M×3 int64
    weights: np.ndarray | None          # N×4 f32 (active lanes first, zero-padded)
    bones: np.ndarray | None            # N×4 int64 entity indices (0 on inactive lanes)
    vertex_ids: np.ndarray | None       # N int64 original window indices (None: absent)
    derived: list[str] = field(default_factory=list)   # attributes synthesised on resolve (tangents, sign, …)

    @property
    def vertex_count(self) -> int:
        return len(self.positions)

    @property
    def name(self) -> str:
        return self.src.name


@dataclass
class ResolvedImport:
    scene: CastScene
    sidecar: dict
    meshes: list[ResolvedMesh]
    entity_map: np.ndarray | None       # Cast bone index → entity index (None: no skeleton in the Cast)
    bones_matched_by: str               # "identity" | "name" | "bp_entity" | "none"
    bone_changes: list[str]             # bones whose transform differs from the sidecar beyond tolerance
    new_materials: list[str]
    warnings: list[str]

    def by_entry(self) -> dict[int, list[ResolvedMesh]]:
        out: dict[int, list[ResolvedMesh]] = {}
        for m in self.meshes:
            out.setdefault(m.entry, []).append(m)
        for v in out.values():
            v.sort(key=lambda m: m.submesh)
        return out


def _side_name(e: dict) -> str:
    """Entity/material name of a sidecar record in the form the Cast carries (UTF-8 with U+FFFD for bytes that
    are not valid UTF-8, see export.cast_text)."""
    hx = e.get("name_hex")
    return bytes.fromhex(hx).decode("utf-8", "replace") if hx else e["name"]


def _resolve_bones(scene: CastScene, sidecar: dict, warnings: list[str]) -> tuple[np.ndarray | None, str, list[str]]:
    ents = sidecar.get("entities", [])
    names = [_side_name(e) for e in ents]
    if not scene.bones:
        return None, "none", []
    cast_names = [b.name for b in scene.bones]
    if len(cast_names) != len(names):
        raise UnsupportedError(f"Cast has {len(cast_names)} bones, the sidecar has {len(names)} entities: adding or "
                               "removing bones is not supported (restore the skeleton from the exported Cast)")
    if cast_names == names:
        emap = np.arange(len(names))
        how = "identity"
    elif len(set(names)) == len(names) and sorted(cast_names) == sorted(names):
        idx = {n: i for i, n in enumerate(names)}
        emap = np.array([idx[n] for n in cast_names], dtype=np.int64)
        how = "name"
        warnings.append("Cast bones are re-ordered relative to the sidecar entities; matched by name")
    elif all(b.entity is not None for b in scene.bones) and sorted(b.entity for b in scene.bones) == list(range(len(names))):
        emap = np.array([b.entity for b in scene.bones], dtype=np.int64)
        how = "bp_entity"
        bad = [(b.name, names[b.entity]) for b in scene.bones if b.name != names[b.entity]]
        if bad:
            warnings.append(f"{len(bad)} bones renamed (matched by bp_entity), e.g. {bad[0][0]!r} → entity {bad[0][1]!r}")
    else:
        missing = sorted(set(names) - set(cast_names))[:5]
        extra = sorted(set(cast_names) - set(names))[:5]
        raise UnsupportedError(f"Cast bone names do not match the sidecar entities (missing {missing}, unexpected {extra}); "
                               "renaming/adding/removing bones is not supported")
    # transform comparison (polar rotation + translation of the native 3×4, both sides)
    changes: list[str] = []
    for b in scene.bones:
        e = ents[int(emap[b.index])]
        exp_parent = int(e["parent"])
        got_parent = -1 if b.parent < 0 or b.parent >= len(emap) else int(emap[b.parent])
        if exp_parent != got_parent:
            changes.append(f"{b.name!r}: parent entity {got_parent} != {exp_parent}")
            continue
        local = np.asarray(e["local_3x4"], dtype=np.float64).reshape(3, 4)
        if b.local_pos is not None and np.abs(np.asarray(b.local_pos) - local[:, 3]).max() > BONE_POS_TOL:
            changes.append(f"{b.name!r}: local position moved by {np.abs(np.asarray(b.local_pos) - local[:, 3]).max():.3g}")
            continue
        if b.local_rot is not None:
            q0 = np.asarray(quaternion_xyzw(polar_rotation(local[:, :3])))
            q1 = np.asarray(b.local_rot, dtype=np.float64)
            if min(np.abs(q1 - q0).max(), np.abs(q1 + q0).max()) > BONE_ROT_TOL:
                changes.append(f"{b.name!r}: local rotation differs by {min(np.abs(q1 - q0).max(), np.abs(q1 + q0).max()):.3g}")
                continue
        if b.scale is not None and np.abs(np.asarray(b.scale) - 1.0).max() > 1e-3:
            changes.append(f"{b.name!r}: scale {np.asarray(b.scale).round(4).tolist()} != 1")
    return emap, how, changes


def _resolve_material(mesh: ImportedMesh, scene: CastScene, sidecar: dict, warnings: list[str]
                      ) -> tuple[str, int | None, str]:
    entries = sidecar.get("materials", {}).get("entries", [])
    names = [_side_name(e) for e in entries]
    slot_prop = mesh.props.get("bp_material_slot")
    name = scene.materials.get(mesh.material_hash) if mesh.material_hash is not None else None
    if name is None and mesh.material_hash is not None:
        raise FormatError(f"mesh {mesh.name!r}: material hash {mesh.material_hash:#x} names no Material node")
    if name is None:
        # no material node: only the recorded slot can say which material it is
        if slot_prop and 0 <= int(slot_prop[0]) < len(names):
            return names[int(slot_prop[0])], int(slot_prop[0]), "bp"
        raise BuildError(f"mesh {mesh.name!r} has no Material and no bp_material_slot: assign a material")
    if slot_prop:
        s = int(slot_prop[0])
        if 0 <= s < len(names) and names[s] == name:
            return name, s, "bp"
    if name in names:
        return name, names.index(name), "exact"
    fold = [i for i, n in enumerate(names) if n.lower() == name.lower()]
    if fold:
        warnings.append(f"mesh {mesh.name!r}: material {name!r} matched slot {fold[0]} ({names[fold[0]]!r}) case-insensitively")
        return names[fold[0]], fold[0], "fold"
    return name, None, "new"


def _match_mesh(mesh: ImportedMesh, sidecar: dict) -> tuple[int, int, str]:
    entries = sidecar.get("geometry_entries", [])
    be, bs = mesh.props.get("bp_entry"), mesh.props.get("bp_submesh")
    if be and bs:
        e, s = int(be[0]), int(bs[0])
        if not 0 <= e < len(entries):
            raise BuildError(f"mesh {mesh.name!r}: bp_entry {e} outside the sidecar's {len(entries)} geometry entries")
        return e, s, "bp"
    m = MESH_NAME_RE.search(mesh.name)
    if not m:
        raise BuildError(f"mesh {mesh.name!r}: no bp_entry/bp_submesh properties and the name does not end in "
                         "'.e<entry>.s<submesh>' — cannot tell which geometry entry/submesh it replaces")
    e, s = int(m.group(1)), int(m.group(2))
    if not 0 <= e < len(entries):
        raise BuildError(f"mesh {mesh.name!r}: entry {e} (from the name) outside the sidecar's {len(entries)} geometry entries")
    return e, s, "name"


def resolve_import(scene: CastScene, sidecar: dict, *, allow_new_submeshes: bool = True) -> ResolvedImport:
    """Match the scene against the sidecar; normalise every mesh's attributes; never modifies the scene."""
    if not schema_matches(sidecar.get("schema"), "nightrunner.mesh/1"):
        raise FormatError(f"sidecar schema {sidecar.get('schema')!r} is not nightrunner.mesh/1")
    warnings = list(scene.warnings)
    emap, bones_how, bone_changes = _resolve_bones(scene, sidecar, warnings)
    entries = sidecar.get("geometry_entries", [])
    resolved: list[ResolvedMesh] = []
    seen: dict[tuple[int, int], str] = {}
    new_materials: list[str] = []
    for mesh in scene.meshes:
        e, s, how = _match_mesh(mesh, sidecar)
        if (e, s) in seen:
            raise BuildError(f"meshes {seen[(e, s)]!r} and {mesh.name!r} both map to entry {e} submesh {s}")
        seen[(e, s)] = mesh.name
        ent = entries[e]
        nsub = len(ent["submeshes"])
        if s >= nsub and not allow_new_submeshes:
            raise BuildError(f"mesh {mesh.name!r}: submesh {s} beyond entry {e}'s {nsub} submeshes")
        fmt = int(ent["format"])
        mat_name, slot, mat_how = _resolve_material(mesh, scene, sidecar, warnings)
        if slot is None and mat_name not in new_materials:
            new_materials.append(mat_name)
        n = mesh.vertex_count
        derived: list[str] = []
        if mesh.uv0 is None:
            raise BuildError(f"mesh {mesh.name!r}: no UV layer 0 (every vertex format stores uv0)")
        if mesh.normals is None:
            raise BuildError(f"mesh {mesh.name!r}: no vertex normal buffer")
        # review F7: non-finite geometry would be encoded silently (NaN normal → zero quad, NaN weight → empty row);
        # non-finite UVs stay allowed (shipped data carries them, e.g. gas_tank_pistol_anm)
        for label, arr in (("position", mesh.positions), ("normal", mesh.normals), ("tangent", mesh.tangents),
                           ("weight", mesh.weights)):
            if arr is not None and not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
                bad = int(np.argmin(np.isfinite(np.asarray(arr, dtype=np.float64)).reshape(n, -1).all(axis=1)))
                raise BuildError(f"mesh {mesh.name!r}: vertex {bad} has a non-finite {label}")
        faces = mesh.faces.astype(np.int64)
        tangents, sign = mesh.tangents, mesh.tangent_sign
        if tangents is None or sign is None:
            t_uv, s_uv = tangents_from_uv(mesh.positions, mesh.normals, mesh.uv0, faces)
            if tangents is None:
                tangents = t_uv
                derived.append("tangents (from UV derivatives)")
            if sign is None:
                sign = s_uv
                derived.append("tangent handedness (from UV derivatives)")
        weights = bones = None
        if is_skinned(fmt):
            if mesh.weights is None or mesh.bones is None:
                raise BuildError(f"mesh {mesh.name!r}: vertex format {fmt} is skinned but the Cast mesh has no weights")
            if emap is None:
                raise BuildError(f"mesh {mesh.name!r}: skinned mesh but the Cast has no Skeleton to map bones to entities")
            w = mesh.weights.astype(np.float32)
            b = mesh.bones.astype(np.int64)
            active = w > 0
            if int(active.sum(axis=1).max(initial=0)) > MAX_INFLUENCES:
                worst = int(np.argmax(active.sum(axis=1)))
                raise BuildError(f"mesh {mesh.name!r}: vertex {worst} has {int(active[worst].sum())} non-zero weights; "
                                 f"the vertex formats carry at most {MAX_INFLUENCES}")
            if not active.any(axis=1).all():
                bad = int(np.argmin(active.any(axis=1)))
                raise BuildError(f"mesh {mesh.name!r}: vertex {bad} has no bone weights (every shipped skinned row sums to 255)")
            if b[active].min(initial=0) < 0 or int(b[active].max(initial=0)) >= len(emap):
                raise BuildError(f"mesh {mesh.name!r}: bone index outside the {len(emap)}-bone skeleton")
            # 4 lanes: the Cast's lane order is kept when it fits (2,904 shipped rows have a zero lane before an
            # active one — census rows_sorted_desc 8,114,394/8,117,298 — and must re-encode bit-exactly); wider
            # buffers are compacted active-first (stable)
            w4 = np.zeros((n, MAX_INFLUENCES), dtype=np.float32)
            b4 = np.zeros((n, MAX_INFLUENCES), dtype=np.int64)
            if w.shape[1] <= MAX_INFLUENCES:
                ws, bs_ = w, b
            else:
                order = np.argsort(~active, axis=1, kind="stable")
                ws = np.take_along_axis(w, order, axis=1)
                bs_ = np.take_along_axis(b, order, axis=1)
            k = min(MAX_INFLUENCES, w.shape[1])
            w4[:, :k] = ws[:, :k]
            b4[:, :k] = bs_[:, :k]
            act4 = w4 > 0
            b4 = np.where(act4, emap[np.clip(b4, 0, len(emap) - 1)], 0)
            weights, bones = w4, b4
        elif mesh.weights is not None:
            warnings.append(f"mesh {mesh.name!r}: weights ignored (vertex format {fmt} is not skinned)")
        ids = None
        idp = mesh.props.get("bp_vertex_id")
        if idp is not None:
            if len(idp) == n:
                ids = np.asarray(idp, dtype=np.int64)
                if ids.min(initial=0) < 0 or int(ids.max(initial=0)) >= int(ent["vertex_count"]):
                    warnings.append(f"mesh {mesh.name!r}: bp_vertex_id outside entry {e}'s {ent['vertex_count']} vertices; ignored")
                    ids = None
            else:
                warnings.append(f"mesh {mesh.name!r}: bp_vertex_id has {len(idp)} values for {n} vertices; ignored")
        resolved.append(ResolvedMesh(
            src=mesh, entry=e, submesh=s, matched_by=how, format=fmt, material_name=mat_name, material_slot=slot,
            material_matched_by=mat_how, positions=mesh.positions.astype(np.float32),
            normals=mesh.normals.astype(np.float32), tangents=np.asarray(tangents, dtype=np.float32),
            tangent_sign=np.asarray(sign, dtype=np.int8), uv0=mesh.uv0.astype(np.float32),
            uv1=None if mesh.uv1 is None else mesh.uv1.astype(np.float32), faces=faces, weights=weights, bones=bones,
            vertex_ids=ids, derived=derived))
    return ResolvedImport(scene=scene, sidecar=sidecar, meshes=resolved, entity_map=emap, bones_matched_by=bones_how,
                          bone_changes=bone_changes, new_materials=new_materials, warnings=warnings)


def load_sidecar(path: Path | str) -> dict:
    from ..util.jsonio import load_json
    side = load_json(Path(path))
    if not schema_matches(side.get("schema"), "nightrunner.mesh/1"):
        raise FormatError(f"{path}: schema {side.get('schema')!r} is not nightrunner.mesh/1")
    return side
