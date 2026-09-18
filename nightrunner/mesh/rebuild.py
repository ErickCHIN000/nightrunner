"""Phase 2: Model + resolved Cast import → new 0x10 / 0x11 / 0xF0 / 0xF1 part bytes (notes/FORMATS/mesh.md §encoder).

Per geometry entry the planner picks one of two paths:

  in place   every Cast mesh of the entry carries `bp_vertex_id` (unique per mesh, inside the window): the original
             vertex window is kept (order, unreferenced vertices, raw holes); only vertices whose float attributes
             differ are re-encoded; faces map back through the ids. Index counts may still change (a deleted
             triangle) — then only the buffer layout is re-planned.
  rebuild    ids missing (a Blender round trip) or vertices added: the window becomes the concatenation of the
             Cast meshes' vertex lists (submesh order); a vertex still identifiable by id — or recovered by an exact
             position + uv0 match — re-encodes from its raw record; new vertices take the hole policy of
             `vertex.hole_defaults`.

Buffers: unchanged counts everywhere ⇒ the original bases and sizes are kept (bit-identical when nothing was
edited); otherwise `encode.plan_new_layout` (census layout rule). The image is patched in place through
`ImagePatch`: entry counts/bases, submesh count, material-slot / index-count / palette arrays (appended when they
grow, pointers retargeted, records relocated), material table (new names appended within capacity), entity
bounds (policy below), embedded `.msh` name. Everything else is carried verbatim and the fixups are re-serialised.

Bounds policy (tools/mesh_bounds_census.py, 2026-09-15, 25,398 geometry-owning entities): for skinned formats
6/8 the stored (centre, half) equals the AABB of the union of the owned entries' raw positions (1,638/1,641
within 1e-6) → rewritten exactly when the geometry changed. For static formats 0/3 the stored box is the exact
AABB in 13,553/23,757 entities but a SUPERSET in most others (19,842/25,398 boxes contain the geometry; extra
extent of unknown origin — not in the vertex buffer) → grow-only: new box = stored box ∪ new AABB. Unchanged
entities keep their bytes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..cast.import_ import ResolvedImport, ResolvedMesh
from ..classreader.graph import MAX_PALETTE
from ..errors import BuildError, UnsupportedError
from .encode import plan_new_layout
from .model import GeometryEntry, Model
from .vertex import (DecodedVertices, STRIDES, VERTEX_DTYPES, decode_qtangent, QTAN_SCALE, encode_block,
                     hole_defaults, is_skinned)
from . import imagepatch

MAX_WINDOW_VERTICES = 65535       # u16 indices (spec: refuse above; corpus max 65,529)


# --------------------------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------------------------

@dataclass
class SubmeshPlan:
    index: int
    mesh: ResolvedMesh
    material_slot: int | None
    material_name: str
    indices: np.ndarray                 # u16, relative to the new window
    palette: np.ndarray                 # u16 entity ids
    palette_changed: bool
    original_index_count: int | None
    moved: int = 0                      # vertices whose position changed (in place / id-matched)
    edited: int = 0                     # vertices with any attribute change
    new_vertices: int = 0               # vertices without a raw record

    @property
    def index_count(self) -> int:
        return int(len(self.indices))


@dataclass
class EntryPlan:
    entry: GeometryEntry
    submeshes: list[SubmeshPlan]
    path: str                           # "in-place" | "rebuild" | "untouched"
    records: np.ndarray | None          # new window records (None: no vertex buffer)
    raw_ids: np.ndarray | None
    changed: bool                       # any byte of the window or index lists differs
    layout_changed: bool                # vertex_count or any index count differs

    @property
    def vertex_count(self) -> int:
        return 0 if self.records is None else int(len(self.records))

    @property
    def index_count(self) -> int:
        return sum(s.index_count for s in self.submeshes)


@dataclass
class RebuildPlan:
    entries: list[EntryPlan]
    new_materials: list[str]
    bone_changes: list[str]
    identity_rename: tuple[bytes, bytes] | None
    layout_changed: bool
    warnings: list[str] = field(default_factory=list)

    def report(self) -> dict:
        ents = []
        for ep in self.entries:
            e = ep.entry
            ents.append({
                "entry": e.index, "path": ep.path, "format": e.format,
                "vertex_count": [e.vertex_count, ep.vertex_count], "index_count": [e.index_count, ep.index_count],
                "submesh_count": [e.submesh_count, len(ep.submeshes)],
                "submeshes": [{"submesh": s.index, "mesh": s.mesh.name if s.mesh else None,
                               "material": s.material_name, "material_slot": s.material_slot,
                               "index_count": [s.original_index_count, s.index_count],
                               "palette": [None if s.original_index_count is None else len(e.submeshes[s.index].palette), len(s.palette)],
                               "palette_changed": s.palette_changed, "moved": s.moved, "edited": s.edited,
                               "new_vertices": s.new_vertices,
                               "matched_by": s.mesh.matched_by if s.mesh else None,
                               "material_matched_by": s.mesh.material_matched_by if s.mesh else None,
                               "derived": list(s.mesh.derived) if s.mesh else []} for s in ep.submeshes],
            })
        return {"entries": ents, "new_materials": list(self.new_materials), "bone_changes": list(self.bone_changes),
                "identity_rename": None if self.identity_rename is None else
                [self.identity_rename[0].decode("utf-8", "replace"), self.identity_rename[1].decode("utf-8", "replace")],
                "layout_changed": self.layout_changed, "warnings": list(self.warnings)}


def _void_keys(*columns: np.ndarray) -> list[bytes]:
    """Row-wise byte keys of the f32-rounded concatenation of *columns* (exact-match hashing)."""
    rows = len(columns[0])
    cat = np.ascontiguousarray(np.concatenate([np.asarray(c, dtype=np.float32).reshape(rows, -1)
                                               for c in columns], axis=1))
    return [row.tobytes() for row in cat]


def _recover_ids(orig: DecodedVertices, positions: np.ndarray, uv0: np.ndarray, normals: np.ndarray) -> np.ndarray:
    """Best-effort original-id recovery for Casts without bp_vertex_id (every Blender round trip).

    1. exact match on the f32 (position, uv0, normal) tuple — UV-seam duplicates share a position but differ in
       uv0/normal, so this picks the right raw record (and with it the right hole bytes: qtangent quad,
       weights/joints, tails, format-8 ext);
    2. otherwise exact position match with the closest (uv0, normal) in float64, NaN-safe (shipped data contains
       inf/NaN uv0 rows, e.g. gas_tank_pistol_anm);
    3. otherwise −1 = new vertex.
    Review 2026-09-15 F2: the previous position-only match with a float32 tie-break picked a neighbouring record
    on 10/15 sample meshes and leaked another vertex's format-8 raw_ext.
    """
    n = len(positions)
    out = np.full(n, -1, dtype=np.int64)
    if orig.count == 0 or n == 0:
        return out
    full: dict[bytes, list[int]] = {}
    for i, key in enumerate(_void_keys(orig.positions, orig.uv0, orig.normals)):
        full.setdefault(key, []).append(i)
    by_pos: dict[bytes, list[int]] = {}
    for i, key in enumerate(_void_keys(orig.positions)):
        by_pos.setdefault(key, []).append(i)
    want_full = _void_keys(positions, uv0, normals)
    want_pos = _void_keys(positions)
    ouv = np.asarray(orig.uv0, dtype=np.float64)
    onr = np.asarray(orig.normals, dtype=np.float64)
    for k in range(n):
        cands = full.get(want_full[k])
        if cands:
            out[k] = cands[0]
            continue
        cands = by_pos.get(want_pos[k])
        if not cands:
            continue
        if len(cands) == 1:
            out[k] = cands[0]
            continue
        c = np.asarray(cands)
        with np.errstate(invalid="ignore"):
            d_uv = np.nan_to_num(np.abs(ouv[c] - np.asarray(uv0[k], dtype=np.float64)).max(axis=1), nan=1e9)
            d_n = 1.0 - np.nan_to_num(np.sum(onr[c] * np.asarray(normals[k], dtype=np.float64), axis=1), nan=-1e9)
        out[k] = c[int(np.argmin(d_uv * 1e3 + d_n))]
    return out


def _plan_palette(sub_orig, mesh: ResolvedMesh, skinned: bool, entry_index: int) -> tuple[np.ndarray, bool]:
    if not skinned:
        return np.zeros(0, dtype=np.uint16), False
    active = mesh.weights > 0
    used = np.unique(mesh.bones[active]) if active.any() else np.zeros(0, dtype=np.int64)
    orig = sub_orig.palette.astype(np.int64) if sub_orig is not None else np.zeros(0, dtype=np.int64)
    if len(orig) and np.isin(used, orig).all():
        return sub_orig.palette.astype(np.uint16), False
    # keep the original order (existing joint bytes stay valid), append new bones in first-use order
    extra = [int(b) for b in mesh.bones[active][np.argsort(np.nonzero(active)[0], kind="stable")] if b not in set(orig.tolist())]
    seen: list[int] = []
    for b in extra:
        if b not in seen:
            seen.append(b)
    pal = np.concatenate([orig, np.asarray(seen, dtype=np.int64)]) if len(seen) else orig
    if len(pal) == 0:
        pal = np.zeros(1, dtype=np.int64)     # a skinned submesh with no active lane cannot happen (import refuses)
    if len(pal) > MAX_PALETTE:
        raise BuildError(f"entry {entry_index} submesh {mesh.submesh} ({mesh.name!r}): {len(pal)} bones in one submesh "
                         f"exceed the {MAX_PALETTE}-entry palette (joint bytes are u8); split the submesh")
    return pal.astype(np.uint16), True


def _joints(mesh: ResolvedMesh, palette: np.ndarray, raw_joints: np.ndarray | None, has_raw: np.ndarray) -> np.ndarray:
    """Joint bytes: palette index of each active bone (raw joint kept when it still names that bone, so unchanged
    rows are bit-identical even with duplicate palette entries); inactive lanes keep the raw byte when there is a
    raw record (census: 1,088,061 inactive lanes carry non-zero joints) else 0."""
    n = len(mesh.positions)
    pal = palette.astype(np.int64)
    first = {}
    for i, b in enumerate(pal.tolist()):
        first.setdefault(b, i)
    active = mesh.weights > 0
    j = np.zeros((n, 4), dtype=np.int64)
    lut = np.full(int(pal.max(initial=0)) + 1, -1, dtype=np.int64)
    for b, i in first.items():
        lut[b] = i
    j[active] = lut[mesh.bones[active]]
    if (j[active] < 0).any():
        raise BuildError(f"mesh {mesh.name!r}: bone outside the submesh palette (internal)")
    if raw_joints is not None and has_raw.any():
        rj = raw_joints.astype(np.int64)
        # inactive lanes: raw byte; active lanes: raw byte when palette[raw] == bone
        in_pal = rj < len(pal)
        named = np.where(in_pal, pal[np.minimum(rj, len(pal) - 1)], -1) if len(pal) else np.full_like(rj, -1)
        # inactive lanes keep the raw byte only while it still indexes the palette being written (review F8;
        # 0 of 147,638 shipped inactive lanes exceed their palette)
        keep = has_raw[:, None] & ((~active & in_pal) | (named == mesh.bones))
        j = np.where(keep, rj, j)
    return j.astype(np.uint8)


def _same_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.all(a == b, axis=1)


def plan_entry(model: Model, e: GeometryEntry, meshes: list[ResolvedMesh], warnings: list[str]) -> EntryPlan:
    if e.vertices is None:
        if meshes:
            raise UnsupportedError(f"entry {e.index}: vertex format {e.format} is not decodable / no vertex buffer; "
                                   "its meshes cannot be rebuilt")
        return EntryPlan(e, [SubmeshPlan(s.index, None, s.material_slot, model.material_name(s.material_slot),
                                         np.ascontiguousarray(s.indices, dtype="<u2"), s.palette.astype(np.uint16), False,
                                         s.index_count) for s in e.submeshes], "untouched", None, None, False, False)
    if not meshes:
        if e.submesh_count == 0:
            return EntryPlan(e, [], "untouched", e.vertices.raw.copy(), np.arange(e.vertex_count), False, False)
        raise UnsupportedError(f"entry {e.index} (LOD {e.element} of entity {e.owner_entity}) has no meshes in the Cast: "
                               "removing a geometry entry is not supported (keep at least submesh 0 of every entry)")
    subs = sorted(meshes, key=lambda m: m.submesh)
    want = [m.submesh for m in subs]
    if want != list(range(len(subs))):
        raise BuildError(f"entry {e.index}: the Cast provides submeshes {want}; they must be 0..{len(subs) - 1} "
                         "without gaps (renumber the mesh names / bp_submesh)")
    fmt = e.format
    skinned = is_skinned(fmt)
    orig = e.vertices
    orig_raw = orig.raw
    n_orig = orig.count
    in_place = all(m.vertex_ids is not None for m in subs) and all(
        len(np.unique(m.vertex_ids)) == len(m.vertex_ids) for m in subs)
    holes = hole_defaults(fmt, orig_raw)
    plans: list[SubmeshPlan] = []

    if in_place:
        # ---- keep the window; overwrite edited vertices -----------------------------------------------------
        v = DecodedVertices(fmt, orig_raw)            # fresh float view of the original window
        owner = np.full(n_orig, -1, dtype=np.int64)   # which mesh last wrote a vertex (conflict detection)
        edited = np.zeros(n_orig, dtype=bool)
        moved = np.zeros(n_orig, dtype=bool)
        new_joints = v.joints.copy() if skinned else None
        for m in subs:
            ids = m.vertex_ids
            sub_orig = e.submeshes[m.submesh] if m.submesh < e.submesh_count else None
            pal, pal_changed = _plan_palette(sub_orig, m, skinned, e.index)
            # attribute comparison against the original decode
            d_pos = ~_same_rows(orig.positions[ids], m.positions)
            d_uv0 = ~_same_rows(orig.uv0[ids], m.uv0)
            d_uv1 = np.zeros(len(ids), dtype=bool) if (m.uv1 is None or orig.uv1 is None) else ~_same_rows(orig.uv1[ids], m.uv1)
            d_frame = (~_same_rows(orig.normals[ids], m.normals) | ~_same_rows(orig.tangents[ids], m.tangents)
                       | (orig.tangent_sign[ids] != m.tangent_sign))
            if skinned:
                d_w = ~_same_rows(orig.weights[ids], m.weights)
                j = _joints(m, pal, orig_raw["joints"][ids], np.ones(len(ids), dtype=bool))
                d_j = ~_same_rows(orig_raw["joints"][ids].astype(np.int64), j.astype(np.int64))
                d_skin = d_w | d_j
            else:
                d_skin = np.zeros(len(ids), dtype=bool)
                j = None
            any_d = d_pos | d_uv0 | d_uv1 | d_frame | d_skin
            # conflicts: a window vertex written by two meshes with different values
            prev = owner[ids]
            clash = (prev >= 0) & (prev != m.submesh) & any_d & edited[ids]
            if clash.any():
                k = int(np.nonzero(clash)[0][0])
                raise BuildError(f"entry {e.index}: window vertex {int(ids[k])} is edited differently by submesh "
                                 f"{int(prev[k])} and submesh {m.submesh} ({m.name!r})")
            sel = ids[any_d]
            v.positions[sel] = m.positions[any_d]
            v.uv0[sel] = m.uv0[any_d]
            if m.uv1 is not None and v.uv1 is not None:
                v.uv1[sel] = m.uv1[any_d]
            v.normals[sel] = m.normals[any_d]
            v.tangents[sel] = m.tangents[any_d]
            v.tangent_sign[sel] = m.tangent_sign[any_d]
            if skinned:
                v.weights[sel] = m.weights[any_d]
                new_joints[ids] = j
            owner[ids[any_d]] = m.submesh
            edited[sel] = True
            moved[ids[d_pos]] = True
            faces = m.faces
            idx = ids[faces.reshape(-1)].astype("<u2")
            plans.append(SubmeshPlan(m.submesh, m, m.material_slot, m.material_name, idx, pal, pal_changed,
                                     None if sub_orig is None else sub_orig.index_count,
                                     moved=int(d_pos.sum()), edited=int(any_d.sum()), new_vertices=0))
        if skinned:
            v.joints = new_joints
        records = encode_block(v, raw=orig_raw, raw_ids=np.arange(n_orig), holes=holes)
        raw_ids = np.arange(n_orig)
        path = "in-place"
    else:
        # ---- rebuild the window from the Cast meshes ---------------------------------------------------------
        chunks = []
        raw_id_chunks = []
        base = 0
        for m in subs:
            n = m.vertex_count
            sub_orig = e.submeshes[m.submesh] if m.submesh < e.submesh_count else None
            pal, pal_changed = _plan_palette(sub_orig, m, skinned, e.index)
            if m.vertex_ids is not None:
                ids = m.vertex_ids.astype(np.int64)
                if len(np.unique(ids)) != len(ids):
                    warnings.append(f"entry {e.index} submesh {m.submesh}: duplicate bp_vertex_id values; window rebuilt")
            else:
                ids = _recover_ids(orig, m.positions, m.uv0, m.normals)
                if n:
                    warnings.append(f"entry {e.index} submesh {m.submesh} ({m.name!r}): no bp_vertex_id; "
                                    f"{int((ids >= 0).sum())}/{n} vertices recovered by exact position match")
            has = ids >= 0
            uv1 = m.uv1
            if uv1 is None and "uv1" in VERTEX_DTYPES[fmt].names:
                uv1 = m.uv0.copy()                       # policy: uv1 = uv0 (census majority) …
                if orig.uv1 is not None and has.any():
                    uv1[has] = orig.uv1[ids[has]]        # … except where the raw record is known
            joints = None
            if skinned:
                joints = _joints(m, pal, orig_raw["joints"][np.maximum(ids, 0)], has)
            fv = DecodedVertices.from_floats(fmt, m.positions, m.uv0, m.normals, m.tangents, m.tangent_sign, uv1=uv1,
                                             weights=m.weights, joints=joints)
            rec = encode_block(fv, raw=orig_raw, raw_ids=ids, holes=holes)
            chunks.append(rec)
            raw_id_chunks.append(ids)
            moved = int((~_same_rows(orig.positions[ids[has]], m.positions[has])).sum()) if has.any() else 0
            edited = int(np.sum(rec[has] != orig_raw[ids[has]])) if has.any() else 0
            idx = (m.faces.reshape(-1) + base).astype(np.int64)
            plans.append(SubmeshPlan(m.submesh, m, m.material_slot, m.material_name, idx, pal, pal_changed,
                                     None if sub_orig is None else sub_orig.index_count,
                                     moved=moved, edited=edited, new_vertices=int((~has).sum())))
            base += n
        total = base
        if total > MAX_WINDOW_VERTICES:
            raise BuildError(f"entry {e.index}: {total} vertices exceed the u16 index range ({MAX_WINDOW_VERTICES}); "
                             "split the geometry")
        new_total = sum(p.new_vertices for p in plans)
        if new_total:
            note = (" (format 8: their 40-byte extension is all zero — 42/42 shipped format-8 entries contain such rows, "
                    "the field's meaning is unknown)" if fmt == 8 else "")
            warnings.append(f"entry {e.index}: {new_total} new vertices take the entry's hole policy{note}")
        for p in plans:
            p.indices = p.indices.astype("<u2")
        records = np.concatenate(chunks) if chunks else np.zeros(0, dtype=VERTEX_DTYPES[fmt])
        raw_ids = np.concatenate(raw_id_chunks) if raw_id_chunks else np.zeros(0, dtype=np.int64)
        path = "rebuild"

    layout_changed = (len(records) != e.vertex_count) or (len(plans) != e.submesh_count) or any(
        p.index_count != e.submeshes[p.index].index_count for p in plans if p.index < e.submesh_count)
    changed = layout_changed or records.tobytes() != orig_raw.tobytes() or any(
        p.indices.tobytes() != np.ascontiguousarray(e.submeshes[p.index].indices, dtype="<u2").tobytes()
        for p in plans if p.index < e.submesh_count) or any(p.palette_changed for p in plans)
    return EntryPlan(e, plans, path, records, raw_ids, changed, layout_changed)


def make_plan(model: Model, imp: ResolvedImport, logical_name: bytes | str) -> RebuildPlan:
    from .identity import expected_embedded_name
    warnings = list(imp.warnings)
    by_entry = imp.by_entry()
    plans = [plan_entry(model, e, by_entry.get(e.index, []), warnings) for e in model.geometry_entries]
    extra = sorted(set(by_entry) - {e.index for e in model.geometry_entries})
    if extra:
        raise BuildError(f"Cast meshes refer to geometry entries {extra} that the mesh does not have; adding "
                         "geometry entries (LOD levels) is not supported")
    exp = expected_embedded_name(logical_name)
    rename = None if exp == model.embedded_name else (model.embedded_name, exp)
    return RebuildPlan(plans, list(imp.new_materials), list(imp.bone_changes), rename,
                       any(p.layout_changed for p in plans), warnings)


# --------------------------------------------------------------------------------------------------------------
# buffers
# --------------------------------------------------------------------------------------------------------------

def _assemble_buffers(model: Model, plan: RebuildPlan, *, pad_index_part: bool) -> tuple[bytes, bytes, list[tuple[int, int]]]:
    """→ (vertex bytes, index bytes, [(vertex_base, index_base)] per entry)."""
    entries = [p.entry for p in plan.entries]
    if not plan.layout_changed:
        bases = [(e.vertex_base, e.index_base) for e in entries]
        vsize, isize = model.vertex_buffer_size, model.index_buffer_size
        vout = bytearray(bytes(model.vertex_buffer) if model.vertex_buffer is not None else b"")
        iout = bytearray(bytes(model.index_buffer) if model.index_buffer is not None else b"")
    else:
        # temporary entries carrying the new counts for the planner
        class _N:
            def __init__(self, fmt, nv, ni):
                self.format, self.vertex_count, self.index_count = fmt, nv, ni
        bases, vsize, isize = plan_new_layout([_N(p.entry.format, p.vertex_count, p.index_count) for p in plan.entries],
                                              pad_index_part=pad_index_part)
        vout = bytearray(vsize)
        iout = bytearray(isize)
    for p, (vb, ib) in zip(plan.entries, bases):
        if p.records is not None and len(p.records):
            data = p.records.tobytes()
            if vb + len(data) > len(vout):
                raise BuildError(f"entry {p.entry.index}: vertex window exceeds the buffer (internal)")
            vout[vb:vb + len(data)] = data
        cur = ib
        for s in p.submeshes:
            data = np.ascontiguousarray(s.indices, dtype="<u2").tobytes()
            if cur + len(data) > len(iout):
                raise BuildError(f"entry {p.entry.index}: index list exceeds the buffer (internal)")
            iout[cur:cur + len(data)] = data
            cur += len(data)
    return bytes(vout), bytes(iout), bases


# --------------------------------------------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------------------------------------------

@dataclass
class RebuildResult:
    image: bytes
    fixups: bytes
    vertex: bytes | None
    index: bytes | None
    plan: RebuildPlan
    report: dict


def refuse_layout(model: Model) -> None:
    """The encoder (imagepatch.py) writes DLTB object offsets and DLTB material-name shapes; a DL2 image
    (notes/FORMATS/mesh-dl2.md) has a different class-6 entry, a class-8 stream object and untagged material names.
    Rebuilding one is refused instead of producing a corrupt mesh."""
    if getattr(model, "layout", "dltb") != "dltb":
        raise UnsupportedError(f"{model.name!r}: rebuilding / importing {model.layout.upper()} meshes is not supported "
                               "(read-only: decode, view and Cast export work; the mesh encoder only knows the "
                               "Dying Light: The Beast layout). Keep the original raw parts for this resource.")


def rebuild(model: Model, imp: ResolvedImport, *, logical_name: bytes | str, pad_index_part: bool = True,
            ignore_bone_changes: bool = False) -> RebuildResult:
    """Model (decoded from the raw parts) + resolved Cast → new parts. Raises BuildError/UnsupportedError rather
    than approximating; see the module docstring for the policies."""
    refuse_layout(model)
    plan = make_plan(model, imp, logical_name)
    if plan.bone_changes and not ignore_bone_changes:
        raise UnsupportedError("bone transforms differ from the sidecar (" + "; ".join(plan.bone_changes[:5])
                               + ("; …" if len(plan.bone_changes) > 5 else "") + "): rewriting entity local/inverse-bind "
                               "matrices from a Cast is not supported (Cast bones carry no bind matrices); restore the "
                               "skeleton, or keep the native skeleton with `nr mesh import --ignore-bones` and replace "
                               "the raw parts")
    if plan.bone_changes:
        plan.warnings.append(f"{len(plan.bone_changes)} bone transforms differ from the sidecar; native skeleton kept")
    has_buffers = model.vertex_buffer is not None or any(p.records is not None for p in plan.entries)
    vbytes = ibytes = None
    bases = [(e.vertex_base, e.index_base) for e in model.geometry_entries]
    if has_buffers:
        vbytes, ibytes, bases = _assemble_buffers(model, plan, pad_index_part=pad_index_part)
    image, fixups, patch_report = imagepatch.apply(model, plan, bases)
    report = plan.report()
    report.update(patch_report)
    report["vertex_size"] = [model.vertex_buffer_size, None if vbytes is None else len(vbytes)]
    report["index_size"] = [model.index_buffer_size, None if ibytes is None else len(ibytes)]
    report["image_size"] = [len(model.image.data), len(image)]
    return RebuildResult(image, fixups, vbytes, ibytes, plan, report)


def verify(result: RebuildResult, model: Model, imp: ResolvedImport, name: str) -> list[str]:
    """Self-check: decode the produced parts and compare with the plan (counts, positions of every Cast mesh)."""
    from .decode import decode_parts
    problems: list[str] = []
    m2 = decode_parts(name, result.image, result.fixups, vertex=result.vertex, index=result.index,
                      skin=model.skin_raw, cloth=model.cloth_raw)
    old_w = set(model.warnings)
    for w in m2.warnings:
        if w not in old_w:
            problems.append(f"decode warning: {w}")
    if len(m2.geometry_entries) != len(result.plan.entries):
        problems.append("geometry entry count changed")
        return problems
    for ep, e2 in zip(result.plan.entries, m2.geometry_entries):
        if e2.vertex_count != ep.vertex_count or e2.submesh_count != len(ep.submeshes):
            problems.append(f"entry {e2.index}: counts {e2.vertex_count}/{e2.submesh_count} != plan {ep.vertex_count}/{len(ep.submeshes)}")
            continue
        for sp, s2 in zip(ep.submeshes, e2.submeshes):
            if s2.index_count != sp.index_count:
                problems.append(f"entry {e2.index} submesh {s2.index}: index count {s2.index_count} != {sp.index_count}")
            if not np.array_equal(s2.palette.astype(np.int64), sp.palette.astype(np.int64)):
                problems.append(f"entry {e2.index} submesh {s2.index}: palette mismatch")
            if sp.mesh is not None and e2.vertices is not None and s2.index_count:
                got = e2.vertices.positions[s2.indices.astype(np.int64)]
                want = sp.mesh.positions[sp.mesh.faces.reshape(-1)]
                fin = np.isfinite(want)
                tol = 1e-2 if e2.format == 0 else 0.0
                if not np.allclose(got[fin], want[fin], atol=tol, rtol=0):
                    problems.append(f"entry {e2.index} submesh {s2.index}: positions differ after re-decode")
    return problems
