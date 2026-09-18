"""Image (0x10) + fixups (0x11) edits for a RebuildPlan, through `classreader.image.ImagePatch`.

What is written (everything else is carried verbatim; notes/FORMATS/mesh.md §encoder):

  geometry entry   +0x10 u16 submesh count; +0x28/+0x2C/+0x30 vertex base / count / index base;
                   u16 material slot[nsub] and u32 index count[nsub]: in place when nsub did not grow, else a new
                   array appended (2-/4-byte aligned) and the +0x08 / +0x18 slot retargeted;
                   PaletteDesc[nsub] (class 7): descriptors reused when nsub did not grow (changed palettes get a
                   new u16 array appended and the descriptor's pointer/count rewritten); when nsub grows a new
                   descriptor array is appended (8-aligned), its pointer slots added, the +0x20 slot retargeted
                   and the class-7 record relocated to it. The class-7 record count follows nsub (census
                   2026-09-15: 27,971/27,971 records equal the entry's submesh count).
  material table   a NEW name goes into entry[count] (the table is allocated with `capacity` entries; the spare
                   entries of shipped meshes already carry a tagged name slot): the string is appended in the
                   shipped shape {u32 len, u32 capacity} + chars + NUL (8-aligned; census material.len_prefix_ok
                   161,363/161,363), the entry's +0x08 tagged slot (tag 0x2100) retargeted, count += 1. No spare
                   entry ⇒ BuildError (a longer class-11 array would need a new record + retargeting the class-10
                   pointer, not done: class-11's other 24 bytes are opaque). No class-0 record is added for the
                   new string — the identity rename (validated in game, survey 02 §3) adds none either (C).
  entity           +0x60 f32[6] bounds when its geometry changed (rebuild.py policy).
  root             +0x00 name pointer → appended "<logical>.msh" when the logical name differs (identity.py rule).
"""

from __future__ import annotations

import struct

import numpy as np

from ..classreader.graph import CLASS_PALETTE, PALETTE_DESC_SIZE, MATERIAL_ENTRY_SIZE, SLOT_TAG_MATERIAL
from ..classreader.image import ImagePatch
from ..errors import BuildError
from .model import Model
from .vertex import is_skinned


def _u16s(vals) -> bytes:
    return struct.pack(f"<{len(vals)}H", *vals)


def _u32s(vals) -> bytes:
    return struct.pack(f"<{len(vals)}I", *vals)


def apply(model: Model, plan, bases: list[tuple[int, int]]) -> tuple[bytes, bytes, dict]:
    if getattr(model, "layout", "dltb") != "dltb":
        raise BuildError(f"{model.name!r}: the image patcher only supports the DLTB mesh layout (got {model.layout})")
    patch = ImagePatch(model.image)
    rep = {"materials_added": [], "arrays_appended": 0, "records_relocated": 0, "slots_added": 0,
           "bounds_rewritten": [], "identity_renamed": False}
    slots_before = len(patch.fixups.slots)

    # ---- materials -----------------------------------------------------------------------------------------
    new_slot: dict[str, int] = {}
    if plan.new_materials:
        _append_materials(model, plan.new_materials, patch, new_slot, rep)

    # ---- geometry entries ------------------------------------------------------------------------------------
    for ep, (vb, ib) in zip(plan.entries, bases):
        e = ep.entry
        patch.write_u32(e.offset + 0x28, vb)
        patch.write_u32(e.offset + 0x2C, ep.vertex_count)
        patch.write_u32(e.offset + 0x30, ib)
        nsub_new, nsub_old = len(ep.submeshes), e.submesh_count
        if nsub_new == 0 and nsub_old == 0:
            continue
        if nsub_new != nsub_old:
            patch.write_u16(e.offset + 0x10, nsub_new)
        slots = []
        for s in ep.submeshes:
            slot = s.material_slot if s.material_slot is not None else new_slot.get(s.material_name)
            if slot is None:
                raise BuildError(f"material {s.material_name!r} has no slot (internal)")
            s.material_slot = slot
            slots.append(slot)
        counts = [s.index_count for s in ep.submeshes]
        grow = nsub_new > nsub_old
        if not grow and e.material_slots_offset is not None:
            patch.write(e.material_slots_offset, _u16s(slots))
            patch.write(e.index_counts_offset, _u32s(counts))
        else:
            off = patch.append(_u16s(slots), 2)
            patch.retarget(e.offset + 0x08, off, kind=0)
            off = patch.append(_u32s(counts), 4)
            patch.retarget(e.offset + 0x18, off, kind=0)
            rep["arrays_appended"] += 2
        # palettes
        if not grow:
            for s in ep.submeshes:
                d = e.submeshes[s.index].palette_desc_offset
                if not s.palette_changed:
                    continue
                if len(s.palette):
                    off = patch.append(_u16s(s.palette.tolist()), 2)
                    patch.retarget(d, off, kind=0)
                    rep["arrays_appended"] += 1
                else:
                    patch.retarget(d, None, kind=0)
                patch.write_u64(d + 8, len(s.palette))
            if nsub_new < nsub_old:
                ri = patch.fixups.record_at(e.submeshes[0].palette_desc_offset)
                if ri is not None and patch.fixups.records[ri].class_id == CLASS_PALETTE:
                    patch.relocate_record(ri, e.submeshes[0].palette_desc_offset, nsub_new)
                    rep["records_relocated"] += 1
        else:
            off_d = patch.append(b"\0" * (PALETTE_DESC_SIZE * nsub_new), 8)
            for s in ep.submeshes:
                d = off_d + PALETTE_DESC_SIZE * s.index
                if s.index < nsub_old and not s.palette_changed:
                    target = e.submeshes[s.index].palette_offset
                else:
                    target = patch.append(_u16s(s.palette.tolist()), 2) if len(s.palette) else None
                    rep["arrays_appended"] += int(target is not None)
                if target is not None:
                    patch.retarget(d, target, kind=0)
                patch.write_u64(d + 8, len(s.palette))
            patch.retarget(e.offset + 0x20, off_d, kind=0)
            ri = patch.fixups.record_at(e.submeshes[0].palette_desc_offset) if nsub_old else None
            if ri is not None and patch.fixups.records[ri].class_id == CLASS_PALETTE:
                patch.relocate_record(ri, off_d, nsub_new)
                rep["records_relocated"] += 1
            else:
                patch.add_record(off_d, CLASS_PALETTE, nsub_new)
            rep["arrays_appended"] += 1

    # ---- bounds ----------------------------------------------------------------------------------------------
    changed_entries = {ep.entry.index for ep in plan.entries if ep.changed}
    by_index = {ep.entry.index: ep for ep in plan.entries}
    for en in model.entities:
        if not en.geometry_entries or not (set(en.geometry_entries) & changed_entries):
            continue
        pts = []
        fmts = []
        for gi in en.geometry_entries:
            ep = by_index[gi]
            fmts.append(ep.entry.format)
            if ep.records is not None and len(ep.records):
                p = ep.records["pos"].astype(np.float64)
                pts.append(p[np.isfinite(p).all(axis=1)])
        pts = [p for p in pts if len(p)]
        if not pts:
            continue
        allp = np.concatenate(pts)
        lo, hi = allp.min(0), allp.max(0)
        if not all(is_skinned(f) for f in fmts):
            c, h = en.bounds_center.astype(np.float64), en.bounds_half.astype(np.float64)
            lo = np.minimum(lo, c - h)
            hi = np.maximum(hi, c + h)
        centre, half = (lo + hi) / 2, (hi - lo) / 2
        patch.write_f32s(en.offset + 0x60, list(centre) + list(half))
        rep["bounds_rewritten"].append(en.index)

    # ---- identity ----------------------------------------------------------------------------------------------
    if plan.identity_rename is not None:
        _, exp = plan.identity_rename
        off = patch.append_string(exp)
        patch.retarget(model.image.records[0].offset, off)
        rep["identity_renamed"] = True

    image, fixups = patch.finish()
    rep["slots_added"] = len(patch.fixups.slots) - slots_before
    return image, fixups, rep


def _append_materials(model: Model, names: list[str], patch: ImagePatch, new_slot: dict[str, int], rep: dict) -> None:
    if model.material_header_offset is None:
        raise BuildError(f"cannot add material(s) {names}: the mesh has no class-10 material table")
    hdr = model.material_header_offset
    count, capacity = len(model.materials), model.material_capacity
    entries_ptr = model.image.pointer(hdr)
    if entries_ptr.target is None:
        raise BuildError("material table pointer is null")
    base = entries_ptr.target
    for name in names:
        if count >= capacity:
            raise BuildError(f"cannot add material {name!r}: the class-11 table is full (count {count} == capacity "
                             f"{capacity}); shipped meshes carry spare entries in 13,721/21,354 cases only — pick an "
                             "existing material name or a mesh with spare capacity")
        raw = name.encode("utf-8")
        if b"\0" in raw or not raw:
            raise BuildError(f"invalid material name {name!r}")
        ent = base + count * MATERIAL_ENTRY_SIZE
        # string object: {u32 len, u32 cap} + chars + NUL (shipped shape), pointer at the chars
        blob = struct.pack("<II", len(raw), len(raw)) + raw + b"\0"
        off = patch.append(blob, 8) + 8
        slot = ent + 0x08
        if patch.slot_kind(slot) is None:
            patch.retarget(slot, off, kind=2, tag=SLOT_TAG_MATERIAL)
        else:
            patch.retarget(slot, off)
        # opaque words of a never-used spare entry: copy +0x18..+0x1F from the last real entry when zero (C)
        if count > 0 and patch.u64(ent + 0x18) == 0:
            patch.write(ent + 0x18, bytes(patch.buf[base + (count - 1) * MATERIAL_ENTRY_SIZE + 0x18: base + (count - 1) * MATERIAL_ENTRY_SIZE + 0x20]))
        new_slot[name] = count
        count += 1
        patch.write_u16(hdr + 0x08, count)
        rep["materials_added"].append({"name": name, "slot": count - 1})
