"""RPACK mesh resource (parts 0x10/0x11/0x12/0xF0/0xF1/0xF3) → Model.

Both object layouts are decoded — DLTB and DL2 (classreader/graph.py `MeshLayout`, notes/FORMATS/mesh-dl2.md);
the layout is detected from the image and recorded in `Model.layout`.

Decodes EVERY class-6 geometry array and every entry in it (multi-entry arrays are LOD chains), every submesh,
every entity, the material table, and keeps every other record as an opaque span. Nothing is skipped and nothing
is guessed: an entry with an unsupported vertex format still appears in the model (vertices = None) and is
reported in `Model.warnings`; a resource without vertex/index parts (skeleton-only) decodes with empty buffers.
"""

from __future__ import annotations

import numpy as np

from ..classreader.fixups import Fixups
from ..classreader.graph import MeshGraph, GEOMETRY_AUX_SIZE, CLASS_GEOMETRY_AUX
from ..classreader.image import Image
from ..container.rp6l import Resource
from ..errors import FormatError
from .model import Entity, GeometryEntry, Material, Model, OpaqueObject, Submesh
from .vertex import STRIDES, decode_block, is_skinned, supported

PART_IMAGE, PART_FIXUPS, PART_SKIN, PART_VERTEX, PART_INDEX, PART_CLOTH = 0x10, 0x11, 0x12, 0xF0, 0xF1, 0xF3
MAX_VERTICES_PER_ENTRY = 1 << 24      # offline bound (u16 indices address 65,536; the buffer address is u32)


def decode_resource(res: Resource, *, layout: str | None = None) -> Model:
    """Decode a type-0x10 resource from an open pack (parts are copied so the model outlives the mmap).
    The DLTB / DL2 layout is detected from the data; *layout* ("dltb" / "dl2") is an optional hint."""
    if res.type != 0x10:
        raise FormatError(f"{res.name!r} is type 0x{res.type:02X}, not a mesh")
    parts = {}
    for t in (PART_IMAGE, PART_FIXUPS, PART_SKIN, PART_VERTEX, PART_INDEX, PART_CLOTH):
        mv = res.read_part_by_type(t)
        parts[t] = None if mv is None else bytes(mv)     # copy: the model must outlive the pack's mmap
    for t in (PART_IMAGE, PART_FIXUPS):
        if parts[t] is None:
            raise FormatError(f"{res.name!r}: missing part 0x{t:02X}")
    return decode_parts(res.name, parts[PART_IMAGE], parts[PART_FIXUPS], vertex=parts[PART_VERTEX],
                        index=parts[PART_INDEX], skin=parts[PART_SKIN], cloth=parts[PART_CLOTH], layout=layout)


def decode_parts(name: str, image_bytes, fixups_bytes, *, vertex=None, index=None, skin=None, cloth=None,
                 layout: str | None = None) -> Model:
    fx = Fixups.parse(fixups_bytes)
    img = Image(image_bytes, fx)
    g = MeshGraph(img, layout)
    warnings: list[str] = []
    if fx.secondary_present:
        warnings.append("fixups declare a secondary image (parsed, not interpreted)")
    bad = fx.unsupported_slots()
    if bad:
        warnings.append(f"{len(bad)} relocation slots of kinds {sorted({s.kind for s in bad})} are not resolvable offline")

    root = g.root
    embedded = root.name
    if embedded is None:
        raise FormatError("root name pointer is null")
    scr = root.scr_name

    # ---- geometry arrays → flat entry list -------------------------------------------------------------------
    vbuf = None if vertex is None else (vertex if isinstance(vertex, (bytes, memoryview)) else memoryview(vertex))
    ibuf = None if index is None else (index if isinstance(index, (bytes, memoryview)) else memoryview(index))
    entries: list[GeometryEntry] = []
    array_first_entry: dict[int, int] = {}
    for ga in g.geometry_arrays:
        array_first_entry[ga.record] = len(entries)
        for k in range(ga.count):
            v = ga.entry(img, k)
            entries.append(_decode_entry(v, len(entries), ga.record, k, vbuf, ibuf, warnings))
    if not g.geometry_arrays and vbuf is not None and len(vbuf):
        warnings.append("vertex buffer present but no class-6 geometry array")

    # ---- entities -------------------------------------------------------------------------------------------
    entities: list[Entity] = []
    owned: dict[int, int] = {}
    for i, ev in enumerate(g.entities()):
        gp = ev.geometry_ptr
        rec = None
        owned_entries: list[int] = []
        if gp is not None and gp.target is not None:
            ga = g.geometry_array_at(gp.target)
            if ga is None:
                warnings.append(f"entity {i}: +0x88 → 0x{gp.target:X} is not the start of a class-6 record")
            else:
                rec = ga.record
                first = array_first_entry[rec]
                owned_entries = list(range(first, first + ga.count))
                if ga.count != ev.geometry_count:
                    warnings.append(f"entity {i}: geometry_count {ev.geometry_count} != class-6 record count {ga.count}")
                for e in owned_entries:
                    if e in owned:
                        warnings.append(f"geometry entry {e} owned by entities {owned[e]} and {i}")
                    owned[e] = i
        elif ev.geometry_count:
            warnings.append(f"entity {i}: geometry_count {ev.geometry_count} but no geometry pointer")
        ap = ev.aux_ptr
        aux_off = None
        raw_aux = b""
        if ap is not None and ap.target is not None:
            aux_off = ap.target
            ri = img.record_at(aux_off)
            if ri is not None and img.records[ri].class_id == CLASS_GEOMETRY_AUX:
                a, b = img.record_span(ri)
                raw_aux = img.raw(a, b - a)
            else:
                raw_aux = img.raw(aux_off, GEOMETRY_AUX_SIZE * max(1, ev.geometry_count))
                warnings.append(f"entity {i}: +0x80 → 0x{aux_off:X} is not a class-5 record")
        if ev.own_index != i:
            warnings.append(f"entity {i}: own index field is {ev.own_index}")
        parent = ev.parent
        if parent < -1 or parent >= g.entity_count or parent == i:
            raise FormatError(f"entity {i}: invalid parent {parent}")
        entities.append(Entity(
            index=i, offset=ev.off, name=ev.name, parent=parent,
            local=np.array(ev.local, dtype=np.float32).reshape(3, 4),
            inv_bind=np.array(ev.inv_bind, dtype=np.float32).reshape(3, 4),
            bounds_center=np.array(ev.bounds_center, dtype=np.float32),
            bounds_half=np.array(ev.bounds_half, dtype=np.float32),
            flags=ev.flags, type=ev.type, geometry_count=ev.geometry_count,
            geometry_array_record=rec, geometry_entries=owned_entries, aux_offset=aux_off, raw_aux=raw_aux,
            raw_90=ev.raw_90, raw_ca=ev.raw_ca,
        ))
    for e in entries:
        e.owner_entity = owned.get(e.index)
        if e.owner_entity is None:
            warnings.append(f"geometry entry {e.index} (record {e.array_record}) is not owned by any entity")
    if root.entity_count != g.entity_count:
        warnings.append(f"root entity count {root.entity_count} != class-4 record count {g.entity_count}")

    # ---- materials -----------------------------------------------------------------------------------------
    materials: list[Material] = []
    capacity = 0
    mh_off = None
    if g.material_header is not None:
        mh_off = g.material_header.off
        capacity = g.material_header.capacity
        for k, mv in enumerate(g.material_header.entries()):
            nm = mv.name
            if nm is None:
                warnings.append(f"material {k}: null name pointer")
                nm = b""
            ptr = mv.name_ptr
            materials.append(Material(k, mv.off, nm, 0 if ptr is None else ptr.tag, mv.raw_bytes, mv.name_inline))
    for e in entries:
        for s in e.submeshes:
            if s.material_slot >= len(materials):
                warnings.append(f"entry {e.index} submesh {s.index}: material slot {s.material_slot} outside the table ({len(materials)})")
            for pe in s.palette:
                if pe >= len(entities):
                    raise FormatError(f"entry {e.index} submesh {s.index}: palette entity {pe} >= {len(entities)}")

    # ---- opaque records -----------------------------------------------------------------------------------
    opaque = [OpaqueObject(ri, cid, a, b - a, img.raw(a, b - a)) for ri, cid, a, b in g.opaque_records()
              if cid != CLASS_GEOMETRY_AUX]      # class 5 is carried on its entity

    return Model(
        name=name, embedded_name=embedded, scr_name=scr, image=img, fixups=fx, entities=entities,
        geometry_entries=entries, materials=materials, material_capacity=capacity, material_header_offset=mh_off,
        root_raw=root.raw_bytes, opaque=opaque, vertex_buffer=vbuf, index_buffer=ibuf,
        skin_raw=None if skin is None else bytes(skin), cloth_raw=None if cloth is None else bytes(cloth),
        warnings=warnings, layout=g.layout.name,
    )


def _decode_entry(v, index: int, record: int, element: int, vbuf, ibuf, warnings: list[str]) -> GeometryEntry:
    fmt = v.format
    nsub = v.submesh_count
    vb, nv, ib = v.vertex_base, v.vertex_count, v.index_base
    if v.stream_offset is not None and v.stream_submesh_count != nsub:
        warnings.append(f"entry {index}: class-8 submesh count {v.stream_submesh_count} != entry count {nsub}")
    slots = v.material_slots()
    counts = v.index_counts()
    descs = v.palette_descs()
    submeshes: list[Submesh] = []
    cursor = ib
    for k in range(nsub):
        d = descs[k]
        pal = np.array(d.entries(), dtype=np.uint16)
        cnt = counts[k]
        if ibuf is not None:
            if cursor + cnt * 2 > len(ibuf):
                raise FormatError(f"entry {index} submesh {k}: indices [{cursor}, +{cnt * 2}) exceed the index buffer ({len(ibuf)})")
            idx = np.frombuffer(ibuf, dtype="<u2", count=cnt, offset=cursor)
        else:
            idx = np.zeros(0, dtype="<u2")
            if cnt:
                warnings.append(f"entry {index} submesh {k}: {cnt} indices declared but no index part")
        if cnt % 3:
            warnings.append(f"entry {index} submesh {k}: index count {cnt} is not a multiple of 3")
        if len(idx) and nv and int(idx.max()) >= nv:
            warnings.append(f"entry {index} submesh {k}: index {int(idx.max())} >= vertex count {nv}")
        ptr = d.entries_ptr
        submeshes.append(Submesh(k, slots[k], cnt, cursor, pal, d.off, ptr.target, idx))
        cursor += cnt * 2
    vertices = None
    if not supported(fmt):
        warnings.append(f"entry {index}: unsupported vertex format {fmt} (vertices kept raw in the buffer)")
    elif vbuf is not None:
        if nv > MAX_VERTICES_PER_ENTRY:
            raise FormatError(f"entry {index}: {nv} vertices exceed the offline bound")
        vertices = decode_block(vbuf, vb, nv, fmt)
    elif nv:
        warnings.append(f"entry {index}: {nv} vertices declared but no vertex part")
    return GeometryEntry(
        index=index, array_record=record, element=element, offset=v.off, owner_entity=None, format=fmt,
        vertex_base=vb, vertex_count=nv, index_base=ib, submeshes=submeshes,
        raw_00=v.raw_00, raw_12=v.raw_12, raw_14=v.raw_14, raw_17=v.raw_17, raw_34=v.raw_34,
        material_slots_offset=v.material_slots_offset,
        index_counts_offset=v.index_counts_offset,
        vertices=vertices, stream_offset=v.stream_offset, raw_stream=v.raw_stream,
    )
