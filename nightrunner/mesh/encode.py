"""Model → vertex (0xF0) / index (0xF1) / fixups (0x11) bytes.

Phase 1 (this file): re-encode the buffers of an unedited or attribute-edited model into the ORIGINAL layout —
every entry keeps its vertex_base/index_base and the buffer sizes; bytes not covered by any entry (padding
between entries, the tail) are copied from the source buffers when present, else zero. This is what
`MeshCodec.roundtrip` runs over the corpus; it is byte-identical for every shipped mesh (see notes/FORMATS/mesh.md).

Layout rule for NEW buffers (phase 2, census 2026-09-15 — see notes/FORMATS/mesh.md §buffers):
    vertex: entries in entry order, each window zero-padded to a multiple of 160 bytes (lcm of all strides)
    index : entries in entry order, each entry base aligned to 4, the part zero-padded to 16
"""

from __future__ import annotations

import numpy as np

from ..errors import FormatError
from .model import GeometryEntry, Model
from .vertex import (INDEX_BASE_ALIGN, INDEX_BUFFER_ALIGN, STRIDES, VERTEX_BLOCK_ALIGN, align_to, encode_block)


def encode_vertex_buffer(model: Model, *, source: bytes | memoryview | None = None, size: int | None = None) -> bytes:
    """Re-encode every decoded entry into a buffer of the original layout. Bytes outside the entry windows come
    from *source* (default: the model's own buffer) so padding is reproduced exactly."""
    src = model.vertex_buffer if source is None else source
    total = (len(src) if src is not None else 0) if size is None else size
    out = bytearray(bytes(src[:total]) if src is not None else b"\0" * total)
    if len(out) < total:
        out += b"\0" * (total - len(out))
    for e in model.geometry_entries:
        if e.vertices is None:
            continue
        rec = encode_block(e.vertices)
        data = rec.tobytes()
        end = e.vertex_base + len(data)
        if end > total:
            raise FormatError(f"entry {e.index}: window [{e.vertex_base}, {end}) exceeds the buffer ({total})")
        out[e.vertex_base:end] = data
    return bytes(out)


def encode_index_buffer(model: Model, *, source: bytes | memoryview | None = None, size: int | None = None) -> bytes:
    src = model.index_buffer if source is None else source
    total = (len(src) if src is not None else 0) if size is None else size
    out = bytearray(bytes(src[:total]) if src is not None else b"\0" * total)
    if len(out) < total:
        out += b"\0" * (total - len(out))
    for e in model.geometry_entries:
        for s in e.submeshes:
            data = np.ascontiguousarray(s.indices, dtype="<u2").tobytes()
            if len(data) != s.index_count * 2:
                raise FormatError(f"entry {e.index} submesh {s.index}: {len(data) // 2} indices != declared {s.index_count}")
            end = s.index_base + len(data)
            if end > total:
                raise FormatError(f"entry {e.index} submesh {s.index}: indices exceed the buffer ({total})")
            out[s.index_base:end] = data
    return bytes(out)


def plan_new_layout(entries: list[GeometryEntry], *, pad_index_part: bool = True) -> tuple[list[tuple[int, int]], int, int]:
    """Phase-2 helper: (vertex_base, index_base) per entry for freshly laid out buffers + total sizes.
    *pad_index_part*: the bit-12 packs store every mesh part zero-padded to 16 bytes (census: 21,297/21,297 meshes
    of the four on-demand packs); engine_pc (field08 = 0) stores index parts at their exact size (57/57)."""
    bases = []
    vcur = icur = 0
    for e in entries:
        vb = vcur
        ib = align_to(icur, INDEX_BASE_ALIGN)
        bases.append((vb, ib))
        vcur = align_to(vb + e.vertex_count * STRIDES[e.format], VERTEX_BLOCK_ALIGN)
        icur = ib + e.index_count * 2
    return bases, vcur, align_to(icur, INDEX_BUFFER_ALIGN) if pad_index_part else icur
