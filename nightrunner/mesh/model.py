"""Decoded mesh model: everything the codec knows about a type-0x10 resource, with every unknown byte carried raw.

The model is the single source for Cast export, the JSON sidecar, the in-memory round-trip and (phase 2) the
encoder. Numbers that the encoder must be able to rewrite in place (image offsets of every decoded object) are
kept next to the decoded values so the phase-2 patcher never has to re-locate anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..classreader.fixups import Fixups
from ..classreader.image import Image
from .vertex import DecodedVertices


@dataclass
class Submesh:
    index: int                  # position within the entry
    material_slot: int          # u16 index into Model.materials
    index_count: int            # u16 indices (triangle list, multiple of 3 in the corpus)
    index_base: int             # byte offset in the index buffer (entry.index_base + Σ previous counts × 2)
    palette: np.ndarray         # u16 entity ordinals (empty for unskinned formats); joints index into this
    palette_desc_offset: int    # image offset of the 16-byte {ptr, count} descriptor
    palette_offset: int | None  # image offset of the u16[] data (None when null)
    indices: np.ndarray         # u16, relative to the entry's vertex window (zero-copy view of the index buffer)

    @property
    def triangle_count(self) -> int:
        return self.index_count // 3


@dataclass
class GeometryEntry:
    index: int                  # global index over Model.geometry_entries
    array_record: int           # fixups record index of the class-6 array
    element: int                # position inside that array
    offset: int                 # image offset of the entry (0x40 bytes DLTB, 0x30 bytes DL2)
    owner_entity: int | None    # entity whose +0x88 points at this array (None = unowned)
    format: int
    vertex_base: int
    vertex_count: int
    index_base: int
    submeshes: list[Submesh]
    raw_00: bytes               # +0x00..0x07 (C)
    raw_12: int                 # +0x12 u16 (C)
    raw_14: int                 # +0x14 u16 (C)
    raw_17: int                 # +0x17 u8 (C)
    raw_34: bytes               # DLTB +0x34..0x3F / DL2 +0x24..0x2F (C); raw_12/14/17 are 0 on DL2
    material_slots_offset: int | None
    index_counts_offset: int | None
    vertices: DecodedVertices | None = None     # None when the resource has no vertex buffer
    stream_offset: int | None = None            # DL2: image offset of the class-8 stream object (None on DLTB)
    raw_stream: bytes = b""                     # DL2: the 32 class-8 bytes (format/bases/counts live there)

    @property
    def submesh_count(self) -> int:
        return len(self.submeshes)

    @property
    def index_count(self) -> int:
        return sum(s.index_count for s in self.submeshes)

    @property
    def vertex_end(self) -> int:
        from .vertex import STRIDES
        return self.vertex_base + self.vertex_count * STRIDES[self.format]

    @property
    def index_end(self) -> int:
        return self.index_base + self.index_count * 2


@dataclass
class Entity:
    index: int
    offset: int                 # image offset of the record (0xE0 bytes DLTB, 0xD0 bytes DL2)
    name: bytes
    parent: int                 # −1 = root
    local: np.ndarray           # 3×4 f32, local-to-parent (row-major, last column = translation)
    inv_bind: np.ndarray        # 3×4 f32 stored inverse bind
    bounds_center: np.ndarray   # 3 f32
    bounds_half: np.ndarray     # 3 f32
    flags: int                  # +0xC0 (C bit meanings)
    type: int                   # +0xC8
    geometry_count: int         # +0xC9
    geometry_array_record: int | None   # fixups record of the class-6 array at +0x88 (None = null / no array)
    geometry_entries: list[int]         # global GeometryEntry indices owned by this entity
    aux_offset: int | None      # class-5 object at +0x80 (opaque)
    raw_aux: bytes              # class-5 bytes (32 × geometry_count; empty when null)
    raw_90: bytes               # +0x90..0xBF (C)
    raw_ca: bytes               # +0xCA..end of the record (22 bytes DLTB, 6 bytes DL2) (C)

    @property
    def name_str(self) -> str:
        return self.name.decode("utf-8", "surrogateescape")

    def local_4x4(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3] = self.local.astype(np.float64)
        return m

    def inv_bind_4x4(self) -> np.ndarray:
        m = np.eye(4, dtype=np.float64)
        m[:3] = self.inv_bind.astype(np.float64)
        return m


@dataclass
class Material:
    index: int
    offset: int                 # image offset of the 32-byte entry
    name: bytes                 # "<name>.mat"
    name_tag: int               # high 16 bits of the tagged name pointer (0x2100 in the corpus; 0 when inline)
    raw: bytes                  # the whole 32-byte entry (C beyond the name pointer)
    name_inline: bool = False   # name ≤ 7 chars stored inline in the 8-byte field (engine_pc 'sky.mat')

    @property
    def name_str(self) -> str:
        return self.name.decode("utf-8", "surrogateescape")


@dataclass
class OpaqueObject:
    record: int
    class_id: int
    offset: int
    size: int
    data: bytes


@dataclass
class Model:
    name: str                               # logical resource name
    embedded_name: bytes                    # "<name>.msh" from the root
    scr_name: bytes | None                  # root+0x40 string when present
    image: Image                            # parsed primary image (bytes + fixups)
    fixups: Fixups
    entities: list[Entity]
    geometry_entries: list[GeometryEntry]
    materials: list[Material]
    material_capacity: int
    material_header_offset: int | None
    root_raw: bytes                         # root bytes (0x70 DLTB, 0x68 DL2)
    opaque: list[OpaqueObject]              # class 5/12/13/... records kept verbatim
    vertex_buffer: bytes | memoryview | None
    index_buffer: bytes | memoryview | None
    skin_raw: bytes | None                  # part 0x12 (material variants; decoded by mesh/variants.py)
    cloth_raw: bytes | None                 # part 0xF3 (opaque)
    warnings: list[str] = field(default_factory=list)
    layout: str = "dltb"                    # object layout detected from the image: "dltb" | "dl2"

    @property
    def is_dl2(self) -> bool:
        return self.layout == "dl2"

    # ---- derived ----------------------------------------------------------------------------------------------

    @property
    def vertex_buffer_size(self) -> int:
        return 0 if self.vertex_buffer is None else len(self.vertex_buffer)

    @property
    def index_buffer_size(self) -> int:
        return 0 if self.index_buffer is None else len(self.index_buffer)

    @property
    def skinned(self) -> bool:
        return any(e.vertices is not None and e.vertices.skinned for e in self.geometry_entries)

    def entity_globals(self) -> np.ndarray:
        """4×4 float64 global transforms: parent_global · local (parents may be forward references)."""
        n = len(self.entities)
        out = np.zeros((n, 4, 4))
        state = np.zeros(n, dtype=np.uint8)
        for start in range(n):
            chain = []
            i = start
            while i >= 0 and state[i] != 2:
                if state[i] == 1:
                    raise ValueError("entity parent cycle")
                state[i] = 1
                chain.append(i)
                i = self.entities[i].parent
            for i in reversed(chain):
                p = self.entities[i].parent
                out[i] = out[p] @ self.entities[i].local_4x4() if p >= 0 else self.entities[i].local_4x4()
                state[i] = 2
        return out

    def material_name(self, slot: int) -> str:
        if 0 <= slot < len(self.materials):
            return self.materials[slot].name_str
        return f"material_{slot}"

    def summary(self) -> dict:
        return {
            "name": self.name, "layout": self.layout, "embedded_name": self.embedded_name.decode("utf-8", "surrogateescape"),
            "entities": len(self.entities), "geometry_entries": len(self.geometry_entries),
            "submeshes": sum(e.submesh_count for e in self.geometry_entries),
            "formats": sorted({e.format for e in self.geometry_entries}),
            "vertices": sum(e.vertex_count for e in self.geometry_entries),
            "triangles": sum(s.triangle_count for e in self.geometry_entries for s in e.submeshes),
            "materials": [m.name_str for m in self.materials],
            "vertex_buffer": self.vertex_buffer_size, "index_buffer": self.index_buffer_size,
            "skin": None if self.skin_raw is None else len(self.skin_raw),
            "cloth": None if self.cloth_raw is None else len(self.cloth_raw),
            "warnings": list(self.warnings),
        }
