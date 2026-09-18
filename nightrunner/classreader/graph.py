"""Typed views over the mesh object graph inside a ClassReader image.

Every view is a thin cursor (image + offset); nothing is copied except through `raw_*` accessors. Field
provenance (survey 02 §2.4/§4/§7 unless marked *census*, which means established by the 2026-09-15 corpus census
recorded in notes/FORMATS/classreader.md; all census claims carry their N/N there):

Class 3 root (0x70 bytes, record 0, count 1)
    0x00 ptr  → "<name>.msh"                 (A)  registration key
    0x08 ptr  → class-4 entity array         (census)
    0x10 ptr  → class-13 object              (census; opaque)
    0x18 ptr  → class-10 material header     (census)
    0x28 ptr  → class-12 object (optional)   (census; opaque)
    0x30 ptr  → class-12 object (optional)   (census; opaque)
    0x40 ptr  → "<name>.scr" string          (census)
    0x48 ptr  → optional object              (census; opaque)
    0x50 ptr  → optional object              (census; opaque)
    0x58 u32  entity count                   (census)
    0x5C..0x6F  raw words                    (C)
Class 4 entity (0xE0 bytes, count = entities)
    0x00 f32[12] local 3×4; 0x30 f32[12] inverse bind 3×4; 0x60 f32[3] bounds centre; 0x6C f32[3] bounds half extent;
    0x78 ptr → name; 0x80 ptr → class-5 object (32 bytes × geometry count, census); 0x88 ptr → class-6 geometry
    array (census); 0x90..0xBF raw (C); 0xC0 u32 flags; 0xC4 u16 own index; 0xC6 i16 parent; 0xC8 u8 type;
    0xC9 u8 geometry count; 0xCA..0xDF raw (C)
Class 6 geometry entry (0x40 bytes, count = entries of one array)
    0x00 raw[8] (C); 0x08 ptr → u16 material slot[nsub]; 0x10 u16 nsub; 0x12 u16 raw; 0x14 u16 raw; 0x16 u8 format;
    0x17 u8 raw; 0x18 ptr → u32 index count[nsub]; 0x20 ptr → class-7 palette descriptors[nsub];
    0x28 u32 vertex byte base; 0x2C u32 vertex count; 0x30 u32 index byte base; 0x34..0x3F raw (C)
Class 7 palette descriptor (16 bytes, count = nsub of the owning entry, census)
    0x00 ptr → u16 entity index[count] (null when count 0); 0x08 u64 count
Class 10 material header (16 bytes): 0x00 ptr → class-11 entries; 0x08 u16 count; 0x0A u16 capacity; 0x0C raw
Class 11 material entry (32 bytes): 0x08 ptr (tagged) → "<name>.mat"; the other 24 bytes raw (C)

Two layouts exist (`MeshLayout`, detected from the data by `detect_layout`, notes/FORMATS/mesh-dl2.md):
    DLTB  root 0x70, entity 0xE0, class-6 entry 0x40 (everything above)
    DL2   root 0x68 (no +0x68 word), entity 0xD0 (raw tail +0xCA is 6 bytes), class-6 entry 0x30 that points to a
          class-8 "stream" object (0x20 bytes) holding format / vertex base / vertex count / index base:
      class 6: 0x00 raw[8]; 0x08 ptr → class-8 stream; 0x10 ptr → u16 material slot[nsub];
               0x18 ptr → class-7 palette descriptors[nsub]; 0x20 u32 nsub; 0x24..0x2F raw (C)
      class 8: 0x00 ptr → u32 index count[nsub]; 0x08 u32 vertex byte base; 0x0C u32 vertex count;
               0x10 u32 index byte base; 0x14 u32 nsub copy (C); 0x18 u32 vertex format; 0x1C u32 raw (C)
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..errors import FormatError
from .image import Image, Pointer

CLASS_ROOT = 3
CLASS_ENTITY = 4
CLASS_GEOMETRY_AUX = 5
CLASS_GEOMETRY = 6
CLASS_PALETTE = 7
CLASS_MATERIAL_HEADER = 10
CLASS_STREAM = 8            # DL2 only
CLASS_MATERIAL_ENTRY = 11

ROOT_SIZE = 0x70
ENTITY_SIZE = 0xE0
GEOMETRY_SIZE = 0x40
PALETTE_DESC_SIZE = 0x10
MATERIAL_HEADER_SIZE = 0x10
MATERIAL_ENTRY_SIZE = 0x20
GEOMETRY_AUX_SIZE = 0x20
DL2_ROOT_SIZE = 0x68
DL2_ENTITY_SIZE = 0xD0
DL2_GEOMETRY_SIZE = 0x30
DL2_STREAM_SIZE = 0x20

SLOT_TAG_MATERIAL = 0x2100  # tag of every class-11 name pointer (census 2026-09-15: 161,363/161,364; the other is inline)

MAX_ENTITIES = 65535        # own index is a u16
MAX_SUBMESHES = 4096        # offline bound (B)
MAX_MATERIALS = 4096        # offline bound (B)
MAX_PALETTE = 256           # joint bytes are u8 (A: vertex layout)


@dataclass(frozen=True)
class MeshLayout:
    """Sizes of the fixed mesh objects of one engine generation."""
    name: str               # "dltb" | "dl2"
    root_size: int
    entity_size: int
    geometry_size: int

    @property
    def is_dl2(self) -> bool:
        return self.name == "dl2"


LAYOUT_DLTB = MeshLayout("dltb", ROOT_SIZE, ENTITY_SIZE, GEOMETRY_SIZE)
LAYOUT_DL2 = MeshLayout("dl2", DL2_ROOT_SIZE, DL2_ENTITY_SIZE, DL2_GEOMETRY_SIZE)
LAYOUTS = {"dltb": LAYOUT_DLTB, "dl2": LAYOUT_DL2}


def _layout_votes(fx) -> set[str]:
    votes: set[str] = set()
    recs = fx.records
    if recs and not recs[0].secondary:
        a, b = fx.record_span(0)
        if b - a == ROOT_SIZE:
            votes.add("dltb")
        elif b - a == DL2_ROOT_SIZE:
            votes.add("dl2")
    for i, r in enumerate(recs):
        if r.secondary or not r.count:
            continue
        if r.class_id == CLASS_ENTITY:
            a, b = fx.record_span(i)
            if b - a == r.count * ENTITY_SIZE:
                votes.add("dltb")
            elif b - a == r.count * DL2_ENTITY_SIZE:
                votes.add("dl2")
        elif r.class_id == CLASS_STREAM:
            votes.add("dl2")
    return votes


def detect_layout(fx, hint: str | MeshLayout | None = None) -> MeshLayout:
    """The mesh layout of an image, from the data: the class-3 root span (0x70 DLTB / 0x68 DL2), the class-4
    entity span (count × 0xE0 / count × 0xD0) and the presence of class-8 records (DL2 only). *hint* ("dltb" /
    "dl2") is used only when the data does not decide; a hint that contradicts the data raises FormatError."""
    if isinstance(hint, MeshLayout):
        hint = hint.name
    if hint is not None and hint not in LAYOUTS:
        raise FormatError(f"unknown mesh layout hint {hint!r}")
    votes = _layout_votes(fx)
    if len(votes) == 1:
        found = votes.pop()
        if hint is not None and hint != found:
            raise FormatError(f"mesh layout hint {hint!r} contradicts the data ({found})")
        return LAYOUTS[found]
    if len(votes) > 1:
        raise FormatError("mesh layout is ambiguous: the root/entity/class-8 records match both DLTB and DL2")
    if hint is not None:
        return LAYOUTS[hint]
    raise FormatError("mesh layout not recognised: record 0 is neither a 0x70 (DLTB) nor a 0x68 (DL2) root")


class View:
    __slots__ = ("img", "off")

    def __init__(self, img: Image, off: int):
        self.img = img
        self.off = off

    def raw(self, rel: int, size: int) -> bytes:
        return self.img.raw(self.off + rel, size)

    def ptr(self, rel: int) -> Pointer:
        return self.img.pointer(self.off + rel)

    def opt_ptr(self, rel: int) -> Pointer | None:
        """Pointer at *rel* if that field is a relocation slot, else None (field holds a plain word)."""
        return self.img.pointer(self.off + rel) if self.img.is_slot(self.off + rel) else None


class RootView(View):
    __slots__ = ("size",)

    def __init__(self, img: Image, off: int, size: int = ROOT_SIZE):
        super().__init__(img, off)
        self.size = size
    @property
    def name(self) -> bytes | None:
        return self.img.string_at(self.off + 0x00)

    @property
    def entities_ptr(self) -> Pointer:
        return self.ptr(0x08)

    @property
    def class13_ptr(self) -> Pointer | None:
        return self.opt_ptr(0x10)

    @property
    def materials_ptr(self) -> Pointer | None:
        return self.opt_ptr(0x18)

    @property
    def scr_name(self) -> bytes | None:
        return self.img.string_at(self.off + 0x40) if self.img.is_slot(self.off + 0x40) else None

    @property
    def entity_count(self) -> int:
        return self.img.u32(self.off + 0x58)

    def pointer_slots(self) -> list[Pointer]:
        return [self.img.pointer(self.off + rel) for rel in range(0, self.size, 8) if self.img.is_slot(self.off + rel)]

    @property
    def raw_bytes(self) -> bytes:
        return self.raw(0, self.size)


class EntityView(View):
    __slots__ = ("size",)

    def __init__(self, img: Image, off: int, size: int = ENTITY_SIZE):
        super().__init__(img, off)
        self.size = size
    @property
    def local(self) -> tuple:
        return self.img.f32s(self.off + 0x00, 12)

    @property
    def inv_bind(self) -> tuple:
        return self.img.f32s(self.off + 0x30, 12)

    @property
    def bounds_center(self) -> tuple:
        return self.img.f32s(self.off + 0x60, 3)

    @property
    def bounds_half(self) -> tuple:
        return self.img.f32s(self.off + 0x6C, 3)

    @property
    def name(self) -> bytes:
        s = self.img.string_at(self.off + 0x78)
        return b"" if s is None else s

    @property
    def aux_ptr(self) -> Pointer | None:
        return self.opt_ptr(0x80)

    @property
    def geometry_ptr(self) -> Pointer | None:
        return self.opt_ptr(0x88)

    @property
    def raw_90(self) -> bytes:
        return self.raw(0x90, 0x30)

    @property
    def flags(self) -> int:
        return self.img.u32(self.off + 0xC0)

    @property
    def own_index(self) -> int:
        return self.img.u16(self.off + 0xC4)

    @property
    def parent(self) -> int:
        return self.img.i16(self.off + 0xC6)

    @property
    def type(self) -> int:
        return self.img.u8(self.off + 0xC8)

    @property
    def geometry_count(self) -> int:
        return self.img.u8(self.off + 0xC9)

    @property
    def raw_ca(self) -> bytes:
        return self.raw(0xCA, self.size - 0xCA)


class PaletteDescView(View):
    @property
    def entries_ptr(self) -> Pointer:
        return self.ptr(0x00)

    @property
    def count(self) -> int:
        return self.img.u64(self.off + 0x08)

    def entries(self) -> tuple:
        n = self.count
        if n == 0:
            return ()
        if n > MAX_PALETTE:
            raise FormatError(f"palette at 0x{self.off:X}: {n} entries exceed {MAX_PALETTE}")
        t = self.entries_ptr.target
        if t is None:
            raise FormatError(f"palette at 0x{self.off:X}: count {n} with a null pointer")
        return self.img.unpack(f"<{n}H", t)


class GeometryEntryView(View):
    @property
    def raw_00(self) -> bytes:
        return self.raw(0x00, 8)

    @property
    def submesh_count(self) -> int:
        return self.img.u16(self.off + 0x10)

    @property
    def raw_12(self) -> int:
        return self.img.u16(self.off + 0x12)

    @property
    def raw_14(self) -> int:
        return self.img.u16(self.off + 0x14)

    @property
    def format(self) -> int:
        return self.img.u8(self.off + 0x16)

    @property
    def raw_17(self) -> int:
        return self.img.u8(self.off + 0x17)

    @property
    def vertex_base(self) -> int:
        return self.img.u32(self.off + 0x28)

    @property
    def vertex_count(self) -> int:
        return self.img.u32(self.off + 0x2C)

    @property
    def index_base(self) -> int:
        return self.img.u32(self.off + 0x30)

    @property
    def raw_34(self) -> bytes:
        return self.raw(0x34, 12)

    def material_slots(self) -> tuple:
        n = self.submesh_count
        if n == 0:
            return ()
        t = self.ptr(0x08).target
        if t is None:
            raise FormatError(f"geometry entry 0x{self.off:X}: null material-slot pointer with {n} submeshes")
        return self.img.unpack(f"<{n}H", t)

    def index_counts(self) -> tuple:
        n = self.submesh_count
        if n == 0:
            return ()
        t = self.ptr(0x18).target
        if t is None:
            raise FormatError(f"geometry entry 0x{self.off:X}: null index-count pointer with {n} submeshes")
        return self.img.unpack(f"<{n}I", t)

    def palette_descs(self) -> list[PaletteDescView]:
        n = self.submesh_count
        if n == 0:
            return []
        t = self.ptr(0x20).target
        if t is None:
            raise FormatError(f"geometry entry 0x{self.off:X}: null palette pointer with {n} submeshes")
        return [PaletteDescView(self.img, t + i * PALETTE_DESC_SIZE) for i in range(n)]

    @property
    def material_slots_offset(self) -> int | None:
        return self.ptr(0x08).target if self.submesh_count else None

    @property
    def index_counts_offset(self) -> int | None:
        return self.ptr(0x18).target if self.submesh_count else None

    @property
    def stream_offset(self) -> int | None:
        return None

    @property
    def raw_stream(self) -> bytes:
        return b""

    @property
    def raw_bytes(self) -> bytes:
        return self.raw(0, GEOMETRY_SIZE)


class Dl2GeometryEntryView(View):
    """DL2 class-6 entry (0x30 bytes) + the class-8 stream object it points to. Same interface as
    GeometryEntryView; the DLTB-only holes (raw_12/raw_14/raw_17) read as 0 and raw_34 carries +0x24..+0x2F."""

    @property
    def raw_00(self) -> bytes:
        return self.raw(0x00, 8)

    @property
    def stream_offset(self) -> int:
        t = self.ptr(0x08).target
        if t is None:
            raise FormatError(f"DL2 geometry entry 0x{self.off:X}: null class-8 stream pointer")
        self.img._check(t, DL2_STREAM_SIZE)
        return t

    def _s32(self, rel: int) -> int:
        return self.img.u32(self.stream_offset + rel)

    @property
    def raw_stream(self) -> bytes:
        return self.img.raw(self.stream_offset, DL2_STREAM_SIZE)

    @property
    def submesh_count(self) -> int:
        n = self.img.u32(self.off + 0x20)
        if n > MAX_SUBMESHES:
            raise FormatError(f"DL2 geometry entry 0x{self.off:X}: {n} submeshes exceed {MAX_SUBMESHES}")
        return n

    raw_12 = 0
    raw_14 = 0
    raw_17 = 0

    @property
    def format(self) -> int:
        f = self._s32(0x18)
        if f > 0xFF:
            raise FormatError(f"DL2 geometry entry 0x{self.off:X}: vertex format {f} out of range")
        return f

    @property
    def vertex_base(self) -> int:
        return self._s32(0x08)

    @property
    def vertex_count(self) -> int:
        return self._s32(0x0C)

    @property
    def index_base(self) -> int:
        return self._s32(0x10)

    @property
    def stream_submesh_count(self) -> int:
        return self._s32(0x14)

    @property
    def raw_34(self) -> bytes:
        return self.raw(0x24, 12)

    @property
    def material_slots_offset(self) -> int | None:
        return self.ptr(0x10).target if self.submesh_count else None

    @property
    def index_counts_offset(self) -> int | None:
        return self.img.pointer(self.stream_offset).target if self.submesh_count else None

    def _array(self, what: str, target: int | None, fmt: str) -> tuple:
        n = self.submesh_count
        if n == 0:
            return ()
        if target is None:
            raise FormatError(f"DL2 geometry entry 0x{self.off:X}: null {what} pointer with {n} submeshes")
        return self.img.unpack(f"<{n}{fmt}", target)

    def material_slots(self) -> tuple:
        return self._array("material-slot", self.material_slots_offset, "H")

    def index_counts(self) -> tuple:
        return self._array("index-count", self.index_counts_offset, "I")

    def palette_descs(self) -> list[PaletteDescView]:
        n = self.submesh_count
        if n == 0:
            return []
        t = self.ptr(0x18).target
        if t is None:
            raise FormatError(f"DL2 geometry entry 0x{self.off:X}: null palette pointer with {n} submeshes")
        return [PaletteDescView(self.img, t + i * PALETTE_DESC_SIZE) for i in range(n)]

    @property
    def raw_bytes(self) -> bytes:
        return self.raw(0, DL2_GEOMETRY_SIZE)


class MaterialEntryView(View):
    """Class-11 entry. The name field at +0x08 is a small-string-optimised string object (census: engine_pc
    'sky' → "sky.mat" stored INLINE in the 8 bytes, no relocation slot; every name ≥ 8 chars is a tagged pointer
    (tag 0x2100) to a NUL-terminated heap string preceded by {u32 length, u32 capacity})."""

    @property
    def name_inline(self) -> bool:
        return not self.img.is_slot(self.off + 0x08)

    @property
    def name(self) -> bytes | None:
        if self.name_inline:
            raw = self.raw(0x08, 8)
            if raw[7] != 0:
                raise FormatError(f"material entry 0x{self.off:X}: inline name without terminator ({raw.hex()})")
            return raw.split(b"\0", 1)[0]
        return self.img.string_at(self.off + 0x08)

    @property
    def name_ptr(self) -> Pointer | None:
        return None if self.name_inline else self.ptr(0x08)

    @property
    def raw_bytes(self) -> bytes:
        return self.raw(0, MATERIAL_ENTRY_SIZE)


class MaterialHeaderView(View):
    @property
    def count(self) -> int:
        return self.img.u16(self.off + 0x08)

    @property
    def capacity(self) -> int:
        return self.img.u16(self.off + 0x0A)

    @property
    def raw_bytes(self) -> bytes:
        return self.raw(0, MATERIAL_HEADER_SIZE)

    def entries(self) -> list[MaterialEntryView]:
        n = self.count
        if n > MAX_MATERIALS:
            raise FormatError(f"material table: {n} entries exceed {MAX_MATERIALS}")
        if n == 0:
            return []
        t = self.ptr(0x00).target
        if t is None:
            raise FormatError("material table: null entry pointer with a non-zero count")
        return [MaterialEntryView(self.img, t + i * MATERIAL_ENTRY_SIZE) for i in range(n)]


@dataclass
class GeometryArray:
    """One class-6 record: `count` consecutive entries starting at `offset`."""
    record: int
    offset: int
    count: int
    layout: MeshLayout = LAYOUT_DLTB

    def entry(self, img: Image, i: int):
        if self.layout.is_dl2:
            return Dl2GeometryEntryView(img, self.offset + i * DL2_GEOMETRY_SIZE)
        return GeometryEntryView(img, self.offset + i * GEOMETRY_SIZE)


class MeshGraph:
    """Locates the known objects of a mesh image. Raises FormatError when the mandatory shape (record 0 = class-3
    root, exactly one class-4 array) is violated; everything else is optional. The layout (DLTB / DL2) is
    detected from the data; *layout* is an optional hint (see `detect_layout`)."""

    def __init__(self, img: Image, layout: str | MeshLayout | None = None):
        self.img = img
        fx = img.fixups
        if not fx.records:
            raise FormatError("empty record table")
        self.layout = lay = detect_layout(fx, layout)
        r0 = fx.records[0]
        if r0.class_id != CLASS_ROOT or r0.secondary or r0.count != 1:
            raise FormatError(f"record 0 is class {r0.class_id} count {r0.count}; expected the class-3 root")
        self.root = RootView(img, r0.offset, lay.root_size)
        ents = fx.records_of_class(CLASS_ENTITY)
        if len(ents) != 1:
            raise FormatError(f"expected exactly one class-4 entity array, found {len(ents)}")
        self.entity_record, erec = ents[0]
        if erec.count > MAX_ENTITIES:
            raise FormatError(f"{erec.count} entities exceed the u16 own-index range")
        self.entity_offset = erec.offset
        self.entity_count = erec.count
        if erec.offset + erec.count * lay.entity_size > img.size:
            raise FormatError("entity array runs past the primary image")
        self.geometry_arrays = [GeometryArray(i, r.offset, r.count, lay)
                                for i, r in fx.records_of_class(CLASS_GEOMETRY)]
        for ga in self.geometry_arrays:
            if ga.offset + ga.count * lay.geometry_size > img.size:
                raise FormatError(f"geometry array record {ga.record} runs past the primary image")
        mats = fx.records_of_class(CLASS_MATERIAL_HEADER)
        self.material_record = mats[0][0] if mats else None
        self.material_header = MaterialHeaderView(img, mats[0][1].offset) if mats else None

    def entity(self, i: int) -> EntityView:
        if not 0 <= i < self.entity_count:
            raise FormatError(f"entity {i} out of range ({self.entity_count})")
        return EntityView(self.img, self.entity_offset + i * self.layout.entity_size, self.layout.entity_size)

    def entities(self) -> list[EntityView]:
        return [self.entity(i) for i in range(self.entity_count)]

    def geometry_array_at(self, offset: int) -> GeometryArray | None:
        for ga in self.geometry_arrays:
            if ga.offset == offset:
                return ga
        return None

    def materials(self) -> list[MaterialEntryView]:
        return self.material_header.entries() if self.material_header else []

    def opaque_records(self) -> list[tuple[int, int, int, int]]:
        """(record index, class id, start, end) of every primary record that is not one of the decoded classes."""
        known = {CLASS_ROOT, CLASS_ENTITY, CLASS_GEOMETRY, CLASS_PALETTE, CLASS_MATERIAL_HEADER, CLASS_MATERIAL_ENTRY}
        if self.layout.is_dl2:
            known.add(CLASS_STREAM)
        out = []
        for i, r in enumerate(self.img.records):
            if r.secondary or r.class_id in known:
                continue
            a, b = self.img.record_span(i)
            out.append((i, r.class_id, a, b))
        return out
