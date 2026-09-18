"""ClassReader metadata blob (mesh part 0x11 `_MESH_FIXUPS_`, prefab part 0x62).

Layout (A: engine_core Initialize@ClassReader +0x7ED40, survey 02 §2.1; every byte re-serialises identically
across the 21,354-mesh corpus — see notes/FORMATS/classreader.md):

    off   size  field
    0x00  u32   primary_size      byte size of the primary image inside part 0x10 (the part may be longer: 16-byte padding)
    0x04  u32   record_count      object records below
    0x08  u32   object_count_raw  bit 31 = secondary stream present; low 31 bits: (C) always 1 in the corpus
    0x0C  12×N  records           {u32 offset, u32 class_raw, u32 flags_raw}
                                   class_raw  low 24 bits = class id; high 8 bits = (C) always 0xB0 in meshes
                                   flags_raw  low 30 bits = element count; bit 30 = object lives in the secondary
                                              image; bit 31 = "reverse prepass" (C, never set in the corpus)
    ...   u32   slot_count        relocation slots
    ...   u32×S slot_offset       byte offsets (in the primary image) of 8-byte pointer slots
    ...   u8×S  slot_kind         kind bits, see SLOT_* below
    [if object_count_raw & 0x80000000]  align 4; u32 secondary_size; align 16; u8 secondary[secondary_size]
    ...   u8×k  trailing          whatever follows (zero padding to the 16-byte part alignment in every shipped mesh)

Slot kinds (A: Resolve@ClassReader +0x90280):
    bit 1  pointer-to-pointer: after pass 1 dereference the resolved location again   (parsed, flagged, not resolved)
    bit 2  tagged: the low 48 bits hold offset+1, the high 16 bits are a tag kept verbatim   (supported)
    bit 4  the slot itself lies in the secondary image                                (parsed, flagged, not resolved)
    bit 8  the resolved pointer targets the secondary image                            (parsed, flagged, not resolved)
    0      plain primary → primary pointer                                             (supported)

Pointer value rule (A): stored u64 S; S == 0 ⇒ null; otherwise target = (S & 0xFFFF_FFFF_FFFF if kind & 2 else S) − 1.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

from ..errors import FormatError
from ..util.binio import align_up

SLOT_INDIRECT = 0x1
SLOT_TAGGED = 0x2
SLOT_IN_SECONDARY = 0x4
SLOT_TO_SECONDARY = 0x8
SLOT_SUPPORTED_MASK = SLOT_TAGGED          # kinds 0 and 2 are fully resolved; any other bit is flagged
SECONDARY_PRESENT = 0x80000000
RECORD_SECONDARY = 0x40000000
RECORD_REVERSE = 0x80000000
CLASS_ID_MASK = 0xFFFFFF
POINTER_OFFSET_MASK = 0xFFFFFFFFFFFF

_HDR = struct.Struct("<3I")
_REC = struct.Struct("<3I")
_U32 = struct.Struct("<I")

# offline sanity bounds (B: DLE MeshImage; generous, never hit by the corpus)
MAX_RECORDS = 1_000_000
MAX_SLOTS = 10_000_000


@dataclass
class Record:
    offset: int
    class_raw: int
    flags_raw: int

    @property
    def class_id(self) -> int:
        return self.class_raw & CLASS_ID_MASK

    @property
    def class_hi(self) -> int:
        """Upper 8 bits of the class word (C: 0xB0 on every mesh record in the corpus)."""
        return self.class_raw >> 24

    @property
    def count(self) -> int:
        return self.flags_raw & 0x3FFFFFFF

    @property
    def secondary(self) -> bool:
        return bool(self.flags_raw & RECORD_SECONDARY)

    @property
    def reverse(self) -> bool:
        return bool(self.flags_raw & RECORD_REVERSE)

    def pack(self) -> bytes:
        return _REC.pack(self.offset, self.class_raw, self.flags_raw)

    def to_json(self) -> dict:
        return {"offset": self.offset, "class_id": self.class_id, "class_hi": f"0x{self.class_hi:02X}",
                "count": self.count, "secondary": self.secondary, "reverse": self.reverse}


@dataclass
class Slot:
    offset: int
    kind: int

    @property
    def supported(self) -> bool:
        return not (self.kind & ~SLOT_SUPPORTED_MASK)

    @property
    def tagged(self) -> bool:
        return bool(self.kind & SLOT_TAGGED)


@dataclass
class Fixups:
    primary_size: int
    object_count_raw: int
    records: list[Record] = field(default_factory=list)
    slots: list[Slot] = field(default_factory=list)
    secondary: bytes | None = None          # secondary image bytes when object_count_raw bit 31 is set
    trailing: bytes = b""                   # bytes after the parsed structure (padding), kept verbatim

    # ---- parse / serialise --------------------------------------------------------------------------------

    @classmethod
    def parse(cls, data) -> "Fixups":
        buf = bytes(data)
        n = len(buf)
        if n < _HDR.size + 4:
            raise FormatError(f"fixups blob too short ({n} bytes)")
        primary_size, record_count, object_count_raw = _HDR.unpack_from(buf, 0)
        if record_count > MAX_RECORDS:
            raise FormatError(f"fixups: record_count {record_count} exceeds the offline bound {MAX_RECORDS}")
        pos = _HDR.size
        end = pos + record_count * _REC.size
        if end + 4 > n:
            raise FormatError(f"fixups: {record_count} records do not fit in {n} bytes")
        records = [Record(*_REC.unpack_from(buf, pos + i * _REC.size)) for i in range(record_count)]
        pos = end
        slot_count = _U32.unpack_from(buf, pos)[0]
        pos += 4
        if slot_count > MAX_SLOTS:
            raise FormatError(f"fixups: slot_count {slot_count} exceeds the offline bound {MAX_SLOTS}")
        if pos + slot_count * 5 > n:
            raise FormatError(f"fixups: {slot_count} slots do not fit in {n} bytes")
        offsets = struct.unpack_from(f"<{slot_count}I", buf, pos)
        pos += 4 * slot_count
        kinds = buf[pos : pos + slot_count]
        pos += slot_count
        slots = [Slot(o, k) for o, k in zip(offsets, kinds)]
        secondary = None
        if object_count_raw & SECONDARY_PRESENT:
            pos = align_up(pos, 4)
            if pos + 4 > n:
                raise FormatError("fixups: secondary size word missing")
            sec_size = _U32.unpack_from(buf, pos)[0]
            pos = align_up(pos + 4, 16)
            if pos + sec_size > n:
                raise FormatError(f"fixups: secondary image {sec_size} bytes does not fit")
            secondary = buf[pos : pos + sec_size]
            pos += sec_size
        trailing = buf[pos:]
        fx = cls(primary_size, object_count_raw, records, slots, secondary, trailing)
        for i, r in enumerate(records):
            if not r.secondary and r.offset > primary_size:
                raise FormatError(f"fixups: record {i} offset 0x{r.offset:X} beyond primary image (0x{primary_size:X})")
        return fx

    def to_bytes(self) -> bytes:
        out = bytearray(_HDR.pack(self.primary_size, len(self.records), self.object_count_raw))
        for r in self.records:
            out += r.pack()
        out += _U32.pack(len(self.slots))
        out += struct.pack(f"<{len(self.slots)}I", *(s.offset for s in self.slots))
        out += bytes(s.kind for s in self.slots)
        if self.object_count_raw & SECONDARY_PRESENT:
            sec = self.secondary or b""
            out += b"\0" * (align_up(len(out), 4) - len(out))
            out += _U32.pack(len(sec))
            out += b"\0" * (align_up(len(out), 16) - len(out))
            out += sec
        out += self.trailing
        return bytes(out)

    # ---- queries ------------------------------------------------------------------------------------------

    @property
    def secondary_present(self) -> bool:
        return bool(self.object_count_raw & SECONDARY_PRESENT)

    @property
    def object_count(self) -> int:
        return self.object_count_raw & 0x7FFFFFFF

    def slot_map(self) -> dict[int, int]:
        return {s.offset: s.kind for s in self.slots}

    def records_of_class(self, class_id: int, *, primary_only: bool = True) -> list[tuple[int, Record]]:
        """(record index, record) for every record of *class_id* (secondary-image records excluded by default)."""
        return [(i, r) for i, r in enumerate(self.records)
                if r.class_id == class_id and not (primary_only and r.secondary)]

    def record_span(self, index: int) -> tuple[int, int]:
        """[start, end) byte span of record *index* in the primary image: end = next record offset (records are
        stored in ascending offset order: census 2026-09-15, 21,354/21,354 meshes) or primary_size for the last."""
        r = self.records[index]
        if r.secondary:
            raise FormatError(f"record {index} lives in the secondary image")
        end = self.primary_size
        for nxt in self.records[index + 1 :]:
            if not nxt.secondary:
                end = nxt.offset
                break
        return r.offset, end

    def record_at(self, offset: int) -> int | None:
        """Index of the primary record starting exactly at *offset*, else None."""
        for i, r in enumerate(self.records):
            if r.offset == offset and not r.secondary:
                return i
        return None

    def class_census(self) -> dict[int, int]:
        h: dict[int, int] = {}
        for r in self.records:
            h[r.class_id] = h.get(r.class_id, 0) + 1
        return dict(sorted(h.items()))

    def unsupported_slots(self) -> list[Slot]:
        return [s for s in self.slots if not s.supported]

    def to_json(self, *, records: bool = False) -> dict:
        d = {
            "primary_size": self.primary_size, "record_count": len(self.records),
            "object_count_raw": f"0x{self.object_count_raw:08X}", "secondary_present": self.secondary_present,
            "secondary_size": None if self.secondary is None else len(self.secondary),
            "slot_count": len(self.slots), "slot_kinds": sorted({s.kind for s in self.slots}),
            "trailing_bytes": len(self.trailing), "class_census": {str(k): v for k, v in self.class_census().items()},
        }
        if records:
            d["records"] = [r.to_json() for r in self.records]
            d["slots"] = [{"offset": s.offset, "kind": s.kind} for s in self.slots]
        return d
