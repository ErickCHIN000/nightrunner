"""Standalone ClassReader "fixups" metadata parser (inspection only; independent of nightrunner.classreader).

Layout — (A) engine_core Initialize@ClassReader +0x7ED40 / Resolve +0x90280 (survey 02 §2.1–2.2); the same stream
frames mesh part 0x11, prefab part 0x62 (after an 8-byte prefix, see prefab.py), AnimGraphBank part 0x48 and
AnimCustomResource part 0x4A (the latter two established by census 2026-09-15, see notes/FORMATS/types.md):

    +0x00 u32 dataSize            byte size of the primary image (part may be longer: 16-byte padding)
    +0x04 u32 tableCount          object records
    +0x08 u32 objectCount         bit31 = secondary stream present; low 31 bits (meaning vs tableCount: C)
    +0x0C ObjRec[tableCount]      {u32 offset, u32 classId (low 24 = class id, high 8 unknown), u32 flags
                                   (low 30 = element count, bit30 = object lives in secondary image, bit31 = "reverse prepass")}
          u32 relocationCount
          u32 slotOffset[relocationCount]   byte offsets of 8-byte pointer slots
          u8  slotKind[relocationCount]     bit1 second pass, bit2 tagged (low 48 bits = offset+1), bit4 slot in
                                            secondary, bit8 target in secondary
          if objectCount & 0x80000000: align4; u32 secondarySize; align16; u8 secondary[secondarySize]
          (both alignments are relative to the start of the PART buffer handed to parse(), not to the stream start:
           for prefab part 0x62 the stream begins at +8 and the secondary image lands on len(part) exactly on
           23/23 packs only with part-relative alignment — QA 2026-09-15, tests/test_types.py::test_prefab_graph)

Pointer value rule (A): stored u64 S; S == 0 → null; else target = S − 1 (with bit2: (S & 0xFFFF_FFFF_FFFF) − 1,
high 16 bits = tag).

`parse()` is strict about bounds; `serialise()` re-emits the stream so a census can prove the parse is
complete (`serialise(parse(b)) == b`). Class layouts beyond the record table are NOT interpreted here.
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass, field

from ..errors import FormatError
from .raw import align_up

_HDR = struct.Struct("<3I")
_REC = struct.Struct("<3I")

SECONDARY_FLAG = 0x80000000
REC_SECONDARY = 0x40000000
REC_REVERSE = 0x80000000
REC_COUNT_MASK = 0x3FFFFFFF
CLASS_MASK = 0xFFFFFF

MAX_TABLE = 5_000_000
MAX_RELOC = 50_000_000


@dataclass
class ObjRec:
    offset: int
    class_word: int
    flags: int

    @property
    def class_id(self) -> int:
        return self.class_word & CLASS_MASK

    @property
    def class_hi(self) -> int:
        return self.class_word >> 24

    @property
    def count(self) -> int:
        return self.flags & REC_COUNT_MASK

    @property
    def secondary(self) -> bool:
        return bool(self.flags & REC_SECONDARY)

    @property
    def reverse(self) -> bool:
        return bool(self.flags & REC_REVERSE)


@dataclass
class Fixups:
    data_size: int
    table_count: int
    object_word: int
    records: list[ObjRec]
    slot_offsets: list[int]
    slot_kinds: bytes
    secondary: bytes | None
    stream_end: int                       # ABSOLUTE end position in the blob passed to parse() (offset included):
                                          # == len(blob) when the stream fills the part (prefab: 23/23, QA 2026-09-15)
    tail: bytes = b""                     # bytes after stream_end (padding), preserved
    pad_after_kinds: bytes = b""          # align4 padding bytes before secondarySize (preserved verbatim)
    pad_after_size: bytes = b""           # align16 padding bytes before the secondary image

    @property
    def object_count(self) -> int:
        return self.object_word & 0x7FFFFFFF

    @property
    def has_secondary(self) -> bool:
        return bool(self.object_word & SECONDARY_FLAG)

    def summary(self) -> dict:
        kinds = Counter(self.slot_kinds)
        classes = Counter(r.class_word for r in self.records)
        return {
            "data_size": self.data_size, "table_count": self.table_count,
            "object_count": self.object_count, "has_secondary": self.has_secondary,
            "secondary_size": (len(self.secondary) if self.secondary is not None else None),
            "relocation_count": len(self.slot_offsets),
            "slot_kinds": {f"0x{k:02X}": v for k, v in sorted(kinds.items())},
            "records_in_secondary": sum(1 for r in self.records if r.secondary),
            "records_reverse": sum(1 for r in self.records if r.reverse),
            "class_histogram": [{"class": f"0x{c:08X}", "class_id": c & CLASS_MASK, "hi": c >> 24, "count": n}
                                for c, n in sorted(classes.items(), key=lambda kv: (-kv[1], kv[0]))],
            "stream_end": self.stream_end, "tail_bytes": len(self.tail),
        }


def parse(blob, *, offset: int = 0) -> Fixups:
    """Parse a fixups stream starting at *offset* in *blob*."""
    mv = memoryview(blob)
    n = len(mv)
    if offset + 12 > n:
        raise FormatError("fixups: shorter than the 12-byte header")
    data_size, table_count, object_word = _HDR.unpack_from(mv, offset)
    if table_count > MAX_TABLE:
        raise FormatError(f"fixups: tableCount {table_count} exceeds sanity limit")
    pos = offset + 12
    end_rec = pos + 12 * table_count
    if end_rec + 4 > n:
        raise FormatError("fixups: record table exceeds blob")
    records = [ObjRec(*_REC.unpack_from(mv, pos + 12 * k)) for k in range(table_count)]
    pos = end_rec
    reloc = struct.unpack_from("<I", mv, pos)[0]
    pos += 4
    if reloc > MAX_RELOC or pos + 5 * reloc > n:
        raise FormatError(f"fixups: relocationCount {reloc} exceeds blob")
    slot_offsets = list(struct.unpack_from(f"<{reloc}I", mv, pos)) if reloc else []
    pos += 4 * reloc
    slot_kinds = bytes(mv[pos : pos + reloc])
    pos += reloc
    secondary = None
    pad1 = b""
    pad2 = b""
    if object_word & SECONDARY_FLAG:
        p4 = align_up(pos, 4)
        pad1 = bytes(mv[pos:p4])
        if p4 + 4 > n:
            raise FormatError("fixups: secondarySize missing")
        sec_size = struct.unpack_from("<I", mv, p4)[0]
        p16 = align_up(p4 + 4, 16)
        pad2 = bytes(mv[p4 + 4 : p16])
        if p16 + sec_size > n:
            raise FormatError(f"fixups: secondary image {sec_size} exceeds blob")
        secondary = bytes(mv[p16 : p16 + sec_size])
        pos = p16 + sec_size
    fx = Fixups(data_size, table_count, object_word, records, slot_offsets, slot_kinds, secondary, pos,
                tail=bytes(mv[pos:]), pad_after_kinds=pad1, pad_after_size=pad2)
    return fx


def serialise(fx: Fixups, *, prefix: bytes = b"") -> bytes:
    out = bytearray(prefix)
    out += _HDR.pack(fx.data_size, fx.table_count, fx.object_word)
    for r in fx.records:
        out += _REC.pack(r.offset, r.class_word, r.flags)
    out += struct.pack("<I", len(fx.slot_offsets))
    out += struct.pack(f"<{len(fx.slot_offsets)}I", *fx.slot_offsets)
    out += fx.slot_kinds
    if fx.has_secondary:
        out += fx.pad_after_kinds
        out += struct.pack("<I", len(fx.secondary or b""))
        out += fx.pad_after_size
        out += fx.secondary or b""
    out += fx.tail
    return bytes(out)


def check_against_image(fx: Fixups, image) -> dict:
    """Offline bounds checks of the fixups stream against its primary image (descriptive, never raises)."""
    n = len(image)
    problems = []
    if fx.data_size > n:
        problems.append(f"dataSize {fx.data_size} > image {n}")
    for k, r in enumerate(fx.records):
        if not r.secondary and r.offset >= max(fx.data_size, 1):
            problems.append(f"record {k} offset 0x{r.offset:X} outside primary dataSize")
            if len(problems) > 20:
                break
    bad_slots = sum(1 for o in fx.slot_offsets if o + 8 > n)
    if bad_slots:
        problems.append(f"{bad_slots} relocation slots outside the image")
    return {"image_size": n, "padding_after_data": n - fx.data_size if fx.data_size <= n else None,
            "problems": problems}


def resolve_pointer(image, slot_off: int, kind: int) -> dict:
    """Decode one stored pointer slot per the native rule; returns target offset (or None) and tag."""
    s = struct.unpack_from("<Q", image, slot_off)[0]
    if s == 0:
        return {"slot": slot_off, "kind": kind, "raw": 0, "target": None, "tag": 0}
    if kind & 2:
        return {"slot": slot_off, "kind": kind, "raw": s, "target": (s & 0xFFFFFFFFFFFF) - 1, "tag": s >> 48}
    return {"slot": slot_off, "kind": kind, "raw": s, "target": s - 1, "tag": 0}


def cstring_at(image, off: int, limit: int = 4096) -> str | None:
    if off < 0 or off >= len(image):
        return None
    mv = bytes(memoryview(image)[off : off + limit])
    nul = mv.find(b"\0")
    if nul < 0:
        return None
    s = mv[:nul]
    if not s or any(b < 0x20 or b > 0x7E for b in s):
        return None
    return s.decode("ascii")
