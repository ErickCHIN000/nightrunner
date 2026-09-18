"""ClassReader primary image (mesh part 0x10 `_MESH_`, prefab part 0x61) with its resolved pointer slots.

`Image` is a read-only view: typed readers plus pointer resolution through the fixups slot table. Nothing is
interpreted beyond what the caller asks for. `ImagePatch` is the in-place editing surface used by encoders
(retarget an existing slot, append data/strings, add new slots) and always re-serialises both parts.

Pointer rule (A, survey 02 §2.2): a pointer field is an 8-byte slot listed in the fixups; stored value 0 ⇒ null,
otherwise target = value − 1 (low 48 bits when the slot kind has bit 2; the high 16 bits are a tag kept verbatim).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from ..errors import FormatError, UnsupportedError
from ..util.binio import align_up
from .fixups import Fixups, Slot, SLOT_TAGGED, POINTER_OFFSET_MASK, Record

_Q = struct.Struct("<Q")
MAX_STRING = 65536


@dataclass(frozen=True)
class Pointer:
    """A resolved pointer slot."""
    slot: int          # offset of the 8-byte slot
    kind: int          # slot kind bits
    raw: int           # stored u64
    target: int | None # resolved byte offset in the primary image (None = null)
    tag: int           # high 16 bits when the slot is tagged, else 0


class Image:
    """Read-only typed access to a primary image + its fixups."""

    def __init__(self, data, fixups: Fixups):
        self.data = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
        self.fixups = fixups
        self.size = fixups.primary_size
        if self.size > len(self.data):
            raise FormatError(f"fixups primary_size {self.size} exceeds the image part ({len(self.data)} bytes)")
        self._slots = fixups.slot_map()

    @classmethod
    def from_parts(cls, image_bytes, fixups_bytes) -> "Image":
        return cls(image_bytes, Fixups.parse(fixups_bytes))

    # ---- raw readers (all bounds-checked against primary_size) ----------------------------------------------

    def _check(self, off: int, size: int) -> None:
        if off < 0 or off + size > self.size:
            raise FormatError(f"read of {size} bytes at 0x{off:X} outside the primary image (0x{self.size:X})")

    def raw(self, off: int, size: int) -> bytes:
        self._check(off, size)
        return bytes(self.data[off : off + size])

    def unpack(self, fmt: str, off: int) -> tuple:
        size = struct.calcsize(fmt)
        self._check(off, size)
        return struct.unpack_from(fmt, self.data, off)

    def u8(self, off: int) -> int:
        return self.unpack("<B", off)[0]

    def u16(self, off: int) -> int:
        return self.unpack("<H", off)[0]

    def i16(self, off: int) -> int:
        return self.unpack("<h", off)[0]

    def u32(self, off: int) -> int:
        return self.unpack("<I", off)[0]

    def u64(self, off: int) -> int:
        return self.unpack("<Q", off)[0]

    def f32s(self, off: int, count: int) -> tuple:
        return self.unpack(f"<{count}f", off)

    def cstring(self, off: int, limit: int = MAX_STRING) -> bytes:
        self._check(off, 0)
        end = self.data.find(b"\0", off, min(self.size, off + limit))
        if end < 0:
            raise FormatError(f"unterminated string at 0x{off:X}")
        return bytes(self.data[off:end])

    # ---- pointers -----------------------------------------------------------------------------------------

    def is_slot(self, off: int) -> bool:
        return off in self._slots

    def slot_kind(self, off: int) -> int | None:
        return self._slots.get(off)

    def pointer(self, off: int, *, allow_unslotted_null: bool = True) -> Pointer:
        """Resolve the pointer slot at *off*. A zero word without a relocation slot is a null pointer (shipped
        meshes store null palette pointers that way); a non-zero word without a slot raises FormatError, as does
        an out-of-image target. UnsupportedError for slot kinds 1/4/8 (indirect / secondary)."""
        kind = self._slots.get(off)
        raw = self.u64(off)
        if kind is None:
            if raw == 0 and allow_unslotted_null:
                return Pointer(off, 0, 0, None, 0)
            raise FormatError(f"0x{off:X} is not a relocation slot (word 0x{raw:X})")
        if kind & ~SLOT_TAGGED:
            raise UnsupportedError(f"slot 0x{off:X}: kind 0x{kind:X} (indirect/secondary) is not resolvable offline")
        if raw == 0:
            return Pointer(off, kind, raw, None, 0)
        if kind & SLOT_TAGGED:
            target = (raw & POINTER_OFFSET_MASK) - 1
            tag = raw >> 48
        else:
            target = raw - 1
            tag = 0
        if not 0 <= target < self.size:
            raise FormatError(f"slot 0x{off:X}: target 0x{target:X} outside the primary image (0x{self.size:X})")
        return Pointer(off, kind, raw, target, tag)

    def target(self, off: int) -> int | None:
        return self.pointer(off).target

    def string_at(self, ptr_off: int) -> bytes | None:
        """The NUL-terminated string a pointer slot refers to (None for a null pointer)."""
        t = self.target(ptr_off)
        return None if t is None else self.cstring(t)

    # ---- records ------------------------------------------------------------------------------------------

    @property
    def records(self) -> list[Record]:
        return self.fixups.records

    def records_of_class(self, class_id: int) -> list[tuple[int, Record]]:
        return self.fixups.records_of_class(class_id)

    def record_span(self, index: int) -> tuple[int, int]:
        return self.fixups.record_span(index)

    def record_bytes(self, index: int) -> bytes:
        a, b = self.record_span(index)
        return self.raw(a, b - a)

    def record_at(self, offset: int) -> int | None:
        return self.fixups.record_at(offset)

    def padding(self) -> bytes:
        """Bytes of the part beyond primary_size (16-byte padding in shipped meshes)."""
        return bytes(self.data[self.size :])


def embedded_mesh_name(image_bytes, fixups_bytes) -> bytes:
    """The `.msh` file name registered by MeshMgr (A: ResourceManagement GetFileName +0x17A60 = **(this+0x68)):
    record 0 must be class 3 (root), non-secondary, count 1; root+0x00 is a pointer slot to the string.
    Imported lazily by nightrunner.container.validate."""
    fx = Fixups.parse(fixups_bytes)
    if not fx.records:
        raise FormatError("no records: missing root object")
    root = fx.records[0]
    if root.class_id != 3 or root.secondary or root.count != 1:
        raise FormatError(f"record 0 is class {root.class_id} count {root.count} (expected class 3 root, count 1)")
    img = Image(image_bytes, fx)
    s = img.string_at(root.offset)
    if s is None:
        raise FormatError("root name pointer is null")
    return s


# --------------------------------------------------------------------------------------------------------------
# In-place patching (used by identity rename and by the phase-2 encoder, mesh/imagepatch.py)
# --------------------------------------------------------------------------------------------------------------

class ImagePatch:
    """Mutable copy of an image + fixups. Writes go through `write_*`; new data is appended after primary_size
    (never overwriting shared storage); pointers are retargeted through their existing slots or new slots are
    registered. `finish()` returns (image_bytes, fixups_bytes) with the primary size updated and the image
    re-padded to 16 bytes, keeping the original padding bytes beyond the old primary size."""

    def __init__(self, image: Image):
        self.src = image
        self.buf = bytearray(image.data[: image.size])
        self.tail = bytes(image.data[image.size :])
        self.fixups = Fixups.parse(image.fixups.to_bytes())   # deep copy
        self._slots = self.fixups.slot_map()

    @property
    def size(self) -> int:
        return len(self.buf)

    def write(self, off: int, data: bytes) -> None:
        if off < 0 or off + len(data) > len(self.buf):
            raise FormatError(f"write of {len(data)} bytes at 0x{off:X} outside the image (0x{len(self.buf):X})")
        self.buf[off : off + len(data)] = data

    def write_u16(self, off: int, v: int) -> None:
        self.write(off, struct.pack("<H", v))

    def write_u32(self, off: int, v: int) -> None:
        self.write(off, struct.pack("<I", v))

    def write_u64(self, off: int, v: int) -> None:
        self.write(off, struct.pack("<Q", v))

    def write_f32s(self, off: int, values) -> None:
        vals = list(values)
        self.write(off, struct.pack(f"<{len(vals)}f", *vals))

    def append(self, data: bytes, align: int = 1) -> int:
        """Append *data* at the end of the primary image, aligned; returns its offset."""
        pad = align_up(len(self.buf), align) - len(self.buf)
        self.buf += b"\0" * pad
        off = len(self.buf)
        self.buf += data
        return off

    def append_string(self, s: bytes, align: int = 1) -> int:
        if b"\0" in s:
            raise FormatError("string contains NUL")
        return self.append(s + b"\0", align)

    def add_slot(self, off: int, kind: int = 0) -> None:
        """Register a new relocation slot (kept sorted by offset: shipped tables are ascending)."""
        if off in self._slots:
            raise FormatError(f"slot 0x{off:X} already exists")
        if off % 8:
            raise FormatError(f"slot 0x{off:X} is not 8-byte aligned")
        self._slots[off] = kind
        self.fixups.slots.append(Slot(off, kind))
        self.fixups.slots.sort(key=lambda s: s.offset)

    def retarget(self, slot_off: int, target: int | None, *, kind: int | None = None, tag: int | None = None) -> None:
        """Point the slot at *slot_off* to *target* (None = null). Tagged slots keep their tag unless *tag* is
        given (needed when a tagged slot is created: the old word carries no tag)."""
        k = self._slots.get(slot_off)
        if k is None:
            if kind is None:
                raise FormatError(f"0x{slot_off:X} is not a relocation slot (pass kind= to add one)")
            self.add_slot(slot_off, kind)
            k = kind
        if k & ~SLOT_TAGGED:
            raise UnsupportedError(f"slot 0x{slot_off:X}: kind 0x{k:X} cannot be retargeted offline")
        old = struct.unpack_from("<Q", self.buf, slot_off)[0]
        if target is None:
            self.write_u64(slot_off, 0)
            return
        if not 0 <= target < len(self.buf):
            raise FormatError(f"target 0x{target:X} outside the image")
        value = target + 1
        if k & SLOT_TAGGED:
            value |= (old & ~POINTER_OFFSET_MASK) if tag is None else ((tag & 0xFFFF) << 48)
        self.write_u64(slot_off, value)

    def add_record(self, offset: int, class_id: int, count: int, *, class_hi: int = 0xB0, flags_hi: int = 0) -> int:
        """Append an object record (kept in ascending offset order); returns its index."""
        rec = Record(offset, (class_hi << 24) | (class_id & 0xFFFFFF), (count & 0x3FFFFFFF) | flags_hi)
        recs = self.fixups.records
        i = len(recs)
        while i > 0 and recs[i - 1].offset > offset:
            i -= 1
        recs.insert(i, rec)
        return i

    def relocate_record(self, index: int, offset: int, count: int) -> int:
        """Move record *index* to a new (offset, count) — used when an array grows and is re-appended at the end
        of the image (the old bytes stay, unreferenced). Records are kept in ascending offset order (census
        2026-09-15: 21,354/21,354 shipped meshes), so the record may change position; returns its new index."""
        recs = self.fixups.records
        rec = recs.pop(index)
        rec.offset = offset
        rec.flags_raw = (rec.flags_raw & ~0x3FFFFFFF) | (count & 0x3FFFFFFF)
        i = len(recs)
        while i > 0 and recs[i - 1].offset > offset:
            i -= 1
        recs.insert(i, rec)
        return i

    def slot_kind(self, off: int) -> int | None:
        return self._slots.get(off)

    def u64(self, off: int) -> int:
        return struct.unpack_from("<Q", self.buf, off)[0]

    def finish(self) -> tuple[bytes, bytes]:
        """(image bytes, fixups bytes). An image whose primary size did not change keeps its exact original
        length (48/21,354 shipped image parts are not 16-byte multiples); a grown image is re-padded to 16 after
        the original padding bytes. Fixups whose record/slot tables did not change re-serialise verbatim; grown
        tables get fresh zero padding to a 16-byte multiple (census: 21,297/21,354 fixups parts are multiples of 16
        and every trailing byte is zero)."""
        grown = len(self.buf) != self.src.size
        self.fixups.primary_size = len(self.buf)
        image = bytes(self.buf) + self.tail
        if grown:
            image += b"\0" * (align_up(len(image), 16) - len(image))
        src_fx = self.src.fixups
        if len(self.fixups.records) != len(src_fx.records) or len(self.fixups.slots) != len(src_fx.slots):
            self.fixups.trailing = b""
            body = self.fixups.to_bytes()
            self.fixups.trailing = b"\0" * (align_up(len(body), 16) - len(body))
        return image, self.fixups.to_bytes()
