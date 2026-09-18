"""RP6L v4 container reader and writer.

Every on-disk record is kept as its raw packed words; derived fields are properties so that unknown bits
are preserved bit-for-bit through read → write. Provenance for the layout: notes/FORMATS/rp6l.md and
notes/survey/01-container-rp6l.md (ResourceCore OpenPack +0x26320, index conversion +0x247F0,
LoadDataTask +0x236D0, preload scan +0x1C290, GetPhysicalResourceParentName +0x1EB00,
FindLogicalResourceUsingName +0x19250; engine +0xCEB1C0/+0xCEB530/+0xCEAA60 for the bit-12 contract).

On-disk layout
--------------
    Header      36 B   9 × u32
    Storage[S]  20 B   type u8 | align_raw u8 | flags u8 | metadata u8 | base_units u32 | size_lo u32 | comp_lo u32 | count u16 | size_hi u8 | comp_hi u8
    Physical[P] 16 B   packed u32 | offset_units u32 | size u32 | fc u32
    Logical[L]  12 B   packed u32 | name_index u32 | first_part u32
    u32 name_offset[N]
    u8  name_blob[name_bytes]
    payload ...        part offset = (storage.base_units + physical.offset_units) << 4
"""

from __future__ import annotations

import io
import mmap
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

from ..errors import FormatError, BuildError, UnsupportedError
from ..util.binio import align_up
from ..util.names import decode_name, engine_fold
from . import catalogue

MAGIC = 0x4C365052  # 'RP6L'
VERSION = 4
HEADER_SIZE = 36
STORAGE_SIZE = 20
PHYSICAL_SIZE = 16
LOGICAL_SIZE = 12
UNIT_SHIFT = 4          # payload addresses are in 16-byte units
MAX_PARTS = 15          # runtime logical word packs part count into 4 bits
MAX_STORAGES = 256      # physical.packed low byte indexes the storage table
MAX_OFFSET = 1 << 36    # (u32 units) << 4

FIELD08_ONDEMAND = 0x1000    # header.field08 bit 12 — single contiguous on-demand read (engine +0xCEB1C0)

# storage.flags bit 3. No decompiled consumer is known; it is set on every mesh-family storage of the on-demand
# packs (0xC9/0xD9/0x59/0x49/0x29) and on the ANM2 stream payload (0x29), clear on textures/areas/prefabs and on
# every method-0 storage. It drives the stock storage-table order (bit3 groups first) and, in bit-12 packs, which
# resources get the per-resource contiguous layout. Name chosen from that behaviour, not from native evidence.
STORAGE_FLAG_STREAM = 0x08

# logical.flags byte values seen in shipped packs (bit 0 always set; semantics otherwise unknown):
LOGICAL_FLAGS_DEFAULT = 0x01   # every texture/anim/prefab/area/envprobe/voxel/method-0 mesh
LOGICAL_FLAGS_ONDEMAND_MESH = 0x81  # type 0x10 in field08 bit-12 packs
LOGICAL_FLAGS_ANM2_STREAM = 0x21    # type 0x40 stored as 0x44+0x45 in *_stream packs

# physical.packed bits
PHYS_STORAGE_MASK = 0xFF
PHYS_BIT8 = 0x0100          # preload scan skips; carried by every mesh part (ResourceCore +0x1C290)
PHYS_PRIORITY_SHIFT = 9     # bits 9..11: on-demand read priority of the FIRST part (engine +0xCEB530)
PHYS_PRIORITY_MASK = 0x7
PHYS_SPECIAL = 0x1000       # bit 12: on-demand compatible path refuses the resource (semantics unknown)
PHYS_CHILD = 0x2000         # bit 13: payload lives in the .rpacz child pack (LoadDataTask +0x236D0)
PHYS_BIT14 = 0x4000         # preload scan skips (meaning unknown)
PHYS_BIT15 = 0x8000         # unknown
PHYS_OWNER_SHIFT = 16       # bits 16..31: owning logical index (GetPhysicalResourceParentName +0x1EB00)
PHYS_FLAG_BITS = 0xFF00     # everything between storage index and owner

_HDR = struct.Struct("<9I")
_STO = struct.Struct("<4B3IH2B")
_PHY = struct.Struct("<4I")
_LOG = struct.Struct("<3I")


# --------------------------------------------------------------------------------------------------------------
# Records
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Header:
    magic: int = MAGIC
    version: int = VERSION
    field08: int = 0
    physical_count: int = 0
    storage_count: int = 0
    name_count: int = 0
    name_bytes: int = 0
    logical_count: int = 0
    flags: int = 1

    @property
    def ondemand(self) -> bool:
        return bool(self.field08 & FIELD08_ONDEMAND)

    @classmethod
    def unpack(cls, buf, off: int = 0) -> "Header":
        return cls(*_HDR.unpack_from(buf, off))

    def pack(self) -> bytes:
        return _HDR.pack(self.magic, self.version, self.field08, self.physical_count, self.storage_count,
                         self.name_count, self.name_bytes, self.logical_count, self.flags)

    def to_json(self) -> dict:
        return {
            "magic": f"0x{self.magic:08X}", "version": self.version, "field08": f"0x{self.field08:08X}",
            "physical_count": self.physical_count, "storage_count": self.storage_count,
            "name_count": self.name_count, "name_bytes": self.name_bytes, "logical_count": self.logical_count,
            "flags": f"0x{self.flags:08X}",
        }


@dataclass
class Storage:
    type: int
    align_raw: int
    flags: int
    metadata: int
    base_units: int
    size: int            # 40-bit
    compressed: int      # 40-bit
    count: int

    # derived (all from the packed first word w = type | align_raw<<8 | flags<<16 | metadata<<24)
    @property
    def word0(self) -> int:
        return self.type | (self.align_raw << 8) | (self.flags << 16) | (self.metadata << 24)

    @property
    def alignment(self) -> int:
        """1 << ((word0 >> 9) & 15) — every native consumer (+0x1C290, +0x236D0, +0xCEB530, +0xCEAA60)."""
        return 1 << ((self.align_raw >> 1) & 0xF)

    @property
    def method(self) -> int:
        return self.flags & 3

    @property
    def version(self) -> int:
        return (self.flags >> 4) | ((self.metadata & 0xF) << 4)

    @property
    def codec(self) -> int:
        return self.metadata >> 4

    @property
    def flag_bit2(self) -> bool:
        return bool(self.flags & 0x04)

    @property
    def flag_bit3(self) -> bool:
        return bool(self.flags & STORAGE_FLAG_STREAM)

    @property
    def stream(self) -> bool:
        return bool(self.flags & STORAGE_FLAG_STREAM)

    @property
    def key(self) -> tuple[int, int, int, int]:
        """Grouping identity used by the writer: (type, align_raw, flags, metadata)."""
        return (self.type, self.align_raw, self.flags, self.metadata)

    @property
    def base_offset(self) -> int:
        return self.base_units << UNIT_SHIFT

    @classmethod
    def unpack(cls, buf, off: int) -> "Storage":
        t, a, f, m, base, size_lo, comp_lo, count, size_hi, comp_hi = _STO.unpack_from(buf, off)
        return cls(t, a, f, m, base, size_lo | (size_hi << 32), comp_lo | (comp_hi << 32), count)

    def pack(self) -> bytes:
        if not (0 <= self.size < 1 << 40 and 0 <= self.compressed < 1 << 40):
            raise BuildError("storage size exceeds 40 bits")
        return _STO.pack(self.type, self.align_raw, self.flags, self.metadata, self.base_units,
                         self.size & 0xFFFFFFFF, self.compressed & 0xFFFFFFFF, self.count,
                         self.size >> 32, self.compressed >> 32)

    def to_json(self) -> dict:
        return {
            "type": f"0x{self.type:02X}", "type_name": catalogue.type_name(self.type),
            "align_raw": self.align_raw, "alignment": self.alignment,
            "flags": f"0x{self.flags:02X}", "metadata": f"0x{self.metadata:02X}",
            "method": self.method, "version": self.version, "codec": self.codec,
            "base_units": self.base_units, "size": self.size, "compressed": self.compressed, "count": self.count,
        }

    @classmethod
    def from_json(cls, d: dict) -> "Storage":
        return cls(int(d["type"], 0), int(d["align_raw"]), int(d["flags"], 0), int(d["metadata"], 0),
                   int(d["base_units"]), int(d["size"]), int(d["compressed"]), int(d["count"]))


@dataclass
class Physical:
    packed: int
    offset_units: int
    size: int
    fc: int

    @property
    def storage_index(self) -> int:
        return self.packed & PHYS_STORAGE_MASK

    @property
    def owner(self) -> int:
        return self.packed >> PHYS_OWNER_SHIFT

    @property
    def flag_bits(self) -> int:
        """bits 8..15 of packed (bit8, priority, special, child, bit14, bit15) — preserved verbatim by the writer."""
        return self.packed & PHYS_FLAG_BITS

    @property
    def bit8(self) -> bool:
        return bool(self.packed & PHYS_BIT8)

    @property
    def priority(self) -> int:
        return (self.packed >> PHYS_PRIORITY_SHIFT) & PHYS_PRIORITY_MASK

    @property
    def special(self) -> bool:
        return bool(self.packed & PHYS_SPECIAL)

    @property
    def child(self) -> bool:
        return bool(self.packed & PHYS_CHILD)

    @property
    def bit14(self) -> bool:
        return bool(self.packed & PHYS_BIT14)

    @property
    def bit15(self) -> bool:
        return bool(self.packed & PHYS_BIT15)

    @classmethod
    def unpack(cls, buf, off: int) -> "Physical":
        return cls(*_PHY.unpack_from(buf, off))

    def pack(self) -> bytes:
        return _PHY.pack(self.packed, self.offset_units, self.size, self.fc)

    def to_json(self) -> dict:
        return {
            "packed": f"0x{self.packed:08X}", "storage_index": self.storage_index, "owner": self.owner,
            "flag_bits": f"0x{self.flag_bits:04X}", "offset_units": self.offset_units, "size": self.size,
            "fc": f"0x{self.fc:08X}",
        }


@dataclass
class Logical:
    packed: int
    name_index: int
    first_part: int

    @property
    def part_count(self) -> int:
        return self.packed & 0xFFFF

    @property
    def type(self) -> int:
        return (self.packed >> 16) & 0xFF

    @property
    def flags(self) -> int:
        return self.packed >> 24

    @classmethod
    def unpack(cls, buf, off: int) -> "Logical":
        return cls(*_LOG.unpack_from(buf, off))

    @classmethod
    def make(cls, part_count: int, type_id: int, flags: int, name_index: int, first_part: int) -> "Logical":
        return cls((part_count & 0xFFFF) | ((type_id & 0xFF) << 16) | ((flags & 0xFF) << 24), name_index, first_part)

    def pack(self) -> bytes:
        return _LOG.pack(self.packed, self.name_index, self.first_part)


# --------------------------------------------------------------------------------------------------------------
# Reader
# --------------------------------------------------------------------------------------------------------------

@dataclass
class Resource:
    """Convenience view over one logical resource."""
    pack: "Pack"
    index: int
    logical: Logical

    @property
    def name_raw(self) -> bytes:
        return self.pack.name_bytes(self.logical.name_index)

    @property
    def name(self) -> str:
        return decode_name(self.name_raw)

    @property
    def type(self) -> int:
        return self.logical.type

    @property
    def flags(self) -> int:
        return self.logical.flags

    @property
    def part_indices(self) -> range:
        return range(self.logical.first_part, self.logical.first_part + self.logical.part_count)

    @property
    def parts(self) -> list[Physical]:
        return [self.pack.physicals[i] for i in self.part_indices]

    @property
    def part_types(self) -> tuple[int, ...]:
        return tuple(self.pack.part_type(i) for i in self.part_indices)

    def part_by_type(self, type_id: int) -> int | None:
        """Physical index of the first part whose storage type is *type_id*, else None."""
        for i in self.part_indices:
            if self.pack.part_type(i) == type_id:
                return i
        return None

    def read_part_by_type(self, type_id: int) -> memoryview | None:
        i = self.part_by_type(type_id)
        return None if i is None else self.pack.read_part(i)

    def to_json(self) -> dict:
        return {
            "index": self.index, "name": self.name, "name_hex": self.name_raw.hex(),
            "type": f"0x{self.type:02X}", "type_name": catalogue.type_name(self.type),
            "flags": f"0x{self.flags:02X}", "name_index": self.logical.name_index,
            "first_part": self.logical.first_part, "part_count": self.logical.part_count,
            "parts": [dict(index=i, type=f"0x{self.pack.part_type(i):02X}", offset=self.pack.part_offset(i),
                           **self.pack.physicals[i].to_json()) for i in self.part_indices],
        }


class Pack:
    """Memory-mapped RP6L v4 pack. Nothing is decoded at open time except the tables."""

    def __init__(self, path: Path | str, data, *, size: int | None = None, mm=None, fh=None):
        self.path = Path(path)
        self._data = data
        self._mm = mm
        self._fh = fh
        self.size = len(data) if size is None else size
        self._parse_tables()

    # ---- lifecycle -------------------------------------------------------------------------------------------

    @classmethod
    def open(cls, path: Path | str) -> "Pack":
        path = Path(path)
        fh = open(path, "rb")
        size = os.fstat(fh.fileno()).st_size
        if size < HEADER_SIZE:
            fh.close()
            raise FormatError(f"{path}: file shorter than the 36-byte header")
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            return cls(path, mm, size=size, mm=mm, fh=fh)
        except Exception:
            mm.close()
            fh.close()
            raise

    @classmethod
    def from_bytes(cls, data: bytes, name: str = "<memory>") -> "Pack":
        return cls(name, memoryview(data))

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Pack":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- tables ----------------------------------------------------------------------------------------------

    def _parse_tables(self) -> None:
        buf = self._data
        self.header = h = Header.unpack(buf, 0)
        if h.magic != MAGIC:
            raise FormatError(f"{self.path}: bad magic 0x{h.magic:08X} (expected RP6L 0x{MAGIC:08X})")
        if h.version != VERSION:
            raise FormatError(f"{self.path}: unsupported version {h.version} (expected {VERSION})")
        if h.storage_count > MAX_STORAGES:
            raise FormatError(f"{self.path}: storage_count {h.storage_count} exceeds the 8-bit physical index")
        off = HEADER_SIZE
        self.storage_offset = off
        end = off + h.storage_count * STORAGE_SIZE
        end2 = end + h.physical_count * PHYSICAL_SIZE
        end3 = end2 + h.logical_count * LOGICAL_SIZE
        end4 = end3 + h.name_count * 4
        end5 = end4 + h.name_bytes
        if end5 > self.size:
            raise FormatError(f"{self.path}: tables (0x{end5:X}) extend past end of file (0x{self.size:X})")
        self.storages = [Storage.unpack(buf, off + i * STORAGE_SIZE) for i in range(h.storage_count)]
        self.physical_offset = end
        self.physicals = [Physical.unpack(buf, end + i * PHYSICAL_SIZE) for i in range(h.physical_count)]
        self.logical_offset = end2
        self.logicals = [Logical.unpack(buf, end2 + i * LOGICAL_SIZE) for i in range(h.logical_count)]
        self.name_offset_offset = end3
        self.name_offsets = list(struct.unpack_from(f"<{h.name_count}I", buf, end3)) if h.name_count else []
        self.name_blob_offset = end4
        self.name_blob = bytes(buf[end4:end5])
        self.table_end = end5
        for i, s in enumerate(self.physicals):
            if s.storage_index >= h.storage_count:
                raise FormatError(f"{self.path}: physical {i} references storage {s.storage_index} of {h.storage_count}")
        for i, l in enumerate(self.logicals):
            if l.first_part + l.part_count > h.physical_count:
                raise FormatError(f"{self.path}: logical {i} parts [{l.first_part}, +{l.part_count}) exceed {h.physical_count}")
            if l.name_index >= h.name_count:
                raise FormatError(f"{self.path}: logical {i} name index {l.name_index} of {h.name_count}")
        self._name_cache: dict[int, bytes] = {}
        self._fold_index: dict[tuple[int, bytes], list[int]] | None = None

    def table_bytes(self) -> bytes:
        """The exact bytes of header + all tables (used for byte-identity checks)."""
        return bytes(self._data[: self.table_end])

    # ---- names -----------------------------------------------------------------------------------------------

    def name_bytes(self, name_index: int) -> bytes:
        cached = self._name_cache.get(name_index)
        if cached is not None:
            return cached
        if name_index >= len(self.name_offsets):
            raise FormatError(f"name index {name_index} out of range")
        start = self.name_offsets[name_index]
        if start >= len(self.name_blob):
            raise FormatError(f"name offset {start} outside the name blob ({len(self.name_blob)})")
        nul = self.name_blob.find(b"\0", start)
        if nul < 0:
            raise FormatError(f"name {name_index} is not NUL-terminated")
        raw = self.name_blob[start:nul]
        self._name_cache[name_index] = raw
        return raw

    def name(self, name_index: int) -> str:
        return decode_name(self.name_bytes(name_index))

    def name_blob_order(self) -> list[int] | None:
        """Resource indices in the order their names appear in the blob, or None when name indices are not a
        bijection onto the resources (then byte-identical name-table reproduction is impossible anyway)."""
        by_name_index: dict[int, int] = {}
        for i, l in enumerate(self.logicals):
            if l.name_index in by_name_index:
                return None
            by_name_index[l.name_index] = i
        if len(by_name_index) != len(self.name_offsets):
            return None
        order = sorted(range(len(self.name_offsets)), key=lambda ni: self.name_offsets[ni])
        return [by_name_index[ni] for ni in order]

    # ---- resources -------------------------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.logicals)

    def resource(self, index: int) -> Resource:
        return Resource(self, index, self.logicals[index])

    def __iter__(self) -> Iterator[Resource]:
        for i, l in enumerate(self.logicals):
            yield Resource(self, i, l)

    def resources_of_type(self, type_id: int) -> Iterator[Resource]:
        for r in self:
            if r.type == type_id:
                yield r

    def find(self, name: str | bytes, type_id: int | None = None, fold: bool = True) -> list[int]:
        """Logical indices matching *name*. With fold=True this mimics FindLogicalResourceUsingName
        (ASCII A-Z folding, linear order = lowest index first). Returns all matches, not just the first."""
        raw = name if isinstance(name, bytes) else name.encode("utf-8", "surrogateescape")
        key = engine_fold(raw) if fold else raw
        out = []
        for i, l in enumerate(self.logicals):
            if type_id is not None and l.type != type_id:
                continue
            n = self.name_bytes(l.name_index)
            if (engine_fold(n) if fold else n) == key:
                out.append(i)
        return out

    # ---- parts -----------------------------------------------------------------------------------------------

    def part_storage(self, phys_index: int) -> Storage:
        return self.storages[self.physicals[phys_index].storage_index]

    def part_type(self, phys_index: int) -> int:
        return self.part_storage(phys_index).type

    def part_offset(self, phys_index: int) -> int:
        p = self.physicals[phys_index]
        return (self.storages[p.storage_index].base_units + p.offset_units) << UNIT_SHIFT

    def part_alignment(self, phys_index: int) -> int:
        return self.part_storage(phys_index).alignment

    def part_is_direct(self, phys_index: int) -> bool:
        """True when the bytes live in this file uncompressed (method 0/1, not a child-pack part)."""
        p = self.physicals[phys_index]
        return self.part_storage(phys_index).method in (0, 1) and not p.child

    def read_part(self, phys_index: int) -> memoryview:
        p = self.physicals[phys_index]
        s = self.storages[p.storage_index]
        if p.child:
            raise UnsupportedError(f"part {phys_index}: payload lives in a .rpacz child pack (bit 13)")
        if s.method not in (0, 1):
            raise UnsupportedError(f"part {phys_index}: storage method {s.method} (compressed) is not supported")
        off = self.part_offset(phys_index)
        if off + p.size > self.size:
            raise FormatError(f"part {phys_index}: span 0x{off:X}+0x{p.size:X} exceeds file size 0x{self.size:X}")
        return memoryview(self._data)[off : off + p.size]

    def owner_of(self, phys_index: int) -> int:
        return self.physicals[phys_index].owner

    # ---- summaries -------------------------------------------------------------------------------------------

    def type_histogram(self) -> dict[int, int]:
        h: dict[int, int] = {}
        for l in self.logicals:
            h[l.type] = h.get(l.type, 0) + 1
        return dict(sorted(h.items()))

    def to_json_tables(self) -> dict:
        return {
            "header": self.header.to_json(),
            "table_end": self.table_end,
            "storages": [s.to_json() for s in self.storages],
        }


# --------------------------------------------------------------------------------------------------------------
# Writer
# --------------------------------------------------------------------------------------------------------------

class PartSource:
    """Where a part's bytes come from: in-memory bytes, or a (path, offset, size) file slice."""

    __slots__ = ("data", "path", "offset", "size")

    def __init__(self, data=None, *, path: Path | str | None = None, offset: int = 0, size: int | None = None):
        if data is not None:
            self.data = memoryview(data) if not isinstance(data, memoryview) else data
            self.path = None
            self.offset = 0
            self.size = len(self.data)
        else:
            if path is None or size is None:
                raise BuildError("PartSource needs data or (path, offset, size)")
            self.data = None
            self.path = Path(path)
            self.offset = offset
            self.size = size

    def write_to(self, out: io.BufferedWriter, chunk: int = 1 << 20) -> None:
        if self.data is not None:
            out.write(self.data)
            return
        with open(self.path, "rb") as fh:
            fh.seek(self.offset)
            remaining = self.size
            while remaining > 0:
                buf = fh.read(min(chunk, remaining))
                if not buf:
                    raise BuildError(f"{self.path}: short read at {self.offset + self.size - remaining}")
                out.write(buf)
                remaining -= len(buf)


@dataclass
class PartSpec:
    """One physical part to write."""
    type: int                      # storage type byte (0x10, 0xF0, ...)
    source: PartSource
    align_raw: int = 8             # storage align_raw byte (8 → 16-byte alignment) — copy from the template storage
    storage_flags: int = 0         # storage flags byte (method | bit3 | version low nibble)
    storage_metadata: int = 0      # storage metadata byte (version high nibble | codec)
    flag_bits: int = 0             # physical.packed bits 8..15 to preserve (bit8, priority, special, ...)
    fc: int = 0                    # physical.fc (unknown word, preserved)
    offset_units: int | None = None  # original offset (only used by the 'preserve' layout)
    storage_index: int | None = None  # original storage index (only used by the 'preserve' layout)

    @property
    def storage_key(self) -> tuple[int, int, int, int]:
        return (self.type, self.align_raw, self.storage_flags, self.storage_metadata)

    @property
    def alignment(self) -> int:
        return 1 << ((self.align_raw >> 1) & 0xF)

    @property
    def stream(self) -> bool:
        return bool(self.storage_flags & STORAGE_FLAG_STREAM)

    @property
    def size(self) -> int:
        return self.source.size

    @classmethod
    def from_pack(cls, pack: Pack, phys_index: int, source: PartSource | None = None) -> "PartSpec":
        p = pack.physicals[phys_index]
        s = pack.storages[p.storage_index]
        src = source or PartSource(path=pack.path, offset=pack.part_offset(phys_index), size=p.size)
        return cls(type=s.type, source=src, align_raw=s.align_raw, storage_flags=s.flags,
                   storage_metadata=s.metadata, flag_bits=p.flag_bits, fc=p.fc,
                   offset_units=p.offset_units, storage_index=p.storage_index)


@dataclass
class ResourceSpec:
    name: bytes
    type: int
    flags: int = 0                 # logical flags byte (bits 24..31)
    parts: list[PartSpec] = field(default_factory=list)
    name_index: int | None = None  # only used by 'preserve' to keep the original name table order

    @classmethod
    def from_pack(cls, pack: Pack, logical_index: int, overrides: dict[int, PartSource] | None = None) -> "ResourceSpec":
        r = pack.resource(logical_index)
        parts = []
        for i in r.part_indices:
            src = (overrides or {}).get(i)
            parts.append(PartSpec.from_pack(pack, i, src))
        return cls(name=r.name_raw, type=r.type, flags=r.flags, parts=parts, name_index=r.logical.name_index)


@dataclass
class LayoutPlan:
    """Result of layout planning: storage records + per-part (storage index, offset_units)."""
    storages: list[Storage]
    part_storage: list[int]          # per physical index
    part_offset_units: list[int]     # per physical index
    payload_start: int               # absolute file offset of first payload byte
    payload_end: int                 # absolute file offset after last payload byte (before final padding)


@dataclass
class WriteReport:
    path: Path
    size: int
    header: Header
    layout: str
    storages: int
    physicals: int
    logicals: int
    warnings: list[str] = field(default_factory=list)


def stock_storage_order(keys) -> list:
    """Stock build-tool order of the storage table: storages with flags bit 3 first, then the rest, each run
    sorted by type id (census 2026-09-14/15, out/reports/census/SUMMARY.md: 47/47 shipped packs; of the three
    non-shipped packs checked, custom_rpacks/rw.rpack follows it while assets_2_pc.rpack and frank.rpack, written by
    other tools, do not). E.g. mixed bit-12 packs = 10,11,12,F0,F1,20,21; common_anims_stream = 45,42,43,44,47,48,
    49,4A; engine_pc = 10,11,12,20,21,40,61,62,F0,F1."""
    return sorted(keys, key=lambda k: (0 if k[2] & STORAGE_FLAG_STREAM else 1, k[0], k[1], k[2], k[3]))


class PackWriter:
    """Assembles an RP6L v4 file from ResourceSpecs.

    layout:
      'contiguous' — the field08 bit-12 family. Non-stream storage groups (textures) first, each a contiguous
                     region in storage-table order with its own base_units; then every resource whose parts all
                     live in stream (flags bit 3) storages, back-to-back in logical order, each part aligned to
                     its storage alignment, those storages' base_units = 0. Matches every shipped bit-12 pack
                     (common_meshes, dlc_frontier, dlc_ft_prologue, menu_level_ft) byte-for-byte.
      'grouped'    — the field08 == 0 family: every storage group is one contiguous region, groups in
                     storage-table order. Matches the other 44 shipped packs byte-for-byte.
      'preserve'   — reuse the original storage indices and offset_units carried by the PartSpecs (requires
                     every part to keep its original size); byte-identical tables and payload placement.
      'auto'       — 'contiguous' when field08 & 0x1000 else 'grouped'.

    Storage-table order defaults to `stock_storage_order`; `storage_order` overrides it (used to reproduce
    custom packs built by other tools). Logical resources are written in the order they were added.
    """

    def __init__(self, field08: int = 0, flags: int = 1, layout: str = "auto", *,
                 storage_order: Sequence[tuple[int, int, int, int]] | None = None,
                 template_storages: Sequence[Storage] | None = None,
                 name_blob_order: Sequence[int] | None = None):
        self.field08 = field08
        self.flags = flags
        self.layout = layout
        self.resources: list[ResourceSpec] = []
        self.storage_order = list(storage_order) if storage_order else None
        self.template_storages = list(template_storages) if template_storages else None
        self.name_blob_order = list(name_blob_order) if name_blob_order is not None else None

    @classmethod
    def from_pack(cls, pack: Pack, layout: str = "auto") -> "PackWriter":
        """Writer pre-configured to reproduce *pack*'s header, storage order and name-blob order."""
        return cls(pack.header.field08, pack.header.flags, layout,
                   storage_order=[s.key for s in pack.storages], template_storages=pack.storages,
                   name_blob_order=pack.name_blob_order())

    def add(self, resource: ResourceSpec) -> None:
        if not 1 <= len(resource.parts) <= MAX_PARTS:
            raise BuildError(f"{decode_name(resource.name)!r}: {len(resource.parts)} parts (runtime allows 1..{MAX_PARTS})")
        for p in resource.parts:
            if p.type not in catalogue.TYPES:
                raise BuildError(f"{decode_name(resource.name)!r}: unknown part type 0x{p.type:02X}")
        self.resources.append(resource)

    # ---- tables ----------------------------------------------------------------------------------------------

    def _effective_layout(self) -> str:
        if self.layout == "auto":
            return "contiguous" if self.field08 & FIELD08_ONDEMAND else "grouped"
        return self.layout

    def _build_names(self) -> tuple[list[int], bytes, list[int]]:
        """Name table. Returns (offsets, blob, name_index per resource).

        Shipped packs store the name strings in the original build tool's insertion order, which is *not* the
        logical order (e.g. dlc_frontier_cb_region_0: blob = [3, 0, 1, 2]); the engine only ever goes through
        name_offsets[name_index], so the blob order is free. `name_blob_order` (resource indices) reproduces
        the original blob byte-for-byte; the original per-resource `name_index` is kept when every resource
        carries one and they form a permutation, otherwise name_index == resource index.
        """
        n = len(self.resources)
        keep_index = all(r.name_index is not None for r in self.resources) and \
            sorted(r.name_index for r in self.resources) == list(range(n))
        name_index = [r.name_index if keep_index else ri for ri, r in enumerate(self.resources)]
        order = list(self.name_blob_order) if self.name_blob_order is not None else list(range(n))
        if sorted(order) != list(range(n)):
            raise BuildError("name_blob_order must be a permutation of the resource indices")
        offsets = [0] * n
        blob = bytearray()
        for ri in order:
            r = self.resources[ri]
            if b"\0" in r.name:
                raise BuildError("resource name contains NUL")
            offsets[name_index[ri]] = len(blob)
            blob += r.name + b"\0"
        return offsets, bytes(blob), name_index

    def _plan(self, table_end: int) -> LayoutPlan:
        layout = self._effective_layout()
        parts: list[tuple[int, int, PartSpec]] = []  # (resource idx, part idx within resource, spec)
        for ri, r in enumerate(self.resources):
            for pi, p in enumerate(r.parts):
                parts.append((ri, pi, p))
        n = len(parts)
        payload_start = align_up(table_end, 16)

        if layout == "preserve":
            return self._plan_preserve(parts, payload_start)

        # storage groups
        keys = self._guess_storage_keys()
        seen = {k: i for i, k in enumerate(keys)}
        if len(keys) > MAX_STORAGES:
            raise BuildError(f"{len(keys)} storage groups exceed {MAX_STORAGES}")
        part_storage = [seen[p.storage_key] for _, _, p in parts]
        part_offset_units = [0] * n
        group_size = [0] * len(keys)
        group_count = [0] * len(keys)
        group_base = [0] * len(keys)

        # storage.size is the sum of the parts' sizes each rounded up to the storage alignment (corpus: every
        # shipped pack; e.g. menu_level_ft_persistent 600968 -> 600976). Payload units are 16 bytes, so the
        # effective alignment is never below 16.
        def lay_group(g: int, cursor: int, members: list[int]) -> int:
            if not members:
                return cursor
            cursor = align_up(cursor, max(16, parts[members[0]][2].alignment))
            group_base[g] = cursor >> UNIT_SHIFT
            for i in members:
                p = parts[i][2]
                a = max(16, p.alignment)
                cursor = align_up(cursor, a)
                part_offset_units[i] = (cursor >> UNIT_SHIFT) - group_base[g]
                group_size[g] += align_up(p.size, a)
                group_count[g] += 1
                cursor += p.size
            return cursor

        if layout == "contiguous":
            # phase 1: non-stream groups as contiguous regions, in storage-table order
            stream_res = [all(p.stream for p in r.parts) for r in self.resources]
            cursor = payload_start
            for g, k in enumerate(keys):
                if k[2] & STORAGE_FLAG_STREAM:
                    continue
                members = [i for i in range(n) if part_storage[i] == g and not stream_res[parts[i][0]]]
                cursor = lay_group(g, cursor, members)
            # phase 2: stream resources back-to-back in logical order, storage base_units stay 0
            i = 0
            while i < n:
                ri = parts[i][0]
                cnt = len(self.resources[ri].parts)
                if not stream_res[ri]:
                    if any(parts[j][2].stream for j in range(i, i + cnt)):
                        raise BuildError(f"{decode_name(self.resources[ri].name)!r}: mixes stream and non-stream storages")
                    i += cnt
                    continue
                max_align = max(parts[j][2].alignment for j in range(i, i + cnt))
                cursor = align_up(cursor, max(16, max_align))
                for _ in range(cnt):
                    p = parts[i][2]
                    a = max(16, p.alignment)
                    cursor = align_up(cursor, a)
                    part_offset_units[i] = cursor >> UNIT_SHIFT
                    g = part_storage[i]
                    group_size[g] += align_up(p.size, a)
                    group_count[g] += 1
                    cursor += p.size
                    i += 1
            payload_end = cursor
        elif layout == "grouped":
            cursor = payload_start
            for g in range(len(keys)):
                cursor = lay_group(g, cursor, [i for i in range(n) if part_storage[i] == g])
            payload_end = cursor
        else:
            raise BuildError(f"unknown layout {layout!r}")

        if payload_end > MAX_OFFSET:
            raise BuildError("payload exceeds the 36-bit addressable range")
        storages = []
        for g, (t, a, f, m) in enumerate(keys):
            # the stock tool stores the low 16 bits of the count (dlc_frontier_envprobes: 123777 -> 58241)
            storages.append(Storage(t, a, f, m, group_base[g], group_size[g], 0, group_count[g] & 0xFFFF))
        return LayoutPlan(storages, part_storage, part_offset_units, payload_start, payload_end)

    def _plan_preserve(self, parts, payload_start: int) -> LayoutPlan:
        if self.template_storages is None:
            raise BuildError("'preserve' layout needs template_storages from the source pack")
        storages = [Storage(*s.key, s.base_units, s.size, s.compressed, s.count) for s in self.template_storages]
        part_storage, part_offset_units = [], []
        end = payload_start
        for _, _, p in parts:
            if p.storage_index is None or p.offset_units is None:
                raise BuildError("'preserve' layout needs original storage_index/offset_units on every part")
            part_storage.append(p.storage_index)
            part_offset_units.append(p.offset_units)
            off = (storages[p.storage_index].base_units + p.offset_units) << UNIT_SHIFT
            if off < payload_start:
                raise BuildError("'preserve' layout: original payload would overlap the (larger) tables")
            end = max(end, off + p.size)
        return LayoutPlan(storages, part_storage, part_offset_units, payload_start, end)

    # ---- write -----------------------------------------------------------------------------------------------

    def render(self) -> tuple[bytes, LayoutPlan, Header, list[str]]:
        """Plan the layout and render header + tables. Returns (table_bytes, plan, header, warnings).
        No payload is touched — this is what `roundtrip`/`validate` use to compare against a source pack."""
        if not self.resources:
            raise BuildError("nothing to write")
        name_offsets, name_blob, name_index = self._build_names()
        nphys = sum(len(r.parts) for r in self.resources)
        table_end = (HEADER_SIZE + nphys * PHYSICAL_SIZE + len(self.resources) * LOGICAL_SIZE
                     + len(name_offsets) * 4 + len(name_blob))
        # storage count is only known after planning; plan with a provisional table_end then fix up (storage count
        # affects table_end, which affects payload_start → iterate to a fixed point, at most twice)
        plan = None
        for _ in range(3):
            trial_end = table_end + (len(plan.storages) if plan else len(self._guess_storage_keys())) * STORAGE_SIZE
            plan = self._plan(trial_end)
            if trial_end == table_end + len(plan.storages) * STORAGE_SIZE:
                table_end = trial_end
                break
        else:
            raise BuildError("layout planning did not converge")

        header = Header(MAGIC, VERSION, self.field08, nphys, len(plan.storages), len(name_offsets), len(name_blob),
                        len(self.resources), self.flags)
        warnings: list[str] = []
        layout = self._effective_layout()
        if layout == "contiguous" and not (self.field08 & FIELD08_ONDEMAND):
            warnings.append("contiguous layout written into a pack without field08 bit 12")
        if layout == "grouped" and (self.field08 & FIELD08_ONDEMAND):
            warnings.append("grouped layout written into a pack WITH field08 bit 12 — mesh resources will not satisfy the one-read contract")

        if len(self.resources) > 0x10000:
            warnings.append(f"{len(self.resources)} logical resources: owner index wraps at 65536 (stock packs do the same; "
                            "GetPhysicalResourceParentName is wrong for those parts)")
        if any(len([1 for r in self.resources for p in r.parts if p.storage_key == k]) > 0xFFFF for k in {p.storage_key for r in self.resources for p in r.parts}):
            warnings.append("a storage group holds more than 65535 parts: count field wraps (stock packs do the same)")

        out = io.BytesIO()
        out.write(header.pack())
        for s in plan.storages:
            out.write(s.pack())
        k = 0
        for ri, r in enumerate(self.resources):
            for p in r.parts:
                if p.flag_bits & ~PHYS_FLAG_BITS:
                    raise BuildError("flag_bits must fit in bits 8..15")
                packed = plan.part_storage[k] | p.flag_bits | ((ri & 0xFFFF) << PHYS_OWNER_SHIFT)
                out.write(Physical(packed, plan.part_offset_units[k], p.size, p.fc).pack())
                k += 1
        k = 0
        for ri, r in enumerate(self.resources):
            out.write(Logical.make(len(r.parts), r.type, r.flags, name_index[ri], k).pack())
            k += len(r.parts)
        out.write(struct.pack(f"<{len(name_offsets)}I", *name_offsets))
        out.write(name_blob)
        tables = out.getvalue()
        assert len(tables) == table_end, (len(tables), table_end)
        return tables, plan, header, warnings

    def part_offsets(self, plan: LayoutPlan) -> list[int]:
        return [(plan.storages[plan.part_storage[i]].base_units + plan.part_offset_units[i]) << UNIT_SHIFT
                for i in range(len(plan.part_storage))]

    def write(self, path: Path | str, *, fill: bytes | Path | str | None = None, final_size: int | None = None,
              replace: bool = True) -> WriteReport:
        """Write the pack. *fill* (preserve layout only) supplies the original file — bytes, or a path read per gap
        (review F11: never the whole file) — so gaps are copied verbatim; otherwise gaps are zero. *final_size* pads
        the tail (preserve layout). With *replace* False the result stays at ``<path>.partial`` (report.path) and
        the caller renames it (build validates first, review F6). The partial file is removed on failure (F13)."""
        path = Path(path)
        tables, plan, header, warnings = self.render()
        nphys = header.physical_count
        layout = self._effective_layout()

        tmp = path.with_name(path.name + ".partial")
        try:
            size = self._write_payload(tmp, tables, plan, nphys, fill, final_size)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if replace:
            os.replace(tmp, path)
            return WriteReport(path, size, header, layout, len(plan.storages), nphys, len(self.resources), warnings)
        return WriteReport(tmp, size, header, layout, len(plan.storages), nphys, len(self.resources), warnings)

    def _write_payload(self, tmp: Path, tables: bytes, plan, nphys: int, fill, final_size: int | None) -> int:
        if isinstance(fill, (str, Path)):
            with open(fill, "rb") as src:
                return self._write_payload_from(tmp, tables, plan, nphys, fill, final_size, src)
        return self._write_payload_from(tmp, tables, plan, nphys, fill, final_size, None)

    def _write_payload_from(self, tmp: Path, tables: bytes, plan, nphys: int, fill, final_size: int | None, src) -> int:
        def gap_bytes(a: int, b: int) -> bytes:
            if src is not None:
                src.seek(a)
                data = src.read(b - a)
                return data + b"\0" * (b - a - len(data))
            return fill[a:b]

        with open(tmp, "wb") as out:
            out.write(tables)
            # payload, in file order
            order = sorted(range(nphys), key=lambda i: (plan.storages[plan.part_storage[i]].base_units + plan.part_offset_units[i]))
            specs = [p for r in self.resources for p in r.parts]
            pos = out.tell()
            for i in order:
                off = (plan.storages[plan.part_storage[i]].base_units + plan.part_offset_units[i]) << UNIT_SHIFT
                if off < pos:
                    raise BuildError(f"part {i} overlaps previous payload (0x{off:X} < 0x{pos:X})")
                if off > pos:
                    gap = off - pos
                    if fill is not None:
                        out.write(gap_bytes(pos, off))
                    else:
                        out.write(b"\0" * gap)
                    pos = off
                specs[i].source.write_to(out)
                pos += specs[i].size
            # shipped packs end on a 16-byte boundary (zero padded)
            end = align_up(pos, 16) if final_size is None else final_size
            if end > pos:
                if fill is not None:
                    out.write(gap_bytes(pos, end))
                else:
                    out.write(b"\0" * (end - pos))
                pos = end
            size = pos
        return size

    def _guess_storage_keys(self) -> list:
        """Storage-table order: explicit `storage_order` (extra keys appended in stock order) or stock order."""
        if self._effective_layout() == "preserve" and self.template_storages is not None:
            return [s.key for s in self.template_storages]
        used = []
        seen = set()
        for r in self.resources:
            for p in r.parts:
                if p.storage_key not in seen:
                    seen.add(p.storage_key)
                    used.append(p.storage_key)
        if not self.storage_order:
            return stock_storage_order(used)
        keys = [k for k in dict.fromkeys(self.storage_order) if k in seen]
        extra = stock_storage_order([k for k in used if k not in self.storage_order])
        return keys + extra
