"""AESP — the Techland container the Chrome Engine keeps its Wwise audio in.

Read-only. Four containers live in `<root>/<data>/work/data/audio/`: `init` (the Wwise init bank), `meta`
(soundbanks plus the `wwisepinhead` XML registry), `sfx` (in-memory `.wem`) and `streams` (streamed `.wem`).

Layout, measured over all four shipped DLTB containers on 2026-09-17 (30,172 members) and cross-checked against
the UTM-AIO tool's `AespExtractor.cs`:

    0x00  4   unknown, `00 00 02 00` on 4/4          (C: preserved, never interpreted)
    0x04  4   unknown, 0 on 4/4                      (C)
    0x08  16  container name, NUL-padded ASCII       (A)
    0x90  4   member count, u32                      (A, B)
    0xA0  4   file-table offset, u32; 0xB8 on 4/4    (A, B)

    file table: `count` entries of 152 bytes
    +0x00 128  member name, NUL-terminated UTF-8     (A, B)
    +0x80 4    Wwise id of the member                (A, reproduced 30,170/30,172 - see `wwise_id`)
    +0x84 4    0 on 30,171/30,172; the exception is        (A, meaning C)
               `wwisepinhead`, which holds 0x25019676
    +0x88 8    payload offset, u64 absolute          (A, B)
    +0x90 8    payload size, u64                     (A, B)

Every payload lies inside its file and starts exactly where the previous one ended, 30,172/30,172: the payload
region is contiguous and in table order.

**Stride.** `AespExtractor.cs` declares `EntryTotalLength = 160` but its reader consumes 152, which is what the
data actually uses. The constant is unused on its read path. Reading at 160 appears to work on `init.aesp` (one
member) and falls apart everywhere else.

Bytes this module does not understand are exposed raw and never reinterpreted. There is no writer: see
`notes/FORMATS/aesp.md` §6 for what a writer would still need (chiefly whether `mods/audio/` is a real engine
path at all).
"""
from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from ..errors import FormatError

MAGIC_OFFSET_COUNT = 0x90
MAGIC_OFFSET_TABLE = 0xA0
HEADER_MIN = 0xB8
ENTRY_SIZE = 152
NAME_SIZE = 128
CONTAINERS = ("init", "meta", "sfx", "streams")
REGISTRY_MEMBER = "wwisepinhead"

_ENTRY_TAIL = struct.Struct("<IIQQ")          # id, reserved, offset, size


def wwise_id(name: str) -> int:
    """The Wwise FNV-1 32-bit hash of *name*, lowercased — the id the engine uses for a named object.

    census 2026-09-17: the file-table id equals this for 123/124 `meta.aesp` members (the miss is the
    `wwisepinhead` XML, which is not a Wwise object), and equals `int(name)` for every `sfx` and `streams` member
    (26,517/26,517 and 3,530/3,530), whose names *are* their ids.
    """
    h = 2166136261
    for ch in name.lower().encode("utf-8"):
        h = (h * 16777619) & 0xFFFFFFFF
        h ^= ch
    return h


def expected_id(name: str) -> int:
    """The id the table should carry for a member called *name*, by the rule above."""
    return int(name) if name.isdigit() else wwise_id(name)


@dataclass(frozen=True)
class Member:
    """One member of a container. `index` is its position in the file table."""
    index: int
    name: str
    id: int                  # the u32 at +0x80
    reserved: int            # the u32 at +0x84; 0 on 30,171/30,172 members, meaning unknown. The one
                             # exception is `wwisepinhead`, the XML registry, which also breaks the id rule
                             # at +0x80 - so that entry seems to use these two words for something else.
    offset: int
    size: int

    @property
    def id_matches_name(self) -> bool:
        """Whether the stored id follows the documented rule. False is worth reporting, not worth refusing."""
        return self.id == expected_id(self.name)

    def to_json(self) -> dict:
        return {"index": self.index, "name": self.name, "id": self.id, "offset": self.offset, "size": self.size,
                "id_matches_name": self.id_matches_name}


class Aesp:
    """A memory-mapped AESP container, opened read-only.

    Game files are never written to, so the mapping is `ACCESS_READ`. `data` is the mapping itself rather than a
    memoryview of it: an exported view would keep the mapping alive and make `close()` raise, the same reason
    `container/rp6l.py` hands the mmap over directly.
    """

    def __init__(self, data, path: Path | str = "<memory>", *, mm=None, fh=None):
        self.data = data
        self.path = Path(path)
        self._mm = mm
        self._fh = fh
        self._members: list[Member] | None = None
        self._parse_header()

    # ---- opening ------------------------------------------------------------------------------------------
    @classmethod
    def open(cls, path: Path | str) -> "Aesp":
        path = Path(path)
        fh = open(path, "rb")
        try:
            mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            fh.close()
            raise
        try:
            return cls(mm, path, mm=mm, fh=fh)
        except Exception:
            mm.close()
            fh.close()
            raise

    def close(self) -> None:
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "Aesp":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- header -------------------------------------------------------------------------------------------
    def _parse_header(self) -> None:
        if len(self.data) < HEADER_MIN:
            raise FormatError(f"{self.path}: {len(self.data)} bytes is too short for an AESP header")
        head = bytes(self.data[:HEADER_MIN])
        self.word00, self.word04 = struct.unpack_from("<II", head, 0)
        self.name = head[0x08:0x18].split(b"\0", 1)[0].decode("utf-8", "replace")
        self.count = struct.unpack_from("<I", head, MAGIC_OFFSET_COUNT)[0]
        self.table_offset = struct.unpack_from("<I", head, MAGIC_OFFSET_TABLE)[0]
        end = self.table_offset + self.count * ENTRY_SIZE
        if self.table_offset < HEADER_MIN or end > len(self.data):
            raise FormatError(f"{self.path}: file table {self.count} x {ENTRY_SIZE} at 0x{self.table_offset:X} "
                              f"does not fit in {len(self.data)} bytes")

    @property
    def header_raw(self) -> bytes:
        """The whole header, for the bytes this module does not interpret."""
        return self.data[:HEADER_MIN]

    # ---- members ------------------------------------------------------------------------------------------
    @property
    def members(self) -> list[Member]:
        if self._members is None:
            out = []
            for i in range(self.count):
                base = self.table_offset + i * ENTRY_SIZE
                raw = bytes(self.data[base:base + ENTRY_SIZE])
                name = raw[:NAME_SIZE].split(b"\0", 1)[0].decode("utf-8", "replace")
                mid, reserved, offset, size = _ENTRY_TAIL.unpack_from(raw, NAME_SIZE)
                if offset + size > len(self.data):
                    raise FormatError(f"{self.path}: member {i} ({name!r}) spans {offset}+{size}, "
                                      f"past the {len(self.data)}-byte file")
                out.append(Member(i, name, mid, reserved, offset, size))
            self._members = out
        return self._members

    def __len__(self) -> int:
        return self.count

    def __iter__(self) -> Iterator[Member]:
        return iter(self.members)

    def find(self, name: str) -> Member | None:
        """The member called *name*, case-insensitively. None when there is none."""
        want = name.casefold()
        for m in self.members:
            if m.name.casefold() == want:
                return m
        return None

    def read(self, member: Member | int | str) -> bytes:
        """A member's payload. Slicing the mapping copies, which is what every caller here wants anyway."""
        if isinstance(member, int):
            member = self.members[member]
        elif isinstance(member, str):
            found = self.find(member)
            if found is None:
                raise KeyError(f"{self.path}: no member called {member!r}")
            member = found
        return self.data[member.offset:member.offset + member.size]

    # ---- whole-file checks --------------------------------------------------------------------------------
    def layout(self) -> dict:
        """What the table says about the payout region, against what this module expects of a shipped file.

        Reports rather than refuses: a container that breaks one of these is interesting, not unreadable.
        """
        ms = self.members
        table_end = self.table_offset + self.count * ENTRY_SIZE
        contiguous = gaps = 0
        prev = table_end
        for m in ms:
            if m.offset == prev:
                contiguous += 1
            else:
                gaps += 1
            prev = m.offset + m.size
        return {
            "members": self.count,
            "table_offset": self.table_offset,
            "table_end": table_end,
            "payload_bytes": sum(m.size for m in ms),
            "file_size": len(self.data),
            "contiguous": contiguous,
            "gaps": gaps,
            "trailing_bytes": len(self.data) - prev,
            "ids_matching_name": sum(1 for m in ms if m.id_matches_name),
            "reserved_nonzero": sum(1 for m in ms if m.reserved),
        }

    def to_json(self) -> dict:
        return {"path": str(self.path), "name": self.name, "size": len(self.data), "members": self.count,
                "table_offset": self.table_offset,
                "header_words": {"0x00": self.word00, "0x04": self.word04},
                "layout": self.layout()}


def audio_dir(root: Path | str, data_dir: str) -> Path:
    """`<root>/<data>/work/data/audio` — where the four containers live."""
    return Path(root) / data_dir / "work" / "data" / "audio"


def open_all(directory: Path | str) -> dict[str, Aesp]:
    """Open every container present in *directory*, keyed by stem. The caller closes them."""
    directory = Path(directory)
    out: dict[str, Aesp] = {}
    try:
        for stem in CONTAINERS:
            p = directory / f"{stem}.aesp"
            if p.is_file():
                out[stem] = Aesp.open(p)
    except Exception:
        for a in out.values():
            a.close()
        raise
    return out
