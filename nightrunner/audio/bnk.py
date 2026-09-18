"""Wwise soundbank (`.bnk`) — chunk walk and HIRC object index.

Read-only, and deliberately shallow. This decodes the structure every bank shares — the chunk list, the header,
the HIRC object table, and the head of a Sound object — and stops there. Object bodies past the fields below are
handed back raw rather than guessed at.

census 2026-09-17 over all 123 banks in DLTB's `meta.aesp`:

* 123/123 parse, and the chunk walk lands exactly on the end of the file
* `BKHD` version 150 on 123/123 — one Wwise version, so nothing here branches on it
* chunks seen: `BKHD` 123, `HIRC` 121, `DIDX` 91, `DATA` 91. Across all four containers (124 banks, the init
  bank included) also `INIT`, `STMG`, `ENVS` and `PLAT`, one each — they are the init bank's own chunks. This
  module exposes every chunk by tag and decodes none of those four (C).
* HIRC is consumed exactly by the object walk in 121/121 banks that have one
* 69,330 Sound objects, referencing 30,461 distinct source ids
* Sound stream types: 68,874 embedded, 239 prefetch, 217 streamed

A Sound object's body begins `u32 plugin id`, `u8 stream type`, `u32 source id`. Everything after that is
`body_raw` (C). The wider bank format is documented by wwiser (https://github.com/bnnm/wwiser), which is
read-only by design and carries no licence, so nothing here is derived from its code — only from the shipped data
and from the structure the UTM-AIO C# tool also reads.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import FormatError

BKHD = b"BKHD"
HIRC = b"HIRC"
DIDX = b"DIDX"
DATA = b"DATA"

#: HIRC object type -> name. Only the ones this module interprets are named; the rest stay numeric on purpose.
OBJECT_SOUND = 2
OBJECT_NAMES = {OBJECT_SOUND: "Sound"}

#: Sound stream type. Names from UTM-AIO (B); the census that makes them plausible is in the module docstring.
STREAM_EMBEDDED, STREAM_PREFETCH, STREAM_STREAMED = 0, 1, 2
STREAM_NAMES = {STREAM_EMBEDDED: "embedded", STREAM_PREFETCH: "prefetch", STREAM_STREAMED: "streamed"}

#: Source plugin ids seen in DLTB, with counts from the 2026-09-17 census. Unnamed ones are left numeric.
PLUGIN_NAMES = {0x00040001: "vorbis"}


@dataclass
class Chunk:
    tag: str
    offset: int          # of the tag itself
    size: int            # of the body, not counting the 8-byte tag+size

    @property
    def body_offset(self) -> int:
        return self.offset + 8

    @property
    def end(self) -> int:
        return self.offset + 8 + self.size


@dataclass
class Sound:
    """The head of a HIRC Sound object. `body_raw` is everything this module does not decode."""
    index: int           # position in the HIRC object table
    id: int
    offset: int          # of the object header, from the start of the bank
    body_offset: int     # of the object body
    size: int            # of the body
    plugin: int
    stream_type: int
    source_id: int
    body_raw: bytes = field(repr=False, default=b"")

    #: Offset of the stream-type byte from the start of the bank. The one byte UTM-AIO patches to redirect a
    #: sound to the stream store; recorded here so a future writer does not have to re-derive it.
    @property
    def stream_type_offset(self) -> int:
        return self.body_offset + 4

    @property
    def stream_type_name(self) -> str:
        return STREAM_NAMES.get(self.stream_type, f"unknown_{self.stream_type}")

    @property
    def plugin_name(self) -> str:
        return PLUGIN_NAMES.get(self.plugin, f"0x{self.plugin:08X}")

    def to_json(self) -> dict:
        return {"index": self.index, "id": self.id, "source_id": self.source_id,
                "stream_type": self.stream_type, "stream_type_name": self.stream_type_name,
                "stream_type_offset": self.stream_type_offset,
                "plugin": self.plugin, "plugin_name": self.plugin_name, "size": self.size}


@dataclass
class HircObject:
    """Any HIRC object. Only Sound is decoded further; the rest carry their bytes and nothing more."""
    index: int
    type: int
    id: int
    offset: int
    size: int

    @property
    def type_name(self) -> str:
        return OBJECT_NAMES.get(self.type, f"type_{self.type}")


class Bank:
    """A parsed soundbank. `data` is the whole bank; nothing is copied unless asked for."""

    def __init__(self, data: bytes | memoryview, name: str = "<memory>"):
        self.data = bytes(data)
        self.name = name
        self.chunks: list[Chunk] = []
        self.version: int | None = None
        self.bank_id: int | None = None
        self._objects: list[HircObject] | None = None
        self._sounds: list[Sound] | None = None
        self._hirc_exact: bool | None = None
        self._parse()

    # ---- chunks -------------------------------------------------------------------------------------------
    def _parse(self) -> None:
        d = self.data
        if len(d) < 8:
            raise FormatError(f"{self.name}: {len(d)} bytes is too short for a bank")
        pos = 0
        while pos + 8 <= len(d):
            tag = d[pos:pos + 4]
            size = struct.unpack_from("<I", d, pos + 4)[0]
            if pos + 8 + size > len(d):
                raise FormatError(f"{self.name}: chunk {tag!r} at {pos} claims {size} bytes, past the end")
            self.chunks.append(Chunk(tag.decode("ascii", "replace"), pos, size))
            pos += 8 + size
        self.trailing = len(d) - pos          # 0 on 123/123 shipped banks
        head = self.chunk(BKHD)
        if head is None:
            raise FormatError(f"{self.name}: no BKHD chunk")
        if head.size >= 8:
            self.version, self.bank_id = struct.unpack_from("<II", d, head.body_offset)

    def chunk(self, tag: bytes | str) -> Chunk | None:
        want = tag.decode("ascii") if isinstance(tag, bytes) else tag
        for c in self.chunks:
            if c.tag == want:
                return c
        return None

    def chunk_data(self, tag: bytes | str) -> bytes | None:
        c = self.chunk(tag)
        return None if c is None else self.data[c.body_offset:c.end]

    # ---- HIRC ---------------------------------------------------------------------------------------------
    def _walk_hirc(self) -> None:
        self._objects, self._sounds = [], []
        c = self.chunk(HIRC)
        if c is None:
            self._hirc_exact = None
            return
        d = self.data
        pos = c.body_offset
        count = struct.unpack_from("<I", d, pos)[0]
        pos += 4
        for i in range(count):
            if pos + 5 > c.end:
                raise FormatError(f"{self.name}: HIRC object {i} header runs past the chunk")
            otype = d[pos]
            osize = struct.unpack_from("<I", d, pos + 1)[0]
            body = pos + 5
            if body + osize > c.end:
                raise FormatError(f"{self.name}: HIRC object {i} body runs past the chunk")
            oid = struct.unpack_from("<I", d, body)[0] if osize >= 4 else 0
            self._objects.append(HircObject(i, otype, oid, pos, osize))
            if otype == OBJECT_SOUND and osize >= 13:
                sb = body + 4                                  # past the object id
                plugin = struct.unpack_from("<I", d, sb)[0]
                stream_type = d[sb + 4]
                source_id = struct.unpack_from("<I", d, sb + 5)[0]
                self._sounds.append(Sound(len(self._sounds), oid, pos, sb, osize - 4, plugin, stream_type,
                                          source_id, bytes(d[sb + 9:body + osize])))
            pos = body + osize
        self._hirc_exact = (pos == c.end)

    @property
    def hirc_exact(self) -> bool | None:
        """Whether the object walk consumed HIRC exactly; None when the bank has no HIRC.

        A property rather than an attribute because the walk is lazy: reading it before touching `objects` used
        to hand back the un-walked None.
        """
        if self._objects is None:
            self._walk_hirc()
        return self._hirc_exact

    @property
    def objects(self) -> list[HircObject]:
        if self._objects is None:
            self._walk_hirc()
        return self._objects

    @property
    def sounds(self) -> list[Sound]:
        if self._sounds is None:
            self._walk_hirc()
        return self._sounds

    def type_histogram(self) -> dict[int, int]:
        out: dict[int, int] = {}
        for o in self.objects:
            out[o.type] = out.get(o.type, 0) + 1
        return dict(sorted(out.items()))

    def stream_histogram(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.sounds:
            out[s.stream_type_name] = out.get(s.stream_type_name, 0) + 1
        return out

    def source_ids(self) -> set[int]:
        return {s.source_id for s in self.sounds}

    def to_json(self) -> dict:
        return {"name": self.name, "size": len(self.data), "version": self.version, "bank_id": self.bank_id,
                "chunks": [{"tag": c.tag, "offset": c.offset, "size": c.size} for c in self.chunks],
                "trailing_bytes": self.trailing, "objects": len(self.objects), "hirc_exact": self.hirc_exact,
                "object_types": self.type_histogram(), "sounds": len(self.sounds),
                "stream_types": self.stream_histogram(), "distinct_source_ids": len(self.source_ids())}


def is_bank(data: bytes | memoryview) -> bool:
    return bytes(data[:4]) == BKHD


def load(path: Path | str) -> Bank:
    path = Path(path)
    return Bank(path.read_bytes(), path.name)
