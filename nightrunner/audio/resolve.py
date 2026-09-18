"""Where a sound's audio actually lives — the join across banks and containers.

A Sound object in a bank names a *source id*. That id is looked up in one of three places, and which one decides
what replacing the sound would take:

| resolves from | sounds | what a replacement would mean |
|---|---|---|
| `sfx.aesp` | 43,017 | the bank holds no audio for it; the container member is the audio |
| the bank's own `DIDX`/`DATA` | 23,340 | the audio is baked into the bank |
| `streams.aesp` | 486 | streamed or prefetched |
| nowhere found | 2,487 | not in the four containers of this install |

census 2026-09-17, DLTB, 69,330 Sound objects over 123 banks in `meta.aesp`.

Two things that census settles:

* **A bank's `DIDX` ids and the container member ids are disjoint** — 0 of 7,953 `DIDX` ids are also members of
  `sfx.aesp`. They are separate stores, not two copies of the same audio.
* **Most sounds are not baked into their bank.** 91 of 123 banks carry a `DIDX`/`DATA` pair at all, and the
  43,017 sounds that resolve out of `sfx.aesp` belong to banks that hold no audio of their own. `menu` is one of
  them: `BKHD` and `HIRC` only, 33 sounds, every one of them resolved from `sfx.aesp`.

That distinction matters for modding. UTM-AIO flips a sound's stream type to "streamed" so the engine fetches it
from the stream store rather than from the bank — which is what a sound baked into `DIDX`/`DATA` needs. A sound
that already resolves out of a container is a different, simpler case: the container member *is* the audio.
Whether the engine can be pointed at a replacement member is still hole E7 (nothing here is confirmed in game),
so this module reports the join and does not act on it.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

from .aesp import CONTAINERS, Aesp, Member, wwise_id
from .bnk import Bank, is_bank
from .pinhead import Pinhead, from_container

OBJECT_ACTION = 3
OBJECT_EVENT = 4

SOURCE_SFX = "sfx"
SOURCE_STREAMS = "streams"
SOURCE_BANK = "bank"
SOURCE_MISSING = "missing"


def didx_ids(bank: Bank) -> dict[int, tuple[int, int]]:
    """`DIDX` as {wem id: (offset into DATA, size)}. Empty when the bank carries no audio of its own."""
    raw = bank.chunk_data("DIDX")
    if not raw:
        return {}
    out = {}
    for i in range(len(raw) // 12):
        wid, off, size = struct.unpack_from("<III", raw, i * 12)
        out[wid] = (off, size)
    return out


@dataclass
class Resolved:
    """One Sound object and where its audio came from."""
    bank: str
    sound_index: int
    sound_id: int
    source_id: int
    stream_type: int
    stream_type_name: str
    stream_type_offset: int
    source: str                      # one of the SOURCE_* constants
    size: int = 0                    # of the wem, when it was found
    container: str | None = None     # container stem, when it came from one

    @property
    def found(self) -> bool:
        return self.source != SOURCE_MISSING

    @property
    def in_bank(self) -> bool:
        """Audio baked into the bank. Replacing it is the case the stream-type flip exists for."""
        return self.source == SOURCE_BANK

    def to_json(self) -> dict:
        return {"bank": self.bank, "sound_index": self.sound_index, "sound_id": self.sound_id,
                "source_id": self.source_id, "stream_type": self.stream_type_name,
                "stream_type_offset": self.stream_type_offset, "source": self.source,
                "container": self.container, "size": self.size}


class AudioIndex:
    """Every container of one install, opened together, with the joins between them.

    Opening is cheap — the containers are memory-mapped and only their file tables are read — but resolving a
    bank parses it, so that is done per bank on demand.
    """

    def __init__(self, directory: Path | str):
        self.dir = Path(directory)
        self.containers: dict[str, Aesp] = {}
        for stem in CONTAINERS:
            p = self.dir / f"{stem}.aesp"
            if p.is_file():
                self.containers[stem] = Aesp.open(p)
        self._by_id: dict[str, dict[int, Member]] = {
            stem: {m.id: m for m in c} for stem, c in self.containers.items() if stem in (SOURCE_SFX, SOURCE_STREAMS)
        }
        self._registry: Pinhead | None = None
        self._banks: dict[str, Bank] = {}

    # ---- lifecycle ----------------------------------------------------------------------------------------
    def close(self) -> None:
        for c in self.containers.values():
            c.close()
        self.containers.clear()
        self._banks.clear()

    def __enter__(self) -> "AudioIndex":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- the pieces ---------------------------------------------------------------------------------------
    @property
    def registry(self) -> Pinhead | None:
        """`wwisepinhead`, parsed once. None when this install has no `meta.aesp`."""
        if self._registry is None and "meta" in self.containers:
            self._registry = from_container(self.containers["meta"])
        return self._registry

    def bank_names(self) -> list[str]:
        """Every soundbank member, across `meta` and `init`."""
        out = []
        for stem in ("meta", "init"):
            c = self.containers.get(stem)
            if c is None:
                continue
            out += [m.name for m in c if is_bank(c.read(m))]
        return out

    def bank(self, name: str) -> Bank | None:
        """A parsed bank by member name, cached for the life of the index."""
        if name in self._banks:
            return self._banks[name]
        for stem in ("meta", "init"):
            c = self.containers.get(stem)
            if c is None:
                continue
            m = c.find(name)
            if m is not None:
                data = c.read(m)
                if is_bank(data):
                    self._banks[name] = Bank(data, m.name)
                    return self._banks[name]
        return None

    def member_of(self, source_id: int) -> tuple[str, Member] | None:
        """The container member an id names, `sfx` before `streams`. None when neither has it."""
        for stem in (SOURCE_SFX, SOURCE_STREAMS):
            m = self._by_id.get(stem, {}).get(source_id)
            if m is not None:
                return stem, m
        return None

    # ---- the join -----------------------------------------------------------------------------------------
    def resolve_bank(self, name: str) -> list[Resolved]:
        """Every Sound of one bank, with where its audio lives."""
        b = self.bank(name)
        if b is None:
            return []
        local = didx_ids(b)
        out = []
        for s in b.sounds:
            if s.source_id in local:
                out.append(Resolved(name, s.index, s.id, s.source_id, s.stream_type, s.stream_type_name,
                                    s.stream_type_offset, SOURCE_BANK, local[s.source_id][1], None))
                continue
            hit = self.member_of(s.source_id)
            if hit is None:
                out.append(Resolved(name, s.index, s.id, s.source_id, s.stream_type, s.stream_type_name,
                                    s.stream_type_offset, SOURCE_MISSING))
            else:
                stem, m = hit
                out.append(Resolved(name, s.index, s.id, s.source_id, s.stream_type, s.stream_type_name,
                                    s.stream_type_offset, stem, m.size, stem))
        return out

    def read_audio(self, r: Resolved) -> bytes | None:
        """The `.wem` bytes behind a resolved sound, wherever they live. None when it was not found."""
        if r.source == SOURCE_MISSING:
            return None
        if r.source == SOURCE_BANK:
            b = self.bank(r.bank)
            data = b.chunk_data("DATA") if b else None
            if data is None:
                return None
            off, size = didx_ids(b)[r.source_id]
            return bytes(data[off:off + size])
        c = self.containers.get(r.container or r.source)
        hit = self._by_id.get(r.container or r.source, {}).get(r.source_id)
        return None if c is None or hit is None else bytes(c.read(hit))

    # ---- events -------------------------------------------------------------------------------------------
    def event_names_by_sound(self, bank_name: str) -> dict[int, list[str]]:
        """{Sound object id: [event name, ...]} for one bank, via the registry.

        The registry's `<Event id>` is Techland's own id, not Wwise's. The Wwise object id is the hash of the
        event *name* — census 2026-09-17: every event whose preload names a bank in `meta.aesp` is found that way,
        10,833/10,833.

        From the Event object (type 4): `u32 id`, `u8 action count`, then that many `u32` action ids. An Action
        (type 3) is `u32 id`, `u16 action type`, `u32 target`. When the target is a Sound in the same bank, the
        event is credited to it here. Often it is not: of 14,430 actions walked, 3,198 target a Sound directly and
        the rest point at container objects (random/sequence/actor-mixer) whose children this module does not
        decode. Those events simply do not appear, rather than being attached to a guess.
        """
        b = self.bank(bank_name)
        ph = self.registry
        if b is None or ph is None:
            return {}
        by_id = {o.id: o for o in b.objects}
        sound_ids = {s.id for s in b.sounds}
        want = {p.name.lower() for p in ph.preloads if p.name.lower() == bank_name.lower()}
        out: dict[int, list[str]] = {}
        for ev in ph.objects:
            if ev.kind != "Event":
                continue
            obj = by_id.get(wwise_id(ev.name))
            if obj is None or obj.type != OBJECT_EVENT or obj.size < 5:
                continue
            body = b.data[obj.offset + 5:obj.offset + 5 + obj.size]
            count = body[4]
            if 5 + 4 * count > len(body):
                continue
            for i in range(count):
                aid = struct.unpack_from("<I", body, 5 + 4 * i)[0]
                act = by_id.get(aid)
                if act is None or act.type != OBJECT_ACTION or act.size < 10:
                    continue
                ab = b.data[act.offset + 5:act.offset + 5 + act.size]
                target = struct.unpack_from("<I", ab, 6)[0]
                if target in sound_ids:
                    out.setdefault(target, []).append(ev.name)
        return {k: sorted(set(v)) for k, v in out.items()}

    def events_for_preload(self, preload_id: int) -> list:
        """Registry events that name a preload id — the way in from "which menu sound is this?"."""
        ph = self.registry
        if ph is None:
            return []
        want = str(preload_id)
        return [o for o in ph.objects if o.kind == "Event" and o.attrs.get("preload_id") == want]

    def summary(self) -> dict:
        return {"dir": str(self.dir),
                "containers": {k: len(v) for k, v in self.containers.items()},
                "banks": len(self.bank_names()),
                "registry_objects": len(self.registry.objects) if self.registry else 0}
