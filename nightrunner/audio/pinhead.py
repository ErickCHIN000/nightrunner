"""`wwisepinhead` — Techland's Wwise integration registry.

A member of `meta.aesp`, 6.7 MB of plain XML rooted at `<Mapping Version="768">`. It is why naming is close to
solved for this game's audio: the engine's ids come with the names they were hashed from.

census 2026-09-17 (DLTB): `Event` 23,904 · `WwiseEvent` 23,904 · `SwitchValue` 4,252 · `WwiseState` 3,667 ·
`WwiseGameParameter` 714 · `WwiseSwitch` 585 · `Switch` 473 · `ParameterRef` 445 · `WwiseStateGroup` 335 ·
`Parameter` 269 · `AuxBus` 163. `<Preloads>` holds 128 entries, every one with `name` and `id`, 88 also
`managed`, 4 also `localized`, and exactly one child element each (`FileData`).

Parsed with `xml.etree` — stdlib, no external parser. The document is read from the game's own container, so it
is treated as data: nothing here executes, resolves entities or follows references out of the document.

Two things worth knowing before building on it:

* A `<Preload>` `id` is **not** the container's file-table id for the same bank (0/123 match), so the two are
  different hashes of different things. Which one the engine looks up is undecoded (hole E4).
* `PinheadPatcher.cs` in UTM-AIO writes `<File id="...">` children into a `<Preload>` as a per-bank wem
  whitelist. DLTB's shipped registry has **none** — 0 of 128 preloads (hole E5).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

MEMBER_NAME = "wwisepinhead"


@dataclass
class Preload:
    name: str
    id: int | None
    managed: bool
    localized: bool
    files: list[str] = field(default_factory=list)       # <FileData name=...>
    whitelist: list[int] = field(default_factory=list)   # <File id=...>, absent in shipped DLTB data

    def to_json(self) -> dict:
        return {"name": self.name, "id": self.id, "managed": self.managed, "localized": self.localized,
                "files": self.files, "whitelist": self.whitelist}


@dataclass
class Named:
    """Any id-carrying named object in the registry: events, switches, states, parameters, buses."""
    kind: str
    name: str
    id: int | None
    attrs: dict[str, str] = field(default_factory=dict)

    def to_json(self) -> dict:
        return {"kind": self.kind, "name": self.name, "id": self.id, **self.attrs}


def _int(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


class Pinhead:
    """The parsed registry. `objects` is every element carrying both a name and an id, in document order."""

    def __init__(self, xml: bytes | str, source: str = MEMBER_NAME):
        self.source = source
        text = xml.decode("utf-8", "replace") if isinstance(xml, (bytes, bytearray, memoryview)) else xml
        self.root = ET.fromstring(text)
        self.version = self.root.get("Version")
        self.preloads: list[Preload] = []
        self.objects: list[Named] = []
        self._by_id: dict[int, Named] = {}
        self._by_name: dict[str, Named] = {}
        self._parse()

    def _parse(self) -> None:
        for el in self.root.findall("./Preloads/*"):
            self.preloads.append(Preload(
                name=el.get("name", ""),
                id=_int(el.get("id")),
                managed=el.get("managed") == "true",
                localized=el.get("localized") == "true",
                files=[f.get("name", "") for f in el.findall("FileData")],
                whitelist=[i for i in (_int(f.get("id")) for f in el.findall("File")) if i is not None],
            ))
        for el in self.root.iter():
            if el.tag in ("Preloads", "Mapping"):
                continue
            name, oid = el.get("name"), _int(el.get("id"))
            if name is None or oid is None:
                continue
            attrs = {k: v for k, v in el.attrib.items() if k not in ("name", "id")}
            obj = Named(el.tag, name, oid, attrs)
            self.objects.append(obj)
            self._by_id.setdefault(oid, obj)
            self._by_name.setdefault(name.casefold(), obj)

    # ---- lookups ------------------------------------------------------------------------------------------
    def name_of(self, obj_id: int) -> str | None:
        """The name behind a Wwise id, or None when the registry does not carry it."""
        o = self._by_id.get(obj_id)
        return None if o is None else o.name

    def by_name(self, name: str) -> Named | None:
        return self._by_name.get(name.casefold())

    def of_kind(self, kind: str) -> list[Named]:
        return [o for o in self.objects if o.kind == kind]

    def names(self) -> dict[int, str]:
        """id -> name, for every object that carries both. Useful as a lookup table for bank contents."""
        return {o.id: o.name for o in self.objects if o.id is not None}

    def kind_histogram(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for o in self.objects:
            out[o.kind] = out.get(o.kind, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def to_json(self) -> dict:
        return {"source": self.source, "version": self.version, "preloads": len(self.preloads),
                "objects": len(self.objects), "kinds": self.kind_histogram(),
                "preloads_with_whitelist": sum(1 for p in self.preloads if p.whitelist)}


def from_container(aesp) -> "Pinhead | None":
    """Read `wwisepinhead` out of an open `Aesp`, or None when that container has no registry."""
    m = aesp.find(MEMBER_NAME)
    if m is None:
        return None
    return Pinhead(bytes(aesp.read(m)), f"{aesp.path.name}:{MEMBER_NAME}")
