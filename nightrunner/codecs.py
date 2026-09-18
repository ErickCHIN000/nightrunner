"""Codec registry: how each logical resource type maps to editable files and back.

A codec turns one logical resource into files inside its resource directory (extract) and, on build, turns
possibly-edited files back into the bytes of specific parts. Anything a codec does not regenerate is taken
from the raw part files recorded in pack.json, so a codec only has to own the parts it understands.

    class Codec:
        kind: str                                   # "raw" | "cast" | "dds" | ...
        def extract(self, ctx: ExtractContext, res: Resource, out_dir: Path) -> dict
            # returns the "editable" record stored in pack.json: {"kind": ..., "files": {rel: {"sha256": ...}}, ...}
        def build(self, ctx: BuildContext, entry: dict, res_dir: Path) -> dict[int, PartSource]
            # returns replacement bytes per *part ordinal within the resource* (0-based) for parts it regenerates.
            # Called only when at least one editable file's sha256 differs from the recorded one.

The registry maps logical type id → codec instance; unknown types fall back to RawCodec, which writes nothing
beyond the raw parts and a small JSON header dump. Mesh/texture/prefab codecs register themselves on import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .container.rp6l import Pack, Resource, PartSource
from .util.hashing import sha256_file


@dataclass
class ExtractContext:
    pack: Pack
    rpx_dir: Path
    options: dict = field(default_factory=dict)      # e.g. {"sdb": Path, "pak": Path, "no_raw": bool}
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class BuildContext:
    rpx_dir: Path
    spec: dict
    options: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def warn(self, msg: str) -> None:
        self.warnings.append(msg)


class Codec:
    kind = "raw"

    def extract(self, ctx: ExtractContext, res: Resource, out_dir: Path) -> dict:  # pragma: no cover - interface
        return {"kind": self.kind, "files": {}}

    def build(self, ctx: BuildContext, entry: dict, res_dir: Path) -> dict[int, PartSource]:  # pragma: no cover
        return {}

    # helpers ------------------------------------------------------------------------------------------------
    @staticmethod
    def file_record(path: Path, rel: str) -> dict:
        return {rel: {"sha256": sha256_file(path), "size": path.stat().st_size}}


class RawCodec(Codec):
    """No editable view: the raw parts are the intermediate. Writes a JSON dump of the leading words of each part
    so the family can be inspected without a hex editor."""

    kind = "raw"

    def extract(self, ctx: ExtractContext, res: Resource, out_dir: Path) -> dict:
        return {"kind": self.kind, "files": {}}


_REGISTRY: dict[int, Codec] = {}
_LOADERS: list[Callable[[], None]] = []


def register(type_id: int, codec: Codec) -> None:
    _REGISTRY[type_id] = codec


def register_loader(fn: Callable[[], None]) -> None:
    """Deferred registration hook so heavy codecs (numpy) load only when needed."""
    _LOADERS.append(fn)


def _ensure_loaded() -> None:
    while _LOADERS:
        fn = _LOADERS.pop(0)
        fn()


def codec_for(type_id: int) -> Codec:
    _ensure_loaded()
    return _REGISTRY.get(type_id) or _RAW


_RAW = RawCodec()


_IMPORT_ERRORS: dict[int, str] = {}
_BUILTIN = (("nightrunner.mesh.codec", (0x10,)), ("nightrunner.texture.codec", (0x20,)),
            ("nightrunner.types.codecs", ()))


def codec_import_error(type_id: int) -> str | None:
    """Why the built-in codec for *type_id* failed to import (None when it loaded or never existed). Review F4:
    build() refuses to silently keep raw parts when an editable file changed but its codec is missing."""
    _ensure_loaded()
    return _IMPORT_ERRORS.get(type_id) if type_id not in _REGISTRY else None


def _load_builtin() -> None:
    # Each optional codec module registers itself; import errors degrade to raw for extract/roundtrip, but are
    # remembered so build() can refuse edits it cannot encode.
    for mod, types in _BUILTIN:
        try:
            __import__(mod)
        except ImportError as exc:
            for t in types:
                _IMPORT_ERRORS[t] = f"{mod}: {exc}"


register_loader(_load_builtin)
