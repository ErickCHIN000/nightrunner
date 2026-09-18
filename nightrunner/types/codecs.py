"""Codec registration for the raw-bundle families.

One `RawTypeCodec` per logical type 0x40, 0x42, 0x47, 0x49, 0x55, 0x56, 0x5A, 0x61:

    extract()   writes the lossless JSON sidecar `<index>_<name>.json` (part list, storage attributes, sizes,
                sha256, head_hex) with the family's structural dump embedded — inspection only
    build()     returns {} — every part is re-emitted from its raw file recorded in pack.json
    roundtrip() returns the stored bytes unchanged so `nr roundtrip` counts these resources as identical

The sidecar is deliberately NOT listed under editable["files"]: editing it must never trigger a rebuild path
(there is none), and deleting it must not fail a build.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from ..codecs import BuildContext, Codec, ExtractContext, register
from ..container.rp6l import PartSource, Resource
from ..extract import part_filename
from ..util.jsonio import dump_json
from . import FAMILY_MODULES
from .raw import generic_dump, read_parts, sidecar, sidecar_name


def family_module(type_id: int):
    name = FAMILY_MODULES.get(type_id)
    return importlib.import_module(f"nightrunner.types.{name}") if name else None


def structural_dump(type_id: int, parts: list[tuple[int, bytes]]) -> dict:
    mod = family_module(type_id)
    if mod is None:
        return generic_dump(parts)
    try:
        return mod.dump(parts)
    except Exception as exc:  # noqa: BLE001 — a dump must never break extraction
        d = generic_dump(parts)
        d["dump_error"] = f"{type(exc).__name__}: {exc}"
        return d


class RawTypeCodec(Codec):
    kind = "raw"

    def __init__(self, type_id: int, family: str):
        self.type_id = type_id
        self.family = family
        self.kind = f"raw:{family}"

    def extract(self, ctx: ExtractContext, res: Resource, out_dir: Path) -> dict:
        parts = read_parts(res)
        dump = structural_dump(self.type_id, parts)
        # names of the raw files the extractor wrote (same naming rule as nightrunner.extract)
        part_files = {}
        seen: dict[str, int] = {}
        for k, i in enumerate(res.part_indices):
            fname = part_filename(k, res.pack.part_type(i), seen)
            part_files[k] = fname if (out_dir / fname).exists() else None
        sc = sidecar(res.pack, res, dump, part_files)
        path = sidecar_name(out_dir)
        dump_json(sc, path)
        return {"kind": self.kind, "files": {}, "sidecar": path.name,
                "dump_form": dump.get("form") or dump.get("kind")}

    def build(self, ctx: BuildContext, entry: dict, res_dir: Path) -> dict[int, PartSource]:
        return {}

    def roundtrip(self, res: Resource) -> dict[int, bytes]:
        pk = res.pack
        out = {}
        for k, i in enumerate(res.part_indices):
            if pk.part_is_direct(i):
                out[k] = bytes(pk.read_part(i))     # a copy: the mmap must be closable after the pack is done
        return out


for _tid, _fam in FAMILY_MODULES.items():
    register(_tid, RawTypeCodec(_tid, _fam))
