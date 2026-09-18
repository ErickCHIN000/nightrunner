"""AnimCustomResource (0x49/0x4A ×2), version 4, compiled extension `.ares_obj`.

* (A) 2,369 resources, each in both common_anims packs; part shape [49,4A,49,4A] (survey 04 §2.5).
* (C → A by census 2026-09-15, `run_census()`): both pairs are (ClassReader image, fixups stream) with
  dataSize == image length. The first image is a small header object that holds two NUL-terminated strings reachable
  through its tagged pointer slots: the resource's own name and a "…Baker" class name (e.g.
  `CAnimCommandsResourceBaker`, `CBehaviorPlaceDefinitionBaker`); the second pair carries the body. The census reports
  how many first images contain the logical name and which baker names occur. Object layouts are (C); the link to the
  2,319 `.ares` text sources in data0.pak is unproven (C).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import FormatError
from . import classreader as cr
from .animgraph import census_classreader_family, dump_pair
from .raw import counter_json


def header_strings(image, blob) -> list[str]:
    """Strings reached through the (non-second-pass) pointer slots of the first image."""
    try:
        fx = cr.parse(blob)
    except FormatError:
        return []
    out = []
    for off, k in zip(fx.slot_offsets, fx.slot_kinds):
        if k & 5 or off + 8 > len(image):
            continue
        p = cr.resolve_pointer(image, off, k)
        if p["target"] is None:
            continue
        s = cr.cstring_at(fx.secondary if (k & 8) else image, p["target"], 256)
        if s:
            out.append(s)
    return out


def dump(parts: list[tuple[int, bytes]]) -> dict:
    types = [t for t, _ in parts]
    out = {"kind": "animcustom", "shape": [f"0x{t:02X}" for t in types], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if types != [0x49, 0x4A, 0x49, 0x4A] or any(d is None for _, d in parts):
        out["form"] = "unexpected"
        return out
    out["header_pair"] = dump_pair(parts[0][1], parts[1][1], kind="animcustom_header")
    out["header_strings"] = header_strings(parts[0][1], parts[1][1])
    out["body_pair"] = dump_pair(parts[2][1], parts[3][1], kind="animcustom_body")
    return out


PACKS = ["common_anims_pc.rpack", "common_anims_stream_pc.rpack"]


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    out = {"family": "animcustom", "type": "0x49"}
    out.update(census_classreader_family(assets, PACKS, 0x49, (0x49, 0x4A, 0x49, 0x4A), pair_offsets=[0, 2],
                                         limit=limit, progress=progress, label="animcustom"))
    # header-image string check (first pack only: the second is byte-identical per the shared census totals)
    bakers = Counter()
    c = Counter()
    p = assets / PACKS[0]
    if p.exists():
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x49):
                if res.part_types != (0x49, 0x4A, 0x49, 0x4A):
                    continue
                idx = list(res.part_indices)
                ss = header_strings(bytes(pk.read_part(idx[0])), bytes(pk.read_part(idx[1])))
                c["checked"] += 1
                c["header_contains_logical_name"] += (res.name in ss)
                bk = [s for s in ss if s.endswith("Baker")]
                c["header_has_baker_string"] += bool(bk)
                for b in bk:
                    bakers[b] += 1
                if limit and c["checked"] >= limit:
                    break
    out["header_strings"] = {"counts": dict(sorted(c.items())), "baker_names": counter_json(bakers)}
    return out
