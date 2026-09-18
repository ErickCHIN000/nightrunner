"""AnimGraphBank (0x47 image + 0x48 fixups), version 140, compiled extension `.awr_obj`.

* (A) 70 banks, each present in both common_anims_pc and common_anims_stream_pc (survey 04 §2.5).
* (C → A by census 2026-09-15, `run_census()`): part 0x48 is a plain ClassReader fixups stream (no prefix) whose
  dataSize equals the 0x47 part length; class words carry 0xB0 in the high byte. Object layouts are (C).
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import FormatError
from . import classreader as cr
from .raw import counter_json, strings, words


def dump_pair(image, blob, *, kind: str) -> dict:
    out = {"kind": kind, "image_size": len(image), "fixups_size": len(blob), "words": words(image, 8)}
    try:
        fx = cr.parse(blob)
    except FormatError as exc:
        out["error"] = str(exc)
        out["strings"] = strings(image, limit=40)
        return out
    out["fixups"] = fx.summary()
    out["image_check"] = cr.check_against_image(fx, image)
    out["reserialise_identical"] = cr.serialise(fx) == bytes(blob)
    out["stream_parsed_to_exact_end"] = fx.stream_end == len(blob)
    slots = list(zip(fx.slot_offsets, fx.slot_kinds))
    ptrs = []
    for off, k in slots[:24]:
        if k & 4:
            continue
        if off + 8 <= len(image):
            p = cr.resolve_pointer(image, off, k)
            tgt = p["target"]
            src = fx.secondary if (k & 8) else image
            p["string"] = cr.cstring_at(src, tgt, 128) if (tgt is not None and src is not None and not (k & 1)) else None
            ptrs.append(p)
    out["pointer_sample"] = ptrs
    out["strings"] = strings(image, limit=40)
    return out


def dump(parts: list[tuple[int, bytes]]) -> dict:
    types = [t for t, _ in parts]
    out = {"kind": "animgraph", "shape": [f"0x{t:02X}" for t in types], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if types != [0x47, 0x48] or any(d is None for _, d in parts):
        out["form"] = "unexpected"
        return out
    out.update(dump_pair(parts[0][1], parts[1][1], kind="animgraph"))
    return out


PACKS = ["common_anims_pc.rpack", "common_anims_stream_pc.rpack"]


def census_classreader_family(assets: Path, packs: list[str], type_id: int, expect_shape: tuple[int, ...],
                              *, pair_offsets: list[int], limit: int | None = None, progress=None, label: str = "") -> dict:
    """Shared census for families whose parts are (image, fixups) pairs at the given part ordinals."""
    c = Counter()
    classes = Counter()
    kinds = Counter()
    per_pack = []
    failures = []
    for name in packs:
        p = assets / name
        if not p.exists():
            continue
        if progress:
            progress(f"{label} census {name}")
        pc = Counter()
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(type_id):
                pc["resources"] += 1
                if res.part_types != expect_shape:
                    pc["unexpected_shape"] += 1
                    continue
                idx = list(res.part_indices)
                for po in pair_offsets:
                    img = bytes(pk.read_part(idx[po]))
                    blob = bytes(pk.read_part(idx[po + 1]))
                    pc["pairs"] += 1
                    try:
                        fx = cr.parse(blob)
                    except FormatError as exc:
                        pc["parse_error"] += 1
                        if len(failures) < 20:
                            failures.append({"pack": name, "index": res.index, "name": res.name, "pair": po, "error": str(exc)})
                        continue
                    pc["parsed"] += 1
                    pc["stream_parsed_to_exact_end"] += (fx.stream_end == len(blob))
                    pc["reserialise_identical"] += (cr.serialise(fx) == bytes(blob))
                    pc["data_size_eq_image"] += (fx.data_size == len(img))
                    pc["data_size_le_image"] += (fx.data_size <= len(img))
                    pc["has_secondary"] += fx.has_secondary
                    pc["no_bounds_problems"] += (not cr.check_against_image(fx, img)["problems"])
                    pc["records_total"] += fx.table_count
                    pc["objects_total"] += fx.object_count
                    pc["relocations_total"] += len(fx.slot_offsets)
                    classes.update(f"0x{r.class_word:08X}" for r in fx.records)
                    kinds.update(fx.slot_kinds)
                if limit and pc["resources"] >= limit:
                    break
        per_pack.append({"pack": name, "counts": dict(sorted(pc.items()))})
        c.update(pc)
    return {"totals": dict(sorted(c.items())), "per_pack": per_pack, "class_histogram": counter_json(classes, limit=80),
            "class_distinct": len(classes), "slot_kinds": {f"0x{k:02X}": v for k, v in sorted(kinds.items())}, "failures": failures}


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    out = {"family": "animgraph", "type": "0x47"}
    out.update(census_classreader_family(assets, PACKS, 0x47, (0x47, 0x48), pair_offsets=[0], limit=limit, progress=progress, label="animgraph"))
    return out
