"""Corpus census: everything we want to know about the shipped packs before trusting a writer policy.

Produces one JSON per pack plus a merged summary (tools/census_corpus.py drives it)."""

from __future__ import annotations

from collections import Counter

from ..util.binio import align_up
from .rp6l import Pack, PackWriter, ResourceSpec


def _tail_cmp(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def census(pack: Pack, *, layouts: tuple[str, ...] = ("auto",)) -> dict:
    h = pack.header
    out: dict = {
        "file": pack.path.name, "size": pack.size, "table_end": pack.table_end,
        "header": h.to_json(), "storages": [s.to_json() for s in pack.storages],
        "types": {f"0x{t:02X}": c for t, c in pack.type_histogram().items()},
    }
    # part shapes per logical type
    shapes: Counter = Counter()
    lflags: Counter = Counter()
    pflags: Counter = Counter()
    fcs: Counter = Counter()
    names_dup: Counter = Counter()
    leading_space = 0
    nonascii = 0
    max_parts = 0
    for r in pack:
        shapes[(r.type, r.part_types)] += 1
        lflags[(r.type, r.flags)] += 1
        max_parts = max(max_parts, r.logical.part_count)
        nm = r.name_raw
        names_dup[(r.type, nm)] += 1
        if nm.startswith(b" "):
            leading_space += 1
        if any(b >= 0x80 for b in nm):
            nonascii += 1
        for i in r.part_indices:
            p = pack.physicals[i]
            pflags[(pack.part_type(i), p.flag_bits)] += 1
            fcs[p.fc] += 1
    out["part_shapes"] = [{"type": f"0x{t:02X}", "parts": [f"0x{x:02X}" for x in sh], "count": c}
                          for (t, sh), c in sorted(shapes.items())]
    out["logical_flags"] = [{"type": f"0x{t:02X}", "flags": f"0x{f:02X}", "count": c} for (t, f), c in sorted(lflags.items())]
    out["physical_flag_bits"] = [{"part_type": f"0x{t:02X}", "bits": f"0x{f:04X}", "count": c} for (t, f), c in sorted(pflags.items())]
    out["fc_values"] = {f"0x{k:X}": v for k, v in sorted(fcs.items())}
    out["max_parts"] = max_parts
    out["duplicate_names"] = sum(c - 1 for c in names_dup.values() if c > 1)
    out["leading_space_names"] = leading_space
    out["nonascii_names"] = nonascii

    # file-order walk: gaps, overlaps, ordering
    order = sorted(range(len(pack.physicals)), key=lambda i: pack.part_offset(i))
    pos = align_up(pack.table_end, 16)
    gaps = overlaps = nonzero_gaps = 0
    max_gap = 0
    file_order_is_logical = True
    prev = (-1, -1)
    for i in order:
        off = pack.part_offset(i)
        p = pack.physicals[i]
        if off < pos:
            overlaps += 1
        elif off > pos:
            gaps += 1
            max_gap = max(max_gap, off - pos)
            if off - pos <= 4096 and any(pack._data[pos:off]):
                nonzero_gaps += 1
        cur = (p.owner, i)
        if cur < prev:
            file_order_is_logical = False
        prev = cur
        pos = off + p.size
    out["file_order"] = dict(gaps=gaps, nonzero_gaps=nonzero_gaps, max_gap=max_gap, overlaps=overlaps,
                             logical_order=file_order_is_logical, tail_bytes=pack.size - pos,
                             payload_start=order and pack.part_offset(order[0]) or None,
                             payload_start_expected=align_up(pack.table_end, 16))
    # storage group facts
    groups = []
    for si, s in enumerate(pack.storages):
        members = [i for i in range(len(pack.physicals)) if pack.physicals[i].storage_index == si]
        offs = [pack.part_offset(i) for i in members]
        contiguous_region = bool(members) and (max(o + pack.physicals[i].size for o, i in zip(offs, members)) - min(offs)
                                               <= sum(align_up(pack.physicals[i].size, 16) for i in members))
        groups.append({"index": si, "type": f"0x{s.type:02X}", "count": len(members),
                       "size_sum_aligned16": sum(align_up(pack.physicals[i].size, max(16, s.alignment)) for i in members),
                       "storage_size": s.size, "base_offset": s.base_offset,
                       "min_offset": min(offs) if offs else None,
                       "region_contiguous": contiguous_region,
                       "offset_units_monotonic": all(pack.physicals[a].offset_units <= pack.physicals[b].offset_units for a, b in zip(members, members[1:]))})
    out["groups"] = groups
    # storage order vs first appearance in logical part order
    first_seen = []
    seen = set()
    for r in pack:
        for i in r.part_indices:
            k = pack.physicals[i].storage_index
            if k not in seen:
                seen.add(k)
                first_seen.append(k)
    out["storage_order_is_first_appearance"] = first_seen == list(range(len(pack.storages)))
    out["storage_order_sorted_by_type"] = [s.type for s in pack.storages] == sorted(s.type for s in pack.storages)
    nbo = pack.name_blob_order()
    out["name_blob_order_is_logical"] = nbo == list(range(len(pack))) if nbo is not None else None

    # on-demand contiguity per resource (all types, not only meshes)
    viol_by_type: Counter = Counter()
    for r in pack:
        idxs = list(r.part_indices)
        base = pack.part_offset(idxs[0])
        cum = 0
        for i in idxs:
            cum = align_up(cum, pack.part_alignment(i))
            if pack.part_offset(i) != base + cum:
                viol_by_type[r.type] += 1
                break
            cum += pack.physicals[i].size
    out["logical_contiguity_violations_by_type"] = {f"0x{t:02X}": c for t, c in sorted(viol_by_type.items())}

    # table-level rebuild identity for the requested layouts (no payload written)
    rebuild = {}
    for layout in layouts:
        try:
            if layout == "preserve":
                w = PackWriter.from_pack(pack, layout)
            else:
                # from scratch: stock storage order rule, only the (unrecoverable) name-blob order is copied
                w = PackWriter(pack.header.field08, pack.header.flags, layout, name_blob_order=pack.name_blob_order())
            for r in pack:
                w.add(ResourceSpec.from_pack(pack, r.index))
            tables, plan, header, warnings = w.render()
            d = _tail_cmp(pack.table_bytes(), tables)
            offs = w.part_offsets(plan)
            orig = [pack.part_offset(i) for i in range(len(pack.physicals))]
            mism = sum(1 for a, b in zip(offs, orig) if a != b)
            rebuild[layout] = {"tables_identical": d is None, "first_table_diff": d, "part_offset_mismatches": mism,
                               "warnings": warnings}
        except Exception as exc:  # noqa: BLE001
            rebuild[layout] = {"error": f"{type(exc).__name__}: {exc}"}
    out["rebuild"] = rebuild
    return out
