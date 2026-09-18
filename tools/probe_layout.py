"""Probe: dump tables and the physical layout of one or more packs (used to establish the stock layout rules)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.util.binio import align_up  # noqa: E402


def probe(path: str, verbose: bool = False) -> None:
    with Pack.open(path) as pk:
        h = pk.header
        print(f"== {pk.path.name}  size={pk.size}  table_end={pk.table_end}")
        print(f"   field08={h.field08:#x} flags={h.flags:#x} S={h.storage_count} P={h.physical_count} L={h.logical_count} N={h.name_count} nb={h.name_bytes}")
        print(f"   types: {{{', '.join(f'0x{t:02X}:{c}' for t, c in pk.type_histogram().items())}}}")
        for i, s in enumerate(pk.storages):
            print(f"   storage[{i}] type=0x{s.type:02X} align_raw={s.align_raw} (A={s.alignment}) flags=0x{s.flags:02X} meta=0x{s.metadata:02X} "
                  f"method={s.method} ver={s.version} codec={s.codec} base_units={s.base_units} size={s.size} comp={s.compressed} count={s.count}")
        # census of physical flag bits / fc / logical flags
        fb: dict[int, int] = {}
        fc: dict[int, int] = {}
        lf: dict[int, int] = {}
        for p in pk.physicals:
            fb[p.flag_bits] = fb.get(p.flag_bits, 0) + 1
            fc[p.fc] = fc.get(p.fc, 0) + 1
        for l in pk.logicals:
            lf[l.flags] = lf.get(l.flags, 0) + 1
        print(f"   phys flag_bits: {{{', '.join(f'0x{k:04X}:{v}' for k, v in sorted(fb.items()))}}}  fc: {{{', '.join(f'0x{k:X}:{v}' for k, v in sorted(fc.items()))}}}  logical flags: {{{', '.join(f'0x{k:02X}:{v}' for k, v in sorted(lf.items()))}}}")
        # per-storage group size/count check
        gsize = [0] * len(pk.storages)
        gcount = [0] * len(pk.storages)
        for p in pk.physicals:
            gsize[p.storage_index] += p.size
            gcount[p.storage_index] += 1
        for i, s in enumerate(pk.storages):
            ok = (gsize[i] == s.size) and (gcount[i] == s.count)
            print(f"   group[{i}] sum(size)={gsize[i]} count={gcount[i]}  -> storage.size/count match: {ok}")
        # file-order walk
        order = sorted(range(len(pk.physicals)), key=lambda i: pk.part_offset(i))
        pos = align_up(pk.table_end, 16)
        gaps = 0
        nonzero_gap = 0
        overlaps = 0
        out_of_logical_order = 0
        prev_owner = -1
        prev_part = -1
        for i in order:
            off = pk.part_offset(i)
            p = pk.physicals[i]
            if off < pos:
                overlaps += 1
            elif off > pos:
                gaps += 1
                if any(pk._data[pos:off]):
                    nonzero_gap += 1
            if (p.owner, i) < (prev_owner, prev_part):
                out_of_logical_order += 1
            prev_owner, prev_part = p.owner, i
            pos = off + p.size
        tail = pk.size - pos
        print(f"   file order: gaps={gaps} nonzero_gaps={nonzero_gap} overlaps={overlaps} out_of_logical_order={out_of_logical_order} tail_bytes={tail} payload_start_16aligned={align_up(pk.table_end,16)}  first_part_off={pk.part_offset(order[0]) if order else None}")
        # owner check
        bad_owner = sum(1 for r in pk for i in r.part_indices if pk.physicals[i].owner != r.index)
        print(f"   owner mismatches: {bad_owner}")
        # contiguity per resource in logical order
        viol = 0
        for r in pk:
            idxs = list(r.part_indices)
            if not idxs:
                continue
            base = pk.part_offset(idxs[0])
            cum = 0
            for i in idxs:
                a = pk.part_alignment(i)
                cum = align_up(cum, a)
                if pk.part_offset(i) != base + cum:
                    viol += 1
                    break
                cum += pk.physicals[i].size
        print(f"   resources violating logical-order contiguity: {viol} / {len(pk)}")
        if verbose:
            for r in pk:
                print(f"     [{r.index}] 0x{r.type:02X} flags=0x{r.flags:02X} {r.name!r} parts={[f'0x{t:02X}' for t in r.part_types]} "
                      f"offs={[pk.part_offset(i) for i in r.part_indices]} sizes={[pk.physicals[i].size for i in r.part_indices]} "
                      f"fb={[f'0x{pk.physicals[i].flag_bits:X}' for i in r.part_indices]}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    verbose = "-v" in sys.argv
    for a in args:
        probe(a, verbose)
