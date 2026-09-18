"""In-memory decode → encode → compare harness.

For every resource in a pack, run its codec's `roundtrip(res)` (if it has one) and compare the re-encoded
part bytes against the stored ones. Nothing is written except the optional JSON report. This is the primary
correctness gate for the mesh and texture codecs across the full corpus."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from .codecs import codec_for
from .container.rp6l import Pack
from .util.jsonio import dump_json


def _first_diff(a, b) -> int | None:
    a = bytes(a)
    b = bytes(b)
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def roundtrip_pack(path: Path | str, *, types: set[int] | None = None, limit: int | None = None,
                   report_path: Path | str | None = None, progress: bool = True) -> dict:
    results = []
    counts = {"checked": 0, "identical": 0, "differs": 0, "failures": 0, "skipped": 0}
    t0 = time.time()
    with Pack.open(path) as pk:
        selected = [r for r in pk if types is None or r.type in types]
        if limit is not None:
            selected = selected[:limit]
        for n, r in enumerate(selected):
            codec = codec_for(r.type)
            rt = getattr(codec, "roundtrip", None)
            if rt is None:
                counts["skipped"] += 1
                continue
            counts["checked"] += 1
            try:
                produced = rt(r)  # dict: part ordinal -> bytes
            except Exception as exc:  # noqa: BLE001
                counts["failures"] += 1
                results.append({"index": r.index, "name": r.name, "type": f"0x{r.type:02X}", "status": "failure",
                                "error": f"{type(exc).__name__}: {exc}"})
                continue
            diffs = []
            for k, i in enumerate(r.part_indices):
                if k not in produced:
                    continue
                d = _first_diff(pk.read_part(i), produced[k])
                if d is not None:
                    diffs.append({"part": k, "type": f"0x{pk.part_type(i):02X}", "first_diff": d,
                                  "orig_size": pk.physicals[i].size, "new_size": len(produced[k])})
            if diffs:
                counts["differs"] += 1
                results.append({"index": r.index, "name": r.name, "type": f"0x{r.type:02X}", "status": "differs", "diffs": diffs})
            else:
                counts["identical"] += 1
            if progress and (n % 100 == 0 or n == len(selected) - 1):
                print(f"\r  roundtrip {n + 1}/{len(selected)}  identical={counts['identical']} differs={counts['differs']} failures={counts['failures']}",
                      end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    rep = {"pack": str(path), "summary": dict(counts, seconds=round(time.time() - t0, 1)), "results": results}
    if report_path:
        dump_json(rep, report_path)
    return rep
