"""`nr types ...` — inspection of the raw-bundle families.

    nr types dump   <rpack> <index|name> [--json-out FILE]      structural dump of one resource (JSON on stdout)
    nr types census <assets dir> --out <dir> [--family f,...] [--limit N]
                                                                run every family census, write <dir>/<family>.json
                                                                and <dir>/summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from ..container import catalogue
from ..container.rp6l import Pack
from ..errors import NightrunnerError
from ..util.jsonio import dump_json
from . import CENSUS_DATE, FAMILY_MODULES
from .codecs import structural_dump
from .raw import read_parts, sidecar


def _resolve(pk: Pack, key: str) -> int:
    if key.isdigit():
        idx = int(key)
        if idx >= len(pk):
            raise NightrunnerError(f"index {idx} out of range (0..{len(pk) - 1})")
        return idx
    hits = pk.find(key)
    if not hits:
        raise NightrunnerError(f"no resource named {key!r}")
    if len(hits) > 1:
        print(f"note: {len(hits)} resources named {key!r}; using logical index {hits[0]}", file=sys.stderr)
    return hits[0]


def cmd_dump(a) -> int:
    with Pack.open(a.pack) as pk:
        idx = _resolve(pk, a.key)
        res = pk.resource(idx)
        parts = read_parts(res)
        dump = structural_dump(res.type, parts)
        out = sidecar(pk, res, dump)
    text = json.dumps(out, indent=2, ensure_ascii=False)
    if a.json_out:
        Path(a.json_out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote {a.json_out} ({res.name!r}, {catalogue.type_name(res.type)})")
    else:
        print(text)
    return 0


def cmd_census(a) -> int:
    import importlib

    assets = Path(a.assets)
    if not assets.is_dir():
        raise NightrunnerError(f"assets dir not found: {assets}")
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = set(a.family.split(",")) if a.family else set(FAMILY_MODULES.values())
    summary = {"census_date": CENSUS_DATE, "assets": str(assets), "families": {}}
    progress = (lambda msg: print(f"  {msg}", file=sys.stderr, flush=True)) if not a.quiet else None
    for tid, fam in FAMILY_MODULES.items():
        if fam not in wanted:
            continue
        mod = importlib.import_module(f"nightrunner.types.{fam}")
        t0 = time.time()
        try:
            rep = mod.run_census(assets, limit=a.limit, progress=progress)
        except Exception as exc:  # noqa: BLE001
            rep = {"family": fam, "type": f"0x{tid:02X}", "error": f"{type(exc).__name__}: {exc}"}
            if a.strict:
                raise
        rep["seconds"] = round(time.time() - t0, 1)
        rep["census_date"] = CENSUS_DATE
        dump_json(rep, out_dir / f"{fam}.json")
        summary["families"][fam] = {k: v for k, v in rep.items() if k in ("type", "totals", "seconds", "error", "packs")}
        print(f"{fam:12s} {rep['seconds']:7.1f}s  {'ERROR ' + rep['error'] if 'error' in rep else ''}")
    dump_json(summary, out_dir / "summary.json")
    print(f"reports written to {out_dir}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr types", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump", help="structural dump of one resource")
    p.add_argument("pack")
    p.add_argument("key", help="logical index or resource name")
    p.add_argument("--json-out")
    p.set_defaults(fn=cmd_dump)
    p = sub.add_parser("census", help="run every family census over the assets dir")
    p.add_argument("assets")
    p.add_argument("--out", required=True)
    p.add_argument("--family", help="comma list of families (anim,animscr,animgraph,animcustom,envprobe,voxelizer,area,prefab)")
    p.add_argument("--limit", type=int, default=None, help="resources (or packs) per family — smoke tests")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--strict", action="store_true", help="re-raise census exceptions")
    p.set_defaults(fn=cmd_census)
    return ap


def run(args) -> int:
    a = build_parser().parse_args(args.args)
    return a.fn(a)
