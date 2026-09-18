"""`nr sdb ...` — read-only SDB material database inspection.

    nr sdb list     <sdb> [--query S] [--limit N]         material names (JSONL: index, name)
    nr sdb material <sdb> <name|index> [--raw]           resolved material as JSON (routes, typed parameters, variants,
                                                          texture bindings); --raw adds hex of the referenced records
    nr sdb textures <sdb> <name|index>                   texture names the material binds (one per line)
    nr sdb stats    <sdb> [--validate] [--json-out F]    block sizes, table counts, program blobs; --validate resolves
                                                          every material and reports the totals
    nr sdb preset   <sdb> <name|index>                   one 0xAA preset record (parameter declarations)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..errors import NightrunnerError
from .reader import Sdb


def _jprint(obj, indent=None) -> None:
    print(json.dumps(obj, indent=indent, ensure_ascii=False))


def _key(s: str):
    return int(s) if s.isdigit() else s


def cmd_list(a) -> int:
    q = a.query.casefold() if a.query else None
    n = 0
    with Sdb.open(a.sdb) as db:
        for i, name in enumerate(db.materials()):
            if q and q not in name.casefold():
                continue
            _jprint({"index": i, "name": name})
            n += 1
            if a.limit and n >= a.limit:
                break
    return 0


def cmd_material(a) -> int:
    with Sdb.open(a.sdb) as db:
        m = db.material(_key(a.name))
        if a.raw:
            refs = {"0xB2": [m["index"]], "0xCA": [], "0xC2": [], "0xB6": [], "0xCE": [], "0xD2": [], "0xAA": [], "0x9A": [], "0xA2": []}
            for r in m["routes"]:
                refs["0xCA"].append(r["index"]); refs["0xC2"].append(r["program"]); refs["0xB6"].append(r["tokens_index"])
                refs["0xCE"].append(r["slots"]); refs["0xD2"].append(r["values"]); refs["0xAA"].extend(r["preset_indices"])
                for v in r["variants"]:
                    refs["0x9A"].append(v["shader"])
                    if v["texture_array"]:
                        refs["0xA2"].append(v["texture_array"])
            m["raw_records"] = [{"table": t, "index": i, "hex": db.record_hex(int(t, 16), i)}
                                for t, idxs in refs.items() for i in sorted(set(idxs))]
        _jprint(m, indent=2)
    return 0


def cmd_textures(a) -> int:
    with Sdb.open(a.sdb) as db:
        for t in sorted(db.textures_for_material(_key(a.name))):
            print(t)
    return 0


def cmd_preset(a) -> int:
    with Sdb.open(a.sdb) as db:
        if a.name.isdigit():
            idx = int(a.name)
        else:
            ids = db.presets_named(a.name)
            if len(ids) != 1:
                raise NightrunnerError(f"{a.name!r}: {len(ids)} preset matches")
            idx = ids[0]
        _jprint(db.preset(idx), indent=2)
    return 0


def cmd_stats(a) -> int:
    with Sdb.open(a.sdb) as db:
        st = db.stats()
        if a.validate:
            st["validation"] = db.validate_all(progress=(lambda m: print(f"\r  {m}", end="", file=sys.stderr, flush=True)))
            print(file=sys.stderr)
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(st, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {a.json_out}")
    else:
        print(f"{Path(a.sdb).name}: {st['size']:,} bytes  A={st['block_a']:,}  B={st['block_b']:,}  materials={st['materials']:,} "
              f"routes={st['routes']:,} presets={st['presets']:,}  compact tags={st['compact_tags']}")
        for t in st["tables"]:
            print(f"  {t['key']}  {t['shape']:12s} w={t['width']:<3d} count={t['count']:>8,}  bytes={t['bytes']:>11,}  {t['meaning']}")
        for k, v in st["programs"].items():
            print(f"  programs kind {k}: {v['count']:,} blobs, {v['bytes']:,} bytes")
        if a.validate:
            print("  validation:", json.dumps(st["validation"]["totals"]))
            for e in st["validation"]["errors"]:
                print("   error:", e)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr sdb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("sdb"); p.add_argument("--query"); p.add_argument("--limit", type=int, default=0); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("material"); p.add_argument("sdb"); p.add_argument("name"); p.add_argument("--raw", action="store_true"); p.set_defaults(fn=cmd_material)
    p = sub.add_parser("textures"); p.add_argument("sdb"); p.add_argument("name"); p.set_defaults(fn=cmd_textures)
    p = sub.add_parser("preset"); p.add_argument("sdb"); p.add_argument("name"); p.set_defaults(fn=cmd_preset)
    p = sub.add_parser("stats"); p.add_argument("sdb"); p.add_argument("--validate", action="store_true"); p.add_argument("--json-out"); p.set_defaults(fn=cmd_stats)
    return ap


def run(args) -> int:
    a = build_parser().parse_args(args.args)
    return a.fn(a)
