"""Command-line front end.  `python nr.py <command> ...` or `python -m nightrunner <command> ...`"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import __version__
from .container import catalogue
from .container.rp6l import Pack
from .container.validate import validate
from .errors import NightrunnerError
from .util.hashing import sha256_bytes
from .util.jsonio import dump_json


def _jprint(obj) -> None:
    print(json.dumps(obj, ensure_ascii=False))


# ---- commands --------------------------------------------------------------------------------------------------

def cmd_info(a) -> int:
    with Pack.open(a.pack) as pk:
        h = pk.header
        out = {
            "path": str(pk.path), "size": pk.size, "table_end": pk.table_end,
            "header": h.to_json(), "ondemand": h.ondemand,
            "types": {f"0x{t:02X} {catalogue.type_name(t)}": c for t, c in pk.type_histogram().items()},
            "storages": [s.to_json() for s in pk.storages],
        }
        if a.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"{pk.path.name}: {pk.size:,} bytes, field08={h.field08:#x} flags={h.flags:#x} "
                  f"storages={h.storage_count} physicals={h.physical_count} logicals={h.logical_count} ondemand={h.ondemand}")
            for k, v in out["types"].items():
                print(f"  {k:32s} {v:>8}")
            for i, s in enumerate(pk.storages):
                print(f"  storage[{i:3d}] type=0x{s.type:02X} {catalogue.type_name(s.type):20s} A={s.alignment:<3d} "
                      f"method={s.method} ver={s.version:<3d} flags=0x{s.flags:02X} meta=0x{s.metadata:02X} "
                      f"base=0x{s.base_offset:X} size={s.size:,} count={s.count}")
    return 0


def _part_sha256(pk: Pack, i: int) -> str | None:
    """sha256 of one part's stored bytes (None for parts that are not directly readable: child/compressed)."""
    if not pk.part_is_direct(i):
        return None
    mv = pk.read_part(i)
    try:
        return sha256_bytes(mv)
    finally:
        mv.release()


def cmd_list(a) -> int:
    t = int(a.type, 0) if a.type else None
    q = a.query.lower() if a.query else None
    n = 0
    with Pack.open(a.pack) as pk:
        for r in pk:
            if t is not None and r.type != t:
                continue
            if q and q not in r.name.lower():
                continue
            if a.offset and n < a.offset:
                n += 1
                continue
            if a.parts:
                rec = r.to_json()
                if a.sha256:
                    for p in rec["parts"]:
                        p["sha256"] = _part_sha256(pk, p["index"])
            else:
                rec = {
                    "index": r.index, "type": f"0x{r.type:02X}", "type_name": catalogue.type_name(r.type),
                    "flags": f"0x{r.flags:02X}", "name": r.name, "parts": [f"0x{x:02X}" for x in r.part_types],
                    "size": sum(pk.physicals[i].size for i in r.part_indices),
                }
                if a.sha256:
                    rec["sha256"] = [_part_sha256(pk, i) for i in r.part_indices]
            _jprint(rec)
            n += 1
            if a.limit and n - (a.offset or 0) >= a.limit:
                break
    return 0


def cmd_validate(a) -> int:
    rc = 0
    for p in a.packs:
        with Pack.open(p) as pk:
            rep = validate(pk, check_mesh_names=not a.no_mesh_names)
        if a.json:
            print(json.dumps(rep.to_json(), indent=2))
        else:
            errs = rep.errors
            warns = [f for f in rep.findings if f.level == "warning"]
            infos = [f for f in rep.findings if f.level == "info"]
            print(f"{'OK ' if rep.ok else 'BAD'} {Path(p).name}: {len(errs)} errors, {len(warns)} warnings, {len(infos)} info  {rep.stats}")
            for f in errs[: a.show] + warns[: a.show]:
                print(f"    {f.level:7s} {f.check:10s} {f.message}")
        if not rep.ok:
            rc = 1
    return rc


def cmd_census(a) -> int:
    from .container.census import census
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for p in a.packs:
        t0 = time.time()
        with Pack.open(p) as pk:
            c = census(pk, layouts=tuple(a.layouts.split(",")))
        c["seconds"] = round(time.time() - t0, 2)
        dump_json(c, out_dir / (Path(p).stem + ".census.json"))
        summary.append({k: c[k] for k in ("file", "size", "types", "file_order", "rebuild", "storage_order_is_first_appearance",
                                            "storage_order_sorted_by_type", "name_blob_order_is_logical",
                                            "logical_contiguity_violations_by_type", "max_parts", "fc_values",
                                            "logical_flags", "physical_flag_bits", "seconds")})
        print(f"{Path(p).name:45s} {c['seconds']:6.1f}s rebuild={c['rebuild']}")
    dump_json(summary, out_dir / "summary.json")
    return 0


def cmd_extract(a) -> int:
    from .extract import extract
    types = {int(t, 0) for t in a.types.split(",")} if a.types else None
    opts = {}
    if a.sdb:
        opts["sdb"] = Path(a.sdb)
    if a.pak:
        opts["pak"] = Path(a.pak)
    with Pack.open(a.pack) as pk:
        spec = extract(pk, a.out, types=types, no_raw=a.no_raw, options=opts, limit=a.limit)
    print(f"extracted {len(spec['resources'])} resources to {a.out} ({len(spec['warnings'])} warnings)")
    for w in spec["warnings"][:20]:
        print("  warning:", w)
    return 0


def cmd_build(a) -> int:
    from .build import build
    # codec options travel through BuildContext.options; the mesh codec reads `ignore_bone_changes`
    # (same switch as `nr mesh import --ignore-bones`: keep the native skeleton when the Cast bones moved)
    options = {"ignore_bone_changes": True} if a.ignore_bones else {}
    res = build(a.spec, a.out, layout=a.layout, field08=int(a.field08, 0) if a.field08 else None,
                force_codec=a.force_codec, options=options, validate_output=not a.no_validate)
    print(json.dumps({k: v for k, v in res.items() if k != "validation"}, indent=2))
    v = res.get("validation")
    if v:
        print(f"validation: {'OK' if v['ok'] else 'FAILED'} ({len([f for f in v['findings'] if f['level']=='error'])} errors)")
    return 0


def cmd_roundtrip(a) -> int:
    from .roundtrip import roundtrip_pack
    types = {int(t, 0) for t in a.types.split(",")} if a.types else None
    rep = roundtrip_pack(a.pack, types=types, limit=a.limit, report_path=a.report)
    print(json.dumps(rep["summary"], indent=2))
    # the gate exists to catch `differs`; exit 1 on either (review 2026-09-15, F17)
    s = rep["summary"]
    return 0 if s.get("failures", 0) == 0 and s.get("differs", 0) == 0 else 1


def cmd_mesh(a) -> int:
    from .mesh import cli as mesh_cli
    return mesh_cli.run(a)


def cmd_texture(a) -> int:
    from .texture import cli as tex_cli
    return tex_cli.run(a)


def cmd_sdb(a) -> int:
    from .sdb import cli as sdb_cli
    return sdb_cli.run(a)


def cmd_model(a) -> int:
    from .pak import cli as pak_cli
    return pak_cli.run(a)


def cmd_audio(a) -> int:
    from .audio import cli as audio_cli
    return audio_cli.run(a)


def cmd_types(a) -> int:
    from .types import cli as types_cli
    return types_cli.run(a)


def cmd_project(a) -> int:
    from .project_cli import run
    return run(a.args)


def cmd_select(a) -> int:
    from .select import run as select_run
    return select_run(a)


# ---- parser ----------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    # help strings are ASCII only: a redirected stdout on Windows is cp1252 and argparse would crash on '->' / em dash
    ap = argparse.ArgumentParser(prog="nr", description=f"nightrunner {__version__} - offline Chrome Engine asset toolkit (DL2, DLTB)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="header, type histogram, storage table")
    p.add_argument("pack")
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("list", help="JSONL of logical resources")
    p.add_argument("pack")
    p.add_argument("--type", help="filter by logical type id, e.g. 0x10")
    p.add_argument("--query", help="case-insensitive substring of the name")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0)
    p.add_argument("--parts", action="store_true", help="include full part records")
    p.add_argument("--sha256", action="store_true",
                   help="sha256 of every part's stored bytes (per part record with --parts, else a list in part order)")
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("validate", help="engine-contract checks")
    p.add_argument("packs", nargs="+")
    p.add_argument("--json", action="store_true")
    p.add_argument("--show", type=int, default=10, help="findings to print per level")
    p.add_argument("--no-mesh-names", action="store_true")
    p.set_defaults(fn=cmd_validate)

    p = sub.add_parser("census", help="corpus statistics + table-level rebuild identity")
    p.add_argument("packs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--layouts", default="auto,preserve")
    p.set_defaults(fn=cmd_census)

    p = sub.add_parser("extract", help="pack -> .rpx tree")
    p.add_argument("pack")
    p.add_argument("out")
    p.add_argument("--types", help="comma list of logical type ids to extract (default all)")
    p.add_argument("--limit", type=int, default=None, help="stop after N resources (testing)")
    p.add_argument("--no-raw", action="store_true", help="skip raw copies for losslessly-encoded families (textures)")
    p.add_argument("--sdb", help="runtime_dx11.sdb / runtime_dx12.sdb for material/texture linking")
    p.add_argument("--pak", help="data0.pak for .model definitions")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("build", help=".rpx tree / pack.json -> RPACK")
    p.add_argument("spec")
    p.add_argument("out")
    p.add_argument("--layout", default="auto", choices=["auto", "preserve", "contiguous", "grouped"])
    p.add_argument("--field08", help="override header field08 (e.g. 0x1000)")
    p.add_argument("--force-codec", action="store_true", help="re-encode every editable resource even if unchanged")
    p.add_argument("--ignore-bones", action="store_true",
                   help="mesh codec: keep the native skeleton when an edited model.cast moved its bones (else refused)")
    p.add_argument("--no-validate", action="store_true")
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser("roundtrip", help="in-memory decode -> encode -> compare for every resource")
    p.add_argument("pack")
    p.add_argument("--types")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--report", help="write a JSON report here")
    p.set_defaults(fn=cmd_roundtrip)

    # sub-CLIs: each area parses `args` itself (`python nr.py mesh import --help` shows its options)
    p = sub.add_parser("mesh", help="mesh helpers: info | export | dump | census | import | diff")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_mesh)

    p = sub.add_parser("texture", help="texture helpers: info | export | ddsinfo | census")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_texture)

    p = sub.add_parser("sdb", help="SDB material database helpers: list | material | textures | stats")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_sdb)

    p = sub.add_parser("audio", help="Wwise audio (AESP containers, soundbanks, registry): info | list | extract "
                                     "| bank | sounds | registry | census")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_audio)

    p = sub.add_parser("model", help=".model definition helpers (PAK): list | show | meshes | write | split")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_model)

    p = sub.add_parser("types", help="non-mesh/non-texture families (anim/prefab/area/...): dump | census")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_types)

    p = sub.add_parser("project", help="mod projects (.nrproj): new | add | model | info | validate | build")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_project)

    p = sub.add_parser("select", help="compose a new pack spec from resources of one or more .rpx trees")
    p.add_argument("args", nargs=argparse.REMAINDER)
    p.set_defaults(fn=cmd_select)
    return ap


def _tolerant_stdio() -> None:
    """A redirected stdout/stderr on Windows defaults to cp1252; resource names and messages may carry characters
    outside it (U+FFFD for the one non-UTF-8 entity name, arrows in reports). Never crash on printing."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure") and (stream.encoding or "").lower() not in ("utf-8", "utf8"):
                stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdio()
    ap = build_parser()
    a = ap.parse_args(argv)
    try:
        return a.fn(a)
    except NightrunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
