"""`nr model ...` — .model definitions inside PAK archives.

    nr model list   <pak> [--query S] [--all] [--limit N]   .model members (JSONL); --all lists every member
    nr model show   <pak> <member|index>                    the JSON document
    nr model meshes <pak> <member|index> [--sdb F]          mesh refs + resolved materials (+ textures when --sdb)
    nr model write  <out.pak> <member=path.json> ... [--text member=file] [--overwrite]
                                                            build a models.pak override from JSON files
    nr model split  <edited.cast> <export dir | model.cast.json> --out DIR [--rpx TREE] [--tol T] [--no-check]
                                                            single-file model Cast (explorer export, e.g. edited in
                                                            Blender) → per-mesh model.cast + mesh.json + raw parts
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..errors import NightrunnerError
from .model_json import PakIndex, mesh_refs, write_models_pak


def _jprint(obj, indent=None) -> None:
    print(json.dumps(obj, indent=indent, ensure_ascii=False))


def cmd_split(a) -> int:
    from ..cast.split import split_model_cast
    src = Path(a.report)
    if src.is_dir():
        found = sorted(src.glob("*.cast.json"))
        if len(found) != 1:
            raise NightrunnerError(f"{src}: expected exactly one *.cast.json export report, found {len(found)}")
        src = found[0]
    res = split_model_cast(a.edited, src, a.out, tol=a.tol, uv_tol=a.tol, check=not a.no_check,
                           rpx=Path(a.rpx) if a.rpx else None)
    for m in res["meshes"]:
        subs = m["submeshes"]
        _jprint({"mesh": m["mesh"], "dir": m["dir"], "check": {k: v for k, v in (m["check"] or {}).items()
                                                                if k != "warnings"},
                 "moved": sum(s.get("moved", 0) for s in subs), "new": sum(s.get("new", 0) for s in subs),
                 "rpx": m.get("rpx"), "problems": m["problems"]})
    for p in res["problems"]:
        print(f"warning: {p}", file=sys.stderr)
    bad = any(m["problems"] for m in res["meshes"])
    return 1 if bad else 0


def cmd_list(a) -> int:
    with PakIndex.open(a.pak) as idx:
        ms = idx.list_members(a.query) if a.all else idx.list_models(a.query)
        for n, m in enumerate(ms):
            if a.limit and n >= a.limit:
                break
            _jprint(m.to_json())
    return 0


def cmd_show(a) -> int:
    with PakIndex.open(a.pak) as idx:
        m = idx.find(a.member)
        doc = idx.load_model(m) if m.name.casefold().endswith((".model", ".models")) else idx.load_json(m)
        _jprint(doc, indent=2)
    return 0


def cmd_meshes(a) -> int:
    with PakIndex.open(a.pak) as idx:
        m = idx.find(a.member)
        doc = idx.load_model(m)
        refs = mesh_refs(doc)
    out = {"member": m.name, "skeleton": (doc.get("preset") or {}).get("skeletonName"), "slots": len(doc.get("slots") or []),
           "meshes": refs}
    if a.sdb:
        from ..sdb.reader import Sdb, SdbError
        with Sdb.open(a.sdb) as db:
            for r in refs:
                for mat in r["materials"]:
                    base = (mat.get("resolved") or {}).get("base") or mat.get("embedded_name")
                    try:
                        mat["textures"] = sorted(db.textures_for_material(base))
                        ov = (mat.get("resolved") or {}).get("overrides") or {}
                        mat["texture_overrides"] = {k: v["value"] for k, v in ov.items() if v.get("type") == 7}
                    except SdbError as exc:
                        mat["textures_error"] = str(exc)
    _jprint(out, indent=2)
    return 0


def cmd_write(a) -> int:
    docs = {}
    for spec in a.entries:
        if "=" not in spec:
            raise NightrunnerError(f"expected member=path.json, got {spec!r}")
        member, path = spec.split("=", 1)
        docs[member] = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    texts = {}
    for spec in a.text or []:
        member, path = spec.split("=", 1)
        texts[member] = Path(path).read_text(encoding="utf-8")
    rep = write_models_pak(a.out, docs, texts=texts, overwrite=a.overwrite)
    _jprint(rep, indent=2)
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr model", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("list"); p.add_argument("pak"); p.add_argument("--query"); p.add_argument("--all", action="store_true"); p.add_argument("--limit", type=int, default=0); p.set_defaults(fn=cmd_list)
    p = sub.add_parser("show"); p.add_argument("pak"); p.add_argument("member"); p.set_defaults(fn=cmd_show)
    p = sub.add_parser("meshes"); p.add_argument("pak"); p.add_argument("member"); p.add_argument("--sdb"); p.set_defaults(fn=cmd_meshes)
    p = sub.add_parser("write"); p.add_argument("out"); p.add_argument("entries", nargs="+"); p.add_argument("--text", action="append"); p.add_argument("--overwrite", action="store_true"); p.set_defaults(fn=cmd_write)
    p = sub.add_parser("split", help="single-file model Cast -> per-mesh model.cast folders")
    p.add_argument("edited")
    p.add_argument("report", help="the export folder (containing <model>.cast.json) or the .cast.json itself")
    p.add_argument("--out", required=True)
    p.add_argument("--tol", type=float, default=1e-4, help="position / UV snap tolerance (default 1e-4)")
    p.add_argument("--no-check", action="store_true", help="skip the mesh-encoder check of each result")
    p.add_argument("--rpx", help="also copy each model.cast into the matching mesh folder of this extracted tree "
                                 "(the original is kept as model.cast.orig); then `nr build` the tree")
    p.set_defaults(fn=cmd_split)
    return ap


def run(args) -> int:
    a = build_parser().parse_args(args.args)
    return a.fn(a)
