"""`nr select` — compose a NEW pack spec (.rpx tree) from resources of one or more extracted trees.

    nr select <tree.rpx>[:<sel>[,<sel>...]] [<tree.rpx>[:<sel>...] ...]
              --out <new.rpx> [--field08 0x1000] [--flags 1] [--link]
              [--rename OLD=NEW ...] [--replace NAME=<tree.rpx>:<index|name> ...]

(nr.py hands `select` its raw argv with argparse.REMAINDER, which only starts collecting at the first positional,
so the trees come first; `nr select -- --out ...` also works.)

Selectors (per tree, comma separated): a decimal logical index, an index range `a-b`, a resource name (exact,
then engine case folding), `type=0xTT`, or `*`. A tree without selector contributes every resource.

The result is an ordinary `nightrunner.rpx/1` spec that `nr build` consumes unchanged:
  * resources renumbered 0..n-1 in selection order (logical order = engine lookup order), `name_index` = index,
    `dir`/`raw` paths rewritten under `<new.rpx>/<family>/<index>_<name>/`, files copied (or hard-linked with
    `--link`) from the source trees;
  * `storages` = the union of the storage records the chosen parts need, deduplicated by
    (type, align_raw, flags, metadata) and listed in stock order (`stock_storage_order`); every part's
    `storage_index` is remapped; base_units/size/count are recomputed by the builder;
  * `complete: false`, no `name_blob_order` (the builder writes names in logical order), header counts recomputed;
  * `source` = list of the contributing trees.

`--rename OLD=NEW` changes the logical name of one selected resource; `--replace NAME=<tree>:<sel>` keeps the
selected resource's name but takes the parts/files of another resource (same type) — the same-name override use
case. A mesh (0x10) whose logical name no longer matches its source is flagged `"needs_identity_fix": true`; `nr build`
rewrites the embedded `.msh` name (mesh.identity.rename) for such entries (not done here: the container layer never
touches payload bytes). Without the fix `nr validate` would report the mismatch as `mesh_name`.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

from . import __version__
from .build import load_spec
from .container import catalogue
from .container.rp6l import Storage, stock_storage_order, STORAGE_FLAG_STREAM, PHYS_SPECIAL
from .errors import BuildError
from .util.binio import align_up
from .util.jsonio import dump_json
from .util.names import engine_fold, resource_dirname

SCHEMA = "nightrunner.rpx/1"


# ---- argument parsing --------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr select", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="new spec directory (<name>.rpx); replaced if it exists")
    ap.add_argument("--field08", default=None, help="header field08 (e.g. 0x1000); default: the sources' common value")
    ap.add_argument("--flags", default=None, help="header flags; default: the sources' common value (1)")
    ap.add_argument("--link", action="store_true", help="hard-link files instead of copying (same volume only)")
    ap.add_argument("--rename", action="append", default=[], metavar="OLD=NEW", help="rename a selected resource")
    ap.add_argument("--replace", action="append", default=[], metavar="NAME=TREE:SEL",
                    help="keep NAME but take parts/files from resource SEL of TREE (same type)")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("trees", nargs="+", metavar="TREE[:SEL,...]", help="extracted .rpx trees with optional selectors")
    return ap


def split_tree_arg(arg: str) -> tuple[Path, str | None]:
    """'<dir>' or '<dir>:<selectors>'. The directory must exist (drive letters contain ':' too)."""
    p = Path(arg)
    if p.is_dir():
        return p, None
    if ":" in arg:
        head, sel = arg.rsplit(":", 1)
        hp = Path(head)
        if hp.is_dir():
            return hp, sel
    raise BuildError(f"select: tree directory not found: {arg!r}")


class Tree:
    def __init__(self, path: Path):
        self.path = path
        self.spec, self.dir = load_spec(path)
        self.resources: list[dict] = self.spec["resources"]
        self.storages = [Storage.from_json(s) for s in self.spec.get("storages", [])]
        self.field08 = int(self.spec["header"]["field08"], 0)
        self.flags = int(self.spec["header"]["flags"], 0)

    def storage_key(self, index: int) -> tuple[int, int, int, int]:
        if index >= len(self.storages):
            raise BuildError(f"select: {self.path}: part references storage {index} of {len(self.storages)}")
        return self.storages[index].key

    def by_name(self, name: str) -> list[dict]:
        exact = [r for r in self.resources if r["name"] == name]
        if exact:
            return exact
        key = engine_fold(name.encode("utf-8", "surrogateescape"))
        return [r for r in self.resources if engine_fold(bytes.fromhex(r["name_hex"])) == key]

    def by_index(self, index: int) -> dict:
        for r in self.resources:
            if r["index"] == index:
                return r
        raise BuildError(f"select: {self.path}: no resource with index {index}")

    def select(self, sel: str | None) -> list[dict]:
        if sel is None or sel.strip() in ("", "*"):
            return list(self.resources)
        out: list[dict] = []
        seen: set[int] = set()

        def take(r: dict) -> None:
            if r["index"] not in seen:
                seen.add(r["index"])
                out.append(r)

        for tok in sel.split(","):
            tok = tok.strip()
            if not tok:
                continue
            if tok == "*":
                for r in self.resources:
                    take(r)
            elif tok.lower().startswith(("type=", "t=")):
                t = int(tok.split("=", 1)[1], 0)
                hits = [r for r in self.resources if int(r["type"], 0) == t]
                if not hits:
                    raise BuildError(f"select: {self.path}: no resource of type 0x{t:02X}")
                for r in hits:
                    take(r)
            elif tok.isdigit():
                take(self.by_index(int(tok)))
            elif "-" in tok and all(x.isdigit() for x in tok.split("-", 1)):
                a, b = (int(x) for x in tok.split("-", 1))
                for i in range(a, b + 1):
                    take(self.by_index(i))
            else:
                hits = self.by_name(tok)
                if not hits:
                    raise BuildError(f"select: {self.path}: no resource named {tok!r}")
                for r in hits:
                    take(r)
        return out


class Pick:
    """One selected resource: its source tree + entry, and the (possibly renamed/replaced) outcome."""

    def __init__(self, tree: Tree, entry: dict):
        self.tree = tree
        self.entry = entry                          # entry whose parts/files/dir are used
        self.name_hex = entry["name_hex"]           # logical name to write
        self.origin = (tree, entry)                 # where the name came from
        self.needs_identity_fix = False

    @property
    def name(self) -> str:
        return bytes.fromhex(self.name_hex).decode("utf-8", "surrogateescape")

    @property
    def type(self) -> int:
        return int(self.entry["type"], 0)


def _find_pick(picks: list[Pick], name: str, what: str) -> Pick:
    hits = [p for p in picks if p.name == name]
    if not hits:
        key = engine_fold(name.encode("utf-8", "surrogateescape"))
        hits = [p for p in picks if engine_fold(bytes.fromhex(p.name_hex)) == key]
    if not hits:
        raise BuildError(f"select: {what}: no selected resource named {name!r}")
    if len(hits) > 1:
        raise BuildError(f"select: {what}: {name!r} matches {len(hits)} selected resources "
                         f"(indices {[picks.index(h) for h in hits]}); rename/replace needs a unique name")
    return hits[0]


def _resolve_one(trees: dict[Path, Tree], spec: str, what: str) -> tuple[Tree, dict]:
    if ":" not in spec:
        raise BuildError(f"select: {what}: expected <tree.rpx>:<index|name>, got {spec!r}")
    tdir, sel = split_tree_arg(spec)
    tree = trees.get(tdir.resolve())
    if tree is None:
        tree = trees[tdir.resolve()] = Tree(tdir)
    hits = tree.select(sel)
    if len(hits) != 1:
        raise BuildError(f"select: {what}: {spec!r} selects {len(hits)} resources (need exactly 1)")
    return tree, hits[0]


# ---- composition -------------------------------------------------------------------------------------------------

def compose(tree_args: list[str], *, field08: int | None = None, flags: int | None = None,
            renames: list[str] = (), replaces: list[str] = ()) -> tuple[list[Pick], dict, list[str]]:
    """Resolve selections/renames/replacements. Returns (picks, header-ish info, warnings)."""
    trees: dict[Path, Tree] = {}
    picks: list[Pick] = []
    for arg in tree_args:
        tdir, sel = split_tree_arg(arg)
        key = tdir.resolve()
        tree = trees.get(key)
        if tree is None:
            tree = trees[key] = Tree(tdir)
        for entry in tree.select(sel):
            picks.append(Pick(tree, entry))
    if not picks:
        raise BuildError("select: nothing selected")

    for spec in replaces:
        if "=" not in spec:
            raise BuildError(f"select: --replace expects NAME=<tree.rpx>:<index|name>, got {spec!r}")
        name, src = spec.split("=", 1)
        pick = _find_pick(picks, name, "--replace")
        tree, entry = _resolve_one(trees, src, "--replace")
        if int(entry["type"], 0) != pick.type:
            raise BuildError(f"select: --replace {name!r}: type {entry['type']} != {pick.entry['type']}")
        pick.tree, pick.entry = tree, entry
        if pick.type == 0x10 and entry["name_hex"] != pick.name_hex:
            pick.needs_identity_fix = True

    for spec in renames:
        if "=" not in spec:
            raise BuildError(f"select: --rename expects OLD=NEW, got {spec!r}")
        old, new = spec.split("=", 1)
        if not new:
            raise BuildError("select: --rename: new name is empty")
        if "\0" in new:
            raise BuildError("select: --rename: new name contains NUL")
        pick = _find_pick(picks, old, "--rename")
        pick.name_hex = new.encode("utf-8", "surrogateescape").hex()
        if pick.type == 0x10 and pick.name_hex != pick.entry["name_hex"]:
            pick.needs_identity_fix = True

    warnings: list[str] = []
    f08s = {t.field08 for t in trees.values()}
    fls = {t.flags for t in trees.values()}
    if field08 is None:
        if len(f08s) != 1:
            raise BuildError(f"select: source trees disagree on field08 ({[hex(x) for x in sorted(f08s)]}); pass --field08")
        field08 = f08s.pop()
    if flags is None:
        flags = fls.pop() if len(fls) == 1 else 1
        if len(fls) > 1:
            warnings.append(f"source trees disagree on header flags {sorted(fls)}; using 1")
    info = {"field08": field08, "flags": flags, "trees": list(trees.values())}
    return picks, info, warnings


def _storage_union(picks: list[Pick]) -> tuple[list[tuple[int, int, int, int]], dict]:
    used: dict[tuple[int, int, int, int], int] = {}
    for p in picks:
        for prec in p.entry["parts"]:
            used.setdefault(p.tree.storage_key(prec["storage_index"]), 0)
    keys = stock_storage_order(list(used))
    return keys, {k: i for i, k in enumerate(keys)}


def _copy_tree(src: Path, dst: Path, link: bool) -> int:
    n = 0
    for root, _dirs, files in os.walk(src):
        rel = Path(root).relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            s = Path(root) / f
            d = dst / rel / f
            if link:
                try:
                    os.link(s, d)
                except OSError as exc:
                    raise BuildError(f"select: --link failed for {s} ({exc}); rerun without --link") from exc
            else:
                shutil.copyfile(s, d)
            n += 1
    return n


def write_selection(picks: list[Pick], info: dict, out_dir: Path, *, link: bool = False, warnings: list[str] | None = None) -> dict:
    out_dir = Path(out_dir)
    warnings = list(warnings or [])
    keys, remap = _storage_union(picks)
    field08, flags = info["field08"], info["flags"]
    ondemand = bool(field08 & 0x1000)

    partial = out_dir.with_name(out_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)

    resources = []
    group_size = [0] * len(keys)
    group_count = [0] * len(keys)
    seen_names: dict[tuple[int, bytes], int] = {}
    phys = 0
    files = 0
    for new_index, p in enumerate(picks):
        e = p.entry
        t = p.type
        family = catalogue.family_dir(t)
        rel_dir = f"{family}/{resource_dirname(new_index, p.name)}"
        src_dir = p.tree.dir / e["dir"]
        if not src_dir.is_dir():
            raise BuildError(f"select: {p.tree.path}: resource directory missing: {src_dir}")
        files += _copy_tree(src_dir, partial / rel_dir, link)

        fkey = (t, engine_fold(bytes.fromhex(p.name_hex)))
        if fkey in seen_names:
            warnings.append(f"duplicate (type 0x{t:02X}, name {p.name!r}) at indices {seen_names[fkey]} and {new_index}: "
                            "the engine's name lookup returns the lower index")
        else:
            seen_names[fkey] = new_index

        parts = []
        stream_flags = []
        for k, prec in enumerate(e["parts"]):
            key = p.tree.storage_key(prec["storage_index"])
            g = remap[key]
            st = p.tree.storages[prec["storage_index"]]
            rec = {
                "ordinal": k, "index": phys, "type": prec["type"], "type_name": prec.get("type_name", catalogue.type_name(int(prec["type"], 0))),
                "storage_index": g, "flag_bits": prec["flag_bits"], "fc": prec["fc"], "size": prec["size"],
            }
            if prec.get("sha256"):
                rec["sha256"] = prec["sha256"]
            raw = prec.get("raw")
            if raw:
                rec["raw"] = f"{rel_dir}/{Path(raw).name}"
                if not (partial / rec["raw"]).exists():
                    raise BuildError(f"select: raw part file missing in source tree: {p.tree.dir / raw}")
            else:
                rec["raw"] = None
                warnings.append(f"{p.name!r}: part {k} ({prec['type']}) has no raw file "
                                "(extracted with --no-raw?); `nr build --force-codec` must regenerate it")
            group_size[g] += align_up(int(prec["size"]), max(16, st.alignment))
            group_count[g] += 1
            phys += 1
            parts.append(rec)
            stream_flags.append(bool(st.flags & STORAGE_FLAG_STREAM))
            if ondemand and t == 0x10 and (st.method != 1 or int(prec["flag_bits"], 0) & PHYS_SPECIAL):
                warnings.append(f"{p.name!r}: mesh part {k} is not on-demand compatible (method {st.method}, "
                                f"flag_bits {prec['flag_bits']}) — a field08 bit-12 pack needs method 1 without 0x1000")
        if ondemand and any(stream_flags) and not all(stream_flags):
            warnings.append(f"{p.name!r}: mixes stream (flags bit 3) and non-stream storages; the contiguous layout refuses it")

        entry = {
            "index": new_index, "name": p.name, "name_hex": p.name_hex, "type": e["type"],
            "type_name": e.get("type_name", catalogue.type_name(t)), "flags": e["flags"], "name_index": new_index,
            "dir": rel_dir, "parts": parts, "editable": e.get("editable", {"kind": "raw", "files": {}}),
            "source": {"tree": str(p.tree.dir), "index": e["index"], "name": e["name"], "dir": e["dir"]},
        }
        if p.needs_identity_fix:
            entry["needs_identity_fix"] = True
        resources.append(entry)

    storages = [Storage(*k, 0, group_size[i], 0, group_count[i] & 0xFFFF).to_json() for i, k in enumerate(keys)]
    names = [bytes.fromhex(r["name_hex"]) for r in resources]
    header = {
        "magic": "0x4C365052", "version": 4, "field08": f"0x{field08:08X}",
        "physical_count": phys, "storage_count": len(keys), "name_count": len(resources),
        "name_bytes": sum(len(n) + 1 for n in names), "logical_count": len(resources), "flags": f"0x{flags:08X}",
    }
    spec = {
        "schema": SCHEMA,
        "tool": f"nightrunner {__version__} select",
        "source": [{"tree": str(t.dir), "pack": t.spec.get("source"), "resources": sum(1 for p in picks if p.tree is t)}
                   for t in info["trees"]],
        "header": header,
        "storages": storages,
        "complete": False,
        "resources": resources,
        "warnings": warnings,
        "note": "composed by `nr select`: storage base_units/size/count and header counts are recomputed by `nr build`; "
                "resources with needs_identity_fix need the mesh codec to rewrite the embedded .msh name",
    }
    dump_json(spec, partial / "pack.json")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    os.replace(partial, out_dir)
    spec["_files"] = files
    return spec


def run(args) -> int:
    ap = build_parser()
    argv = list(args.args)
    if argv and argv[0] == "--":
        argv = argv[1:]
    a = ap.parse_args(argv)
    field08 = int(a.field08, 0) if a.field08 is not None else None
    flags = int(a.flags, 0) if a.flags is not None else None
    picks, info, warnings = compose(a.trees, field08=field08, flags=flags, renames=a.rename, replaces=a.replace)
    spec = write_selection(picks, info, Path(a.out), link=a.link, warnings=warnings)
    if not a.quiet:
        h = spec["header"]
        print(f"selected {h['logical_count']} resources / {h['physical_count']} parts / {h['storage_count']} storages "
              f"from {len(spec['source'])} tree(s) into {a.out} (field08={h['field08']}, {spec['_files']} files "
              f"{'linked' if a.link else 'copied'})")
        for w in spec["warnings"]:
            print("  warning:", w, file=sys.stderr)
    return 0
