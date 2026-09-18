"""`nr project` — build projects from the command line (the GUI Build tab edits the same .nrproj files).

    nr project new <file.nrproj> [--out DIR] [--rpack NAME] [--pak NAME]
    nr project add <file.nrproj> <file|folder> ... [--target NAME] [--pack LABEL] [--as NEW_NAME]
    nr project model <file.nrproj> --appearance [CHAR/]NAME | --member ROLE=MEMBER ... [--label L] [--path root|original|both]
    nr project info <file.nrproj>
    nr project validate <file.nrproj>
    nr project build <file.nrproj> [--out DIR] [--keep-work]

Every command takes --game dltb|dl2 and --root DIR (before or after the command name).
The game is --game (profile id), else the project's own game (DLTB for old files). Its install comes from --root,
NIGHTRUNNER_GAME_ROOT (when it holds that game), or the Steam libraries. `--game <folder>` still works as --root.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import games as G
from .project import (GameEnv, Project, ProjectError, add_files, build_project, load_override, output_names,
                      parse_appearances, validate)


def _split_game(game: str | None, root: str | None) -> tuple[str | None, str | None]:
    """--game takes a profile id; anything else is taken as a root folder (pre-profile usage)."""
    if game and game.lower() not in G.PROFILES:
        return None, root or game
    return (game.lower() if game else None), root


def _env(game: str | None, root: str | None = None) -> GameEnv:
    """GameEnv for profile *game* (None = detect from the root / first found)."""
    want = G.profile(game) if game else None
    if root:
        r = Path(root)
        if r.joinpath(*G.ASSETS_SUB).is_dir() and r.name.lower() in {d for p in G.PROFILES.values() for d in p.data_dirs}:
            r = r.parent                              # the data folder itself was given
        gi = G.GameInstall(r, want) if want else G.GameInstall(r)
        if gi.valid():
            return GameEnv.from_game(gi)
        raise ProjectError(f"no game data under {root}")
    g = G.find_game(None, want)
    if g is None:
        what = want.name if want else "game"
        raise ProjectError(f"{what} install not found (pass --root or set NIGHTRUNNER_GAME_ROOT)")
    return GameEnv.from_game(g)


def run(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="nr project", description=__doc__.split("\n\n")[0])
    ap.add_argument("--game", help="game: " + " | ".join(G.PROFILES) + " (default: the project's)")
    ap.add_argument("--root", help="game root folder (or its ph / ph_ft folder)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("new")
    p.add_argument("file")
    p.add_argument("--out", default="")
    p.add_argument("--rpack", default="")
    p.add_argument("--pak", default="")
    p = sub.add_parser("add")
    p.add_argument("file")
    p.add_argument("paths", nargs="+")
    p.add_argument("--target")
    p.add_argument("--pack", default="")
    p.add_argument("--as", dest="new_name")
    p.add_argument("--format", choices=["auto", "rgba8", "r8", "normal"])
    p = sub.add_parser("model")
    p.add_argument("file")
    p.add_argument("--appearance", help="[Character/]Appearance from scripts/playerappearances.scr")
    p.add_argument("--member", action="append", default=[], metavar="ROLE=MEMBER")
    p.add_argument("--label")
    p.add_argument("--path", default="root", choices=["root", "original", "both"])
    p.add_argument("--pak", default="")
    for n in ("info", "validate"):
        p = sub.add_parser(n)
        p.add_argument("file")
    p = sub.add_parser("build")
    p.add_argument("file")
    p.add_argument("--out")
    p.add_argument("--keep-work", action="store_true")
    for sp in sub.choices.values():
        sp.add_argument("--game", dest="sub_game", default=None, help=argparse.SUPPRESS)
        sp.add_argument("--root", dest="sub_root", default=None, help=argparse.SUPPRESS)
    a = ap.parse_args(argv[1:] if argv and argv[0] == "--" else argv)
    game, root = _split_game(a.sub_game or a.game, a.sub_root or a.root)

    if a.cmd == "new":
        if game is None and root:
            game = (G.detect_profile(root) or G.profile(None)).id
        pr = Project(name=Path(a.file).stem, output_dir=a.out, rpack_name=a.rpack, pak_name=a.pak,
                     game=game or G.DEFAULT_ID)
        print(pr.save(a.file))
        return 0
    pr = Project.load(a.file)
    env = _env(game or (None if root else pr.game), root)
    if env.profile.id != pr.game:
        print(f"warning: project is for {pr.game}, building against {env.profile.id}", file=sys.stderr)
    if a.cmd == "add":
        new = add_files(pr, [Path(x) for x in a.paths], env)
        for it in new:
            if a.target:
                it.target_name = a.target
            if a.pack:
                it.target_pack = a.pack
            if a.new_name:
                it.new_name = a.new_name
            if a.format:
                it.options["format"] = a.format
            print(f"{it.id} {it.kind:<8} {Path(it.source).name} -> {it.output_name} "
                  f"(target {it.target_name} in {it.target_pack or '?'})")
        pr.save()
        return 0
    if a.cmd == "model":
        pak_name = a.pak or env.profile.base_pak
        pak = env.pak(pak_name)
        if pak is None:
            raise ProjectError(f"{pak_name} not found in the game")
        members = {}
        label = a.label
        if a.appearance:
            from .pak.model_json import PakIndex
            if not env.profile.appearances_script:
                raise ProjectError(f"--appearance is not supported for {env.profile.name}")
            with PakIndex.open(pak) as ix:
                text = ix.read(env.profile.appearances_script).decode("utf-8", "replace")
            char, _, name = a.appearance.rpartition("/")
            hits = [x for x in parse_appearances(text) if x["appearance"] == name and (not char or x["character"] == char)]
            if not hits:
                raise ProjectError(f"appearance {a.appearance!r} not found")
            members = {k: hits[0][k] for k in ("tpp", "fpp", "lodcc", "ui") if hits[0].get(k)}
            label = label or f"{hits[0]['character']} · {name}"
        for m in a.member:
            role, _, mem = m.partition("=")
            members[role] = mem
        if not members:
            raise ProjectError("give --appearance or --member")
        mo = load_override(label or "model", members, pak, pak_name)
        mo.path_mode = a.path
        pr.models.append(mo)
        pr.save()
        print(f"{mo.id} {mo.label}: " + ", ".join(f"{k}={r.member}" for k, r in mo.roles.items()))
        return 0
    if a.cmd == "info":
        rp, pk = output_names(pr, env)
        print(f"{pr.name}: {len(pr.items)} items, {len(pr.models)} model overrides -> {rp} + {pk}")
        for it in pr.items:
            print(f"  {it.id} {'on ' if it.enabled else 'off'} {it.kind:<8} {it.output_name}  <- {it.source}")
        for mo in pr.models:
            print(f"  {mo.id} model    {mo.label}: " + ", ".join(f"{k}={r.member}" for k, r in mo.roles.items()))
        return 0
    if a.cmd == "validate":
        probs = validate(pr, env)
        for p in probs:
            print(f"{p.level:<7} {p.where:<10} {p.message}")
        return 1 if any(p.level == "error" for p in probs) else 0
    if a.cmd == "build":
        if a.keep_work:
            pr.options["keep_work"] = True
        rep = build_project(pr, env, progress=lambda m: print(" ", m), out_dir=Path(a.out) if a.out else None)
        for k, v in rep["outputs"].items():
            print(f"{k}: {v['path']} ({v['size']:,} bytes)")
        for w in rep["warnings"][:20]:
            print("  warning:", w, file=sys.stderr)
        return 0
    return 2
