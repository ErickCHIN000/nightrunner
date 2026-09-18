"""`nr audio ...` — read-only inspection of the Wwise audio containers.

    nr audio info    <file.aesp>                        header, member count, layout check
    nr audio list    <file.aesp> [--query S] [--limit N] members as JSONL (index, name, id, offset, size)
    nr audio extract <file.aesp> <out dir> [--query S] [--limit N] [--ext auto|none]
                                                        write members to disk
    nr audio bank    <file.aesp> <member>               parse one soundbank: chunks, HIRC, sounds
    nr audio sounds  <file.aesp> <member> [--streamed]  Sound objects as JSONL (source id, stream type)
    nr audio registry <meta.aesp> [--kind K] [--query S] [--limit N]
                                                        the wwisepinhead names (events, switches, states, buses)
    nr audio census  <audio dir> [--json-out F]         every container and every bank, with totals
    nr audio towem   <in.wav> <out.wem> [--no-junk]     build a PCM .wem from a WAV (no Wwise needed)
    nr audio weminfo <file.wem>                         what a .wem is: codec, rate, channels, chunks

Nothing here writes into the game. `extract` is the only command that writes at all, and only where you say.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..errors import NightrunnerError
from . import bnk, wem
from .aesp import CONTAINERS, Aesp
from .pinhead import MEMBER_NAME, Pinhead


def _jprint(obj, indent=None) -> None:
    print(json.dumps(obj, indent=indent, ensure_ascii=False))


def _members(a, container: Aesp):
    q = a.query.casefold() if getattr(a, "query", None) else None
    n = 0
    for m in container:
        if q and q not in m.name.casefold():
            continue
        yield m
        n += 1
        if getattr(a, "limit", None) and n >= a.limit:
            return


def _suffix(name: str, data: memoryview) -> str:
    """What a member should be called on disk. Members carry no extension of their own."""
    head = bytes(data[:4])
    if head == bnk.BKHD:
        return ".bnk"
    if head == b"RIFF":
        return ".wem"
    if head[:2] == b"<?":
        return ".xml"
    return ".bin"


def cmd_info(a) -> int:
    with Aesp.open(a.file) as c:
        _jprint(c.to_json(), indent=2)
    return 0


def cmd_list(a) -> int:
    with Aesp.open(a.file) as c:
        for m in _members(a, c):
            _jprint(m.to_json())
    return 0


def cmd_extract(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    written = total = 0
    with Aesp.open(a.file) as c:
        for m in _members(a, c):
            data = c.read(m)
            ext = "" if a.ext == "none" else _suffix(m.name, data)
            dest = out / f"{m.name}{ext}"
            dest.write_bytes(bytes(data))
            written += 1
            total += m.size
    _jprint({"extracted": written, "bytes": total, "out": str(out)})
    return 0


def _bank(container: Aesp, member: str) -> bnk.Bank:
    m = container.find(member)
    if m is None:
        raise NightrunnerError(f"no member called {member!r} in {container.path.name}")
    data = container.read(m)
    if not bnk.is_bank(data):
        raise NightrunnerError(f"member {member!r} is not a soundbank (magic {bytes(data[:4])!r})")
    return bnk.Bank(data, m.name)


def cmd_bank(a) -> int:
    with Aesp.open(a.file) as c:
        _jprint(_bank(c, a.member).to_json(), indent=2)
    return 0


def cmd_sounds(a) -> int:
    with Aesp.open(a.file) as c:
        for s in _bank(c, a.member).sounds:
            if a.streamed and s.stream_type == bnk.STREAM_EMBEDDED:
                continue
            _jprint(s.to_json())
    return 0


def cmd_registry(a) -> int:
    with Aesp.open(a.file) as c:
        m = c.find(MEMBER_NAME)
        if m is None:
            raise NightrunnerError(f"{c.path.name} has no {MEMBER_NAME} member")
        ph = Pinhead(bytes(c.read(m)), f"{c.path.name}:{MEMBER_NAME}")
    if not (a.kind or a.query):
        _jprint(ph.to_json(), indent=2)
        return 0
    q = a.query.casefold() if a.query else None
    n = 0
    for o in ph.objects:
        if a.kind and o.kind.casefold() != a.kind.casefold():
            continue
        if q and q not in o.name.casefold():
            continue
        _jprint(o.to_json())
        n += 1
        if a.limit and n >= a.limit:
            break
    return 0


def cmd_census(a) -> int:
    """Every container in a folder, every bank inside them. The numbers in `notes/FORMATS/aesp.md` come from here."""
    d = Path(a.dir)
    report: dict = {"dir": str(d), "containers": {}, "banks": {}}
    banks = {"parsed": 0, "failed": 0, "versions": {}, "chunks": {}, "object_types": {},
             "stream_types": {}, "hirc_exact": 0, "sounds": 0, "source_ids": 0}
    src: set[int] = set()
    fails: list[dict] = []
    for stem in CONTAINERS:
        p = d / f"{stem}.aesp"
        if not p.is_file():
            continue
        with Aesp.open(p) as c:
            report["containers"][stem] = c.to_json()
            for m in c:
                data = c.read(m)
                if not bnk.is_bank(data):
                    continue
                try:
                    b = bnk.Bank(data, m.name)
                except Exception as exc:                       # a bank that will not parse is reported, not fatal
                    banks["failed"] += 1
                    fails.append({"member": m.name, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                banks["parsed"] += 1
                banks["versions"][str(b.version)] = banks["versions"].get(str(b.version), 0) + 1
                for ch in b.chunks:
                    banks["chunks"][ch.tag] = banks["chunks"].get(ch.tag, 0) + 1
                if b.hirc_exact:
                    banks["hirc_exact"] += 1
                for t, n in b.type_histogram().items():
                    banks["object_types"][str(t)] = banks["object_types"].get(str(t), 0) + n
                for k, n in b.stream_histogram().items():
                    banks["stream_types"][k] = banks["stream_types"].get(k, 0) + n
                banks["sounds"] += len(b.sounds)
                src |= b.source_ids()
    banks["source_ids"] = len(src)
    banks["object_types"] = dict(sorted(banks["object_types"].items(), key=lambda kv: int(kv[0])))
    report["banks"] = banks
    if fails:
        report["failures"] = fails
    if a.json_out:
        Path(a.json_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        _jprint({"written": a.json_out})
    else:
        _jprint(report, indent=2)
    return 0


def cmd_towem(a) -> int:
    """WAV -> PCM .wem. No Wwise and no encoder: the engine ships PCM wems of its own (see audio/wem.py)."""
    data = Path(a.input).read_bytes()
    blob = wem.wav_to_wem(data, junk=not a.no_junk)
    Path(a.out).write_bytes(blob)
    _jprint({"out": a.out, **wem.describe(blob)})
    return 0


def cmd_weminfo(a) -> int:
    _jprint(wem.describe(Path(a.file).read_bytes()), indent=2)
    return 0


def build(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="nr audio", description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="container header and layout")
    p.add_argument("file")
    p.set_defaults(fn=cmd_info)

    p = sub.add_parser("list", help="members as JSONL")
    p.add_argument("file")
    p.add_argument("--query")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_list)

    p = sub.add_parser("extract", help="write members to a folder")
    p.add_argument("file")
    p.add_argument("out")
    p.add_argument("--query")
    p.add_argument("--limit", type=int)
    p.add_argument("--ext", choices=("auto", "none"), default="auto",
                   help="auto adds .bnk/.wem/.xml by magic (default); none keeps the bare member name")
    p.set_defaults(fn=cmd_extract)

    p = sub.add_parser("bank", help="parse one soundbank")
    p.add_argument("file")
    p.add_argument("member")
    p.set_defaults(fn=cmd_bank)

    p = sub.add_parser("sounds", help="Sound objects of one bank as JSONL")
    p.add_argument("file")
    p.add_argument("member")
    p.add_argument("--streamed", action="store_true", help="only sounds that are not embedded")
    p.set_defaults(fn=cmd_sounds)

    p = sub.add_parser("registry", help="wwisepinhead names")
    p.add_argument("file")
    p.add_argument("--kind", help="Event, Switch, SwitchValue, AuxBus, Parameter, Preload, ...")
    p.add_argument("--query")
    p.add_argument("--limit", type=int)
    p.set_defaults(fn=cmd_registry)

    p = sub.add_parser("towem", help="build a PCM .wem from a WAV")
    p.add_argument("input")
    p.add_argument("out")
    p.add_argument("--no-junk", action="store_true", help="omit the junk padding chunk the shipped wems carry")
    p.set_defaults(fn=cmd_towem)

    p = sub.add_parser("weminfo", help="codec, rate, channels and chunks of a .wem")
    p.add_argument("file")
    p.set_defaults(fn=cmd_weminfo)

    p = sub.add_parser("census", help="every container and bank in a folder")
    p.add_argument("dir")
    p.add_argument("--json-out")
    p.set_defaults(fn=cmd_census)

    return ap.parse_args(argv)


def run(args) -> int:
    a = build(list(args.args))
    try:
        return a.fn(a)
    except NightrunnerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
