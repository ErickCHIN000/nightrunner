"""AnimationScr (0x42) + AnimationScrFixups (0x43): the compiled animation-script pair.

Layout — tier (L) legacy corpus contract (survey 04 §2.4: eight structural claims passed over all 217 resources
by the old tooling; no native decompile) — re-verified here by `census()` (A, census 2026-09-15 when the reported
`parsed_to_exact_end` equals the resource count):

Part 0 (0x42)
    R × 56-byte records            R = name_count read from part 1
    N × 12-byte entries (3 × u32)  N = Σ record.owned_count ; middle u32 uses 0xFFFFFFFF as "absent"
    R × NUL-terminated UTF-8 names ; no trailer
    record: +00 u32 name_offset (into the names block) | +04 u32 w04 (few distinct values, meaning unknown)
            +08 u32 zero | +0C u32 zero | +10 u32 w10 ∈ {0,1,3} | +14 f32 | +18 f32 (commonly 30.0)
            +1C f32 | +20 f32 | +24..+2F zero | +30 u32 owned_count | +34 u32 zero
Part 1 (0x43) — a command script, NOT a relocation table despite the type name
    u32 name_count ; u32 group_count
    group_count × { u32 command_count ; command_count × { cstr name ; u32 argc ; argc × (u8 tag, value) } }
    name_count × cstr
    tag 'f' f32 | 'i' i32 | 's' cstr | 'v' 3 × f32

Field meanings beyond the framing (what a record *is*, which name a group belongs to) are (C).
"""

from __future__ import annotations

import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import FormatError
from .raw import counter_json

RECORD = struct.Struct("<IIIIIffffIIIII")   # 56 bytes: 5 u32, 4 f32, 5 u32
ENTRY = struct.Struct("<3I")
ABSENT = 0xFFFFFFFF


@dataclass
class ScrRecord:
    name_offset: int
    w04: int
    w08: int
    w0C: int
    w10: int
    f14: float
    f18: float
    f1C: float
    f20: float
    w24: int
    w28: int
    w2C: int
    owned_count: int
    w34: int


def _cstr(buf, pos: int) -> tuple[bytes, int]:
    end = buf.find(b"\0", pos)
    if end < 0:
        raise FormatError(f"animscr: unterminated string at 0x{pos:X}")
    return bytes(buf[pos:end]), end + 1


def parse_script(part1) -> dict:
    """Part 1: {name_count, groups: [[{name, args:[(tag, value)]}]], names: [bytes], end}"""
    buf = bytes(part1)
    if len(buf) < 8:
        raise FormatError("animscr part 1 shorter than 8 bytes")
    name_count, group_count = struct.unpack_from("<II", buf, 0)
    pos = 8
    groups = []
    for _ in range(group_count):
        if pos + 4 > len(buf):
            raise FormatError("animscr: truncated group")
        cc = struct.unpack_from("<I", buf, pos)[0]
        pos += 4
        cmds = []
        for _ in range(cc):
            name, pos = _cstr(buf, pos)
            argc = struct.unpack_from("<I", buf, pos)[0]
            pos += 4
            args = []
            for _ in range(argc):
                tag = buf[pos : pos + 1]
                pos += 1
                if tag == b"f":
                    args.append(("f", struct.unpack_from("<f", buf, pos)[0])); pos += 4
                elif tag == b"i":
                    args.append(("i", struct.unpack_from("<i", buf, pos)[0])); pos += 4
                elif tag == b"s":
                    s, pos = _cstr(buf, pos); args.append(("s", s.decode("utf-8", "surrogateescape")))
                elif tag == b"v":
                    args.append(("v", list(struct.unpack_from("<3f", buf, pos)))); pos += 12
                else:
                    raise FormatError(f"animscr: unknown arg tag {tag!r} at 0x{pos - 1:X}")
            cmds.append({"name": name.decode("utf-8", "surrogateescape"), "args": args})
        groups.append(cmds)
    names = []
    for _ in range(name_count):
        s, pos = _cstr(buf, pos)
        names.append(s)
    return {"name_count": name_count, "group_count": group_count, "groups": groups, "names": names, "end": pos, "size": len(buf)}


def parse_records(part0, name_count: int) -> dict:
    buf = bytes(part0)
    recs = []
    pos = 0
    for k in range(name_count):
        if pos + 56 > len(buf):
            raise FormatError(f"animscr: record {k} truncated")
        recs.append(ScrRecord(*RECORD.unpack_from(buf, pos)))
        pos += 56
    n_entries = sum(r.owned_count for r in recs)
    entries = []
    if pos + 12 * n_entries > len(buf):
        raise FormatError("animscr: entries truncated")
    for k in range(n_entries):
        entries.append(ENTRY.unpack_from(buf, pos))
        pos += 12
    names_start = pos
    names = []
    for k in range(name_count):
        s, pos = _cstr(buf, pos)
        names.append(s)
    return {"records": recs, "entries": entries, "names": names, "names_start": names_start, "end": pos, "size": len(buf)}


def serialise_script(sc: dict) -> bytes:
    out = bytearray(struct.pack("<II", sc["name_count"], sc["group_count"]))
    for cmds in sc["groups"]:
        out += struct.pack("<I", len(cmds))
        for c in cmds:
            out += c["name"].encode("utf-8", "surrogateescape") + b"\0" + struct.pack("<I", len(c["args"]))
            for tag, v in c["args"]:
                out += tag.encode()
                if tag == "f":
                    out += struct.pack("<f", v)
                elif tag == "i":
                    out += struct.pack("<i", v)
                elif tag == "s":
                    out += v.encode("utf-8", "surrogateescape") + b"\0"
                else:
                    out += struct.pack("<3f", *v)
    for n in sc["names"]:
        out += n + b"\0"
    return bytes(out)


def dump(parts: list[tuple[int, bytes]]) -> dict:
    out = {"kind": "animscr", "shape": [f"0x{t:02X}" for t, _ in parts], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if [t for t, _ in parts] != [0x42, 0x43] or any(d is None for _, d in parts):
        out["form"] = "unexpected"
        return out
    p0, p1 = parts[0][1], parts[1][1]
    try:
        sc = parse_script(p1)
        rc = parse_records(p0, sc["name_count"])
    except FormatError as exc:
        out["error"] = str(exc)
        return out
    out["form"] = "legacy_contract"
    out["script"] = {"name_count": sc["name_count"], "group_count": sc["group_count"],
                     "commands": sum(len(g) for g in sc["groups"]), "parsed_to_exact_end": sc["end"] == sc["size"],
                     "command_names": counter_json(Counter(c["name"] for g in sc["groups"] for c in g), limit=40),
                     "first_group": sc["groups"][0][:6] if sc["groups"] else []}
    out["records"] = {"count": len(rc["records"]), "entries": len(rc["entries"]),
                      "entries_absent_middle": sum(1 for e in rc["entries"] if e[1] == ABSENT),
                      "parsed_to_exact_end": rc["end"] == rc["size"],
                      "names_match_script": rc["names"] == sc["names"],
                      "name_offsets_consistent": all(rc["names_start"] + r.name_offset < rc["size"] and
                                                     bytes(p0)[rc["names_start"] + r.name_offset:].split(b"\0", 1)[0] == rc["names"][k]
                                                     for k, r in enumerate(rc["records"])),
                      "w04_values": counter_json(Counter(r.w04 for r in rc["records"])),
                      "w10_values": counter_json(Counter(r.w10 for r in rc["records"])),
                      "f18_values": counter_json(Counter(round(r.f18, 4) for r in rc["records"]), limit=8),
                      "zero_fields_hold": all(r.w08 == 0 and r.w0C == 0 and r.w24 == 0 and r.w28 == 0 and r.w2C == 0 and r.w34 == 0 for r in rc["records"]),
                      "first_records": [{"name": rc["names"][k].decode("utf-8", "replace"), "w04": r.w04, "w10": r.w10,
                                         "f14": r.f14, "f18": r.f18, "f1C": r.f1C, "f20": r.f20, "owned": r.owned_count}
                                        for k, r in enumerate(rc["records"][:5])]}
    return out


PACKS = ["common_anims_pc.rpack", "common_anims_stream_pc.rpack", "player_anims_pc.rpack",
         "player_anims_static_pc.rpack", "player_anims_stream_pc.rpack", "lang_speech_en_pc.rpack"]


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    c = Counter()
    cmd_names = Counter()
    w04 = Counter()
    w10 = Counter()
    tags = Counter()
    per_pack = []
    failures = []
    for name in PACKS:
        p = assets / name
        if not p.exists():
            continue
        if progress:
            progress(f"animscr census {name}")
        pc = Counter()
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x42):
                parts = [(pk.part_type(i), bytes(pk.read_part(i))) for i in res.part_indices]
                pc["resources"] += 1
                if [t for t, _ in parts] != [0x42, 0x43]:
                    pc["unexpected_shape"] += 1
                    continue
                try:
                    sc = parse_script(parts[1][1])
                    rc = parse_records(parts[0][1], sc["name_count"])
                except FormatError as exc:
                    pc["parse_error"] += 1
                    failures.append({"pack": name, "index": res.index, "name": res.name, "error": str(exc)})
                    continue
                pc["parsed"] += 1
                pc["script_exact_end"] += (sc["end"] == sc["size"])
                pc["records_exact_end"] += (rc["end"] == rc["size"])
                pc["names_match"] += (rc["names"] == sc["names"])
                pc["reserialise_identical"] += (serialise_script(sc) == bytes(parts[1][1]))
                pc["zero_fields_hold"] += all(r.w08 == 0 and r.w0C == 0 and r.w24 == 0 and r.w28 == 0 and r.w2C == 0 and r.w34 == 0 for r in rc["records"])
                pc["records"] += len(rc["records"])
                pc["entries"] += len(rc["entries"])
                pc["entries_absent_middle"] += sum(1 for e in rc["entries"] if e[1] == ABSENT)
                pc["commands"] += sum(len(g) for g in sc["groups"])
                pc["groups"] += sc["group_count"]
                for g in sc["groups"]:
                    for cm in g:
                        cmd_names[cm["name"]] += 1
                        for t, _ in cm["args"]:
                            tags[t] += 1
                for r in rc["records"]:
                    w04[r.w04] += 1
                    w10[r.w10] += 1
                if limit and pc["resources"] >= limit:
                    break
        per_pack.append({"pack": name, "counts": dict(sorted(pc.items()))})
        c.update(pc)
    return {"family": "animscr", "type": "0x42", "totals": dict(sorted(c.items())), "per_pack": per_pack,
            "command_names": counter_json(cmd_names), "arg_tags": counter_json(tags),
            "record_w04_values": counter_json(w04), "record_w10_values": counter_json(w10), "failures": failures[:20]}
