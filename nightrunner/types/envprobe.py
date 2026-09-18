"""EnvprobeBin (0x55): single part, storage flags 0x21 (method 1, version 2); 124,163 resources in three
`*_envprobes_pc` packs (dlc_frontier 123,777 / menu_level_ft 240 / dlc_ft_prologue 146 — survey 04 §4.2).

Body (C → structural facts A by census 2026-09-15, `run_census()`): ASCII section tags `ENVBIN_HEADER`,
`ENVBIN_COEFFS`, `ENVBIN_RFLPROBE`, …, `ENVBIN_END` at increasing offsets, *not* NUL-terminated and not aligned;
the HEADER tag is followed by 7 u32 words in every sample (0x859, a timestamp-like value, 0, 5, 4, 3, 0x3C —
reported as words, meaning (C)). Section payloads are (C). The dump lists the tags,
their offsets and the bytes after each; the census histograms the tag sequences, the HEADER bytes (with the
timestamp-like u32 masked) and the body sizes.
"""

from __future__ import annotations

import re
import struct
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from .raw import counter_json, words

_TAG = re.compile(rb"ENVBIN_(?:HEADER|COEFFS|RFLPROBE|END|[A-Z]+)")
HEADER_TAG = b"ENVBIN_HEADER"
HEADER_TAIL = 28


def sections(data) -> list[dict]:
    buf = bytes(data)
    out = []
    for m in _TAG.finditer(buf):
        tag = m.group().decode("ascii")
        # the regex is greedy over [A-Z]+; known tags are matched first by alternation order
        out.append({"tag": tag, "offset": m.start(), "end": m.end()})
    for k, s in enumerate(out):
        nxt = out[k + 1]["offset"] if k + 1 < len(out) else len(buf)
        s["payload_bytes"] = nxt - s["end"]
        s["next24_hex"] = buf[s["end"] : s["end"] + 24].hex()
    return out


def header_bytes(data) -> bytes | None:
    buf = bytes(data)
    if not buf.startswith(HEADER_TAG):
        return None
    return buf[len(HEADER_TAG) : len(HEADER_TAG) + HEADER_TAIL]


def dump(parts: list[tuple[int, bytes]]) -> dict:
    out = {"kind": "envprobe", "shape": [f"0x{t:02X}" for t, _ in parts], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if [t for t, _ in parts] != [0x55] or parts[0][1] is None:
        out["form"] = "unexpected"
        return out
    d = bytes(parts[0][1])
    secs = sections(d)
    out["form"] = "tagged" if secs and secs[0]["offset"] == 0 else "untagged"
    out["sections"] = secs
    hb = header_bytes(d)
    out["header_bytes_hex"] = hb.hex() if hb else None
    if hb and len(hb) == HEADER_TAIL:
        out["header_words"] = {f"u32@{4 * k}": v for k, v in enumerate(struct.unpack_from("<7I", hb, 0))}
    out["words"] = words(d, 4)
    return out


PACKS = ["dlc_frontier_envprobes_pc.rpack", "menu_level_ft_envprobes_pc.rpack", "dlc_ft_prologue_envprobes_pc.rpack"]


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    c = Counter()
    tag_seqs = Counter()
    hdr_masked = Counter()
    sizes = Counter()
    coeff_sizes = Counter()
    per_pack = []
    for name in PACKS:
        p = assets / name
        if not p.exists():
            continue
        if progress:
            progress(f"envprobe census {name}")
        pc = Counter()
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x55):
                d = bytes(pk.read_part(res.part_indices[0]))
                pc["resources"] += 1
                sizes[len(d)] += 1
                if not d.startswith(HEADER_TAG):
                    pc["no_header_tag_at_0"] += 1
                    continue
                pc["header_tag_at_0"] += 1
                secs = sections(d)
                tag_seqs["|".join(s["tag"] for s in secs)] += 1
                pc["ends_with_END_tag"] += (secs[-1]["tag"] == "ENVBIN_END")
                pc["END_payload_1_byte"] += (secs[-1]["tag"] == "ENVBIN_END" and secs[-1]["payload_bytes"] == 1)
                hb = header_bytes(d)
                if hb and len(hb) == HEADER_TAIL:
                    hdr_masked[(hb[:4] + b"\0\0\0\0" + hb[8:]).hex()] += 1
                for s in secs:
                    if s["tag"] == "ENVBIN_COEFFS":
                        coeff_sizes[s["payload_bytes"]] += 1
                if limit and pc["resources"] >= limit:
                    break
        per_pack.append({"pack": name, "counts": dict(sorted(pc.items()))})
        c.update(pc)
    return {"family": "envprobe", "type": "0x55", "totals": dict(sorted(c.items())), "per_pack": per_pack,
            "tag_sequences": counter_json(tag_seqs, limit=20), "header_bytes_masked_timestamp": counter_json(hdr_masked, limit=20),
            "body_sizes": counter_json(sizes, limit=30), "coeffs_payload_sizes": counter_json(coeff_sizes, limit=20)}
