"""VoxelizerBin (0x56): single part, storage flags 0x21 (method 1, version 2); 3,689 resources in
dlc_frontier_voxelizer_pc.rpack, named `chunk_<x>_<y>_<z>`.

Body (C → structural facts A by census 2026-09-15, `run_census()`):

    +0x00 i32 w00   +0x04 u32 w04   +0x08 u32 w08   (w04/w08 unpacked unsigned here; all three are small signed
          integers, census 2026-09-15 over 3,689 bodies — notes/FORMATS/types.md §8: w00 spread over roughly
          −20 … 20 (2 × 160, 1 × 159, −1 × 152, 0 × 151, …); w04 0 × 1,213, 1 × 874, −1 × 534, 2 × 418, … up to 8;
          w08 2 × 166, −1 × 164, 4 × 162, 0 × 161, …; consistent in range with the `chunk_x_y_z` grid coordinates,
          correspondence not checked — C)
    +0x0C … a byte stream of gzip members (RFC 1952, `1F 8B 08`), each immediately preceded by a compact integer
          (same encoding as the SDB compact int: tag = b & 3, width (1,2,4,1)[tag], value = int >> 2) equal to the
          member's compressed length. Between members: short runs holding compact-length-prefixed ASCII names
          (weather presets such as `dawn_clear_a`, `sunrise_overcast_a`) and small compact ints.
    Every member in the samples inflates to 4 + 2^n bytes whose first four bytes are `02 00 08 00` (131,076 B) or
    `02 00 04 00` (65,540 B). Contents (C).

The dump lists members (offset, compressed/decompressed size, first bytes) and gap bytes; nothing is decoded.
"""

from __future__ import annotations

import struct
import zlib
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from .raw import counter_json, words

GZIP_MAGIC = b"\x1f\x8b\x08"
HEADER = struct.Struct("<iII")
_WIDTH = (1, 2, 4, 1)


def compact_before(buf: bytes, pos: int) -> list[int]:
    """Values of every self-consistent compact int that ends exactly at *pos* (widths 1, 2, 4 are ambiguous when
    read backwards, so all candidates are returned; the census asks whether any equals the member length)."""
    out = []
    for w in (1, 2, 4):
        s = pos - w
        if s < 0:
            continue
        tag = buf[s] & 3
        if _WIDTH[tag] != w:
            continue
        out.append(int.from_bytes(buf[s:pos], "little") >> 2)
    return out


def gap_items(gap: bytes) -> list:
    """Sequential compact-int decode of a gap: compact ints, and compact-length-prefixed printable strings."""
    items = []
    pos = 0
    n = len(gap)
    while pos < n:
        tag = gap[pos] & 3
        w = _WIDTH[tag]
        if pos + w > n:
            items.append({"raw_hex": gap[pos:].hex()})
            break
        v = int.from_bytes(gap[pos:pos + w], "little") >> 2
        pos += w
        if 0 < v <= 128 and pos + v <= n and all(0x20 <= b < 0x7F for b in gap[pos:pos + v]):
            items.append(gap[pos:pos + v].decode("ascii"))
            pos += v
        else:
            items.append(v)
    return items


def members(data, *, inflate: bool = True, max_members: int = 100000) -> dict:
    buf = bytes(data)
    out = {"size": len(buf)}
    if len(buf) >= 12:
        w00, w04, w08 = HEADER.unpack_from(buf, 0)
        out.update({"w00": w00, "w04": w04, "w08": w08})
    pos = 12
    mem = []
    gaps = []
    ok_len = 0
    while len(mem) < max_members:
        i = buf.find(GZIP_MAGIC, pos)
        if i < 0:
            break
        gap = buf[pos:i]
        do = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            raw = do.decompress(buf[i:]) if inflate else None
        except zlib.error as exc:
            out["error"] = f"zlib at 0x{i:X}: {exc}"
            break
        if not inflate:
            # cannot know the member length without inflating; stop
            break
        consumed = len(buf) - i - len(do.unused_data)
        pre = compact_before(buf, i)
        m = {"offset": i, "compressed": consumed, "decompressed": len(raw), "head_hex": raw[:8].hex(),
             "compact_before": pre, "compact_eq_len": consumed in pre, "gap_hex": gap.hex() if len(gap) <= 48 else gap[:48].hex() + "…",
             "gap_len": len(gap), "gap_items": gap_items(gap)}
        ok_len += (consumed in pre)
        mem.append(m)
        gaps.append(gap)
        pos = i + consumed
        if not do.eof:
            out["error"] = f"member at 0x{i:X} did not reach EOF"
            break
    out["members"] = mem
    out["member_count"] = len(mem)
    out["members_compact_eq_len"] = ok_len
    out["tail_after_last_hex"] = buf[pos:pos + 32].hex()
    out["tail_after_last_len"] = len(buf) - pos
    out["gap_strings"] = sorted({it for m in mem for it in m["gap_items"] if isinstance(it, str)})
    return out


def dump(parts: list[tuple[int, bytes]]) -> dict:
    out = {"kind": "voxelizer", "shape": [f"0x{t:02X}" for t, _ in parts], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if [t for t, _ in parts] != [0x56] or parts[0][1] is None:
        out["form"] = "unexpected"
        return out
    d = parts[0][1]
    m = members(d)
    out["form"] = "gzip_members" if m["member_count"] else "no_gzip_members"
    out.update(m)
    out["words"] = words(d, 3)
    return out


PACKS = ["dlc_frontier_voxelizer_pc.rpack"]


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    c = Counter()
    w00 = Counter()
    w04 = Counter()
    w08 = Counter()
    member_counts = Counter()
    dec_sizes = Counter()
    heads = Counter()
    names = Counter()
    sizes = []
    per_pack = []
    for name in PACKS:
        p = assets / name
        if not p.exists():
            continue
        if progress:
            progress(f"voxelizer census {name}")
        pc = Counter()
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x56):
                d = bytes(pk.read_part(res.part_indices[0]))
                pc["resources"] += 1
                sizes.append(len(d))
                m = members(d)
                if "error" in m:
                    pc["errors"] += 1
                w00[m.get("w00")] += 1
                w04[m.get("w04")] += 1
                w08[m.get("w08")] += 1
                member_counts[m["member_count"]] += 1
                pc["members_total"] += m["member_count"]
                pc["members_compact_eq_len"] += m["members_compact_eq_len"]
                pc["bodies_all_members_compact_eq_len"] += (m["member_count"] > 0 and m["members_compact_eq_len"] == m["member_count"])
                pc["tail_zero"] += (not any(bytes(d)[len(d) - m["tail_after_last_len"]:]))
                for mm in m["members"]:
                    dec_sizes[mm["decompressed"]] += 1
                    heads[mm["head_hex"]] += 1
                for s in m["gap_strings"]:
                    names[s] += 1
                if limit and pc["resources"] >= limit:
                    break
        per_pack.append({"pack": name, "counts": dict(sorted(pc.items()))})
        c.update(pc)
    s = sorted(sizes)
    return {"family": "voxelizer", "type": "0x56", "totals": dict(sorted(c.items())), "per_pack": per_pack,
            "w00": counter_json(w00), "w04": counter_json(w04, limit=20), "w08": counter_json(w08, limit=20),
            "members_per_body": counter_json(member_counts, limit=20), "decompressed_sizes": counter_json(dec_sizes, limit=20),
            "decompressed_head8": counter_json(heads, limit=20), "gap_strings": counter_json(names, limit=60),
            "sizes": ({"count": len(s), "min": s[0], "median": s[len(s) // 2], "max": s[-1], "total": sum(s)} if s else {})}
