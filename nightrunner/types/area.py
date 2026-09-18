"""Area (0x5A): single part, storage flags 0x21 (method 1, version 2); 2,875 resources in 18 `reg_*` packs.

Body (C → structural facts A by census 2026-09-15, `run_census()`): a tagged chunk stream.

    +0x00 char[4] 'AHDr'   u32 w04 (2 on 2,875/2,875)   u32 w08 (32 on 2,875)   u32 w0C (56 on 2,853, 0 on 22)
    +0x10 28 bytes: zero on 1,723 of 2,875 bodies (NOT always zero; contents undecoded — census 2026-09-15,
          notes/FORMATS/types.md §9; the dump reports them as `gap_0x10_0x2C_zero`)
    +0x2C chunks: { char[4] tag ; u32 version ; u32 size ; u8 payload[size] }  — payloads that tile exactly into
          further chunks are walked recursively (AREA ⊃ MSTs UDTs CLTs SPTs SPTv XFTs BNDc BNDL ⊃ BNDt ENTs TREe …)
    trailing bytes after the last top-level chunk: zero padding to a 16-byte multiple

Tag meanings are (C); the dump reports the chunk tree, versions and sizes, and a strings sample (mesh names
`*.msh`, `Default`, `Null`, `*_lod0` appear inside MSTs/BNDt payloads as u16-length-prefixed strings — observed,
not decoded).
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from .raw import counter_json, strings, words

MAGIC = b"AHDr"
CHUNK_START = 0x2C
_CH = struct.Struct("<4sII")


def _is_tag(b: bytes) -> bool:
    return len(b) == 4 and all(0x20 <= x < 0x7F for x in b) and b[0:1].isalpha()


def walk(buf, start: int, end: int, depth: int = 0, max_depth: int = 6) -> list[dict] | None:
    """Parse [start, end) as a sequence of chunks; None when it does not tile exactly."""
    pos = start
    out = []
    while pos < end:
        if pos + 12 > end:
            return None
        tag, ver, size = _CH.unpack_from(buf, pos)
        if not _is_tag(tag) or pos + 12 + size > end:
            return None
        rec = {"tag": tag.decode("ascii"), "version": ver, "size": size, "offset": pos}
        if size >= 12 and depth < max_depth:
            sub = walk(buf, pos + 12, pos + 12 + size, depth + 1, max_depth)
            if sub:
                rec["children"] = sub
        out.append(rec)
        pos += 12 + size
    return out if out else None


def parse(data) -> dict:
    buf = bytes(data)
    out = {"size": len(buf), "magic": buf[:4].decode("latin-1")}
    if len(buf) < CHUNK_START or buf[:4] != MAGIC:
        out["form"] = "no_AHDr_magic"
        return out
    w04, w08, w0C = struct.unpack_from("<3I", buf, 4)
    out.update({"w04": w04, "w08": w08, "w0C": w0C, "gap_0x10_0x2C_zero": not any(buf[0x10:CHUNK_START])})
    # top-level chunks until the tiling stops; the remainder must be zero padding
    pos = CHUNK_START
    top = []
    while pos + 12 <= len(buf):
        tag, ver, size = _CH.unpack_from(buf, pos)
        if not _is_tag(tag) or pos + 12 + size > len(buf):
            break
        rec = {"tag": tag.decode("ascii"), "version": ver, "size": size, "offset": pos}
        sub = walk(buf, pos + 12, pos + 12 + size, 1)
        if sub:
            rec["children"] = sub
        top.append(rec)
        pos += 12 + size
    out["chunks"] = top
    out["chunks_end"] = pos
    out["trailing_bytes"] = len(buf) - pos
    out["trailing_zero"] = not any(buf[pos:])
    out["form"] = "chunked" if top else "no_chunks"
    return out


def _flatten(chunks, path="", acc=None):
    acc = [] if acc is None else acc
    for c in chunks or []:
        p = f"{path}/{c['tag']}"
        acc.append((p, c["version"], c["size"]))
        _flatten(c.get("children"), p, acc)
    return acc


def dump(parts: list[tuple[int, bytes]]) -> dict:
    out = {"kind": "area", "shape": [f"0x{t:02X}" for t, _ in parts], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if [t for t, _ in parts] != [0x5A] or parts[0][1] is None:
        out["form"] = "unexpected"
        return out
    d = parts[0][1]
    out.update(parse(d))
    out["words"] = words(d, 4)
    out["strings"] = strings(d, limit=60)
    return out


def _find_area_packs(assets: Path) -> list[Path]:
    out = []
    for p in sorted(assets.rglob("*.rpack")):
        if "custom_rpacks" in p.parts or p.name.lower().startswith("assets_"):
            continue
        try:
            with Pack.open(p) as pk:
                if 0x5A in pk.type_histogram():
                    out.append(p)
        except Exception:  # noqa: BLE001
            continue
    return out


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    c = Counter()
    tags = Counter()
    tag_versions = Counter()
    top_tags = Counter()
    w04 = Counter()
    sizes = []
    per_pack = []
    for p in _find_area_packs(assets):
        if progress:
            progress(f"area census {p.name}")
        pc = Counter()
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x5A):
                data = bytes(pk.read_part(res.part_indices[0]))
                pc["resources"] += 1
                sizes.append(len(data))
                a = parse(data)
                if a.get("form") != "chunked":
                    pc[f"form_{a.get('form')}"] += 1
                    continue
                pc["chunked"] += 1
                pc["gap_zero"] += a["gap_0x10_0x2C_zero"]
                pc["trailing_zero"] += a["trailing_zero"]
                pc["trailing_lt_16"] += (a["trailing_bytes"] < 16)
                pc["size_multiple_of_16"] += (len(data) % 16 == 0)
                w04[(a["w04"], a["w08"], a["w0C"])] += 1
                top_tags["|".join(ch["tag"] for ch in a["chunks"])] += 1
                for path, ver, size in _flatten(a["chunks"]):
                    tags[path] += 1
                    tag_versions[f"{path}@v{ver}"] += 1
                if limit and pc["resources"] >= limit:
                    break
        per_pack.append({"pack": p.name, "counts": dict(sorted(pc.items()))})
        c.update(pc)
    s = sorted(sizes)
    return {"family": "area", "type": "0x5A", "totals": dict(sorted(c.items())), "per_pack": per_pack,
            "header_words_w04_w08_w0C": counter_json(w04), "top_level_tag_sequences": counter_json(top_tags, limit=20),
            "chunk_paths": counter_json(tags, limit=120), "chunk_versions": counter_json(tag_versions, limit=160),
            "sizes": ({"count": len(s), "min": s[0], "median": s[len(s) // 2], "max": s[-1], "total": sum(s)} if s else {})}
