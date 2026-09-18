"""Generic lossless raw bundle: JSON sidecar + shared word/strings dump helpers.

The raw parts (already written by nightrunner.extract as `<family>/<index>_<name>/<partfile>.bin`) are the
intermediate. The sidecar `<index>_<name>.json` written next to them records everything needed to audit the
bundle: part list with storage attributes, sizes, sha256, the first 64 bytes of each part (`head_hex`) and the
family's structural dump. Nothing beyond the raw parts is required to rebuild — build() of every raw family
returns {} and the container writer re-emits the recorded raw files.

Shared helpers used by every family module:

    words(data, n)        first n 32-bit words as u32 / i32 / f32 / ascii — a *dump*, not a decode
    strings(data, ...)    printable ASCII runs (NUL- or length-delimited scan), for name discovery
    counter_json(c)       Counter → sorted JSON-able list
"""

from __future__ import annotations

import math
import re
import struct
from collections import Counter
from pathlib import Path

from ..container import catalogue
from ..container.rp6l import Pack, Resource
from ..util.hashing import sha256_bytes

SIDECAR_SCHEMA = "nightrunner.raw/1"
HEAD_BYTES = 64

_PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")


def words(data, n: int = 16, offset: int = 0) -> list[dict]:
    """First *n* little-endian 32-bit words starting at *offset*, rendered every way a reader might want them.
    Purely descriptive: the caller must not infer field meaning from this."""
    mv = memoryview(data)
    out = []
    for k in range(n):
        off = offset + 4 * k
        if off + 4 > len(mv):
            break
        raw = bytes(mv[off : off + 4])
        u = struct.unpack("<I", raw)[0]
        i = struct.unpack("<i", raw)[0]
        f = struct.unpack("<f", raw)[0]
        rec = {"off": f"0x{off:X}", "hex": raw.hex(), "u32": u, "u16": [u & 0xFFFF, u >> 16]}
        if i < 0:
            rec["i32"] = i
        if math.isfinite(f) and (f == 0.0 or 1e-6 <= abs(f) <= 1e7):
            rec["f32"] = round(f, 6)
        asc = "".join(chr(b) if 32 <= b < 127 else "." for b in raw)
        if asc.count(".") <= 1:
            rec["ascii"] = asc
        out.append(rec)
    return out


def strings(data, min_len: int = 4, limit: int = 200, unique: bool = True) -> list[dict]:
    """Printable-ASCII runs (≥ min_len) with offsets. A discovery aid, not a parse of any string table."""
    mv = bytes(memoryview(data))
    seen = set()
    out = []
    for m in _PRINTABLE.finditer(mv):
        s = m.group()
        if len(s) < min_len:
            continue
        txt = s.decode("ascii")
        if unique:
            if txt in seen:
                continue
            seen.add(txt)
        out.append({"off": f"0x{m.start():X}", "len": len(s), "text": txt})
        if len(out) >= limit:
            out.append({"truncated": True})
            break
    return out


def counter_json(c: Counter, key=None, limit: int | None = None) -> list:
    items = sorted(c.items(), key=(lambda kv: (-kv[1], str(kv[0]))) if key is None else key)
    if limit is not None:
        items = items[:limit]
    return [{"value": (k if isinstance(k, (int, str, float)) else str(k)), "count": v} for k, v in items]


def align_up(v: int, a: int) -> int:
    return (v + a - 1) & ~(a - 1)


# ---- sidecar ---------------------------------------------------------------------------------------------------

def part_records(pack: Pack, res: Resource, part_files: dict[int, str] | None = None) -> list[dict]:
    """One record per physical part: storage attributes, size, sha256, head_hex. `part_files` maps ordinal → the
    raw file name written by the extractor (None when the part is not directly readable)."""
    out = []
    for k, i in enumerate(res.part_indices):
        p = pack.physicals[i]
        s = pack.storages[p.storage_index]
        rec = {
            "ordinal": k, "physical_index": i, "type": f"0x{s.type:02X}", "type_name": catalogue.type_name(s.type),
            "size": p.size, "offset": pack.part_offset(i), "offset_units": p.offset_units,
            "flag_bits": f"0x{p.flag_bits:04X}", "fc": f"0x{p.fc:08X}",
            "storage": {
                "index": p.storage_index, "align_raw": s.align_raw, "alignment": s.alignment,
                "flags": f"0x{s.flags:02X}", "metadata": f"0x{s.metadata:02X}",
                "method": s.method, "version": s.version, "codec": s.codec,
            },
            "file": (part_files or {}).get(k),
        }
        if pack.part_is_direct(i):
            data = pack.read_part(i)
            rec["sha256"] = sha256_bytes(data)
            rec["head_hex"] = bytes(data[:HEAD_BYTES]).hex()
        else:
            rec["sha256"] = None
            rec["head_hex"] = None
            rec["unsupported"] = "child-pack or compressed storage"
        out.append(rec)
    return out


def sidecar(pack: Pack, res: Resource, dump: dict | None, part_files: dict[int, str] | None = None) -> dict:
    return {
        "schema": SIDECAR_SCHEMA,
        "index": res.index, "name": res.name, "name_hex": res.name_raw.hex(),
        "type": f"0x{res.type:02X}", "type_name": catalogue.type_name(res.type), "flags": f"0x{res.flags:02X}",
        "lossless": "raw parts only — nothing in this sidecar is needed to rebuild the resource byte-identically",
        "parts": part_records(pack, res, part_files),
        "dump": dump,
    }


def sidecar_name(res_dir: Path) -> Path:
    return res_dir / (res_dir.name + ".json")


def read_parts(res: Resource) -> list[tuple[int, bytes]]:
    """[(storage type, bytes)] for every directly readable part, in logical order. Copies (bytes), so the pack's
    mmap can be closed while a dump still holds the data."""
    pk = res.pack
    out = []
    for i in res.part_indices:
        if pk.part_is_direct(i):
            out.append((pk.part_type(i), bytes(pk.read_part(i))))
        else:
            out.append((pk.part_type(i), None))
    return out


def generic_dump(parts: list[tuple[int, bytes]], n_words: int = 16) -> dict:
    """Family-agnostic dump: per part the leading words and a strings sample."""
    return {
        "kind": "generic",
        "parts": [
            {"type": f"0x{t:02X}", "size": (len(d) if d is not None else None),
             "words": (words(d, n_words) if d is not None else None),
             "strings": (strings(d, limit=40) if d is not None else None)}
            for t, d in parts
        ],
    }
