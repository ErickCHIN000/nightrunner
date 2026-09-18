"""Mesh part 0x12 `_SKIN_`: named material variants ("Default", "loot", "olive_plastic", ...).

Layout (see notes/FORMATS/mesh-variants.md for tiers and evidence; verified byte-exact on 25/25 available meshes):

    off   size   field
    0x00  u32    count             variant records
    0x04  u32    entries_rel       offset of the record array (8 on 25/25)
    0x08  32×N   records           VariantRecord below; every offset in a record is relative to the RECORD start
    ...          payload           pair arrays, remap arrays, NUL-terminated names, tail objects, ref arrays —
                                   in no fixed order, possibly shared between records, possibly with dead bytes
    ...          padding           zero bytes up to a 16-byte multiple

    VariantRecord (32 bytes)
    +0x00 u32   name_rel:20 | n_nib:4 | a_hi:8  name (0 = unnamed); n_nib raw (C: 0 on the 25 DLTB meshes, any
                                                 value on 56/33,728 DL2 parts); a_hi raw (C)
    +0x04 u32   flags                           0x10000000 on 25/25 meshes; 0x40000000 = modifier record (C name)
    +0x08 u32   tail_rel                        8-byte tail object (`00 00 00 ff 00 00 00 ff` on every record)
    +0x0C u32   tail_flags                      0 or 0x10000000 (C)
    +0x10 u32   pairs_rel:21 | c_hi:11          material pair array (0 = none); c_hi raw (C)
    +0x14 u32   d_lo:8 | remap_rel:20 | e_hi:4   remap array (0 = none); d_lo raw (0 on 25/25); e_hi raw (0 on the
                                                 25 DLTB meshes; 0x1 = bit 28 on DL2 records, cf. the 0x10000000 flags)
    +0x18 u32   refs_rel                        modifier reference array (0 = none)
    +0x1C u8    pair_count
    +0x1D u8    f1                              (C) 1 on every record of wn_pistol_b_b, 0 elsewhere
    +0x1E u8    remap_count
    +0x1F u8    ref_count

    pair  = {u16 slot, u16 material}   slot: material-table index used by submeshes; material: index into the FULL
                                       class-11 table (count .. capacity holds the variant-only materials)  (A)
    remap = {u8 key, u8 value, u16 hi}                                                          (C: meaning unknown)
    ref   = {u32 record:24 | hi:8, u32 raw}                                  index of a modifier record (A: structure)

`decode()` never raises; `encode()` re-serialises a decoded dict byte-identically (same layout; list lengths and
name lengths must not change) and raises ValueError for anything it cannot place without moving data.
"""

from __future__ import annotations

import struct
from typing import Any

HEADER_SIZE = 8
RECORD_SIZE = 0x20
PAIR_SIZE = 4
REMAP_SIZE = 4
REF_SIZE = 8
TAIL_SIZE = 8
MAX_RECORDS = 65535
MAX_NAME = 4096
FLAG_BASE = 0x10000000
FLAG_MODIFIER = 0x40000000
SCHEMA = "nightrunner.mesh.variants/1"

_HDR = struct.Struct("<2I")
_REC = struct.Struct("<7I4B")
_PAIR = struct.Struct("<2H")
_REMAP = struct.Struct("<2BH")
_REF = struct.Struct("<2I")


class _Bad(Exception):
    """Internal: structural violation (turned into an error dict)."""


def _cstr(data: bytes, off: int) -> bytes:
    end = data.find(b"\0", off, min(len(data), off + MAX_NAME))
    if end < 0:
        raise _Bad(f"unterminated name at 0x{off:X}")
    return data[off:end]


def _span(data: bytes, what: str, off: int, size: int) -> None:
    if off < HEADER_SIZE or off + size > len(data):
        raise _Bad(f"{what}: span 0x{off:X}+{size} outside the part ({len(data)} bytes)")


def _parse_record(data: bytes, base: int) -> dict:
    w0, flags, tail_rel, tail_flags, w4, w5, refs_rel, n_pairs, f1, n_remap, n_refs = _REC.unpack_from(data, base)
    return {
        "name_rel": w0 & 0xFFFFF, "n_nib": (w0 >> 20) & 0xF, "a_hi": w0 >> 24, "flags": flags, "tail_rel": tail_rel, "tail_flags": tail_flags,
        "pairs_rel": w4 & 0x1FFFFF, "c_hi": w4 >> 21, "d_lo": w5 & 0xFF, "remap_rel": (w5 >> 8) & 0xFFFFF, "e_hi": w5 >> 28,
        "refs_rel": refs_rel, "pair_count": n_pairs, "f1": f1, "remap_count": n_remap, "ref_count": n_refs,
    }


def _check_count_ptr(what: str, rel: int, n: int, notes: list[str], idx: int) -> None:
    if n and not rel:
        raise _Bad(f"record {idx}: {what} count {n} with a null offset")
    if rel and not n:
        notes.append(f"record {idx}: {what} offset set with count 0")


def material_table(model: Any) -> list[str]:
    """Full class-11 material table of a decoded Model, including the entries between `count` and `capacity`
    (variant-only materials that `Model.materials` does not list). Returns [] when unavailable."""
    try:
        from ..classreader.graph import MaterialEntryView
        img, hdr = model.image, model.material_header_offset
        if hdr is None:
            return []
        cap = img.u16(hdr + 0x0A)
        first = img.pointer(hdr).target
        if first is None:
            return []
        out = []
        for i in range(cap):
            try:
                nm = MaterialEntryView(img, first + i * 32).name
                out.append("" if nm is None else nm.decode("utf-8", "surrogateescape"))
            except Exception:  # noqa: BLE001 - defensive: one bad entry must not hide the rest
                out.append(f"material_{i}")
        return out
    except Exception:  # noqa: BLE001
        return []


def _submeshes_by_slot(model: Any) -> dict[int, list[list[int]]]:
    out: dict[int, list[list[int]]] = {}
    try:
        for g in model.geometry_entries:
            for s in g.submeshes:
                out.setdefault(int(s.material_slot), []).append([int(g.index), int(s.index)])
    except Exception:  # noqa: BLE001
        return {}
    return out


def variant_names(data: bytes | memoryview) -> list[str]:
    """Record names in record order ('' for an unnamed modifier record); [] when the part is unreadable."""
    try:
        d = bytes(data)
        count, rel = _HDR.unpack_from(d, 0)
        if count > MAX_RECORDS or rel < HEADER_SIZE or rel + count * RECORD_SIZE > len(d):
            return []
        out = []
        for k in range(count):
            base = rel + k * RECORD_SIZE
            nrel = struct.unpack_from("<I", d, base)[0] & 0xFFFFFF
            out.append(_cstr(d, base + nrel).decode("utf-8", "surrogateescape") if nrel else "")
        return out
    except Exception:  # noqa: BLE001
        return []


def decode(data: bytes | memoryview, model: Any = None) -> dict:
    """Decode part 0x12. With *model* (a nightrunner.mesh Model) material indices are resolved to names and each
    pair lists the (geometry entry, submesh) pairs drawn with that slot. Never raises: returns {"error": ...}."""
    try:
        return _decode(bytes(data), model)
    except _Bad as e:
        return {"error": str(e), "size": len(data) if data is not None else 0}
    except Exception as e:  # noqa: BLE001 - garbage input must never raise
        return {"error": f"{type(e).__name__}: {e}", "size": 0}


def _decode(d: bytes, model: Any) -> dict:
    size = len(d)
    if size < HEADER_SIZE:
        raise _Bad(f"part too small ({size} bytes)")
    count, rel = _HDR.unpack_from(d, 0)
    notes: list[str] = []
    if count > MAX_RECORDS:
        raise _Bad(f"record count {count} exceeds {MAX_RECORDS}")
    if rel < HEADER_SIZE or rel % 4:
        raise _Bad(f"record array offset {rel} invalid")
    if rel != HEADER_SIZE:
        notes.append(f"record array at {rel} (8 on every known mesh)")
    if count:
        _span(d, "record array", rel, count * RECORD_SIZE)
    used = bytearray(size)          # 1 = byte belongs to a referenced object
    used[0:HEADER_SIZE] = b"\1" * HEADER_SIZE
    used[rel:rel + count * RECORD_SIZE] = b"\1" * (count * RECORD_SIZE)
    mats = material_table(model) if model is not None else []
    base_count = len(getattr(model, "materials", []) or []) if model is not None else 0
    by_slot = _submeshes_by_slot(model) if model is not None else {}

    def mname(i: int) -> str | None:
        return mats[i] if 0 <= i < len(mats) else None

    variants = []
    for k in range(count):
        base = rel + k * RECORD_SIZE
        r = _parse_record(d, base)
        v: dict[str, Any] = {"index": k, "offset": base, "raw_hex": d[base:base + RECORD_SIZE].hex()}
        # name
        if r["name_rel"]:
            no = base + r["name_rel"]
            _span(d, f"record {k} name", no, 1)
            nb = _cstr(d, no)
            used[no:no + len(nb) + 1] = b"\1" * (len(nb) + 1)
            v["name"], v["name_offset"] = nb.decode("utf-8", "surrogateescape"), no
        else:
            v["name"], v["name_offset"] = "", None
        if not r["flags"] & FLAG_BASE or r["flags"] & ~(FLAG_BASE | FLAG_MODIFIER):
            notes.append(f"record {k}: unseen flags 0x{r['flags']:08X}")
        v["modifier"] = bool(r["flags"] & FLAG_MODIFIER)
        # tail object
        to = base + r["tail_rel"]
        if r["tail_rel"]:
            _span(d, f"record {k} tail", to, TAIL_SIZE)
            used[to:to + TAIL_SIZE] = b"\1" * TAIL_SIZE
            v["tail_hex"] = d[to:to + TAIL_SIZE].hex()
        else:
            v["tail_hex"] = None
            notes.append(f"record {k}: null tail object")
        # material pairs
        _check_count_ptr("pair", r["pairs_rel"], r["pair_count"], notes, k)
        mm = []
        if r["pair_count"]:
            po = base + r["pairs_rel"]
            _span(d, f"record {k} pairs", po, r["pair_count"] * PAIR_SIZE)
            used[po:po + r["pair_count"] * PAIR_SIZE] = b"\1" * (r["pair_count"] * PAIR_SIZE)
            for j in range(r["pair_count"]):
                slot, mat = _PAIR.unpack_from(d, po + j * PAIR_SIZE)
                e: dict[str, Any] = {"slot": slot, "material": mat}
                if model is not None:
                    e["slot_name"], e["material_name"] = mname(slot), mname(mat)
                    e["variant_only"] = mat >= base_count
                    e["submeshes"] = by_slot.get(slot, [])
                    if mats and mat >= len(mats):
                        notes.append(f"record {k}: material {mat} outside the table ({len(mats)})")
                mm.append(e)
        v["material_map"] = mm
        # remap items
        _check_count_ptr("remap", r["remap_rel"], r["remap_count"], notes, k)
        rm = []
        if r["remap_count"]:
            ro = base + r["remap_rel"]
            _span(d, f"record {k} remap", ro, r["remap_count"] * REMAP_SIZE)
            used[ro:ro + r["remap_count"] * REMAP_SIZE] = b"\1" * (r["remap_count"] * REMAP_SIZE)
            for j in range(r["remap_count"]):
                a, b, hi = _REMAP.unpack_from(d, ro + j * REMAP_SIZE)
                rm.append({"key": a, "value": b, "hi": hi})
        v["remap"] = rm
        # modifier references
        _check_count_ptr("ref", r["refs_rel"], r["ref_count"], notes, k)
        refs = []
        if r["ref_count"]:
            fo = base + r["refs_rel"]
            _span(d, f"record {k} refs", fo, r["ref_count"] * REF_SIZE)
            used[fo:fo + r["ref_count"] * REF_SIZE] = b"\1" * (r["ref_count"] * REF_SIZE)
            for j in range(r["ref_count"]):
                lo, hi = _REF.unpack_from(d, fo + j * REF_SIZE)
                idx = lo & 0xFFFFFF
                if idx >= count:
                    notes.append(f"record {k}: ref {j} to record {idx} out of range")
                refs.append({"record": idx, "lo_hi": lo >> 24, "raw": hi})
        v["refs"] = refs
        v["raw"] = r
        variants.append(v)
    # resolve reference names after all records are known
    for v in variants:
        for ref in v["refs"]:
            ref["name"] = variants[ref["record"]]["name"] if ref["record"] < count else None
    consumed = max(used.rfind(1) + 1, HEADER_SIZE)
    trailing = d[consumed:]
    gaps = []
    i = used.find(0, 0, consumed)
    while i >= 0:
        j = used.find(1, i, consumed)
        j = consumed if j < 0 else j
        gaps.append({"offset": i, "hex": d[i:j].hex()})
        i = used.find(0, j, consumed)
    pad_ok = not any(trailing) and len(trailing) < 16 and size % 16 == 0
    if not pad_ok:
        notes.append(f"trailing {len(trailing)} bytes after the last object are not zero padding to 16")
    if gaps:
        notes.append(f"{len(gaps)} unreferenced span(s), {sum(len(g['hex']) // 2 for g in gaps)} bytes "
                     "(kept verbatim; C: dead allocations)")
    return {
        "schema": SCHEMA, "count": count, "entries_rel": rel, "variants": variants,
        "material_table": mats, "material_count": base_count if model is not None else None,
        "unreferenced": gaps, "trailing_hex": trailing.hex(),
        "complete": pad_ok, "consumed": consumed, "size": size, "notes": notes,
    }


def encode(decoded: dict) -> bytes:
    """Re-serialise a dict produced by `decode()` into the same layout. Edits of pair/remap/ref values, names of
    equal or shorter length (which must not clash with other bytes) and raw fields are allowed; list lengths are
    taken from the lists and must fit the original spans. Raises ValueError when data would collide."""
    if not isinstance(decoded, dict) or "error" in decoded:
        raise ValueError("encode needs a successful decode() result")
    size = int(decoded["size"])
    buf = bytearray(size)
    owner: list[str | None] = [None] * size

    def put(off: int, blob: bytes, what: str) -> None:
        if off < 0 or off + len(blob) > size:
            raise ValueError(f"{what}: 0x{off:X}+{len(blob)} outside {size} bytes")
        for i, b in enumerate(blob):
            p = off + i
            if owner[p] is not None and buf[p] != b:
                raise ValueError(f"{what}: byte 0x{p:X} collides with {owner[p]}")
            buf[p] = b
            owner[p] = what

    variants = decoded["variants"]
    rel = int(decoded.get("entries_rel", HEADER_SIZE))
    put(0, _HDR.pack(len(variants), rel), "header")
    for k, v in enumerate(variants):
        r = v["raw"]
        base = rel + k * RECORD_SIZE
        counts = (len(v["material_map"]), len(v["remap"]), len(v["refs"]))
        for c, lim in zip(counts, ("pair", "remap", "ref")):
            if c > 255:
                raise ValueError(f"record {k}: {lim} count {c} > 255")
        for c, orig, key in zip(counts, (r["pair_count"], r["remap_count"], r["ref_count"]),
                                ("pairs_rel", "remap_rel", "refs_rel")):
            if c and not r[key]:
                raise ValueError(f"record {k}: cannot add a {key[:-4]} array to a record without one")
            if c > orig:
                raise ValueError(f"record {k}: {key[:-4]} array grows ({orig} → {c}); layout changes are unsupported")
        put(base, _REC.pack(
            (r["name_rel"] & 0xFFFFF) | ((r.get("n_nib", 0) & 0xF) << 20) | (r["a_hi"] << 24), r["flags"], r["tail_rel"], r["tail_flags"],
            (r["pairs_rel"] & 0x1FFFFF) | (r["c_hi"] << 21), (r["d_lo"] & 0xFF) | ((r["remap_rel"] & 0xFFFFF) << 8) | ((r.get("e_hi", 0) & 0xF) << 28),
            r["refs_rel"], counts[0], r["f1"], counts[1], counts[2]), f"record {k}")
        if r["name_rel"]:
            put(base + r["name_rel"], v["name"].encode("utf-8", "surrogateescape") + b"\0", f"record {k} name")
        if r["tail_rel"] and v.get("tail_hex") is not None:
            put(base + r["tail_rel"], bytes.fromhex(v["tail_hex"]), f"record {k} tail")
        if counts[0]:
            put(base + r["pairs_rel"], b"".join(_PAIR.pack(e["slot"], e["material"]) for e in v["material_map"]),
                f"record {k} pairs")
        if counts[1]:
            put(base + r["remap_rel"], b"".join(_REMAP.pack(e["key"], e["value"], e["hi"]) for e in v["remap"]),
                f"record {k} remap")
        if counts[2]:
            put(base + r["refs_rel"], b"".join(_REF.pack((e["record"] & 0xFFFFFF) | (e["lo_hi"] << 24), e["raw"])
                                               for e in v["refs"]), f"record {k} refs")
    for g in decoded.get("unreferenced", []):
        put(int(g["offset"]), bytes.fromhex(g["hex"]), "unreferenced")
    tail = bytes.fromhex(decoded.get("trailing_hex", ""))
    put(size - len(tail), tail, "trailing")
    return bytes(buf)
