"""Animation (0x40) bodies: plain part [0x40] or the stream pair [0x44 ANM2Header, 0x45 ANM2Payload].

What is known (all tiers stated; see notes/FORMATS/types.md, section "0x40 Animation"):

* (A) compiled extension `.anm_obj`, memory category 5 (ResourceCore catalogue).
* (A, census 2026-09-15 — see `census()` output) the plain 0x40 part equals the concatenation of the 0x44 and
  0x45 parts of the same-named resource in the *_stream pack; the 0x44 part begins with ASCII `ANM2` and a
  fixed 0x20-byte word block; the 0x45 part length is 16 × the u32 at header offset 0x08.
* (C) field semantics: nothing in the current vault names any of these words for DLTB; the DL2 ANM2 reader is
  explicitly NOT to be applied by lineage (survey 04 §2.3). This module therefore reports words by offset and
  measured relations only.

Header word block dumped by `dump_header()` (offsets are bytes; names are *positions*, not meanings):

    +0x00 char[4]  'ANM2'
    +0x04 u16      w04       (42 in every sample)
    +0x06 u16      w06       (3 in every sample)
    +0x08 u32      w08       payload size / 16 (measured relation R3)
    +0x0C u32      w0C       0x100000xx (low byte varies)
    +0x10 u16      w10
    +0x12 u16      w12       == w14 (measured relation R2)
    +0x14 u16      w14
    +0x16 u16      w16
    +0x18 u16      n18       count of u32 words in list A at +0x20 (R3)
    +0x1A u16      w1A       (reappears as the first u16 after the payload's leading run — reported, not a claim)
    +0x1C u32      n1C       count of u32 words in list B following list A (R3)
    +0x20 u32[n18] list A
          u32[n1C] list B
          u16[w10] seg      — sum(seg) == w12 (R4)
          u16[3]   (w16, w14, w16) (R4)
          pad to 16
"""

from __future__ import annotations

import struct
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack, Resource
from .raw import align_up, counter_json, words

MAGIC = b"ANM2"
_HDR = struct.Struct("<4sHHIIHHHHHHI")   # 0x20 bytes
HEADER_WORDS = 0x20


def header_words(data) -> dict | None:
    if len(data) < HEADER_WORDS:
        return None
    magic, w04, w06, w08, w0C, w10, w12, w14, w16, n18, w1A, n1C = _HDR.unpack_from(data, 0)
    return {"magic": magic.decode("latin-1"), "w04": w04, "w06": w06, "w08": w08, "w0C": w0C, "w10": w10,
            "w12": w12, "w14": w14, "w16": w16, "n18": n18, "w1A": w1A, "n1C": n1C}


def header_layout(data) -> dict | None:
    """Measured header geometry: list A/B spans, trailer, computed header length (relations R2–R5)."""
    h = header_words(data)
    if h is None or h["magic"] != "ANM2":
        return None
    a_off = HEADER_WORDS
    a_end = a_off + 4 * h["n18"]
    b_end = a_end + 4 * h["n1C"]
    n_tr = h["w10"] + 3
    tr_end = b_end + 2 * n_tr
    hdr_len = align_up(tr_end, 16)
    n = len(data)
    lay = {"list_a": [a_off, a_end], "list_b": [a_end, b_end], "trailer": [b_end, tr_end], "header_len": hdr_len,
           "fits": tr_end <= n}
    if tr_end <= n:
        la = struct.unpack_from(f"<{h['n18']}I", data, a_off) if h["n18"] else ()
        lb = struct.unpack_from(f"<{h['n1C']}I", data, a_end) if h["n1C"] else ()
        tr = struct.unpack_from(f"<{n_tr}H", data, b_end)
        seg, tail3 = tr[: h["w10"]], tr[h["w10"]:]
        lay["list_a_head"] = [f"0x{x:08X}" for x in la[:12]]
        lay["list_a_all_small"] = all(x < 0x10000 for x in la) if la else None
        lay["list_a_sorted"] = all(la[i] < la[i + 1] for i in range(len(la) - 1)) if len(la) > 1 else None
        lay["list_b_head"] = [f"0x{x:08X}" for x in lb[:8]]
        lay["list_b_sorted"] = all(lb[i] < lb[i + 1] for i in range(len(lb) - 1)) if len(lb) > 1 else None
        lay["trailer_words"] = list(tr)
        lay["R2_w12_eq_w14"] = h["w12"] == h["w14"]
        lay["R4_seg_sum_eq_w12"] = sum(seg) == h["w12"]
        lay["R4_tail3_eq_w16_w14_w16"] = tuple(tail3) == (h["w16"], h["w14"], h["w16"])
        lay["pad_zero"] = not any(data[tr_end:hdr_len])
    return lay


def payload_lead(data) -> dict | None:
    """Leading u16 words of a 0x45 payload (or of the payload region of a plain body). Reports the run of
    non-zero u16 words after the first one and whether its last value equals header w08 (relation R6)."""
    if len(data) < 4:
        return None
    n16 = min(2048, len(data) // 2)
    ws = list(struct.unpack_from(f"<{n16}H", data, 0))
    run = []
    for x in ws[1:]:
        if x == 0:
            break
        run.append(x)
    after = align_up(2 + 2 * len(run) + 2, 16)     # first 16-byte boundary after the zero terminator
    return {"u16": ws[:16], "w0": ws[0], "run": run[:16], "run_len": len(run),
            "run_ascending": all(run[i] <= run[i + 1] for i in range(len(run) - 1)) if len(run) > 1 else None,
            "after_run_u16": ws[after // 2: after // 2 + 4] if after // 2 + 4 <= len(ws) else None}


def dump(parts: list[tuple[int, bytes]]) -> dict:
    """Structural dump for one logical 0x40 resource given its (type, bytes) parts."""
    types = [t for t, _ in parts]
    out = {"kind": "anim", "shape": [f"0x{t:02X}" for t in types], "sizes": [len(d) if d is not None else None for _, d in parts]}
    if types == [0x40] and parts[0][1] is not None:
        d = parts[0][1]
        h = header_words(d)
        lay = header_layout(d)
        out.update({"form": "plain", "header": h, "layout": lay, "words": words(d, 8)})
        if lay and lay["fits"]:
            hl = lay["header_len"]
            out["R3_size_eq_header_plus_16xw08"] = (hl + 16 * h["w08"] == len(d))
            out["payload"] = payload_lead(memoryview(d)[hl:])
            out["R6_payload_run_last_eq_w08"] = bool(out["payload"] and out["payload"]["run"] and out["payload"]["run"][-1] == h["w08"])
    elif types == [0x44, 0x45] and parts[0][1] is not None:
        hd, pd = parts[0][1], parts[1][1]
        h = header_words(hd)
        lay = header_layout(hd)
        out.update({"form": "stream_pair", "header": h, "layout": lay, "words": words(hd, 8)})
        if lay:
            out["R3_header_len_eq_part44"] = (lay["header_len"] == len(hd))
            out["R3_part45_eq_16xw08"] = (pd is not None and 16 * h["w08"] == len(pd))
        if pd is not None:
            out["payload"] = payload_lead(pd)
            out["R6_payload_run_last_eq_w08"] = bool(h and out["payload"] and out["payload"]["run"] and out["payload"]["run"][-1] == h["w08"])
    else:
        out["form"] = "unexpected"
        out["words"] = [words(d, 8) if d is not None else None for _, d in parts]
    return out


# ---- census ------------------------------------------------------------------------------------------------

class _Stats:
    def __init__(self):
        self.sizes: list[int] = []
        self.hdr_sizes: Counter = Counter()
        self.c = Counter()

    def size_summary(self) -> dict:
        s = sorted(self.sizes)
        if not s:
            return {}
        pct = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
        return {"count": len(s), "min": s[0], "p10": pct(0.10), "median": pct(0.5), "p90": pct(0.90), "max": s[-1],
                "mean": round(sum(s) / len(s), 1), "total": sum(s)}


def _tally(st: _Stats, parts, name: str) -> None:
    types = [t for t, _ in parts]
    total = sum(len(d) for _, d in parts if d is not None)
    st.sizes.append(total)
    st.c["resources"] += 1
    st.c[f"shape_{'_'.join(f'{t:02X}' for t in types)}"] += 1
    head = parts[0][1]
    if head is None:
        st.c["unreadable"] += 1
        return
    h = header_words(head)
    if h is None or h["magic"] != "ANM2":
        st.c["no_ANM2_magic"] += 1
        return
    st.c["ANM2_magic"] += 1
    st.c[f"w04={h['w04']}"] += 1
    st.c[f"w06={h['w06']}"] += 1
    st.c[f"w0C_hi24=0x{h['w0C'] >> 8:06X}"] += 1
    st.c[f"w10={h['w10']}"] += 1
    st.c[f"w16={h['w16']}"] += 1
    lay = header_layout(head)
    if not lay["fits"]:
        st.c["layout_does_not_fit"] += 1
        return
    st.c["R2_w12_eq_w14"] += lay["R2_w12_eq_w14"]
    st.c["R4_seg_sum_eq_w12"] += lay["R4_seg_sum_eq_w12"]
    st.c["R4_tail3_eq_w16_w14_w16"] += lay["R4_tail3_eq_w16_w14_w16"]
    st.c["pad_zero"] += lay["pad_zero"]
    if lay["list_b_sorted"] is not None:
        st.c["list_b_sorted"] += lay["list_b_sorted"]
        st.c["list_b_checked"] += 1
    if lay["list_a_all_small"] is not None:
        st.c["list_a_all_small"] += lay["list_a_all_small"]
        st.c["list_a_sorted"] += bool(lay["list_a_sorted"])
        st.c["list_a_checked"] += 1
    st.c[f"n18={h['n18']}"] += 1
    st.c[f"n1C_zero"] += (h["n1C"] == 0)
    if types == [0x40]:
        hl = lay["header_len"]
        st.hdr_sizes[hl] += 1
        st.c["R3_size_eq_header_plus_16xw08"] += (hl + 16 * h["w08"] == len(head))
        pl = payload_lead(memoryview(head)[hl:])
    else:
        hd, pd = parts[0][1], parts[1][1]
        st.hdr_sizes[len(hd)] += 1
        st.c["R3_header_len_eq_part44"] += (lay["header_len"] == len(hd))
        st.c["R3_part45_eq_16xw08"] += (pd is not None and 16 * h["w08"] == len(pd))
        pl = payload_lead(pd) if pd is not None else None
    if pl:
        st.c["R6_payload_run_last_eq_w08"] += bool(pl["run"] and pl["run"][-1] == h["w08"])
        st.c[f"R6_holds_for_w10={h['w10']}"] += bool(pl["run"] and pl["run"][-1] == h["w08"])
        st.c["R6_run_ascending"] += bool(pl["run_ascending"])
        st.c["R7_after_run_u16_eq_w1A"] += bool(pl["after_run_u16"] and pl["after_run_u16"][0] == h["w1A"])
        st.c[f"payload_w0={pl['w0']}"] += 1


def census_pack(pack: Pack, *, limit: int | None = None) -> dict:
    st = _Stats()
    n = 0
    for res in pack.resources_of_type(0x40):
        parts = [(pack.part_type(i), bytes(pack.read_part(i)) if pack.part_is_direct(i) else None) for i in res.part_indices]
        _tally(st, parts, res.name)
        n += 1
        if limit and n >= limit:
            break
    return {"pack": pack.path.name, "counts": dict(sorted(st.c.items())), "sizes": st.size_summary(),
            "header_sizes": counter_json(st.hdr_sizes, key=lambda kv: kv[0], limit=60)}


def compare_static_stream(static: Pack, stream: Pack, *, limit: int | None = None) -> dict:
    """Relation R1: for every 0x40 resource of *static* (plain form) whose name matches a 0x40 resource of
    *stream* (pair form) at the same logical index, plain == part44 + part45 byte for byte."""
    c = Counter()
    mismatches = []
    idx_stream = {}
    for r in stream.resources_of_type(0x40):
        idx_stream.setdefault(r.name_raw, []).append(r)
    n = 0
    for r in static.resources_of_type(0x40):
        cands = idx_stream.get(r.name_raw)
        if not cands:
            c["no_stream_twin"] += 1
            continue
        # duplicates: pair by order of appearance
        s = cands.pop(0)
        c["paired"] += 1
        plain = static.read_part(r.part_indices[0])
        st_types = s.part_types
        if st_types != (0x44, 0x45):
            c["twin_not_44_45"] += 1
            continue
        hd = stream.read_part(s.part_indices[0])
        pd = stream.read_part(s.part_indices[1])
        if len(plain) == len(hd) + len(pd) and plain[: len(hd)] == hd and plain[len(hd):] == pd:
            c["R1_plain_eq_44_concat_45"] += 1
        else:
            c["R1_mismatch"] += 1
            if len(mismatches) < 20:
                mismatches.append({"index": r.index, "name": r.name, "plain": len(plain), "p44": len(hd), "p45": len(pd)})
        n += 1
        if limit and n >= limit:
            break
    return {"static": static.path.name, "stream": stream.path.name, "counts": dict(sorted(c.items())), "mismatches": mismatches}


PACKS = ["common_anims_pc.rpack", "common_anims_stream_pc.rpack", "player_anims_pc.rpack",
         "player_anims_static_pc.rpack", "player_anims_stream_pc.rpack", "lang_speech_en_pc.rpack", "engine_pc.rpack"]
PAIRS = [("player_anims_static_pc.rpack", "player_anims_stream_pc.rpack"), ("common_anims_pc.rpack", "common_anims_stream_pc.rpack")]


def run_census(assets: Path, *, limit: int | None = None, progress=None) -> dict:
    out = {"family": "anim", "type": "0x40", "packs": [], "pairs": []}
    for name in PACKS:
        p = assets / name
        if not p.exists():
            continue
        if progress:
            progress(f"anim census {name}")
        with Pack.open(p) as pk:
            out["packs"].append(census_pack(pk, limit=limit))
    for a, b in PAIRS:
        pa, pb = assets / a, assets / b
        if pa.exists() and pb.exists():
            if progress:
                progress(f"anim static/stream compare {a} vs {b}")
            with Pack.open(pa) as ps, Pack.open(pb) as pt:
                out["pairs"].append(compare_static_stream(ps, pt, limit=limit))
    return out
