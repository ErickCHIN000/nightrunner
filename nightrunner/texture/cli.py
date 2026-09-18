"""`nr texture …` — info / export / census / ddsinfo.

    nr texture info    <rpack> <index|name>                      header, levels, DDS mapping
    nr texture export  <rpack> <index|name> <out.dds|out.png> [--mip N] [--face N] [--slice N] [--force]
    nr texture ddsinfo <file.dds>                                parse a DDS the way the importer will
    nr texture census  <assets dir> --out <dir> [--packs a,b,…]  corpus statistics over every texture pack
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from ..container.rp6l import Pack
from ..errors import NightrunnerError, FormatError, UnsupportedError
from ..util.jsonio import dump_json
from . import formats
from .dds import dds_header_for, iter_dds, read_dds
from .imgc import LEVEL_PADDING_STOCK, LEVEL_PADDING_TIGHT, check_payload, detect_level_padding, find_level, level_layout, mip_dims, parse_header
from .codec import part_ordinals

# every shipped pack that contains type-0x20 resources (census 2026-09-15) + the two custom packs
TEXTURE_PACKS = [
    "common_textures_0_pc.rpack", "gui_common_pc.rpack", "gui_hud_pc.rpack", "gui_ingame_menu_pc.rpack",
    "gui_main_menu_pc.rpack", "gui_menu_common_pc.rpack",
    "dlc_frontier_heightmaps_pc.rpack", "dlc_ft_prologue_heightmaps_pc.rpack", "menu_level_ft_heightmaps_pc.rpack",
    "dlc_frontier_envprobes_pc.rpack", "dlc_ft_prologue_envprobes_pc.rpack", "menu_level_ft_envprobes_pc.rpack",
    "dlc_frontier_pc.rpack", "dlc_ft_prologue_pc.rpack", "menu_level_ft_pc.rpack", "engine_pc.rpack",
    "assets_2_pc.rpack", "custom_rpacks/rw.rpack", "custom_rpacks/frank.rpack",
]


def resolve(pk: Pack, key: str) -> int:
    if key.isdigit():
        i = int(key)
        if i >= len(pk):
            raise NightrunnerError(f"index {i} out of range ({len(pk)} resources)")
        return i
    hits = pk.find(key, 0x20)
    if not hits:
        raise NightrunnerError(f"no texture named {key!r} in {pk.path.name}")
    if len(hits) > 1:
        raise NightrunnerError(f"{len(hits)} textures named {key!r}: indices {hits}; pass an index")
    return hits[0]


def _load(pk: Pack, index: int):
    """(resource, header, physical index of the bitmap part or None). The bitmap view itself is opened by the
    caller inside `_view` so it is released before the mmap closes."""
    res = pk.resource(index)
    if res.type != 0x20:
        raise NightrunnerError(f"resource {index} {res.name!r} is type 0x{res.type:02X}, not a texture")
    hdr_k, bmp_k = part_ordinals(res.part_types)
    if hdr_k is None:
        raise FormatError(f"{res.name!r}: no 0x20 part")
    h = parse_header(pk.read_part(res.part_indices[hdr_k]))
    bmp_i = None if (bmp_k is None or h.header_only) else res.part_indices[bmp_k]
    return res, h, bmp_i


class _view:
    """Context manager around pk.read_part(i) that releases the memoryview (mmap refuses to close otherwise)."""

    def __init__(self, pk: Pack, index: int | None):
        self.pk, self.index, self.mv = pk, index, None

    def __enter__(self):
        if self.index is not None:
            self.mv = self.pk.read_part(self.index)
        return self.mv

    def __exit__(self, *exc):
        if self.mv is not None:
            self.mv.release()
            self.mv = None


# ---- info ----------------------------------------------------------------------------------------------------

def cmd_info(a) -> int:
    with Pack.open(a.pack) as pk:
        res, h, bmp_i = _load(pk, resolve(pk, a.key))
        out = {"index": res.index, "name": res.name, "parts": [f"0x{t:02X}" for t in res.part_types],
               "imgc": h.to_json()}
        f = formats.FORMATS.get(h.format)
        out["format"] = {"il": h.format, "name": h.format_name, "dxgi": f.dxgi if f else None,
                         "dxgi_name": f.dxgi_name if f else None, "unit": f.unit if f else None,
                         "block": f.block if f else None, "tier": f.tier if f else None}
        if bmp_i is not None:
            size = pk.physicals[bmp_i].size
            padding = detect_level_padding(h, size)
            levels = check_payload(h, size, padding)
            out["payload_size"] = size
            out["level_padding"] = padding
            out["levels"] = [{"mip": lv.mip, "face": lv.face, "w": lv.width, "h": lv.height, "d": lv.depth,
                              "offset": lv.offset, "size": lv.size, "padded": lv.padded_size} for lv in levels]
            out["dds_size"] = 148 + sum(lv.size for lv in levels)
        if a.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"[{res.index}] {res.name}  {h.width}x{h.height}x{h.depth} {h.type_name} {h.format_name} (IL {h.format}, "
                  f"DXGI {out['format']['dxgi']}) mips={h.mip_count} flags=0x{h.flags:02X} header_size={h.header_size} "
                  f"mip_split=0x{h.mip_split:X} reserved={h.reserved}")
            print(f"  stats min={h.minimum} max={h.maximum} mean={h.mean}")
            if h.header_only:
                print(f"  header-only (flag 0x02); reference = {h.reference!r}")
            else:
                print(f"  payload {out['payload_size']:,} bytes, {len(out['levels'])} levels, level_padding {out['level_padding']}"
                      f"{'' if out['level_padding'] == LEVEL_PADDING_STOCK else ' (third-party tight layout; stock = 16)'}")
                for lv in out["levels"][: a.levels]:
                    print(f"    mip {lv['mip']:2d} face {lv['face']} {lv['w']:5d}x{lv['h']:<5d}x{lv['d']:<3d} @ {lv['offset']:<10d} {lv['size']:>10,} (+{lv['padded'] - lv['size']})")
    return 0


# ---- export --------------------------------------------------------------------------------------------------

def _write_dds(out: Path, h, bitmap) -> int:
    """Stream the DDS to *out*; a separate function so every mmap slice is dropped before the pack closes."""
    n = 0
    with open(out, "wb") as fh:
        for chunk in iter_dds(h, bitmap):
            fh.write(chunk)
            n += len(chunk)
    return n


def cmd_export(a) -> int:
    out = Path(a.out)
    if out.exists() and not a.force:
        raise NightrunnerError(f"{out} exists (use --force)")
    with Pack.open(a.pack) as pk:
        res, h, bmp_i = _load(pk, resolve(pk, a.key))
        if bmp_i is None:
            raise UnsupportedError(f"{res.name!r} is a header-only record (reference {h.reference!r}); nothing to export")
        suffix = out.suffix.lower()
        if suffix not in (".dds", ".png"):
            raise NightrunnerError("export target must end in .dds (lossless) or .png (8-bit preview)")
        with _view(pk, bmp_i) as bitmap:
            if suffix == ".dds":
                out.parent.mkdir(parents=True, exist_ok=True)
                n = _write_dds(out, h, bitmap)
                print(f"wrote {out} ({n:,} bytes, {h.format_name} {h.width}x{h.height}x{h.depth} {h.type_name} mips={h.mip_count})")
                return 0
            from .png import PREVIEW_NOTE, decode_preview, png_bytes
            levels = check_payload(h, len(bitmap))
            lv = find_level(levels, a.mip, a.face)
            if not 0 <= a.slice < lv.depth:
                raise NightrunnerError(f"slice {a.slice} outside depth {lv.depth}")
            data = bytes(bitmap[lv.offset + a.slice * lv.slice_size: lv.offset + (a.slice + 1) * lv.slice_size])
            rgba = decode_preview(data, lv.width, lv.height, h.format)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(png_bytes(rgba))
        print(f"wrote {out} ({lv.width}x{lv.height}, mip {a.mip} face {a.face} slice {a.slice}; {PREVIEW_NOTE})")
    return 0


# ---- ddsinfo -------------------------------------------------------------------------------------------------

def cmd_ddsinfo(a) -> int:
    buf = Path(a.file).read_bytes()
    d = read_dds(buf, srgb_to_linear=a.srgb_to_linear, allow_unobserved=a.allow_unobserved)
    print(json.dumps(d.to_json(), indent=2))
    return 0


# ---- census --------------------------------------------------------------------------------------------------

def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def _r(x: float) -> float:
    return round(x, 4)


def census_pack(pk: Pack) -> dict:
    c = {
        "file": pk.path.name, "textures": 0, "formats": Counter(), "flags": Counter(), "flags_by_format": Counter(),
        "header_size": Counter(), "version": Counter(), "extension": Counter(), "tail_nonzero": 0,
        "mip_split": Counter(), "reserved": Counter(), "types": Counter(), "mip_count": Counter(),
        "chain": Counter(), "part_shapes": Counter(), "npot": 0, "depth_gt1": 0,
        "min_w": None, "max_w": 0, "min_h": None, "max_h": 0, "max_d": 0, "min_bitmap": None, "max_bitmap": 0,
        "total_bitmap": 0, "stats_pattern_by_format": {}, "stats_range_by_format": {},
        "payload_mismatch": [], "tight_layout": [], "header_only": [], "errors": [], "unmapped_formats": Counter(),
        "storage_flags": Counter(), "stats_convention_by_format": {}, "flag10": 0, "flag10_and_split": 0,
        "mip_split_records": [],
    }
    for s in pk.storages:
        if s.type in (0x20, 0x21):
            c["storage_flags"][f"0x{s.type:02X}:flags=0x{s.flags:02X}:meta=0x{s.metadata:02X}:align={s.alignment}"] += 1
    for res in pk.resources_of_type(0x20):
        c["textures"] += 1
        pts = res.part_types
        c["part_shapes"][",".join(f"0x{t:02X}" for t in pts)] += 1
        hdr_k, bmp_k = part_ordinals(pts)
        if hdr_k is None:
            c["errors"].append({"index": res.index, "name": res.name, "error": "no 0x20 part"})
            continue
        try:
            raw = pk.read_part(res.part_indices[hdr_k])
            h = parse_header(raw)
        except NightrunnerError as exc:
            c["errors"].append({"index": res.index, "name": res.name, "error": str(exc)})
            continue
        fname = h.format_name
        c["formats"][f"{h.format}:{fname}"] += 1
        c["flags"][f"0x{h.flags:02X}"] += 1
        c["flags_by_format"][f"{fname}|0x{h.flags:02X}"] += 1
        c["header_size"][h.header_size] += 1
        c["version"][f"0x{h.version:08X}"] += 1
        c["extension"][h.extension.hex() if h.extension else ""] += 1
        if any(h.tail):
            c["tail_nonzero"] += 1
        c["mip_split"][f"0x{h.mip_split:X}"] += 1
        c["reserved"][h.reserved] += 1
        c["types"][h.type_name] += 1
        c["mip_count"][h.mip_count] += 1
        if not h.header_only:
            full = max(h.width, h.height, h.depth if h.is_volume else 1).bit_length()
            c["chain"]["full" if h.mip_count == full else ("single" if h.mip_count == 1 else "partial")] += 1
        if not (_is_pow2(h.width) and _is_pow2(h.height)):
            c["npot"] += 1
        if h.depth > 1:
            c["depth_gt1"] += 1
        c["min_w"] = h.width if c["min_w"] is None else min(c["min_w"], h.width)
        c["min_h"] = h.height if c["min_h"] is None else min(c["min_h"], h.height)
        c["max_w"], c["max_h"], c["max_d"] = max(c["max_w"], h.width), max(c["max_h"], h.height), max(c["max_d"], h.depth)
        if h.format not in formats.FORMATS:
            c["unmapped_formats"][h.format] += 1
        # statistics conventions: what do channels beyond the format's channel count hold?
        f = formats.FORMATS.get(h.format)
        nch = len(f.channels) if f else 4
        mn, mx, me = h.minimum, h.maximum, h.mean
        pat = c["stats_pattern_by_format"].setdefault(fname, Counter())
        if nch < 4:
            pat[f"min{tuple(_r(x) for x in mn[nch:])} max{tuple(_r(x) for x in mx[nch:])} mean{tuple(_r(x) for x in me[nch:])}"] += 1
        else:
            pat[f"alpha min={_r(mn[3])} max={_r(mx[3])} mean={_r(me[3])}"] += 1
        rng = c["stats_range_by_format"].setdefault(fname, {"min": None, "max": None})
        lo, hi = min(mn[:nch] + me[:nch]), max(mx[:nch] + me[:nch])
        rng["min"] = lo if rng["min"] is None else min(rng["min"], lo)
        rng["max"] = hi if rng["max"] is None else max(rng["max"], hi)
        conv = c["stats_convention_by_format"].setdefault(fname, Counter())
        conv["textures"] += 1
        if all(x == 0.0 for x in mn[nch:] + mx[nch:]):
            conv["minmax_unused_channels_zero"] += 1
        if all(x == 0.0 for x in me):
            conv["vec3_all_zero"] += 1
        if all(0.0 <= x <= 1.0 for x in me):
            conv["vec3_within_0_1"] += 1
        lo_s = -1.0 if (f and f.kind == "snorm") or h.format in (62, 64, 67) else 0.0
        if all(lo_s <= x <= 1.0 for x in mn[:nch] + mx[:nch]):
            conv["minmax_within_stored_range"] += 1
        if h.tex_type != 0:
            conv["non2d"] += 1
            if all(x == 0.0 for x in me):
                conv["non2d_vec3_zero"] += 1
        # flag bit 0x10 ↔ mip_split (census 2026-09-15: 17/17 both ways); decode the split arithmetic
        if h.flags & 0x10:
            c["flag10"] += 1
        if h.mip_split:
            lo32, hi32 = h.mip_split & 0xFFFFFFFF, h.mip_split >> 32
            n = hi32 >> 28                       # top nibble = number of mips in the first group
            rest = hi32 & 0x0FFFFFFF
            tight = [formats.level_bytes(h.format, *mip_dims(h.width, h.height, h.depth, m)) * h.faces
                     for m in range(h.mip_count)] if h.format in formats.FORMATS and h.mip_count else []
            ok = bool(tight) and n <= len(tight) and sum(tight[:n]) == lo32 and sum(tight[n:]) == rest
            c["mip_split_records"].append({"index": res.index, "name": res.name, "format": fname, "w": h.width,
                                           "h": h.height, "mips": h.mip_count, "flags": f"0x{h.flags:02X}",
                                           "mip_split": f"0x{h.mip_split:016X}", "first_group_mips": n,
                                           "first_group_bytes": lo32, "rest_bytes": rest, "arithmetic_matches": ok})
            if h.flags & 0x10:
                c["flag10_and_split"] += 1
        if h.header_only:
            c["header_only"].append({"index": res.index, "name": res.name, "header_size": h.header_size,
                                     "stored": len(raw), "reference": h.reference, "flags": f"0x{h.flags:02X}",
                                     "dims": [h.width, h.height, h.depth], "format": fname, "mips": h.mip_count,
                                     "has_bitmap_part": bmp_k is not None})
            continue
        if bmp_k is None:
            c["errors"].append({"index": res.index, "name": res.name, "error": "no 0x21 part and flag 0x02 clear"})
            continue
        size = pk.physicals[res.part_indices[bmp_k]].size
        try:
            levels = level_layout(h)
            total = levels[-1].offset + levels[-1].padded_size
        except NightrunnerError as exc:
            c["errors"].append({"index": res.index, "name": res.name, "error": str(exc)})
            continue
        if total != size:
            rec = {"index": res.index, "name": res.name, "computed": total, "part": size}
            if sum(lv.size for lv in levels) == size:
                # levels back-to-back without the 16-byte stride (custom_rpacks/frank.rpack, 12 textures): decodable
                # with level_padding 0 (imgc.detect_level_padding); still listed here because it is not stock
                rec["level_padding"] = LEVEL_PADDING_TIGHT
                c["tight_layout"].append(rec)
            c["payload_mismatch"].append(rec)
        c["min_bitmap"] = size if c["min_bitmap"] is None else min(c["min_bitmap"], size)
        c["max_bitmap"] = max(c["max_bitmap"], size)
        c["total_bitmap"] += size
    # Counters → sorted dicts; cap the stats-pattern histograms
    for k in ("formats", "flags", "flags_by_format", "header_size", "version", "extension", "mip_split", "reserved",
              "types", "mip_count", "chain", "part_shapes", "unmapped_formats", "storage_flags"):
        c[k] = dict(sorted(c[k].items(), key=lambda kv: (-kv[1], str(kv[0]))))
    for fname, conv in c["stats_convention_by_format"].items():
        c["stats_convention_by_format"][fname] = dict(conv)
    for fname, pat in c["stats_pattern_by_format"].items():
        top = pat.most_common(8)
        rest = sum(pat.values()) - sum(n for _, n in top)
        c["stats_pattern_by_format"][fname] = dict(top, **({"other": rest} if rest else {}))
    return c


def _merge(total: dict, c: dict) -> None:
    for k in ("formats", "flags", "flags_by_format", "header_size", "version", "extension", "mip_split", "reserved",
              "types", "mip_count", "chain", "part_shapes", "unmapped_formats", "storage_flags"):
        t = total.setdefault(k, Counter())
        for kk, v in c[k].items():
            t[kk] += v
    for k in ("textures", "npot", "depth_gt1", "tail_nonzero", "total_bitmap", "flag10", "flag10_and_split"):
        total[k] = total.get(k, 0) + c[k]
    sc = total.setdefault("stats_convention_by_format", {})
    for fname, conv in c["stats_convention_by_format"].items():
        d = sc.setdefault(fname, Counter())
        for kk, v in conv.items():
            d[kk] += v
    total["max_w"] = max(total.get("max_w", 0), c["max_w"])
    total["max_h"] = max(total.get("max_h", 0), c["max_h"])
    total["max_d"] = max(total.get("max_d", 0), c["max_d"])
    total["max_bitmap"] = max(total.get("max_bitmap", 0), c["max_bitmap"])
    for k in ("min_w", "min_h", "min_bitmap"):
        if c[k] is not None:
            total[k] = c[k] if total.get(k) is None else min(total[k], c[k])
    for k in ("payload_mismatch", "tight_layout", "header_only", "errors", "mip_split_records"):
        total.setdefault(k, []).extend(dict(x, pack=c["file"]) for x in c[k])
    sp = total.setdefault("stats_pattern_by_format", {})
    for fname, pat in c["stats_pattern_by_format"].items():
        d = sp.setdefault(fname, Counter())
        for kk, v in pat.items():
            d[kk] += v
    sr = total.setdefault("stats_range_by_format", {})
    for fname, r in c["stats_range_by_format"].items():
        d = sr.setdefault(fname, {"min": None, "max": None})
        d["min"] = r["min"] if d["min"] is None else min(d["min"], r["min"])
        d["max"] = r["max"] if d["max"] is None else max(d["max"], r["max"])


def cmd_census(a) -> int:
    assets = Path(a.assets)
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    packs = a.packs.split(",") if a.packs else TEXTURE_PACKS
    total: dict = {"packs": []}
    t_all = time.time()
    for rel in packs:
        p = assets / rel
        if not p.exists():
            print(f"  missing: {p}", file=sys.stderr)
            continue
        t0 = time.time()
        with Pack.open(p) as pk:
            c = census_pack(pk)
        c["seconds"] = round(time.time() - t0, 2)
        dump_json(c, out_dir / f"census_{p.stem}.json")
        total["packs"].append({"file": p.name, "textures": c["textures"], "types": c["types"], "seconds": c["seconds"],
                               "payload_mismatch": len(c["payload_mismatch"]), "errors": len(c["errors"])})
        _merge(total, c)
        print(f"{p.name:42s} {c['textures']:>7} textures  {c['seconds']:6.1f}s  types={c['types']} "
              f"mismatch={len(c['payload_mismatch'])} errors={len(c['errors'])}")
    for k, v in list(total.items()):
        if isinstance(v, Counter):
            total[k] = dict(sorted(v.items(), key=lambda kv: (-kv[1], str(kv[0]))))
    for fname, pat in total.get("stats_pattern_by_format", {}).items():
        total["stats_pattern_by_format"][fname] = dict(sorted(pat.items(), key=lambda kv: -kv[1]))
    for fname, conv in total.get("stats_convention_by_format", {}).items():
        total["stats_convention_by_format"][fname] = dict(conv)
    total["seconds"] = round(time.time() - t_all, 1)
    dump_json(total, out_dir / "census_summary.json")
    print(f"total {total.get('textures', 0):,} textures in {len(total['packs'])} packs, {total['seconds']}s -> {out_dir / 'census_summary.json'}")
    return 0


# ---- parser --------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="nr texture")
    sub = ap.add_subparsers(dest="sub", required=True)
    p = sub.add_parser("info")
    p.add_argument("pack")
    p.add_argument("key", help="logical index or exact name")
    p.add_argument("--json", action="store_true")
    p.add_argument("--levels", type=int, default=16, help="levels to print")
    p.set_defaults(fn=cmd_info)
    p = sub.add_parser("export")
    p.add_argument("pack")
    p.add_argument("key")
    p.add_argument("out")
    p.add_argument("--mip", type=int, default=0)
    p.add_argument("--face", type=int, default=0)
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_export)
    p = sub.add_parser("ddsinfo")
    p.add_argument("file")
    p.add_argument("--srgb-to-linear", action="store_true")
    p.add_argument("--allow-unobserved", action="store_true")
    p.set_defaults(fn=cmd_ddsinfo)
    p = sub.add_parser("census")
    p.add_argument("assets")
    p.add_argument("--out", required=True)
    p.add_argument("--packs", help="comma list of pack paths relative to the assets dir (default: every texture pack)")
    p.set_defaults(fn=cmd_census)
    return ap


def run(args) -> int:
    a = build_parser().parse_args(args.args)
    return a.fn(a)
