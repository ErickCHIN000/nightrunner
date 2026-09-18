"""Pack → .rpx tree (pack.json + per-type folders)."""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

from . import __version__
from .codecs import ExtractContext, codec_for
from .container import catalogue
from .container.rp6l import Pack
from .util.hashing import sha256_bytes, sha256_file
from .util.jsonio import dump_json
from .util.names import resource_dirname

SCHEMA = "nightrunner.rpx/1"

PART_FILE_NAMES = {
    0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin", 0xF3: "cloth.bin",
    0x20: "header.imgc", 0x21: "bitmap.bin",
    0x61: "prefab.bin", 0x62: "prefab_fixups.bin",
    0x40: "anim.bin", 0x44: "anm2_header.bin", 0x45: "anm2_payload.bin",
    0x42: "animscr.bin", 0x43: "animscr_fixups.bin", 0x47: "animgraph.bin", 0x48: "animgraph_fixups.bin",
    0x49: "animcustom.bin", 0x4A: "animcustom_fixups.bin",
    0x55: "envprobe.bin", 0x56: "voxelizer.bin", 0x5A: "area.bin",
}


def part_filename(ordinal: int, type_id: int, seen: dict[str, int]) -> str:
    base = PART_FILE_NAMES.get(type_id, f"part_{type_id:02X}.bin")
    n = seen.get(base, 0)
    seen[base] = n + 1
    if n == 0:
        return base
    stem, ext = os.path.splitext(base)
    return f"{stem}.{n}{ext}"


def extract(pack: Pack, rpx_dir: Path | str, *, types: set[int] | None = None, no_raw: bool = False,
            options: dict | None = None, progress: bool = True, limit: int | None = None,
            indices: set[int] | None = None) -> dict:
    """Extract *pack* into *rpx_dir*. Returns the pack.json dict."""
    rpx_dir = Path(rpx_dir)
    partial = rpx_dir.with_name(rpx_dir.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    ctx = ExtractContext(pack, partial, dict(options or {}, no_raw=no_raw))
    t0 = time.time()
    resources = []
    selected = [r for r in pack if (types is None or r.type in types) and (indices is None or r.index in indices)]
    if limit is not None:
        selected = selected[:limit]
    total = len(selected)
    for n, r in enumerate(selected):
        family = catalogue.family_dir(r.type)
        rel_dir = f"{family}/{resource_dirname(r.index, r.name)}"
        res_dir = partial / rel_dir
        res_dir.mkdir(parents=True, exist_ok=True)
        parts = []
        seen: dict[str, int] = {}
        skipped: list[tuple[dict, int, str]] = []   # (--no-raw) part record, physical index, file name
        for k, i in enumerate(r.part_indices):
            p = pack.physicals[i]
            s = pack.storages[p.storage_index]
            rec = {
                "ordinal": k, "index": i, "type": f"0x{s.type:02X}", "type_name": catalogue.type_name(s.type),
                "storage_index": p.storage_index, "flag_bits": f"0x{p.flag_bits:04X}", "fc": f"0x{p.fc:08X}",
                "offset_units": p.offset_units, "offset": pack.part_offset(i), "size": p.size,
            }
            if pack.part_is_direct(i):
                data = pack.read_part(i)
                rec["sha256"] = sha256_bytes(data)
                fname = part_filename(k, s.type, seen)
                if not no_raw or r.type not in _RAW_OPTIONAL_TYPES:
                    with open(res_dir / fname, "wb") as fh:
                        fh.write(data)
                    rec["raw"] = f"{rel_dir}/{fname}"
                else:
                    rec["raw"] = None            # --no-raw: the codec regenerates this part from the editable file
                    rec["no_raw"] = True
                    skipped.append((rec, i, fname))
            else:
                rec["raw"] = None
                rec["unsupported"] = "child-pack or compressed storage"
                ctx.warn(f"{r.name!r}: part {i} not extractable (method {s.method}, child={p.child})")
            parts.append(rec)
        codec = codec_for(r.type)
        try:
            editable = codec.extract(ctx, r, res_dir)
        except Exception as exc:  # noqa: BLE001
            editable = {"kind": "raw", "files": {}, "error": f"{type(exc).__name__}: {exc}"}
            ctx.warn(f"{r.name!r}: codec {codec.kind} failed: {exc}")
        if editable.get("kind") == "raw" or "error" in editable:
            # --no-raw relies on a lossless editable; without one the raw parts are the only copy (review F5)
            for rec, i, fname in skipped:
                with open(res_dir / fname, "wb") as fh:
                    fh.write(pack.read_part(i))
                rec["raw"] = f"{rel_dir}/{fname}"
                del rec["no_raw"]
        resources.append({
            "index": r.index, "name": r.name, "name_hex": r.name_raw.hex(), "type": f"0x{r.type:02X}",
            "type_name": catalogue.type_name(r.type), "flags": f"0x{r.flags:02X}", "name_index": r.logical.name_index,
            "dir": rel_dir, "parts": parts, "editable": editable,
        })
        if progress and (n % 200 == 0 or n == total - 1):
            el = time.time() - t0
            print(f"\r  extracted {n + 1}/{total} ({el:.0f}s)", end="", file=sys.stderr, flush=True)
    if progress:
        print(file=sys.stderr)
    spec = {
        "schema": SCHEMA,
        "tool": f"nightrunner {__version__}",
        "source": {"path": str(pack.path), "size": pack.size, "sha256": sha256_file(pack.path) if pack.path.exists() else None},
        "header": pack.header.to_json(),
        "storages": [s.to_json() for s in pack.storages],
        "name_blob_order": pack.name_blob_order(),
        "complete": types is None and limit is None and indices is None,
        "resources": resources,
        "warnings": ctx.warnings,
    }
    dump_json(spec, partial / "pack.json")
    if rpx_dir.exists():
        shutil.rmtree(rpx_dir)
    os.replace(partial, rpx_dir)
    return spec


# Types whose editable intermediate is lossless on its own, so --no-raw may skip the raw copies.
_RAW_OPTIONAL_TYPES = {0x20}
