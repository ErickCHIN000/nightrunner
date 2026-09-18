"""Rebuild every given pack from its own parts with each layout policy and compare with the original bytes.

    python tools/rebuild_check.py PACK... [--out DIR] [--layouts auto,preserve]

Prints one line per (pack, layout): identical / first difference offset with a short diagnosis.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner.container.rp6l import Pack, PackWriter, ResourceSpec  # noqa: E402


def first_diff(a: bytes, b: bytes) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def rebuild(src: Path, dst: Path, layout: str) -> None:
    with Pack.open(src) as pk:
        w = PackWriter.from_pack(pk, layout)
        for r in pk:
            w.add(ResourceSpec.from_pack(pk, r.index))
        fill = bytes(pk._data) if layout == "preserve" else None
        w.write(dst, fill=fill, final_size=pk.size if layout == "preserve" else None)


def diagnose(pk: Pack, off: int) -> str:
    if off < 36:
        return "header"
    if off < pk.physical_offset:
        return f"storage[{(off - pk.storage_offset) // 20}] byte {(off - pk.storage_offset) % 20}"
    if off < pk.logical_offset:
        return f"physical[{(off - pk.physical_offset) // 16}] byte {(off - pk.physical_offset) % 16}"
    if off < pk.name_offset_offset:
        return f"logical[{(off - pk.logical_offset) // 12}]"
    if off < pk.table_end:
        return "name table"
    for i in range(len(pk.physicals)):
        o = pk.part_offset(i)
        if o <= off < o + pk.physicals[i].size:
            return f"payload part {i} (+{off - o})"
    return "gap/padding"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("packs", nargs="+")
    ap.add_argument("--out", default=None)
    ap.add_argument("--layouts", default="auto,preserve")
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(tempfile.mkdtemp(prefix="bp_rebuild_"))
    out.mkdir(parents=True, exist_ok=True)
    failures = 0
    for p in args.packs:
        src = Path(p)
        for layout in args.layouts.split(","):
            dst = out / f"{src.stem}.{layout}.rpack"
            try:
                rebuild(src, dst, layout)
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL  {src.name:45s} {layout:10s} exception: {exc}")
                failures += 1
                continue
            a = src.read_bytes()
            b = dst.read_bytes()
            d = first_diff(a, b)
            if d is None:
                print(f"OK    {src.name:45s} {layout:10s} identical ({len(a)} bytes)")
            else:
                with Pack.open(src) as pk:
                    where = diagnose(pk, d)
                print(f"DIFF  {src.name:45s} {layout:10s} first diff @0x{d:X} ({where}); sizes {len(a)} vs {len(b)}")
                failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
