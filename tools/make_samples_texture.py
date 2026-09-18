"""Build the texture unit-test samples: out/samples/textures.rpack (a small real RPACK assembled from shipped
resources with PackWriter) and out/samples/textures.rpx (its extracted tree), plus textures.samples.json listing
where every sample came from and why it was picked.

Selection (smallest bitmap that satisfies each criterion, scanned over every texture pack):
  * per observed IL format: one multi-mip texture and, where one exists, one single-mip texture
  * one cube, one volume of each observed volume format, one non-power-of-two multi-mip texture
  * one texture per observed flags value (0x64 / 0x44 / 0x04 / 0x54), two with non-zero mip_split
  * both header-only (flag 0x02) records

    python tools/make_samples_texture.py [--assets <dir>] [--out out/samples]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack, PackWriter, PartSource, ResourceSpec  # noqa: E402
from nightrunner.extract import extract  # noqa: E402
from nightrunner.texture.codec import part_ordinals  # noqa: E402
from nightrunner.texture.imgc import parse_header  # noqa: E402
from nightrunner.util.jsonio import dump_json  # noqa: E402
from tests.paths import ASSETS  # noqa: E402

PACKS = ["common_textures_0_pc.rpack", "engine_pc.rpack", "gui_common_pc.rpack", "dlc_frontier_pc.rpack",
         "dlc_frontier_envprobes_pc.rpack", "dlc_frontier_heightmaps_pc.rpack", "menu_level_ft_pc.rpack"]


def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


def scan(assets: Path) -> dict[str, tuple]:
    """criterion → (bitmap size, pack rel, index, name, description)."""
    best: dict[str, tuple] = {}

    def offer(key: str, size: int, rel: str, res, desc: str):
        cur = best.get(key)
        if cur is None or size < cur[0]:
            best[key] = (size, rel, res.index, res.name, desc)

    for rel in PACKS:
        p = assets / rel
        if not p.exists():
            continue
        with Pack.open(p) as pk:
            for res in pk.resources_of_type(0x20):
                hk, bk = part_ordinals(res.part_types)
                if hk is None:
                    continue
                h = parse_header(pk.read_part(res.part_indices[hk]))
                size = pk.physicals[res.part_indices[bk]].size if bk is not None else 0
                if h.header_only:
                    offer(f"header_only:{res.name}", 0, rel, res, "header-only record (flag 0x02)")
                    continue
                if h.mip_count == 0:
                    continue
                fname = h.format_name
                if h.is_cube:
                    offer("cube", size, rel, res, f"cube {fname}")
                elif h.is_volume:
                    offer(f"volume:{fname}", size, rel, res, f"volume {fname}")
                else:
                    if h.mip_count > 1:
                        offer(f"format_mips:{fname}", size, rel, res, f"{fname} multi-mip")
                    else:
                        offer(f"format_1mip:{fname}", size, rel, res, f"{fname} single mip")
                    if h.mip_count > 1 and not (_is_pow2(h.width) and _is_pow2(h.height)) and size >= 4096:
                        offer("npot", size, rel, res, "non-power-of-two multi-mip")
                offer(f"flags:0x{h.flags:02X}", size, rel, res, f"flags 0x{h.flags:02X}")
                if h.mip_split:
                    offer(f"mip_split:{h.mip_split:016X}", size, rel, res, "non-zero mip_split")
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default=str(ASSETS))
    ap.add_argument("--out", default=str(ROOT / "out" / "samples"))
    a = ap.parse_args()
    assets = Path(a.assets)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    best = scan(assets)
    # de-duplicate (pack, index) while keeping every criterion that picked it
    chosen: dict[tuple[str, int], dict] = {}
    for key, (size, rel, index, name, desc) in sorted(best.items()):
        c = chosen.setdefault((rel, index), {"pack": rel, "index": index, "name": name, "size": size, "reasons": []})
        c["reasons"].append(f"{key}: {desc}")
    picks = sorted(chosen.values(), key=lambda c: (c["pack"], c["index"]))
    writer = PackWriter(field08=0, flags=1, layout="grouped")
    by_pack: dict[str, list[dict]] = {}
    for c in picks:
        by_pack.setdefault(c["pack"], []).append(c)
    for rel, items in by_pack.items():
        with Pack.open(assets / rel) as pk:
            for c in items:
                spec = ResourceSpec.from_pack(pk, c["index"])
                # materialise the bytes now: the writer streams from (path, offset, size) otherwise
                for ps in spec.parts:
                    src = ps.source
                    with open(src.path, "rb") as fh:
                        fh.seek(src.offset)
                        data = fh.read(src.size)
                    ps.source = PartSource(data)
                writer.add(spec)
    rep = writer.write(out / "textures.rpack")
    print(f"wrote {rep.path} ({rep.size:,} bytes, {rep.logicals} textures, {rep.storages} storages)")
    with Pack.open(out / "textures.rpack") as pk:
        spec = extract(pk, out / "textures.rpx", progress=False)
    print(f"extracted {len(spec['resources'])} resources to {out / 'textures.rpx'} ({len(spec['warnings'])} warnings)")
    for w in spec["warnings"]:
        print("  warning:", w)
    dump_json({"assets": str(assets), "samples": picks}, out / "textures.samples.json")
    for c in picks:
        print(f"  [{c['index']:>6}] {c['pack']:36s} {c['size']:>10,}  {c['name']}  <- {'; '.join(c['reasons'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
