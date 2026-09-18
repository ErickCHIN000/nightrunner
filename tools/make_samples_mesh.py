"""Build the mesh sample set used by tests/test_classreader.py, tests/test_mesh_decode.py, tests/test_cast_export.py.

    python tools/make_samples_mesh.py            # needs the game install (tests/paths.py)

Writes
    out/samples/meshes.rpack   a small RP6L pack (field08 0x1000, contiguous layout) with the resources below,
                               parts cloned byte-for-byte from the shipped packs
    out/samples/meshes.rpx     `nr extract` of that pack (mesh codec: model.cast + mesh.json + raw parts)
    out/samples/meshes.json    the sample list (source pack, index, name, why it was picked)

The picks cover every vertex format (0/3/6/8), a multi-entry (LOD) hair/beard mesh, a cloth mesh (0xF3), a
skeleton-only mesh (0x10+0x12+0x11), rigged player/NPC meshes, static props, the two meshes with non-finite
vertex data, and a method-0 mesh from engine_pc.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack, PackWriter, ResourceSpec  # noqa: E402
from tests.paths import ASSETS, SAMPLES, have_game  # noqa: E402

# (pack file, logical name, reason)
PICKS = [
    ("common_meshes_pc.rpack", " npc_b_man_pants_b_holster_bag_c", "format 6, leading-space name"),
    ("common_meshes_pc.rpack", " prp_hotel_number_10_a", "format 3 static, leading-space name"),
    ("common_meshes_pc.rpack", "dlc_ft_safe_zone_cable_e", "format 0 (f16 positions)"),
    ("common_meshes_pc.rpack", "dummybox_025m", "format 0"),
    ("common_meshes_pc.rpack", "sh2_npc_aiden_beast", "format 8 (80-byte vertices), rigged NPC"),
    ("common_meshes_pc.rpack", "sh2_npc_crane", "format 8 rigged NPC"),
    ("common_meshes_pc.rpack", "sh_npc_ft_crane_hair_a", "multi-entry geometry (LOD chain)"),
    ("common_meshes_pc.rpack", "sh2_npc_ft_crane_beard_a", "multi-entry geometry (LOD chain)"),
    ("common_meshes_pc.rpack", "dlc_ft_freak_banshee_clothes_matriarch", "cloth part 0xF3"),
    ("common_meshes_pc.rpack", "man_basic_skeleton", "skeleton-only (no vertex/index parts)"),
    ("common_meshes_pc.rpack", "sh2_player_tpp_phx_skeleton", "skeleton-only player skeleton"),
    ("common_meshes_pc.rpack", "anim_hammer_a", "2-entry animated prop"),
    ("common_meshes_pc.rpack", "bdp_ce_a_ornament_str_d", "2-entry static prop"),
    ("common_meshes_pc.rpack", "alarm_siren_anm", "format 6 animated prop"),
    ("common_meshes_pc.rpack", "gas_tank_pistol_anm", "non-finite vertex data (#6184)"),
    ("common_meshes_pc.rpack", "wn_pistol_b_b", "non-finite vertex data (#10264)"),
    ("engine_pc.rpack", None, "first mesh of engine_pc (method 0 / field08 0)"),
]


def main() -> int:
    if not have_game():
        print("game not found; nothing to do", file=sys.stderr)
        return 1
    SAMPLES.mkdir(parents=True, exist_ok=True)
    out_pack = SAMPLES / "meshes.rpack"
    manifest = []
    writer = PackWriter(field08=0x1000, flags=1, layout="contiguous")
    packs: dict[str, Pack] = {}
    try:
        for pack_name, name, why in PICKS:
            pk = packs.get(pack_name)
            if pk is None:
                pk = packs[pack_name] = Pack.open(ASSETS / pack_name)
            if name is None:
                idx = next(r.index for r in pk.resources_of_type(0x10))
            else:
                hits = pk.find(name, type_id=0x10, fold=False)
                if not hits:
                    print(f"  missing: {name!r} in {pack_name}", file=sys.stderr)
                    continue
                idx = hits[0]
            r = pk.resource(idx)
            spec = ResourceSpec.from_pack(pk, idx)
            # the sample pack is a bit-12 pack: every mesh part must sit in a method-1 stream storage
            for p in spec.parts:
                p.storage_flags = (p.storage_flags & ~0x3) | 0x1 | 0x8
            spec.name_index = None
            writer.add(spec)
            manifest.append({"pack": pack_name, "index": idx, "name": r.name, "why": why,
                             "parts": [f"0x{t:02X}" for t in r.part_types]})
            print(f"  {pack_name:24s} #{idx:<6d} {r.name!r:48s} {why}")
        rep = writer.write(out_pack)
    finally:
        for pk in packs.values():
            pk.close()
    with open(SAMPLES / "meshes.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, ensure_ascii=False)
    print(f"wrote {out_pack} ({rep.size:,} bytes, {rep.logicals} resources)")

    # extract with whatever codecs are registered (mesh codec when available)
    from nightrunner.extract import extract
    with Pack.open(out_pack) as pk:
        spec = extract(pk, SAMPLES / "meshes.rpx", progress=False)
    print(f"extracted {len(spec['resources'])} resources to {SAMPLES / 'meshes.rpx'} ({len(spec['warnings'])} warnings)")
    for w in spec["warnings"][:10]:
        print("  warning:", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
