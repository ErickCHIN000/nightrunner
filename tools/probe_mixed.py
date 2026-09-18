"""Probe the open questions from the corpus census (mixed bit-12 packs, u16 wraps, storage order, logical flags)."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "F:/DLTB")
from nightrunner.container.rp6l import Pack  # noqa: E402

A = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Dying Light The Beast\ph_ft\work\data_platform\pc\assets")

print("### dlc_ft_prologue_pc: logical order vs file order")
with Pack.open(A / "dlc_ft_prologue_pc.rpack") as pk:
    for r in pk:
        print(f"  [{r.index:3d}] 0x{r.type:02X} flags=0x{r.flags:02X} parts={[f'0x{t:02X}' for t in r.part_types]} "
              f"offs={[pk.part_offset(i) for i in r.part_indices]} sizes={[pk.physicals[i].size for i in r.part_indices]} {r.name!r}")
    print("  storages:", [(f"0x{s.type:02X}", s.base_units, s.count) for s in pk.storages])
    print("  name blob order:", pk.name_blob_order())

print("### envprobes: owner wrap / storage count wrap")
with Pack.open(A / "dlc_frontier_envprobes_pc.rpack") as pk:
    bad = 0
    wrap_ok = 0
    for r in pk:
        for i in r.part_indices:
            o = pk.physicals[i].owner
            if o != r.index:
                bad += 1
                if o == (r.index & 0xFFFF):
                    wrap_ok += 1
    print(f"  owner != index: {bad}; of which == index & 0xFFFF: {wrap_ok}; logicals={len(pk)}")
    print("  storages:", [(f"0x{s.type:02X}", s.count, s.size) for s in pk.storages])
    # logical order: are types grouped? first index of each type
    first = {}
    for r in pk:
        first.setdefault(r.type, r.index)
    print("  first index per type:", {f"0x{t:02X}": i for t, i in first.items()})

print("### storage order + logical flags + logical order per pack")
for name in ["common_meshes_pc", "engine_pc", "common_textures_0_pc", "gui_common_pc", "common_anims_pc", "common_anims_stream_pc",
             "player_anims_static_pc", "common_prefabs_pc", "dlc_frontier_voxelizer_pc", "lang_speech_en_pc", "assets_2_pc", "dlc_frontier_heightmaps_pc"]:
    p = A / f"{name}.rpack"
    with Pack.open(p) as pk:
        lf = Counter((r.type, r.flags) for r in pk)
        types_seq = []
        for r in pk:
            if not types_seq or types_seq[-1] != r.type:
                types_seq.append(r.type)
        # are names sorted (case-insensitively) within the logical table?
        names = [r.name_raw.lower() for r in pk]
        sorted_names = names == sorted(names)
        sorted_by_type_name = [(r.type, r.name_raw.lower()) for r in pk] == sorted((r.type, r.name_raw.lower()) for r in pk)
        print(f"  {name:28s} field08={pk.header.field08:#x} storages={[f'0x{s.type:02X}' for s in pk.storages]} "
              f"lflags={{{', '.join(f'0x{t:02X}:0x{f:02X}={c}' for (t, f), c in sorted(lf.items()))}}} "
              f"type_seq={[f'0x{t:02X}' for t in types_seq[:8]]} names_sorted={sorted_names} type_name_sorted={sorted_by_type_name}")
