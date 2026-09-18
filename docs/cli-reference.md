# CLI reference

Run from the repository root as `python nr.py <command>` (or `python -m nightrunner <command>`). Every command
prints JSON or JSONL on stdout; errors go to stderr as `error: …` with exit code 2.

`--help` works at every level: `python nr.py --help`, `python nr.py mesh --help`, `python nr.py mesh export --help`.

In the examples, `$ASSETS` is an install's assets folder and `$SOURCE` its `source` folder — see
[getting-started.md](getting-started.md).

## Surface

```
nr info      <rpack> [--json]
nr list      <rpack> [--type 0xTT] [--query STR] [--limit N] [--offset N] [--parts] [--sha256]
nr validate  <rpack>... [--json] [--show N] [--no-mesh-names]
nr census    <rpack>... --out DIR [--layouts auto,preserve]
nr extract   <rpack> <out.rpx> [--types 0x10,0x20] [--limit N] [--no-raw] [--sdb FILE] [--pak FILE]
nr build     <tree.rpx|pack.json> <out.rpack> [--layout auto|preserve|contiguous|grouped] [--field08 0x1000]
             [--force-codec] [--ignore-bones] [--no-validate]
nr roundtrip <rpack> [--types ...] [--limit N] [--report FILE.json]
nr select    <tree.rpx>[:sel,...]... --out <new.rpx> [--field08 X] [--flags X] [--link] [--rename OLD=NEW]
             [--replace NAME=TREE:SEL] [--quiet]
nr project   new <f.nrproj> | add <f> <files…> [--target N] [--pack L] [--as NEW]
             | model <f> --appearance [CHAR/]NAME | info | validate | build <f> [--out DIR]
             (any subcommand: [--game dltb|dl2] [--root DIR]; the default game is the project's)

nr mesh      info <rpack> <index|name>
             | export <rpack> <index|name> <out.cast|.glb|.gltf> [--no-sidecar]
             | dump <rpack> <index|name> [--vertices N]
             | census <rpack> [--limit N]
             | import <model.cast|.glb|.gltf> <mesh.json> --out-dir DIR [--parts-dir DIR] [--name NAME]
                      [--no-pad-index] [--ignore-bones]
             | diff <resource dir> [--name NAME] [--ignore-bones] [--json]
nr texture   info <rpack> <index|name> [--json] [--levels N]
             | export <rpack> <index|name> <out.dds|out.png> [--mip N] [--face N] [--slice N] [--force]
             | ddsinfo <file.dds> [--srgb-to-linear] [--allow-unobserved]
             | census <assets dir> --out DIR [--packs a,b]
nr types     dump <rpack> <index|name> [--json-out FILE]
             | census <assets dir> --out DIR [--family a,b] [--limit N] [--quiet] [--strict]
nr sdb       list <sdb> [--query STR] [--limit N] | material <sdb> <name> [--raw] | textures <sdb> <name>
             | stats <sdb> [--validate] [--json-out FILE]
nr model     list <pak> [--query STR] [--all] [--limit N] | show <pak> <member> | meshes <pak> <member> [--sdb FILE]
             | write <out.pak> NAME=FILE... [--text NAME=FILE] [--overwrite]
             | split <edited.cast|.glb|.gltf> <export dir> --out DIR [--rpx TREE] [--tol T] [--no-check]
```

## Container

```powershell
python nr.py info     "$ASSETS\common_meshes_pc.rpack"
python nr.py list     "$ASSETS\common_meshes_pc.rpack" --type 0x10 --query crane --parts --sha256
python nr.py validate "$ASSETS\common_meshes_pc.rpack" "$ASSETS\engine_pc.rpack"
python nr.py census   "$ASSETS\reg_in_pc.rpack" --out .\reports\census
```

`info` prints the header, the type histogram and the storage table. `list` prints one JSONL record per resource;
`--parts` adds the part records, `--sha256` the sha256 of the stored bytes. `validate` replays the engine's read
contracts and exits 1 on an error. `census` measures the layout rules and rebuild identity across packs.

## Extract, build, verify

```powershell
python nr.py extract   "$ASSETS\menu_level_ft_pc.rpack" .\menu_level_ft.rpx
python nr.py build     .\menu_level_ft.rpx .\menu_level_ft_new.rpack
python nr.py validate  .\menu_level_ft_new.rpack
python nr.py roundtrip "$ASSETS\dlc_ft_prologue_pc.rpack" --types 0x10,0x20
```

`roundtrip` decodes and re-encodes in memory and compares against the stored parts, writing nothing. It is the
primary correctness gate for every codec.

## Compose

```powershell
python nr.py extract "$ASSETS\common_textures_0_pc.rpack" .\ct0.rpx --types 0x20 --limit 200
python nr.py select  .\ct0.rpx:alarm_lamp_a_gre.png .\menu_level_ft.rpx:type=0x20 --out .\mix.rpx --field08 0
python nr.py build   .\mix.rpx .\assets_2_pc.rpack
```

## Textures

```powershell
python nr.py texture info    "$ASSETS\engine_pc.rpack" default_org.png
python nr.py texture export  "$ASSETS\engine_pc.rpack" default_org.png .\default_org.dds   # lossless DX10 DDS
python nr.py texture export  "$ASSETS\engine_pc.rpack" default_org.png .\default_org.png   # 8-bit preview
python nr.py texture ddsinfo .\default_org.dds                                             # what the importer sees
python nr.py texture census  "$ASSETS" --out .\reports\texture
```

## Meshes

```powershell
python nr.py mesh info   "$ASSETS\common_meshes_pc.rpack" sh2_npc_crane
python nr.py mesh export "$ASSETS\common_meshes_pc.rpack" sh2_npc_crane .\crane.cast    # + crane.mesh.json
python nr.py mesh dump   "$ASSETS\common_meshes_pc.rpack" 9132 --vertices 8
python nr.py mesh census "$ASSETS\engine_pc.rpack"
python nr.py mesh diff   .\menu_level_ft.rpx\mesh\000000_<name>
python nr.py mesh import .\model.cast .\mesh.json --out-dir .\parts
```

`mesh diff` prints what an edited `model.cast` would change, and what the build would refuse, without building
anything. `mesh import` is the stand-alone re-encode: it writes the four parts plus `import_report.json` without
producing a pack.

## Raw families

```powershell
python nr.py types dump   "$ASSETS\common_anims_pc.rpack" 0
python nr.py types dump   "$ASSETS\reg_in_pc.rpack" Prefabs --json-out .\prefabs.json
python nr.py types census "$ASSETS" --out .\reports\types --family anim,area
```

## SDB (read-only)

```powershell
python nr.py sdb list     "$ASSETS\runtime_dx11.sdb" --query barrel
python nr.py sdb material "$ASSETS\runtime_dx11.sdb" barrel_b.mat --raw
python nr.py sdb textures "$ASSETS\runtime_dx11.sdb" player_kc_basic_torso_a_tpp.mat
python nr.py sdb stats    "$ASSETS\runtime_dx11.sdb" --validate
```

## PAK and `.model`

```powershell
python nr.py model list   "$SOURCE\data0.pak" --query player
python nr.py model show   "$SOURCE\data0.pak" models/player/player_tpp_skeleton.model
python nr.py model meshes "$SOURCE\data0.pak" player_tpp_skeleton.model --sdb "$ASSETS\runtime_dx11.sdb"
python nr.py model write  .\models.pak player_tpp_skeleton.model=.\my_skeleton.model --text playerappearances.scr=.\pa.scr
python nr.py model split  .\edited.cast .\export_dir --out .\per_mesh --rpx .\tree.rpx
```

## Mod projects

```powershell
python nr.py project new      .\mymod.nrproj
python nr.py project add      .\mymod.nrproj .\face_dif.png --target 12345
python nr.py project model    .\mymod.nrproj --appearance PLAYER
python nr.py project validate .\mymod.nrproj
python nr.py project build    .\mymod.nrproj --out .\build
```

Same engine as the GUI's Build tab. See [workflow.md](workflow.md) for what the pieces mean.
