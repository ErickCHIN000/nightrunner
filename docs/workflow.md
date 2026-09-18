# Workflow

Two routes lead to an installable mod. Use the CLI route when you are editing shipped resources in place; use the
project route when you are building a character or an outfit out of a Blender scene.

Both end the same way: you get files in an output folder, and you install them yourself.

---

## Route 1 — extract, edit, build (CLI)

### 1. Find what you want

```powershell
python nr.py info "$ASSETS\menu_level_ft_pc.rpack"
python nr.py list "$ASSETS\menu_level_ft_pc.rpack" --type 0x20 --query lamp --parts
```

### 2. Extract

```powershell
python nr.py extract "$ASSETS\menu_level_ft_pc.rpack" .\menu_level_ft.rpx
```

Narrow it down with `--types 0x10,0x20` and `--limit N` — extracting a whole large pack takes a while and a lot of
disk.

You now have an `.rpx` tree:

```
<pack>.rpx\
  pack.json                       the build spec: source, header, storages, every resource in logical order
  mesh\000123_<name>\             image.bin skin.bin fixups.bin vertex.bin index.bin [cloth.bin]   raw parts
                                  model.cast  mesh.json                                            editable view
  texture\000456_<name>\          header.imgc bitmap.bin                                           raw parts
                                  <stem>.dds  <stem>.tex.json                                      editable view
  anim\ animscr\ animgraph\ animcustom\ prefab\ envprobe\ voxelizer\ area\
       000789_<name>\             <part>.bin …  000789_<name>.json                  raw + structural dump
  other\                          any type the catalogue has no family name for
```

**Raw is truth.** `pack.json` records the sha256 of every editable file. On build, a resource whose editable files
still match their recorded hashes is emitted straight from its raw parts. Only a file you actually changed
triggers its codec, and only for that resource.

### 3. Edit

**Texture.** Replace `<stem>.dds` — it must be DX10 or legacy-FourCC, tightly packed, exactly the sum of the level
sizes. Geometry, format and mip count may all change. Alternatively set `"source": "<file>.png"` in
`<stem>.tex.json` to import a PNG as RGBA8 / R8 / RG8_SNORM with a full mip chain. Flags, `mip_split` and the
header extension come from the sidecar; statistics are recomputed for uncompressed and BC4/BC5 formats and kept
otherwise.

There is no BC encoder. For BC formats, produce the DDS in an external tool such as `texconv`.

**Mesh.** Edit `model.cast` in place (Blender with the official Cast plugin, or `nightrunner.cast.castlib`) and
keep `mesh.json` beside it. Native engine coordinates, Y up, no axis conversion.

Check the plan before you build:

```powershell
python nr.py mesh diff .\menu_level_ft.rpx\mesh\000000_<name>
```

That prints, per entry: the path it will take (`in-place` while every mesh still carries `bp_vertex_id`, else
`rebuild`), vertex / index / submesh counts before and after, how many vertices moved, were edited or are new,
palette growth, new material names, bone changes — and anything the build would refuse.

**Anything else.** The raw `.bin` is the part. A replacement of any size builds; the container re-lays out. Nothing
validates the body.

### 4. Build

```powershell
python nr.py build .\menu_level_ft.rpx .\menu_level_ft_new.rpack
```

The output is validated before it replaces anything; a failing build leaves your previous output untouched and
keeps the bad file as `.invalid`.

Layout policies: `auto` (default), `preserve` (byte-identical tables from the source), `contiguous`, `grouped`.
`--field08 0x1000` switches to the on-demand contiguous layout, which you need when the pack carries meshes that
must load through the on-demand path — that is, meshes copied out of `common_meshes_pc` or the `dlc_*` packs.

Other useful flags: `--force-codec` re-encodes everything instead of only changed files; `--ignore-bones` keeps
the native skeleton when the Cast bones moved only by Blender rounding.

### 5. Verify

```powershell
python nr.py validate  .\menu_level_ft_new.rpack
python nr.py roundtrip .\menu_level_ft_new.rpack
python nr.py texture export .\menu_level_ft_new.rpack <name> .\check.png
```

`validate` replays the engine's decompiled read contracts. `roundtrip` makes every codec re-decode what it just
wrote. Exporting the edited resource back out is the cheapest way to confirm the edit actually landed.

### Composing a pack from several trees

```powershell
python nr.py select .\a.rpx:type=0x20 .\b.rpx:sh2_npc_crane --out .\mix.rpx --field08 0x1000 --rename old=new
python nr.py build  .\mix.rpx .\assets_2_pc.rpack
```

Selectors are an index, a range `a-b`, a name, `type=0xTT`, or `*`. `select` renumbers resources, unions the
storage records in stock order and copies (or `--link`s) the files. `build` then rewrites the embedded `.msh` name
of any renamed or replaced mesh.

---

## Route 2 — a mod project (GUI Build tab)

This is the route for "I modelled something in Blender and I want to wear it in game".

1. **Export from the game.** Models tab → pick the character → export as a **single Cast** (or `.glb`). You get a
   merged rig, every mesh, PNG materials, and a report file beside the scene.
2. **Edit in Blender.** Keep the rig. New geometry may only use bones the target mesh already has — this is a hard
   engine limit, not a tool limit. Weight your new meshes to those bones.
3. **New project.** Build tab → Project ▾ → New, then Save as… somewhere outside the game folder.
4. **Resources page.** Add your edited scene as a `scene` item and pick its report. Map each Blender object onto
   an export submesh, or Skip it. Choose targets deliberately — skin belongs on a skin material, hair on an opaque
   cloth material. Add your textures as `texture` items, each targeting the game texture whose header it should
   borrow, with a new name if it is a new texture.
5. **Models page.** Add the player model override. Switch slots on or off, point each slot at your mesh, swap
   textures through `rttiValues`. Leave **No gear** on unless you know you want equipped gear to keep overwriting
   your slots. Run the checks; the preview shows what the engine will join.
6. **Build.** You get `assets_N_pc.rpack` and `dataN.pak` in the project's output folder, both validated.
7. **Install by hand.** Copy the rpack into `<data>\work\data_platform\pc\assets\` and the PAK into
   `<data>\source\`. **The game must be closed** — it holds `assets_*.rpack` and `dataN.pak` open while running,
   so the files cannot be replaced until it exits.
8. **Start the game and look at it.** Offline verification proves the bytes are right; only the running game
   proves the mod is right.

The CLI equivalent is `nr project new | add | model | validate | build`.

---

## Things that will bite you

* **One mesh entry per slot.** The game ignores the `selected` flag. Two entries in a slot means the game draws
  the first one, which is the stock mesh. The builder writes only the selected entry for exactly this reason.
* **Gear overwrites player slots.** Without the empty `player_outfit_slots.scr`, equipped or default gear replaces
  your head, hair and shoes at runtime.
* **The material join is by name.** `materialsData` rows match the mesh's embedded submesh material names. If you
  rename a material in Blender without telling the model, the join silently misses.
* **New bones are impossible.** Rig edits are refused; re-weighting to existing bones is fine.
* **Install order is a convention, not a decoded rule.** Which numbered pack wins is confirmed by testing, not by
  reading the engine. If a mod does not show up, that is the first thing to suspect.
* **Back up before you overwrite.** Nothing in this tool writes into a game folder, so nothing in this tool can
  undo a copy you made yourself.
