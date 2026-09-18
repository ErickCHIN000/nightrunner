# What Nightrunner can do

Nightrunner is an offline toolkit for the RPACK (RP6L v4) containers of Chrome Engine games. It opens any shipped
pack, extracts every resource into an editable tree, rebuilds valid packs, validates them against the engine's
decompiled read contracts, and documents every structure it touches with its provenance.

Pure file work: no process injection, no memory patching, no runtime hooks. Game files are memory-mapped
read-only and never modified.

## Format coverage

| family                                                                                                                          | read                                                                                                               | editable intermediate                                                  | write                                                                                             |
| ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| container (RP6L v4)                                                                                                             | every table, every raw bit                                                                                         | `pack.json` build spec                                                 | byte-identical or engine-valid packs, 3 layout policies                                           |
| textures `0x20` (IMGC)                                                                                                          | 18 shipped formats plus the native writer's 21; stock 16-byte-padded levels and the third-party tight level layout | DDS (DX10) + `tex.json`                                                | DDS / PNG → IMGC, any mapped format, geometry may change                                          |
| meshes `0x10` (ClassReader)                                                                                                     | every entry, submesh, entity, palette, material name                                                               | Cast scene (`model.cast`, or `model.glb` / `model.gltf`) + `mesh.json` | Cast import + native re-encode: image / fixups / vertex / index parts rebuilt from an edited Cast |
| anim `0x40`, animscr `0x42`, animgraph `0x47`, animcustom `0x49`, prefab `0x61`, envprobe `0x55`, voxelizer `0x56`, area `0x5A` | raw parts + structural dump                                                                                        | none — raw is the intermediate                                         | raw re-emission (copy or move between packs)                                                      |
| SDB material database                                                                                                           | full table walk, material → textures resolver                                                                      | none                                                                   | **read-only**                                                                                     |
| PAK (`data0.pak` …) `.model`                                                                                                    | ZIP index, `.model` v6 join rule                                                                                   | JSON                                                                   | `.model` override PAK writer                                                                      |

Everywhere a Cast is written or read, `.glb` and `.gltf` work too, and are equally lossless — the exact data lives
in `mesh.json` and the export report. Do not leave both a `.cast` and a `.glb` in the same resource folder.

## Mesh edits the encoder accepts

Moved or edited vertices (position, normal, tangent, UV0/UV1, weights, joints); changed topology per submesh;
added or removed vertices; submesh count shrink or grow within an entry; material re-assignment by name; **new**
material names while the class-11 table still has a spare entry; palette growth (new bones appended, ≤ 256 per
submesh); a renamed resource (the embedded `.msh` name follows); Blender round trips (no `bp_*` properties, no
tangents, `.001` names).

## What it refuses, and why

The project's rule is to refuse rather than approximate. Every refusal names its exact limitation.

* Adding or removing LOD entries; removing every mesh of an entry. The class-5 / root layout that drives LOD
  selection is not decoded.
* Bone add, remove, rename, or any bone transform edit. Cast carries no inverse-bind matrices, so a rest-pose edit
  cannot be encoded. `--ignore-bones` keeps the native skeleton when the Cast bones only moved by rounding.
  Skin re-weighting works fine.
* More than 65,535 vertices per entry, more than 256 bones per submesh, more than 4 weights or a zero-weight
  vertex on a skinned format, a skinned mesh without a skeleton.
* A new material name when the class-11 table is full (7,633 of 21,354 shipped meshes have no spare entry).
* A mesh without UV0, normals or a material; vertex formats other than 0, 3, 6 and 8.
* Any result that fails the encoder's own re-decode self-check.
* **No BC encoder.** Edits to BC-compressed textures need an external encoder (such as `texconv`) that writes
  DX10 DDS. PNG import only produces RGBA8, R8 or RG8_SNORM. This is a deliberate choice, not an oversight.
* **No SDB writer.** New materials are impossible; only `.model` `rttiValues` overrides of existing materials are
  available.
* **DL2 meshes are decode-only.** DL2 packs and SDBs read; building DL2 mods is not implemented.
* Storage compression methods 2 and 3, `.rpacz` / `.rpaco` overlays, and packs with more than 15 parts per
  resource.

## What is verified, with numbers

Corpus: 47 shipped packs, 301,292 logical resources, 491,053 parts, 50,922,835,168 bytes (census 2026-09-14).

* **Container tables, from scratch.** Rebuilding every shipped pack from its own resources with the stock layout
  rules reproduces the header and all four tables byte-for-byte, and every part offset — **47/47**. Only the
  name-blob string order is copied from the source; it is the original tool's insertion order and not derivable.
  Under the `preserve` policy the whole file is byte-identical.
* **Engine contract.** All 21,297 meshes in the four `field08 = 0x1000` packs satisfy the decompiled one-read
  slicing contract — 47/47 packs. `nr validate` replays it.
* **Meshes, in-memory round trip.** Decode every resource, re-encode fixups / vertex / index parts, compare —
  **21,354/21,354 identical** (census 2026-09-15). Covers every class-6 geometry entry including LOD chains of 2–4
  entries, 142,391 submeshes, 206,960 entities and 39,804,375 vertices in formats 3, 0, 6 and 8. The embedded
  `.msh` name equals the logical name on 21,354/21,354.
* **Meshes, Cast export → import → rebuild.** All four regenerated parts compared with the stored bytes over the
  whole corpus — **21,354/21,354 byte-identical** (2026-09-15). On disk: extract 200 meshes, rebuild every one of
  them through the Cast importer, validate — 200/200 byte-identical. One moved vertex changes 3 bytes, all inside
  that vertex's position.
* **Textures, in-memory round trip** (IMGC → DDS bytes → IMGC, both parts): **49,887/49,887 stock textures
  identical**, the two header-only records included.
* **Raw families.** Identical by construction. The structural census parsed 217/217 animscr, 140/140 animgraph,
  9,476/9,476 animcustom and 23/23 prefab streams to their exact end and re-serialised them byte-for-byte;
  39,906/39,906 ANM2 stream pairs equal their plain-form twin.
* **SDB.** Both databases parse to the exact block ends; the resolver reproduces 348,164 parameters and
  1,854,984 of 1,854,986 texture bindings with 0 unresolved (2026-09-15).
* **DL2.** Meshes decode 33,728/33,728; SDBs resolve 100 %.

## Confirmed in the running game

These are the claims backed by a mod actually loading and rendering, not by offline comparison:

* A pack built by Nightrunner loads. `assets_2_pc.rpack` and `assets_3_pc.rpack` both load from the assets
  folder, side by side with a third-party pack.
* Meshes rebuilt with **wholly new geometry** render on the player through a `.model` override. New geometry may
  only use bones the target mesh already has.
* **One mesh entry per slot.** The game ignores the `selected` flag: with the stock mesh listed first and a new
  one second it draws the stock one. No shipped `.model` has more than one entry per slot, and the builder writes
  only the selected entry.
* `dataN.pak` members override `data0.pak` ones, and root-level bare member names work.
* Gear overwrites player slots through `scripts/player/player_outfit_slots.scr`; shipping that script with an
  empty `main()` stops it.
* The material join is **by name**: `materialsData` rows match the mesh's embedded submesh material names. The
  `number` is only an id linking to `materialsResources`. Unknown `rttiValues` parameters are harmless.

# 

## Provenance discipline

Every structural claim in this project carries a tier:

| tier   | meaning                                           |
| ------ | ------------------------------------------------- |
| **A**  | verified by disassembly or by a corpus census     |
| **B**  | behaviour of the older tooling this work replaces |
| **C**  | a hole — not established                          |
| **RT** | confirmed in the running game                     |

Unknown bytes are preserved verbatim and labelled, never reinterpreted. New measurements are stated with their
numbers.
