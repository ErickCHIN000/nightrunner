# Nightrunner

Offline asset toolkit for the RPACK (RP6L v4) containers of Chrome Engine games: **Dying Light: The Beast** and
**Dying Light 2**.

It opens any shipped pack, extracts every resource into an editable tree, rebuilds valid packs, validates them
against the engine's decompiled read contracts, and documents every structure it touches with provenance. A
PySide6 GUI adds an explorer for textures, meshes, materials and `.model` documents, and a Build tab that turns a
Blender export into an installable mod rpack and PAK.

Pure offline file work: no process injection, no memory patching, no runtime hooks. Game files are memory-mapped
read-only and never modified. Nothing is ever written into a game folder — a build produces files in an output
folder you name, and you install them yourself.

Version 0.1.0.

## Documentation

| | |
|---|---|
| [docs/getting-started.md](docs/getting-started.md) | install, first commands, how an install is detected, tests |
| [docs/capabilities.md](docs/capabilities.md) | what it reads and writes, what it refuses, what is verified |
| [docs/gui-tabs.md](docs/gui-tabs.md) | the GUI tab by tab |
| [docs/workflow.md](docs/workflow.md) | extract → edit → build → install, end to end |
| [docs/cli-reference.md](docs/cli-reference.md) | every command, with examples |

## What it reads and writes

| family | read | editable intermediate | build |
|---|---|---|---|
| container (RP6L v4) | every table, every raw bit | `pack.json` build spec | byte-identical or engine-valid packs, 3 layout policies |
| textures `0x20` (IMGC) | 18 shipped formats plus the native writer's 21 | DDS (DX10) + `tex.json` | DDS / PNG → IMGC, any mapped format |
| meshes `0x10` (ClassReader) | every entry, submesh, entity, palette, material name | Cast (`model.cast`, or `.glb` / `.gltf`) + `mesh.json` | Cast import + native re-encode |
| anim, animscr, animgraph, animcustom, prefab, envprobe, voxelizer, area | raw parts + structural dump | raw is the intermediate | raw re-emission |
| SDB material database | full table walk, material → textures resolver | — | **read-only** |
| PAK `.model` documents | ZIP index, `.model` v6 join rule | JSON | `.model` override PAK writer |

Deliberately not implemented: a BC texture encoder (use `texconv` and hand it a DX10 DDS), an SDB writer (so no
new materials — only `rttiValues` overrides of existing ones), and a DL2 mesh writer (DL2 is decode-only).

## Requirements

Windows, Python ≥ 3.11, `numpy` and `Pillow`, plus `PySide6 >= 6.7` for the GUI and `pyvgmstream` for audio
preview. That is the whole list.

```bat
setup.bat            :: creates .venv and installs requirements-gui.txt
Nightrunner.pyw      :: the GUI; re-launches itself inside .venv
python nr.py --help  :: the CLI
```

## Quick start

`$ASSETS` below is an install's `…\<data>\work\data_platform\pc\assets` folder.

```powershell
python nr.py info     "$ASSETS\common_meshes_pc.rpack"
python nr.py list     "$ASSETS\common_meshes_pc.rpack" --type 0x10 --query crane --parts
python nr.py validate "$ASSETS\engine_pc.rpack"

python nr.py extract  "$ASSETS\menu_level_ft_pc.rpack" .\menu_level_ft.rpx
#   edit the .dds / model.cast inside the tree
python nr.py build    .\menu_level_ft.rpx .\menu_level_ft_new.rpack
python nr.py validate .\menu_level_ft_new.rpack
```

## How it decides what to re-encode

Extraction writes both the raw parts and an editable view, and records the sha256 of every editable file in
`pack.json`. **Raw is truth:** on build, a resource whose editable files still match their recorded hashes is
emitted straight from its raw bytes. A changed DDS or PNG triggers the texture codec, a changed `model.cast` or
`mesh.json` triggers the mesh codec, for that resource only. `--force-codec` re-encodes everything.

That is why an untouched extract → build cycle is byte-identical rather than merely equivalent.

## Verified, with numbers

Corpus: 47 shipped packs, 301,292 logical resources, 491,053 parts, 50,922,835,168 bytes (census 2026-09-14).

* **Container tables rebuilt from scratch — 47/47 packs** byte-identical in header and all four tables, and every
  part offset. Only the name-blob string order is copied from the source; it is not derivable.
* **Meshes, decode → re-encode — 21,354/21,354 identical.** Covers every class-6 geometry entry including LOD
  chains, 142,391 submeshes, 206,960 entities, 39,804,375 vertices.
* **Meshes through the full Cast export → import → rebuild path — 21,354/21,354 byte-identical.**
* **Textures, IMGC → DDS → IMGC — 49,887/49,887 identical**, the two header-only records included.
* **SDB** — both databases parse to the exact block ends; 348,164 parameters and 1,854,984 of 1,854,986 texture
  bindings resolved, 0 unresolved.
* **DL2** — meshes decode 33,728/33,728, SDBs resolve 100 %.

Confirmed in the running game: a Nightrunner-built pack loads; meshes rebuilt with wholly new geometry render on
the player through a `.model` override; `dataN.pak` members override `data0.pak` ones. See
[docs/capabilities.md](docs/capabilities.md) for the full list, including what is **not** confirmed.

## Layout

```
nr.py                    CLI entry point → nightrunner.cli
Nightrunner.pyw          GUI launcher (re-runs itself inside .venv)
setup.bat                creates .venv, installs requirements-gui.txt
nightrunner/
  container/   rp6l.py (reader + PackWriter), validate.py, catalogue.py, census.py
  classreader/ fixups.py, image.py, graph.py          the object graph inside a mesh's primary image
  mesh/        decode, encode, rebuild, imagepatch, variants, identity, sidecar, codec, cli
  cast/        castlib.py (vendored, MIT), export, import_, assemble, split
  texture/     imgc, formats, dds, png, codec, cli
  types/       raw, classreader, anim, animscr, animgraph, animcustom, prefab, area, envprobe, voxelizer
  sdb/         reader.py, cli.py           pak/  model_json.py, cli.py
  gui/         app, mainwindow, context, catalog, services, tasks, widgets, game, meshview, matpreview,
               modelresolve, sdbscan, theme,  tabs/ raw, textures, meshes, models, build, sdb
  project.py   .nrproj mod projects        games.py  game profiles and install detection
  extract.py   build.py  select.py  roundtrip.py  codecs.py  cli.py  errors.py  util/
docs/                    this documentation
tests/                   unittest suites (run.py, paths.py, synth.py, test_*.py)
tools/                   corpus censuses, layout probes, sample builders
```

## Known limits

Everything preserved raw but not understood — container flag bits, IMGC flags, several mesh class layouts and
vertex holes, LOD semantics, most of the SDB's tables, `.model` field semantics, every animation, prefab and area
body, and the pack registration order — is tracked with its location, current handling, and the step that would
resolve it. Unknown bytes are copied verbatim and labelled, never reinterpreted.

The most consequential open item: **which mechanism makes a numbered override pack win is not decompiled.** The
install recipe is a convention confirmed by testing, not a rule read out of the engine.

## AI assistance

This project was built with AI assistance throughout: writing and refactoring the code, writing this
documentation, and the reverse-engineering research behind it — reading disassembly, forming hypotheses about
undocumented structures, and designing the censuses that test them.

That does not mean the findings are guesses. The project's rule is that no structural claim ships without
provenance, and every claim here is one of four things: read out of disassembly, measured across the whole
shipped corpus with the numbers stated, inherited from earlier community tooling and labelled as such, or
confirmed by loading a mod in the running game. Anything that is none of those is recorded as an open hole rather
than filled in with a plausible-sounding answer. Unknown bytes are copied verbatim, never reinterpreted.

The measurements are reproducible — `nr census`, `nr roundtrip` and the scripts in `tools/` re-run them against
your own install. If a number in these docs disagrees with what you measure, trust your measurement and open an
issue.

## Credits

`nightrunner/cast/castlib.py` is vendored from DTZxPorter's Cast library (MIT, see CAST-LICENSE.txt). Findings from
earlier community tooling for these games were used as a starting point and are labelled as such wherever they
inform a structural claim.

## Thanks

None of this was worked out in a vacuum. To the people who were around for it:

CHAINSAW · EricPlayZ · Gamer5700 · Goose · Light · nelson01023 · ODST · SDavidLee · Steffen · Teo ·
Viktor Neres · xlhs

## Licence

MIT — see [LICENSE](LICENSE). The vendored Cast library keeps its own MIT notice in
[CAST-LICENSE.txt](CAST-LICENSE.txt).
