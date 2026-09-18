# Nightrunner — orientation for an agent

Offline asset toolkit for **Chrome Engine** games (Dying Light: The Beast, Dying Light 2): RPACK containers,
meshes, textures, the shader database (SDB), `.model` definitions in PAKs, plus a PySide6 GUI whose Build tab
turns Blender exports into an installable mod rpack + `dataN.pak`.

Python package `nightrunner/`, CLI `nr.py`, GUI launcher `Nightrunner.pyw`.

All paths below are relative to the repository root. The repo can live anywhere on disk; nothing in the code or
the docs may hard-code an absolute path to it, to a game install, or to anyone's machine.

## Read these first

| When | Read |
|---|---|
| Orientation, capabilities, CLI surface | `docs/` — `capabilities.md`, `workflow.md`, `cli-reference.md`, `gui-tabs.md` |
| Any code work | `notes/DEV-BRIEFING.md` — conventions, container/codec APIs, how to add a type |
| Format questions | `notes/FORMATS/*.md` — rp6l, mesh, mesh-dl2, imgc, sdb, pak-and-model, classreader, mesh-variants |
| Why something is the way it is | `notes/DECISIONS.md`; open unknowns: `notes/HOLES.md` |
| GUI work | `notes/GUI.md`, `notes/GUI-models.md`, `notes/GUI-build.md` |
| The mod-building workflow | `notes/GUIDE-build-rpack.md` (Blender → in game, worked example) |

`notes/` is the maintainer's working record and is **not published** — it is git-ignored. If it is not in the
working tree, work from `docs/` and the code, and say plainly when a format detail is not available to you rather
than guessing it.

## Environment

- Windows. Interpreter: `.venv\Scripts\python.exe` (created by `setup.bat`; numpy, Pillow, PySide6).
  **Never install extra dependencies**: stdlib + numpy + Pillow (+ PySide6 for the GUI) only, and keep the code
  3.11-compatible.
- Tests: `.venv\Scripts\python.exe tests\run.py -q` (set `QT_QPA_PLATFORM=offscreen` first).
  475 tests collected, 2 expected failures, 0 failures on a correct tree. GUI tests skip without Qt; corpus tests
  skip without a game install; sample-based tests skip unless `out\samples\` exists (build it with
  `tools\make_samples_mesh.py` / `make_samples_texture.py`); the DL2 decode fixtures under `tests\data\` are not
  published, so those always skip. A clone with the game present reports `Ran 472 ... OK (skipped=135,
  expected failures=2)` — 472 because three skip at class level, which unittest omits from its total.
  **Run the suite before claiming a change works.**
- Single module: `.venv\Scripts\python.exe -m unittest tests.test_project -q`.
- CLI: `.venv\Scripts\python.exe nr.py <command>` — `info list validate census extract build roundtrip select`,
  plus areas `mesh|texture|types|sdb|model|project <sub>`.
- GUI: `Nightrunner.pyw` (re-launches itself in the venv).
- Hash cache: whole-file sha256 of game packs is cached in `.cache\sha256.json` (`NIGHTRUNNER_HASH_CACHE`
  overrides). Without it a project build re-hashes multi-gigabyte packs and takes minutes instead of seconds.
- Long jobs: write output to a file under `out\` and read that, rather than streaming huge stdout.

## Games

Both installs are read-only reference data. Detection lives in `nightrunner/games.py`: an explicit root, then
`$NIGHTRUNNER_GAME_ROOT`, then the Steam libraries.

| Game | Steam folder | Data folder | Notes |
|---|---|---|---|
| DLTB (`dltb`) | `Dying Light The Beast` | `ph_ft` | primary target, most verified |
| DL2 (`dl2`) | `Dying Light 2` | `ph` | mesh + SDB read support; also ships `DevTools\` |

Assets: `<root>\<data>\work\data_platform\pc\assets\*.rpack` + `runtime_dx11.sdb` / `runtime_dx12.sdb`.
PAKs (`.model` definitions, scripts): `<root>\<data>\source\dataN.pak`.

## Hard rules

1. **Everything created or modified lives inside this repository.** Game installs and any external reference
   sources are read-only.
2. **The tool never writes into a game folder.** A build produces files in its output folder; the user installs
   them. Only copy into a game folder when asked in that turn, and note that the game locks `assets_*.rpack` and
   `dataN.pak` while it runs — files can only be swapped with the game closed.
3. **Offline file work only.** No process injection, no memory patching, no runtime hooks.
4. **No guessing.** Every structural claim carries provenance: (A) disassembly / corpus census, (B) behaviour of
   the older tooling, (C) hole, (RT) confirmed in the running game. Unknown bytes are preserved verbatim and
   labelled, never reinterpreted. New measurements are stated with numbers ("census: 21,354/21,354").
5. **Refuse rather than approximate.** An encoder that cannot honour its input raises `BuildError` /
   `UnsupportedError` naming the exact limitation (e.g. DL2 meshes are decode-only today).
6. **Nothing personal ships.** No absolute paths from a developer's machine, no user names, no local folder
   layouts in `README.md`, `docs/`, this file, or the source.
7. Scratch scripts and test output go in `out\` (git-ignored, not part of the deliverable).

## Game facts confirmed in game (RT) — don't re-derive, don't contradict

- `assets_2_pc.rpack` and `assets_3_pc.rpack` both load from the assets folder; `dataN.pak` members override
  `data0.pak` ones, root-level bare member names work.
- Meshes with entirely new geometry render. New geometry may only use bones the target mesh already has.
- **One mesh entry per slot.** The game ignores `"selected"`: with the stock mesh listed first and the new one
  second it draws the stock one. No stock `.model` has more than one entry per slot. `project.game_doc()` writes
  only the selected entry into the built PAK — this was the bug that made a custom head, torso and hair
  invisible (fixed and confirmed in game 2026-09-16).
- Gear overwrites player slots through `scripts/player/player_outfit_slots.scr`; shipping it with an empty
  `main()` ("No gear") stops that.
- The material join is **by name**: `materialsData` rows match the mesh's embedded submesh material names; the
  `number` is only an id linking to `materialsResources`. Unknown `rttiValues` params are harmless.
- Older finding, now doubtful: "new meshes in HEAD/LEGS slots are not drawn" was probably the one-entry bug.
  Re-test before trusting the warning the builder still emits.

## Where things are

```
nightrunner/
  container/rp6l.py     RPACK reader/writer (byte-exact rebuild)
  mesh/                 decode, encode, rebuild, codec, variants, identity   (DL2: decode-only)
  texture/              IMGC <-> DDS/PNG
  sdb/reader.py         shader database (DLTB + DL2 layouts)
  pak/model_json.py     .model documents in PAKs + PAK writer
  cast/                 Cast + glTF import/export, split_model_cast
  classreader/          the object graph inside a mesh's primary image
  project.py            .nrproj mod projects: items, model overrides, build_project
  games.py              game profiles and install detection
  gui/                  PySide6 app: tabs raw/textures/meshes/models/sdb/build
docs/                   published documentation (getting-started, capabilities, gui-tabs, workflow, cli-reference)
notes/                  maintainer's working record — not published, may be absent
tests/                  unittest; run.py is the runner
tools/                  census + sample-making scripts
out/                    scratch, samples, build output, diagnostics (not part of the deliverable)
```

## Working style

- Direct and terse. Honesty over politeness, no padding.
- **UI text is terse**: single words where possible, no legends, info banners or over-explaining.
- Don't delegate small work to subagents; do it directly. Parallel subagents are for genuinely large work.
- Verify with real data before claiming something works; say plainly what is untested.

## State as of 2026-09-17

- The package was renamed from "beastpack" on 2026-09-16 (package, CLI, launcher, `.bpproj` → `.nrproj`). Old
  schema ids and `.bpproj` files still load (`nightrunner/util/schema.py`).
- DL2 support: meshes decode (33,728/33,728) and SDBs resolve 100 %; building DL2 mods is not implemented.
- A full custom character builds and renders in game through the Build tab (documented in
  `notes/GUIDE-build-rpack.md`).
- Open items live in `notes/HOLES.md`; the biggest ones: DL2 mesh writer, DL2 mod loading untested in game,
  the DL2 DLC folder (`ph\dlc_opera`) not indexed, custom material creation (no SDB writer).
