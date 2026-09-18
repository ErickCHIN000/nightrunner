# Getting started

## Requirements

* Windows.
* Python 3.11 or newer (developed on 3.14; the source stays 3.11-compatible).
* `numpy` and `Pillow`. `PySide6 >= 6.7` as well if you want the GUI. Nothing else — no other dependency is
  permitted in this project.
* One of the supported games installed. Every command takes explicit paths, and nothing is ever written into a
  game folder.

## Install

Clone the repository, then from its root:

```bat
setup.bat
```

`setup.bat` creates `.venv` next to itself and installs `requirements-gui.txt` (numpy, Pillow, PySide6). Re-run it
to update. If you only want the CLI and already have numpy and Pillow, you can skip it and use your own
interpreter.

## Run the GUI

```
Nightrunner.pyw
```

Double-click it, or run it with any Python — it re-launches itself with `.venv\Scripts\pythonw.exe`. The
equivalent module entry point is `python -m nightrunner.gui`.

## Run the CLI

From the repository root:

```powershell
python nr.py --help
python nr.py info "<assets>\common_meshes_pc.rpack"
```

`python -m nightrunner` works the same way. Every command prints JSON or JSONL on stdout; errors go to stderr as
`error: …` with exit code 2.

## Supported games and how they are found

| id | game | Steam folder | data folder |
|---|---|---|---|
| `dltb` | Dying Light: The Beast | `Dying Light The Beast` | `ph_ft` (the `ph` beside it only holds `fs.ini`) |
| `dl2` | Dying Light 2 | `Dying Light 2` | `ph` (`ph_ft` is accepted too, e.g. a renamed test copy) |

Inside an install:

```
<root>\<data>\work\data_platform\pc\assets\    *.rpack, runtime_dx11.sdb, runtime_dx12.sdb
<root>\<data>\source\                          data0.pak, data1.pak, …
```

Detection (`nightrunner/games.py`) scores each profile against a candidate root: the executable under
`<data>\work\bin\x64` (+8), a marker folder (+4), the Steam folder name (+3), the preferred data folder (+2).
Only data folders that actually contain the assets folder count. Pointing at the data folder itself also works —
its parent is used.

Lookup order, in this order:

1. an explicit root (`--root` on the CLI, **Game › Browse…** in the GUI),
2. the `NIGHTRUNNER_GAME_ROOT` environment variable (the legacy `BEASTPACK_GAME_ROOT` is still read),
3. every Steam library from `libraryfolders.vdf`.

The GUI remembers the last used install per game and reopens it on start.

## Set up shell variables

The examples in these docs assume two variables. Set them once per session to your own install:

```powershell
$ASSETS = "<steam library>\steamapps\common\Dying Light The Beast\ph_ft\work\data_platform\pc\assets"
$SOURCE = "<steam library>\steamapps\common\Dying Light The Beast\ph_ft\source"
```

## First five minutes

```powershell
python nr.py info     "$ASSETS\engine_pc.rpack"
python nr.py list     "$ASSETS\common_meshes_pc.rpack" --type 0x10 --query crane --limit 20
python nr.py validate "$ASSETS\engine_pc.rpack"
python nr.py texture export "$ASSETS\engine_pc.rpack" default_org.png .\default_org.png
python nr.py mesh    export "$ASSETS\common_meshes_pc.rpack" sh2_npc_crane .\crane.cast
```

Then read [workflow.md](workflow.md) for the extract → edit → build loop, or open the GUI and read
[gui-tabs.md](gui-tabs.md).

## Tests

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python tests\run.py -q
python tests\run.py -k texture -k select      # only ids containing a substring (OR-ed)
python -m unittest tests.test_texture -v      # one module
```

432 tests are collected, 2 of them expected failures. A correct tree never reports a failure.

Much of the suite needs data that is not in the repository, and skips cleanly without it: tests that need a game
install, GUI tests without PySide6, and tests that need the sample fixture tree. Build the samples with
`tools\make_samples_mesh.py` and `tools\make_samples_texture.py`, which write into `out\samples\`. The DL2 decode
fixtures under `tests\data\` are not published at all, so those tests always skip on a clone.

On a fresh clone with a game installed, expect roughly `Ran 429 ... OK (skipped=135, expected failures=2)` —
429 rather than 432 because three of them skip at class level, which unittest does not count in its total.

Corpus-wide gates are run by hand and take minutes: `nr census`, `nr roundtrip`, `nr texture census`,
`nr types census`, plus `tools\census_mesh.py`, `tools\mesh_bounds_census.py`, `tools\mesh_phase2_acceptance.py`
and `tools\rebuild_check.py`.

## Where things are written

Nothing is written into a game install, ever. Outputs go where you name them. The GUI additionally keeps its
settings, a resource cache and a crash log under `%LOCALAPPDATA%\nightrunner`, and whole-file hashes are cached in
`.cache\sha256.json` in the repository (`NIGHTRUNNER_HASH_CACHE` overrides the location). Without that cache a
project build re-hashes multi-gigabyte game packs on every run.
