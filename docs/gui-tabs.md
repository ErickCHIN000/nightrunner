# The GUI, tab by tab

Launch with `Nightrunner.pyw` (or `python -m nightrunner.gui`). PySide6, dark Fusion theme.

The GUI is read-only towards the game. It writes only the exports you ask for, the build output of a mod project,
and its own settings, cache and crash log under `%LOCALAPPDATA%\nightrunner`.

On start it opens the last used install, or the first one it detects. It then indexes **every `.rpack` under
`<data>\work\data_platform\pc\assets`**, sub-folders included, on a background thread, and opens
`runtime_dx11.sdb` and every `<data>\source\dataN.pak`. Every list is virtual and every search, decode and export
runs off the GUI thread, so nothing is capped — scroll or search to any of the ~150,000 resources.

Common to all tabs:

* **Game** menu — one entry per detected install, plus **Browse…** for any root. Switching rebuilds every tab in
  place; an unsaved Build project asks first.
* **File › Open Extra Packs…**, or drag and drop, adds packs from outside the install.
* **File › SDB** switches between `runtime_dx11.sdb` and `runtime_dx12.sdb` (the SDB tab has the same dropdown).
* Double-clicking a texture, mesh, material or model anywhere opens it in its own tab.

## The update indicator

Top right of the tab row: `v0.1.0 (+3) 89ef0b6` — the version, how many commits are waiting, and the commit this
copy is on. It turns amber when an update is available and is grey otherwise; hovering explains the state.

The check runs on a worker thread shortly after start and every six hours after that. It never blocks the window,
and when it cannot reach GitHub it says "offline" and carries on — Nightrunner works entirely offline.

Clicking it when an update is waiting opens a dialog listing what changed, grouped into FIXED / IMPROVED / OTHER
from the commit subjects, with a count of anything not listed. **Update now** fast-forwards your checkout and tells
you to restart. Clicking it when you are up to date just opens the repository in a browser.

The update is a git fast-forward and nothing else. It refuses up front, naming the reason, when the tree has
uncommitted changes, HEAD is detached, git is not installed, or this is not a clone — so it can never merge,
rebase, or throw away work of yours. In those cases the button becomes **Open on GitHub** instead.

Two environment variables matter if you run a fork: `NIGHTRUNNER_UPDATE_REPO` (default `ErickCHIN000/nightrunner`)
and `NIGHTRUNNER_UPDATE_BRANCH` (default `main`). A private repository needs a token, read from
`NIGHTRUNNER_GITHUB_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN`, or the GitHub CLI's stored auth. A public repository needs
none.

---

## Raw

The container itself, for when you want the bytes rather than a decoded view.

A tree of all packs: header, storage and name tables, then one group per resource type, then resources, then
parts. Groups load in growing chunks as you scroll. Tri-state check boxes on every level.

**Export Checked / Selected / All** (the whole install, with a size confirmation) writes the raw part blobs to
`<out>/<pack>/<family>/<index>_<name>/<part>.bin`. Existing files are skipped, never overwritten.

Search: all words must match; `#123` looks up a logical index; plus a type filter.

## Textures

Every texture in every pack, as a list or a thumbnail grid, with search, a pack filter and column sorting.

The preview has zoom and pan, R/G/B/A channel toggles and mip / face / slice selectors. It decodes BC1–BC7 and
the uncompressed formats.

The info pane shows every IMGC header field, any other pack carrying the same name, and the SDB materials that
use the texture.

Export checked items as lossless DDS, as 8-bit PNG, or raw.

## Meshes

Every mesh, with a QtQuick3D viewer — orbit, pan, zoom, **F** to frame, **W** wireframe, **N** normals.

Controls: LOD selection, per-submesh visibility, SDB diffuse textures applied to the model, and a **variant**
selector (mesh part `0x12`, e.g. `olive_plastic`).

Sub-tabs: overview, submeshes, materials (SDB resolution, with textures marked found or missing), skeleton,
variants, parts.

Export checked items as Cast + `mesh.json` — the same output as `nr mesh export` — or raw.

## Models

The `.model` documents from every `dataN.pak`, including any later PAK that overrides them.

Selecting one resolves the whole chain: skeleton → slots → meshes (the selected one and its alternates, with
every pack that provides each) → submesh materials → mesh-variant remap → `materialsData` / `materialsResources`
join → SDB material, preset, parameters and texture bindings → the `.model`'s own `rttiValues` overrides → every
texture, marked found or missing.

Four sub-tabs: **Mapping** (Item | Value | Source | Status), **3D** (the found meshes), **Raw JSON**, **Summary**
(copyable lists of unique textures, materials and meshes). Filters for "only missing" and "only overrides".

Export as **split files** (the `.model`, `mapping.json`, one Cast per mesh, DDS textures) or as a **single Cast**
(merged rig plus every chosen mesh, PNG materials, and a small Blender script for hair and eye alpha), or both —
each in Cast, `.glb` or `.gltf`.

**Split edited…** turns a single Cast that you edited in Blender back into per-mesh `model.cast` folders,
optionally dropped straight into an extracted `.rpx` tree ready for `nr build`. The CLI equivalent is
`nr model split`.

## Build

Mod projects (`.nrproj`). This is the tab that turns a Blender export into something installable. One build
produces one mod rpack plus, when the project has model overrides, one PAK.

| file | default name | install to |
|---|---|---|
| rpack | first free `assets_N_pc.rpack`, N from 2 | `<data>\work\data_platform\pc\assets\` |
| PAK | first free `data0.pak` … `data8.pak`, never a stock archive | `<data>\source\` |

Both names are editable, and after the first successful build the auto-chosen names are pinned into the project —
otherwise the files you just installed would push the next build to a different N. **The tool never writes into
the game folder; installing is your own copy step.**

### Resources page

Four item kinds, each targeting a game resource. Same name → the item overrides it; a new name → a new resource
cloned from the target.

* **texture** — PNG / JPG / TGA / BMP (imported as RGBA8, R8 or RG8 normal; there is no BC encoder) or a DDS,
  which is kept as-is. The target is the header template. Names carry `.png` / `.dds`, exactly what `rttiValues`
  reference. A dropped `x.png` also matches a game `x.dds`.
* **mesh** — a per-mesh Cast / GLB / glTF edited from the target's own export.
* **scene** — a single-model export, split at build time. **Objects** maps Blender objects onto export submeshes
  (several objects can merge into one submesh); **Targets** picks what each replaces; **Hide** collapses a submesh
  to one degenerate triangle; **LODs** (on by default) applies the same replacement to the entity's other LOD
  entries with the same material slot; **2-sided** appends a reversed copy of the faces, for thin cloth seen from
  the inside.
* **raw** — replaces one part byte for byte.

Target choice matters for shading: put skin on a skin material, hair on an opaque cloth material. A jacket
material over skin UVs showed holes in game; hair shaders cut pixels with their own dither map.

### Models page

`.model` overrides per role (TPP / FPP / LodCC / UI, paired from `playerappearances.scr`). Slot on/off, mesh
swaps, texture swaps through `rttiValues`, a diff against the original, checks, and a live preview rendered by the
Models tab pipeline.

Turning a slot **off** empties that slot's `meshResources.resources` — stock models ship empty slots such as ARMS
and HANDS, so this is a shape the engine already sees. Removed entries are stashed in the project and restored
when the slot comes back on.

Player overrides default to **No gear**: the PAK also gets `player_outfit_slots.scr` with an empty `main()`.
Without it, equipped (or default) gear overwrites the player slots at runtime and your override loses its head,
hair and shoes.

Checks verify only the slots that differ from the original: that the selected mesh and every `*_tex` value either
exist in the game or are produced by an item in this project.

## SDB

Materials, presets, and a texture reverse index (which materials bind a given texture).

The material view has an overview and features page, the parameters shown against their preset defaults,
per-variant texture bindings with catalog status, which models and meshes use the material (a background scan),
the raw record hex, and JSON export. A Stats page runs Validate over the whole database.
