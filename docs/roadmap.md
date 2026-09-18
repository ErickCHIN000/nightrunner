# Roadmap — planned tab sections

The GUI groups its tabs into sections (`nightrunner/gui/tabs/__init__.py`, `SECTIONS`). Two exist; three are
planned. This file records what each planned section is meant to cover and what is known about it, so the work
can start without re-deriving the plan.

Nothing below is implemented. Nothing below has been verified against the games. Treat every claim here as a
starting hypothesis to check, not as a finding — the project's provenance rule (A / B / C / RT, see
[capabilities.md](capabilities.md)) applies to this file as much as to the code.

## Shipping today

| section | tabs | state |
|---|---|---|
| **RPACK** | Raw, Textures, Meshes, Models, Build | the container, its resources, and mod building |
| **SDB** | SDB | read-only: browse and dump materials, presets, textures |

Adding a section means writing its tab modules under `nightrunner/gui/tabs/` and adding one row to `SECTIONS`.
The window needs no other change: it builds the outer bar from that table, gives a section with several tabs an
inner bar, and shows a single-tab section directly.

---

## SDB — writing

The SDB section is read-only today. The intended next step is **generating a modified SDB with new materials and
presets injected**, which would turn the section from a browser into an editor.

This is the largest open item in the whole project, not a small addition. What stands in the way is recorded as
holes S8 and S9 in the format notes: there is no writer, table growth and new strings are not understood, the
shared-D2 blob would need copy-on-write, the runtime material-name lookup is undecoded, and it is unknown whether
new `AE` / `B2` strings need `0xBE` companions. It is also unknown where a replacement database has to live for
the game to load it.

Until that is resolved, new materials are impossible and the only material customisation available is a `.model`
`rttiValues` override of a material that already exists.

## Audio — Wwise

Audio modding goes through Wwise. An existing community tool already covers a large part of this ground, written
in C#; the intent here is a Python implementation, which suits this codebase and the rest of the toolchain
better.

To check before starting: whether a usable Python library or binding for the Wwise container formats already
exists, and whether it is permissively licensed and dependency-free enough for this project's rules (stdlib +
numpy + Pillow only — a new runtime dependency needs a deliberate decision, not a quiet `pip install`). If the
answer is a vendored single file, it follows the same pattern as `cast/castlib.py`: copied in, attributed, with
its licence beside it.

Unknown until someone looks: which Wwise version the games ship, whether the banks sit in the RPACKs or in the
PAKs, and whether audio can be replaced without touching an event/ID table that would need a writer of its own.

## GUI modding

The game's interface is described in a markup format — XML or something close to it. This section would read,
present and rebuild those documents.

Unknown until someone looks: the exact format and where it lives (`gui_*.rpack` are the obvious candidates given
the pack names — `gui_common_pc`, `gui_hud_pc`, `gui_ingame_menu_pc`, `gui_main_menu_pc`, `gui_menu_common_pc`),
whether the markup is plain text or a compiled form, and how it references the textures and fonts it draws.

Likely the cheapest of the three to reach something useful, because a text-ish format needs no decoder before it
can be displayed.

## Data PAK modding

The `dataN.pak` archives hold `.model` documents, scripts and other data files. The reader, the `.model` join
rule and the PAK writer already exist (`nightrunner/pak/`), and the Build tab writes a mod PAK — but there is no
tab for browsing or editing PAK contents in their own right.

This section would cover the rest of what a PAK holds: scripts (`.scr`), and the `.to` / `.rscr` / behaviour-tree
formats that are currently listed as unknown (hole P7).

Worth knowing before starting: which `dataN` the game enumerates and in what order is **not decoded** (hole P1).
Only `N = 3` with root-level member names is proven, by testing rather than by reading the engine. Anything this
section does about load order rests on that convention.

---

## Order these are likely worth doing

1. **GUI modding** — probably a text format, so the shortest path to something usable.
2. **Data PAK** — half the machinery exists already; this is mostly a tab over code that is written.
3. **Audio** — needs a format investigation first, and possibly a dependency decision.
4. **SDB writing** — blocked on real reverse-engineering work (S8, S9), and the most likely to break a game
   install if it is got wrong.
