# Roadmap — planned tab sections

The GUI groups its tabs into sections (`nightrunner/gui/tabs/__init__.py`, `SECTIONS`). Two exist; three are
planned. This file records what each planned section is meant to cover and what is known about it, so the work
can start without re-deriving the plan.

Nothing below is implemented, and nothing below has been confirmed in a running game. Where a section says a
number was measured, it was; everything else is a starting hypothesis to check, not a finding. The project's
provenance rule (A / B / C / RT, see [capabilities.md](capabilities.md)) applies to this file as much as to the
code.

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

Investigated 2026-09-17. The container format is measured and documented in the maintainer's format notes; what
follows is the shape of the work and what blocks it.

**Where the audio is.** Four `.aesp` containers in `<root>/<data>/work/data/audio/`, 9.2 GB in total: `init`
(the Wwise init bank), `meta` (123 soundbanks plus an XML registry), `sfx` (26,517 in-memory `.wem`) and
`streams` (3,530 streamed `.wem`). AESP is a Techland wrapper, not a Wwise format: a header, a table of
152-byte entries (128-byte name, a u32 the other tooling calls a CRC, then u64 offset and size), and a payload
region that is contiguous and in table order — 30,172/30,172 members measured.

**Naming is already solved.** One member of `meta.aesp`, `wwisepinhead`, is 6.7 MB of plain XML mapping Wwise
IDs to names: 23,904 events, plus switches, states, game parameters and aux buses, and a `<Preload>` list tying
bank names to their IDs. Most Wwise projects need a guessed wordlist to recover names; this game ships them.

**Two references, and what each is good for.**

[wwiser](https://github.com/bnnm/wwiser) is the mature `.bnk` parser — pure Python, active, 369 stars. It is
explicitly read-only: *"This tool is not, and will never be, a `.bnk` editor (can't replace files)."* So it can
teach us the bank format and generate playable `.txtp` for vgmstream, but it cannot do the half of the job that
modding needs. **It also declares no licence at all**, which means default copyright and no permission to
vendor. Reading it to learn the format is fine; copying it into this repository is not, unless the author adds
a licence. That closes the "just vendor a library" option the earlier draft of this roadmap hoped for.

UTM-AIO (the existing C# toolkit) covers what wwiser will not: `AespExtractor`, `BnkParser` (4,168 lines),
`AudioInjectorService`, `AudioPackerService`, `BankCloner`, `BankLocator`, `PinheadPatcher`. Its approach is to
extract, replace a `.wem`, patch the bank, patch the `wwisepinhead` registry, and repack. That is the recipe to
follow; it is (B)-tier evidence, to be re-verified rather than trusted, and one of its assumptions already does
not hold on this game — see below.

**How modding it actually works.** The author of UTM-AIO (ODST) described the technique, and the shipped data
backs it up. Banks are Wwise version 150, and every Sound object carries a stream-type byte: 0 embedded,
1 prefetch, 2 streamed. The game embeds nearly everything — 68,874 embedded against 239 prefetch and 217 streamed
across 69,330 Sound objects. Flipping that one byte from 0 to 2 makes the engine fetch the audio from the stream
store instead of from inside the bank, so the bank keeps its exact size and layout and never has to be rebuilt.
The replacement `.wem` is supplied in a small separate `.aesp` alongside a patched `meta.aesp`; the 2.2 GB `sfx`
and 6.6 GB `streams` containers are left alone.

That patching is his own invention rather than anything wwiser does, and he reports it works universally as far
as he has tested. He deliberately stopped there: real bank editing "has much more potential" but was skipped
"because it is hard to simplify … banks are complicated and nested".

**What is settled.** The AESP container is fully measured, and the field the other tool calls a CRC turned out to
be the Wwise id — the numeric name for `sfx`/`streams` members, the FNV-1 hash of the lowercased name for banks,
reproduced on 30,170/30,172 members. A writer can compute it, so building a container from scratch is not
blocked. Bank parsing is also in reach: all 123 banks walk cleanly to their exact end, on one Wwise version.

**What is still open.**

1. Whether `mods/audio/` is a real engine path and how an extra `.aesp` is mounted. The entire delivery mechanism
   rests on this, and it is the audio equivalent of the numbered-pack convention — confirmed by someone else's
   testing, not by reading the engine.
2. `.bnk` beyond the chunk and object walk: the rest of a Sound body, the other 18 object types, `DIDX`/`DATA`
   pairing. Needed for anything past stream-type patching.
3. `PinheadPatcher` rewrites a `<File id="...">` whitelist inside each `<Preload>`. DLTB's shipped registry has
   **no `<File>` elements at all** — 0 of 128 preloads, measured. Reading the caller settles what it is: it runs in
   "update mode" (same bank name in and out), removes any `<File>` children and adds one per new wem id. So the
   whitelist is the tool's own invention, which matches its author describing the patching side as something he
   made up rather than ported. Whether the engine reads those elements at all, or whether the work is really done
   by the stream-type flip and the extra mod container, is untested.
4. **Encoding a `.wem` may not need Audiokinetic's tooling after all.** The game ships two PCM wems (source
   plugin `0x00010001`, `fx` bank), so the engine's Wwise runtime carries the PCM codec, and PCM is writable with
   `struct`. `nightrunner/audio/wem.py` builds that shape: its `fmt` chunk is byte-identical to the shipped one,
   and vgmstream reads what it writes bit-exactly. Two unknowns remain before this is an import path — the
   16-byte content-derived `hash` chunk that nothing here can compute (E12), and that swapping a sound to PCM
   also means patching its plugin id in the bank (E13). Neither is confirmed in game. The original note below
   still stands for Vorbis specifically.

5. **Encoding Wwise Vorbis needs Audiokinetic's own tooling.** This was worth checking properly, because it changes
   what the dependency question even is. UTM-AIO does not encode audio with ffmpeg: its build notes say ffmpeg is
   there "only to normalise WAV files" — resample to 44.1 kHz, mix to stereo, write `pcm_s16le`. The actual
   conversion is `WwiseConsole.exe convert-external-source` against a template `.wproj`, i.e. the Wwise authoring
   application. So the missing piece is not a library anyone can vendor; it is a licensed third-party toolchain
   the user must install. Reading audio out needs vgmstream; writing audio in needs Wwise.
6. `<data>/work/data_lang/speech_en/` has not been looked at.

**A reasonable first slice** is read-only and needs none of the above settled: an AESP reader, the registry
parsed into names, a bank chunk/HIRC walk, and extraction of banks and `.wem` to disk. It is squarely what this
project already does well, has no dependency problem, and would let an Audio tab list, search and export audio —
23,904 named events make that genuinely useful on its own. Stream-type patching is a small step after it; real
bank editing is the frontier ODST left open.

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
3. **Audio** — the container is now measured, so a read-only slice (browse and extract) could start today; the
   injection half waits on the CRC and resize questions, and on a `.wem` encoder decision.
4. **SDB writing** — blocked on real reverse-engineering work (S8, S9), and the most likely to break a game
   install if it is got wrong.
