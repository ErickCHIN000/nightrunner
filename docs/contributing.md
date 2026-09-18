# Working on this repository

Where the code lives, what has to pass before a change is done, and how to get it onto GitHub.

## The repository

`https://github.com/ErickCHIN000/nightrunner` — **private** as of 2026-09-17. `main` is the only branch and is
pushed to directly; there is no PR flow set up.

Published: `nightrunner/`, `tests/`, `tools/`, `docs/`, the launchers and `README.md` / `CLAUDE.md`. Git-ignored:
`.venv/`, `.cache/`, `notes/`, `out/`, `exports/`, `sdb_exports/`, `.remove_this/`, `tests/data/`, and asset
artefacts (`*.rpack`, `*.pak`, `*.dds`, `*.cast`, …) wherever they land.

Two of those matter more than they look:

* **`notes/` is not published.** It is the maintainer's working record — format specs, decisions, open holes. A
  fresh clone does not have it, so anything in `docs/` that a stranger needs must not point there.
* **Dumps and exports are output, not source.** The SDB dump alone is ~112 MB; GitHub rejects any file over
  100 MB. Keep export output out of the repository, or in a git-ignored folder.

## Before a change is done

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
python tests\run.py -q
```

489 tests collected, 2 expected failures, **0 failures**. A clone with the game installed reports
`Ran 486 ... OK (skipped=135, expected failures=2)` — 486 rather than 489 because three skip at class level,
which unittest does not count in its total.

Run the whole suite, not just the file you touched. Several things in this project are only reachable through a
tab or a worker, and a passing unit test has more than once sat next to a real bug — see the notes on GUI work
below.

For anything touching real game data, run a census or a round trip over the corpus and quote the numbers in the
commit message. "It works" is not a result; "21,354/21,354 byte-identical" is.

## Committing

Conventional commit messages are not used here. Write a subject line that says what changed, then a body that
says **why**, with the measurements behind any decision that rests on one. Look at
`git log` for the shape — e.g. the SDB export commit records the census that justified flattening `routes` and
the three materials that stopped a de-duplication from being safe.

Every commit made with Claude Code ends with:

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```

Identity is set per repository so no personal email is attached:

```bash
git config user.name "ErickCHIN000"
git config user.email "61078393+ErickCHIN000@users.noreply.github.com"
```

## Pushing

```bash
git add -A
git status --short          # read this: an export folder is easy to sweep up by accident
git commit -F -             # heredoc the message
git push origin main
```

`gh` is installed at `C:\Program Files\GitHub CLI\gh.exe` and authenticated. It is not on every shell's PATH; a
new terminal picks it up, or call it by full path. Git pushes through Git Credential Manager, so no token is
handled in the shell.

If a push is rejected for file size, nothing reached GitHub. Recover with `git reset --mixed HEAD~1`, add the
offending path to `.gitignore`, re-stage and commit again — `reset --mixed` never touches the working tree, so
the files themselves are safe.

## The update indicator

The GUI shows the commit it is running and offers to fast-forward to the latest one
([gui-tabs.md](gui-tabs.md)). It reads `main` of the repository above. While the repository is private it needs a
token, which it picks up from `NIGHTRUNNER_GITHUB_TOKEN`, `GITHUB_TOKEN`, `GH_TOKEN` or the GitHub CLI's stored
auth; once the repository is public no token is needed at all. A fork points it elsewhere with
`NIGHTRUNNER_UPDATE_REPO`.

So a push is also what makes the update appear in everyone's window. Push work that is finished, not work in
progress.

## Notes on GUI work

Tabs live in `nightrunner/gui/tabs/`, one module each, registered in `nightrunner/gui/tabs/__init__.py` under
`SECTIONS`. Adding a section is one row in that table plus the tab modules.

Two habits worth keeping, both learned the hard way:

* **Split building a menu or a dialog from showing it.** A test that has to monkey-patch `QMenu.exec` to stop a
  modal blocking will eventually hang the suite for real. `item_menu(gid) -> QMenu` is testable; `exec` inside
  the builder is not.
* **Drive a new feature through the real tab once.** Unit tests with stubs pass happily against
  `self.svc.sdb()` when `sdb` is a property, and against a bare `TITLE` that is actually a class attribute. Both
  of those shipped into a commit and were caught only by building the tab and clicking the thing.

## Working rules

The full set is in `CLAUDE.md`. The ones that bite most often:

* Everything created stays inside the repository. Game installs are read-only reference data.
* The tool never writes into a game folder. A build produces files; installing them is the user's step.
* No new dependencies. Stdlib, numpy, Pillow, and PySide6 for the GUI.
* No guessing. Every structural claim carries its provenance tier, and unknown bytes are preserved verbatim
  rather than reinterpreted.
* Refuse rather than approximate. An encoder that cannot honour its input raises with the exact limitation named.
* Nothing personal ships: no machine paths, no user names, no local folder layouts in the published files.
