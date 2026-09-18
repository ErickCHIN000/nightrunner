"""One module per main-window tab. Each exposes `class Tab(QWidget)` with `TITLE`, `__init__(self, ctx)`, and
optional navigation slots (`open_gid(gid)`, `open_material(name)`, `open_model(name)`) plus `shutdown()`.

Tabs are grouped into sections, shown as an outer tab bar with the tabs of the current section below it. A
section holding a single tab shows that tab directly, without a second bar that would have nothing to switch
between.

Planned sections that do not exist yet are listed in `docs/roadmap.md`; adding one means writing its tab modules
and adding a row here, and nothing else in the window.
"""

#: ((section title, (tab module name, ...)), ...) - the order the window shows them in.
SECTIONS = (
    ("RPACK", ("raw", "textures", "meshes", "models", "build")),
    ("SDB", ("sdb",)),
    ("AUDIO", ("audio",)),
)

#: Every tab module, in window order. Kept flat for the places that only care about the set of tabs.
TAB_MODULES = tuple(name for _, names in SECTIONS for name in names)


def section_of(tab: str) -> str | None:
    """The section title a tab module belongs to, or None when it is not in any."""
    for title, names in SECTIONS:
        if tab in names:
            return title
    return None
