"""Shared state of the Build tab: the open project, its game view, dirty tracking and change signals.

Both Build pages (Resources = tabs/build.py, Model overrides = tabs/build_models.py) hold the same BuildState and
talk only through it: edit `state.project`, then call `state.touch("items" | "models" | "settings")`.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..project import GameEnv, Project, next_free_pak, next_free_rpack


class CatalogEnv(GameEnv):
    """GameEnv backed by the explorer's catalog (same packs, same order, no second table scan). Only packs that
    are not user-opened count as game packs, so a previously built mod pack never matches as a target."""

    def __init__(self, ctx):
        game = ctx.game
        super().__init__(game.assets if game else None, game.source if game else None, packs={},
                         profile=game.profile if game else None)
        self.ctx = ctx

    def _entries(self):
        return [e for e in self.ctx.catalog.packs if e.pack is not None and not e.user]

    @property
    def packs(self):                       # label -> path, live
        return {e.label: e.path for e in self._entries()}

    @packs.setter
    def packs(self, _v):                   # GameEnv.__init__ assigns; ignored
        pass

    def find(self, name: str, type_id: int, pack: str = "") -> list[tuple[str, int]]:
        cat = self.ctx.catalog
        out = []
        for gid in cat.lookup(name, type_id):
            e, i = cat.split(gid)
            if e.user or (pack and e.label != pack):
                continue
            out.append((e.label, i))
        return out


class BuildState(QObject):
    itemsChanged = Signal()
    modelsChanged = Signal()
    settingsChanged = Signal()
    projectReplaced = Signal()             # new / open: pages rebuild everything
    dirtyChanged = Signal(bool)
    built = Signal(dict)                   # build report
    problemsChanged = Signal(list)         # [project.Problem]

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.env = CatalogEnv(ctx)
        self.project = Project()
        self.dirty = False
        self.last_report: dict | None = None

    # ---- lifecycle ----------------------------------------------------------------------------------------------
    def new(self, name: str = "untitled") -> None:
        self.project = Project(name=name, game=self.env.profile.id)
        self.last_report = None
        self._set_dirty(False)
        self.projectReplaced.emit()

    def open(self, path: str | Path) -> None:
        self.project = Project.load(path)
        self.last_report = None
        self._set_dirty(False)
        self.ctx.settings.setValue("build/last_project", str(path))
        self.projectReplaced.emit()

    def save(self, path: str | Path | None = None) -> Path:
        p = self.project.save(path)
        self.ctx.settings.setValue("build/last_project", str(p))
        self._set_dirty(False)
        return p

    def touch(self, what: str = "items") -> None:
        self._set_dirty(True)
        {"items": self.itemsChanged, "models": self.modelsChanged}.get(what, self.settingsChanged).emit()

    def _set_dirty(self, on: bool) -> None:
        if on != self.dirty:
            self.dirty = on
            self.dirtyChanged.emit(on)

    # ---- output names ------------------------------------------------------------------------------------------
    def default_rpack(self) -> str:
        return next_free_rpack(self.env.assets, profile=self.env.profile)

    def default_pak(self) -> str:
        return next_free_pak(self.env.source, profile=self.env.profile)

    def game_mismatch(self) -> bool:
        return (self.project.game or "dltb") != self.env.profile.id

    def rpack_name(self) -> str:
        return self.project.rpack_name or self.default_rpack()

    def pak_name(self) -> str:
        return self.project.pak_name or self.default_pak()

    def output_dir(self) -> Path:
        if self.project.output_dir:
            return Path(self.project.output_dir)
        if self.project.path:
            return Path(self.project.path).parent / "out"
        return Path.cwd() / "out" / self.project.name
