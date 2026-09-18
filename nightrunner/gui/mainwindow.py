"""Nightrunner main window: one tab per asset view, a shared AppContext, background indexing status."""
from __future__ import annotations

import importlib
import traceback
from pathlib import Path

from PySide6.QtGui import QAction, QActionGroup, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QLabel, QMainWindow, QMessageBox, QProgressBar,
                               QTabWidget, QTextEdit)

from .. import __version__
from . import APP_NAME
from ..games import PROFILES, GameInstall, detect_profile
from .context import AppContext, detected_installs, remember_game
from .tabs import TAB_MODULES
from .theme import apply_theme


class MainWindow(QMainWindow):
    def __init__(self, ctx: AppContext | None = None):
        super().__init__()
        self.ctx = ctx or AppContext()
        self.settings = self.ctx.settings
        self._retired: list[AppContext] = []      # switched-away contexts (their index threads may still run)
        self._installs: dict[str, GameInstall] | None = None
        self.resize(1600, 950)
        self.setAcceptDrops(True)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.setCentralWidget(self.tabs)
        self.tab_objs: dict[str, object] = {}

        self.prog = QProgressBar()
        self.prog.setMaximumWidth(260)
        self.prog.setFormat("indexing packs %v / %m")
        self.game_label = QLabel()
        self.statusBar().addPermanentWidget(self.game_label)
        self.statusBar().addPermanentWidget(self.prog)

        self._build_menu()
        self._attach(self.ctx)
        g = self.settings.value("window/geometry")
        if g is not None:
            self.restoreGeometry(g)
        self.tabs.setCurrentIndex(int(self.settings.value("window/tab", 0)))
        if self.ctx.game is not None and ctx is None:
            remember_game(self.settings, self.ctx.game)

    # ---- context ----------------------------------------------------------------------------------------------
    def _attach(self, ctx: AppContext) -> None:
        """Wire *ctx* to the window and build the tabs on it."""
        self.ctx = ctx
        cat = ctx.catalog
        cat.progress.connect(self._on_progress)
        cat.ready.connect(self._on_ready)
        ctx.status.connect(self._on_status)
        ctx.openTexture.connect(lambda g: self._goto("textures", "open_gid", g))
        ctx.openMesh.connect(lambda g: self._goto("meshes", "open_gid", g))
        ctx.openRaw.connect(lambda g: self._goto("raw", "open_gid", g))
        ctx.openMaterial.connect(lambda n: self._goto("sdb", "open_material", n))
        ctx.openModel.connect(lambda n: self._goto("models", "open_model", n))
        ctx.openModelDoc.connect(lambda n, d, p: self._goto("models", "open_model_doc", n, d, p))
        ctx.sdbChanged.connect(self._on_sdb_changed)
        for name in TAB_MODULES:
            w = self._make_tab(name)
            self.tab_objs[name] = w
            self.tabs.addTab(w, getattr(w, "TITLE", name.capitalize()))
        self._update_game_label()
        self.prog.setVisible(bool(cat.packs) and not cat.is_ready)
        if ctx.game is None:
            self.statusBar().showMessage("No game — Game > Browse…", 0)
        else:
            self.statusBar().clearMessage()

    def _detach(self) -> None:
        """Shut the tabs down and retire the current context (closed, signals cut)."""
        self._shutdown_tabs()
        self.tabs.clear()
        for w in self.tab_objs.values():
            w.setParent(None)
            w.deleteLater()
        self.tab_objs = {}
        old = self.ctx
        for sig in (old.catalog.progress, old.catalog.ready, old.catalog.packAdded, old.sdb.reverseReady,
                    old.sdb.failed, old.status, old.sdbChanged, old.openTexture,
                    old.openMesh, old.openRaw, old.openMaterial, old.openModel, old.openModelDoc):
            try:
                sig.disconnect()
            except (RuntimeError, TypeError):
                pass
        old.runner.pool.waitForDone(2000)
        old.close()
        self._retired.append(old)

    def installs(self, refresh: bool = False) -> dict[str, GameInstall]:
        if self._installs is None or refresh:
            self._installs = detected_installs(self.settings)
            if self.ctx.game is not None:
                self._installs.setdefault(self.ctx.game.id, self.ctx.game)
        return self._installs

    def switch_game(self, game: GameInstall, force: bool = False) -> bool:
        """Reload every tab on *game* (same window). False when unchanged or the user cancels."""
        if not force and self.ctx.game is not None and game == self.ctx.game:
            return False
        if not self._confirm_switch():
            self._sync_game_menu()
            return False
        remember_game(self.settings, game)
        if self._installs is not None:
            self._installs[game.id] = game
        tab = self.tabs.currentIndex()
        self._detach()
        self._attach(AppContext(self.settings, game, autoload=True))
        self.tabs.setCurrentIndex(max(0, tab))
        self._sync_game_menu()
        return True

    def _confirm_switch(self) -> bool:
        """Unsaved Build project: Save / Discard / Cancel."""
        st = getattr(self.tab_objs.get("build"), "state", None)
        if st is None or not st.dirty:
            return True
        r = QMessageBox.question(self, APP_NAME, "Save project?",
                                 QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel)
        if r == QMessageBox.Save:
            return bool(self.tab_objs["build"].save_project())
        return r == QMessageBox.Discard

    # ---- tabs -------------------------------------------------------------------------------------------------
    def _make_tab(self, name: str):
        try:
            mod = importlib.import_module(f".tabs.{name}", __package__)
            return mod.Tab(self.ctx)
        except Exception:  # noqa: BLE001 - one broken tab must not take the window down
            w = QTextEdit()
            w.setReadOnly(True)
            w.setPlainText(f"The {name} tab failed to load:\n\n{traceback.format_exc()}")
            w.TITLE = f"{name.capitalize()} (error)"
            return w

    def _goto(self, name: str, slot: str, *args) -> None:
        w = self.tab_objs.get(name)
        fn = getattr(w, slot, None)
        if fn is None:
            return
        self.tabs.setCurrentWidget(w)
        fn(*args)

    # ---- status -----------------------------------------------------------------------------------------------
    def _on_progress(self, done: int, total: int) -> None:
        self.prog.show()
        self.prog.setRange(0, total)
        self.prog.setValue(done)

    def _on_ready(self) -> None:
        cat = self.ctx.catalog
        failed = [p for p in cat.packs if p.error]
        self.prog.hide()
        msg = f"{len(cat.packs) - len(failed)} packs, {len(cat):,} resources indexed"
        if failed:
            msg += f"; {len(failed)} failed: " + ", ".join(p.label for p in failed[:3])
        self.statusBar().showMessage(msg, 15000)

    def _on_status(self, m: str) -> None:
        self.statusBar().showMessage(m, 8000)

    def _update_game_label(self) -> None:
        g = self.ctx.game
        self.game_label.setText(str(g.root) if g else "")
        self.setWindowTitle(f"{APP_NAME} — {g.name}" if g else APP_NAME)

    # ---- menu -------------------------------------------------------------------------------------------------
    def _build_menu(self) -> None:
        mb = self.menuBar()
        m = mb.addMenu("&File")
        a = m.addAction("Open Extra Packs…", self._open_packs)
        a.setShortcut(QKeySequence.Open)
        m.addSeparator()
        sm = m.addMenu("SDB")
        grp = QActionGroup(self)
        cur = self.ctx.sdb_api()
        self._sdb_actions = {}
        for api in ("dx11", "dx12"):
            act = QAction(f"runtime_{api}.sdb", self, checkable=True)
            act.setChecked(api == cur)
            act.triggered.connect(lambda _=False, api=api: self.ctx.set_sdb_api(api))
            grp.addAction(act)
            sm.addAction(act)
            self._sdb_actions[api] = act
        m.addSeparator()
        q = m.addAction("Quit", self.close)
        q.setShortcut(QKeySequence.Quit)
        self.game_menu = mb.addMenu("&Game")
        self.game_menu.aboutToShow.connect(lambda: self._sync_game_menu(refresh=True))
        self._game_group = QActionGroup(self)
        self._game_actions: dict[str, QAction] = {}
        for pid, pr in PROFILES.items():
            act = QAction(pr.name, self, checkable=True)
            act.triggered.connect(lambda _=False, pid=pid: self._pick_game(pid))
            self._game_group.addAction(act)
            self.game_menu.addAction(act)
            self._game_actions[pid] = act
        self.game_menu.addSeparator()
        self.game_menu.addAction("Browse…", self._set_game)
        v = mb.addMenu("&View")
        dark = QAction("Dark Theme", self, checkable=True)
        dark.setChecked(self.settings.value("view/dark", True, type=bool))
        dark.toggled.connect(self._toggle_dark)
        v.addAction(dark)
        h = mb.addMenu("&Help")
        h.addAction("About", lambda: QMessageBox.about(
            self, APP_NAME, f"<b>{APP_NAME}</b> {__version__}<br>Offline asset explorer for Chrome Engine games "
            "(" + ", ".join(p.name for p in PROFILES.values()) + ")."))
        self._sync_game_menu()

    def _toggle_dark(self, on: bool) -> None:
        self.settings.setValue("view/dark", on)
        apply_theme(QApplication.instance(), on)

    def _open_packs(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Open extra RPACK files", "", "RPACK (*.rpack)")
        if files:
            self.ctx.open_user_packs([Path(f) for f in files])

    def _sync_game_menu(self, refresh: bool = False) -> None:
        found = self.installs(refresh)
        cur = self.ctx.game.id if self.ctx.game is not None else None
        for pid, act in self._game_actions.items():
            gi = found.get(pid)
            act.setVisible(gi is not None)
            act.setToolTip(str(gi.root) if gi else "")
            act.setChecked(pid == cur)

    def _pick_game(self, pid: str) -> None:
        gi = self.installs().get(pid)
        if gi is not None:
            self.switch_game(gi)
        else:
            self._sync_game_menu()

    def _browse_dir(self) -> str:
        """Folder dialog (tests replace this)."""
        start = str(self.ctx.game.root.parent) if self.ctx.game else ""
        return QFileDialog.getExistingDirectory(self, "Game folder", start)

    def _set_game(self, d: str | None = None) -> bool:
        d = d or self._browse_dir()
        if not d:
            return False
        pr = detect_profile(d)
        gi = GameInstall(d, pr) if pr else None
        if gi is None or not gi.valid():
            QMessageBox.warning(self, APP_NAME, "No game data here.")
            return False
        return self.switch_game(gi, force=True)

    def _on_sdb_changed(self, api: str) -> None:
        act = self._sdb_actions.get(api)
        if act is not None and not act.isChecked():
            act.setChecked(True)
        for w in self.tab_objs.values():
            fn = getattr(w, "sdb_changed", None)
            if fn:
                fn()
        self.statusBar().showMessage(f"SDB: runtime_{api}.sdb", 6000)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        packs = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.toLocalFile().lower().endswith(".rpack")]
        if packs:
            self.ctx.open_user_packs(packs)

    def _shutdown_tabs(self) -> None:
        for w in self.tab_objs.values():
            fn = getattr(w, "shutdown", None)
            if fn:
                try:
                    fn()
                except Exception:  # noqa: BLE001
                    pass

    def closeEvent(self, e):
        self.settings.setValue("window/geometry", self.saveGeometry())
        self.settings.setValue("window/tab", self.tabs.currentIndex())
        self._shutdown_tabs()
        self.ctx.runner.pool.waitForDone(2000)
        self.ctx.close()
        super().closeEvent(e)
