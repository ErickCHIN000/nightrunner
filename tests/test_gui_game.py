"""Game picker (gui/mainwindow.py + gui/context.py): choice resolution from QSettings, Game menu, in-place switch
between two fake installs (DLTB `ph_ft`, DL2 `ph`), Browse… detection, Build tab mismatch warning. Offscreen."""
from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

import tests.synth as S
from nightrunner import games as G
from tests.test_games import dl2, dltb

try:
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:  # pragma: no cover
    HAVE_QT = False


def _pump(app, cond=lambda: False, n=200):
    import time
    for _ in range(n):
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    return False


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class GamePickerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._tmp = S.tmpdir("gui_game_")
        d = self.d = Path(self._tmp.name)
        self.lib = d / "lib"
        common = self.lib / "steamapps" / "common"
        self.b = dltb(common / "Dying Light The Beast")
        self.t = dl2(common / "Dying Light 2")
        self.settings = QSettings(str(d / "s.ini"), QSettings.IniFormat)
        self._patches = [mock.patch.object(G, "_libraries", lambda: [self.lib]),
                         mock.patch.dict(os.environ, {}, clear=False)]
        for p in self._patches:
            p.start()
        os.environ.pop("NIGHTRUNNER_GAME_ROOT", None)
        os.environ.pop("BEASTPACK_GAME_ROOT", None)

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self._tmp.cleanup()

    def test_resolve(self):
        from nightrunner.gui.context import remembered_roots, resolve_game
        self.assertEqual(resolve_game(self.settings).id, "dltb")          # first run: DLTB
        self.settings.setValue("game/current", "dl2")
        self.assertEqual(resolve_game(self.settings).root, self.t)
        other = dl2(self.d / "moved")
        self.settings.setValue("game/root/dl2", str(other))
        self.assertEqual(resolve_game(self.settings).root, other)
        self.settings.setValue("game/current", "dl9")                      # unknown: DLTB
        self.assertEqual(resolve_game(self.settings).id, "dltb")
        s2 = QSettings(str(self.d / "legacy.ini"), QSettings.IniFormat)
        s2.setValue("paths/game_root", str(self.b))                        # pre-profile setting
        self.assertEqual(remembered_roots(s2), {"dltb": str(self.b)})
        with mock.patch.object(G, "_libraries", lambda: []):
            self.assertEqual(resolve_game(s2).root, self.b)
            self.assertIsNone(resolve_game(QSettings(str(self.d / "e.ini"), QSettings.IniFormat)))

    def test_switch(self):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.mainwindow import MainWindow
        ctx = AppContext(self.settings)
        self.assertEqual(ctx.game.id, "dltb")
        win = MainWindow(ctx)
        try:
            self.assertEqual(win.windowTitle(), "Nightrunner — Dying Light: The Beast")
            win._sync_game_menu(refresh=True)
            acts = win._game_actions
            self.assertTrue(acts["dltb"].isChecked() and acts["dl2"].isVisible())
            old_tabs = dict(win.tab_objs)
            build = win.tab_objs["build"]
            self.assertEqual(build.state.env.profile.id, "dltb")
            self.assertEqual(build.state.project.game, "dltb")

            acts["dl2"].trigger()
            self.assertIsNot(win.ctx, ctx)
            self.assertTrue(ctx.catalog.closed)
            self.assertEqual(win.ctx.game.root, self.t)
            self.assertEqual(win.windowTitle(), "Nightrunner — Dying Light 2")
            self.assertEqual(win.settings.value("game/current"), "dl2")
            self.assertEqual(win.settings.value("game/root/dl2"), str(self.t))
            self.assertTrue(acts["dl2"].isChecked() and not acts["dltb"].isChecked())
            self.assertEqual(win.tabs.count(), len(old_tabs))
            self.assertTrue(all(win.tab_objs[k] is not old_tabs[k] for k in old_tabs))
            self.assertEqual(win.ctx.sdb.path, self.t / "ph/work/data_platform/pc/assets/runtime_dx11.sdb")
            self.assertEqual([p.name for p in win.ctx.paks.paths], ["data0.pak", "data1.pak"])
            st = win.tab_objs["build"].state
            self.assertEqual((st.env.profile.id, st.env.source), ("dl2", self.t / "ph" / "source"))
            self.assertEqual(st.default_pak(), "data2.pak")
            self.assertFalse(win.switch_game(win.ctx.game))                 # same game: no-op

            # a DLTB project opened under DL2 -> Problems warning
            from nightrunner.project import Project
            f = Project(name="p", game="dltb").save(self.d / "p.nrproj")
            tab = win.tab_objs["build"]
            tab.open_project(str(f))
            tab.validate()
            _pump(self.app, lambda: any("current game" in p.message for p in tab.problems))
            self.assertTrue(any("made for Dying Light: The Beast" in p.message for p in tab.problems))

            # unsaved project + cancel: stays
            st = win.tab_objs["build"].state
            st.touch("settings")
            win._confirm_switch = lambda: False
            self.assertFalse(win.switch_game(G.GameInstall(self.b)))
            self.assertEqual(win.ctx.game.id, "dl2")
            del win._confirm_switch

            # Browse…: detection from the picked folder (the data folder itself works too)
            st.dirty = False
            self.assertTrue(win._set_game(str(self.b / "ph_ft")))
            self.assertEqual((win.ctx.game.id, win.ctx.game.root), ("dltb", self.b))
            with mock.patch("nightrunner.gui.mainwindow.QMessageBox.warning") as warn:
                self.assertFalse(win._set_game(str(self.d / "nothing")))
                warn.assert_called_once()
            self.assertEqual(win.ctx.game.id, "dltb")
        finally:
            win.close()
            win.deleteLater()
            _pump(self.app, n=5)

    def test_profile_features(self):
        """A profile without the player scripts hides Player… / No gear instead of failing."""
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.tabs import build_models as BM
        bare = G.GameProfile(id="dltb", name="Bare", steam_dir="x", data_dirs=("ph_ft",))
        gi = G.GameInstall(self.b, bare)
        ctx = AppContext(self.settings, gi, autoload=False)
        self.assertEqual(BM.read_appearances(ctx), [])
        self.assertIsNone(BM._profile(ctx).outfit_script)
        from nightrunner.gui.tabs.build import Tab
        tab = Tab(ctx)
        try:
            page = tab.models_page
            self.assertTrue(page.no_gear.isHidden())
            self.assertFalse(page.no_gear.isEnabled())
            from PySide6.QtWidgets import QToolButton
            texts = [a.text() for b in page.findChildren(QToolButton) if b.menu() for a in b.menu().actions()]
            self.assertIn("Pick .model…", texts)
            self.assertNotIn("Player…", texts)
        finally:
            tab.shutdown()
            tab.deleteLater()
            ctx.close()


if __name__ == "__main__":
    unittest.main()
