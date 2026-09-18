"""Tab sections: the outer bar, the tabs inside each section, and navigation across them.

Headless; skips without PySide6. The registry tests need nothing at all; the window tests build a real
MainWindow and so need a game install like the other window tests.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QTabWidget
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

from nightrunner.gui.tabs import SECTIONS, TAB_MODULES, section_of  # noqa: E402
from tests.paths import have_game  # noqa: E402


class RegistryTests(unittest.TestCase):
    def test_flat_list_is_every_section_in_order(self):
        self.assertEqual(TAB_MODULES, tuple(n for _, names in SECTIONS for n in names))

    def test_no_tab_is_in_two_sections(self):
        self.assertEqual(len(TAB_MODULES), len(set(TAB_MODULES)))

    def test_section_titles_are_unique(self):
        titles = [t for t, _ in SECTIONS]
        self.assertEqual(len(titles), len(set(titles)))

    def test_no_section_is_empty(self):
        self.assertTrue(all(names for _, names in SECTIONS))

    def test_section_of(self):
        self.assertEqual(section_of("raw"), "RPACK")
        self.assertEqual(section_of("sdb"), "SDB")
        self.assertIsNone(section_of("not-a-tab"))

    def test_every_tab_has_a_module(self):
        import importlib
        for name in TAB_MODULES:
            mod = importlib.import_module(f"nightrunner.gui.tabs.{name}")
            self.assertTrue(hasattr(mod, "Tab"), name)


@unittest.skipUnless(HAVE_QT and have_game(), "needs PySide6 and a game install")
class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.mainwindow import MainWindow
        cls.ctx = AppContext()
        cls.win = MainWindow(cls.ctx)

    @classmethod
    def tearDownClass(cls):
        cls.win._shutdown_tabs()
        cls.ctx.close()

    def test_outer_bar_is_the_sections(self):
        w = self.win
        self.assertEqual([w.tabs.tabText(i) for i in range(w.tabs.count())], [t for t, _ in SECTIONS])

    def test_a_multi_tab_section_has_an_inner_bar(self):
        page = self.win.tabs.widget(0)
        self.assertIsInstance(page, QTabWidget)
        self.assertEqual([page.tabText(i) for i in range(page.count())],
                         ["Raw", "Textures", "Meshes", "Models", "Build"])

    def test_a_single_tab_section_shows_its_tab_directly(self):
        """No inner bar for SDB: one tab has nothing to switch between."""
        page = self.win.tabs.widget(1)
        self.assertNotIsInstance(page, QTabWidget)
        self.assertIs(page, self.win.tab_objs["sdb"])

    def test_show_tab_reaches_every_tab(self):
        for name in TAB_MODULES:
            self.assertTrue(self.win.show_tab(name), name)
            self.assertEqual(self.win.current_tab_name(), name)

    def test_show_tab_rejects_an_unknown_name(self):
        self.assertFalse(self.win.show_tab("not-a-tab"))

    def test_navigation_crosses_sections(self):
        """openMaterial lands on SDB from an RPACK tab, and openMesh comes back."""
        self.win.show_tab("raw")
        self.win.ctx.openMaterial.emit("barrel_b.mat")
        self.app.processEvents()
        self.assertEqual(self.win.current_tab_name(), "sdb")
        self.win.ctx.openMesh.emit(0)
        self.app.processEvents()
        self.assertEqual(self.win.current_tab_name(), "meshes")

    def test_update_corner_survives_the_grouping(self):
        from nightrunner.gui.updater import UpdateCorner
        self.assertIsInstance(self.win.tabs.cornerWidget(), UpdateCorner)

    def test_the_open_tab_is_remembered_by_name(self):
        """Against a throwaway settings object: the real one belongs to the user's installed app."""
        from PySide6.QtCore import QSettings
        real = self.win.settings
        self.win.settings = QSettings(QSettings.IniFormat, QSettings.UserScope,
                                      "nightrunner-tests", "sections-test")
        try:
            self.win.settings.clear()
            self.win.show_tab("models")
            self.win._remember_tab()
            self.assertEqual(self.win.settings.value(self.win.TAB_KEY), "models")
            self.assertEqual(self.win._remembered_tab(), "models")
            self.assertEqual(int(self.win.settings.value("window/tab")), TAB_MODULES.index("models"))
            self.win.settings.clear()
        finally:
            self.win.settings = real



@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class RememberedTabTests(unittest.TestCase):
    """QSettings is per user, not per folder, so two builds on one machine share these keys.

    Regression: writing a tab *name* into `window/tab` made older builds crash on startup with
    `int('audio')`. The name now lives in its own key and `window/tab` is kept a valid index.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def win(self, values: dict):
        from PySide6.QtWidgets import QWidget
        from nightrunner.gui.mainwindow import MainWindow
        w = QWidget()
        w.settings = type("S", (), {
            "_d": dict(values),
            "value": lambda self, k, d=None: self._d.get(k, d),
            "setValue": lambda self, k, v: self._d.__setitem__(k, v),
        })()
        w.TAB_KEY = MainWindow.TAB_KEY
        return w

    def test_prefers_the_name_key(self):
        from nightrunner.gui.mainwindow import MainWindow
        w = self.win({MainWindow.TAB_KEY: "sdb", "window/tab": 0})
        self.assertEqual(MainWindow._remembered_tab(w), "sdb")

    def test_reads_a_legacy_index(self):
        from nightrunner.gui.mainwindow import MainWindow
        w = self.win({"window/tab": 1})
        self.assertEqual(MainWindow._remembered_tab(w), TAB_MODULES[1])

    def test_tolerates_a_name_left_in_the_old_key(self):
        """Exactly the value that crashed the older build."""
        from nightrunner.gui.mainwindow import MainWindow
        w = self.win({"window/tab": "audio"})
        self.assertEqual(MainWindow._remembered_tab(w), "audio")

    def test_tolerates_junk(self):
        from nightrunner.gui.mainwindow import MainWindow
        for junk in ("", None, "nonsense", 999):
            w = self.win({"window/tab": junk})
            self.assertIsInstance(MainWindow._remembered_tab(w), str)

    def test_saving_keeps_the_old_key_an_integer(self):
        from nightrunner.gui.mainwindow import MainWindow
        w = self.win({})
        w.current_tab_name = lambda: "audio"
        MainWindow._remember_tab(w)
        self.assertEqual(w.settings._d[MainWindow.TAB_KEY], "audio")
        self.assertEqual(w.settings._d["window/tab"], TAB_MODULES.index("audio"))
        self.assertIsInstance(w.settings._d["window/tab"], int)

    def test_saving_with_no_tab_still_writes_an_integer(self):
        from nightrunner.gui.mainwindow import MainWindow
        w = self.win({})
        w.current_tab_name = lambda: None
        MainWindow._remember_tab(w)
        self.assertEqual(w.settings._d["window/tab"], 0)


if __name__ == "__main__":
    unittest.main()
