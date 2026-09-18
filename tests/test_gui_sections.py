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
        self.win.show_tab("models")
        self.win.settings.setValue("window/tab", self.win.current_tab_name() or "")
        self.assertEqual(self.win.settings.value("window/tab"), "models")


if __name__ == "__main__":
    unittest.main()
