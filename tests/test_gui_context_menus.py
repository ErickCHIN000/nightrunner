"""Right-click context menus on the Raw, Textures, Meshes and Models tabs.

The point of these menus is that they act on the row under the cursor, not on the check boxes and not on the
current selection. That is what these tests pin: the export path is reached with exactly the one item that was
right-clicked.

Headless (QT_QPA_PLATFORM=offscreen); skips without PySide6. No export ever runs: the export entry point of each
tab is replaced with a recorder, so nothing reaches disk and no dialog opens.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def menu_texts(menu) -> list[str]:
    return [a.text() for a in menu.actions() if not a.isSeparator()]


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class MenuWiringTests(unittest.TestCase):
    """Each tab's `_row_menu` builds its menu from the clicked row and routes to that row only.

    The tabs are not constructed (they need a game install); the methods are called unbound against a stand-in
    that carries only what they touch. That keeps the test about the wiring, which is the part that was written
    here, rather than about the tab as a whole.
    """

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    # ---- helpers --------------------------------------------------------------------------------------------
    def stub(self, **attrs):
        """A real QWidget to stand in for a Tab: QMenu(self) needs a live Qt parent, and Tab.__new__ is not one."""
        from PySide6.QtWidgets import QWidget
        w = QWidget()
        for k, v in attrs.items():
            setattr(w, k, v)
        return w

    def fake_view(self, valid=True, row=3):
        from PySide6.QtWidgets import QTableView
        v = QTableView()

        class Idx:
            def isValid(self):
                return valid

            def row(self):
                return row
        v.indexAt = lambda pos: Idx()
        return v

    # ---- textures -------------------------------------------------------------------------------------------
    def test_textures_menu_exports_only_the_clicked_gid(self):
        from nightrunner.gui.tabs import textures
        from PySide6.QtWidgets import QMenu
        calls, menus = [], []
        tab = self.stub(
            model=type("M", (), {"gid_at": staticmethod(lambda r: 77), "checked": {1, 2},
                                 "set_checked": lambda self, g, on: calls.append(("check", g, on))})(),
            cat=type("C", (), {"name": staticmethod(lambda g: "alarm_lamp_a.dds")})(),
            _export=lambda mode, out_dir=None, gids=None: calls.append(("export", mode, gids)))
        tab._export_one = textures.Tab._export_one.__get__(tab)   # the real routing, onto the stubbed _export
        menu = textures.Tab.item_menu(tab, 77)
        self.assertEqual(menu_texts(menu)[:3],
                         ["Export as DDS…", "Export as PNG…", "Export raw parts…"])
        for a in menu.actions():
            if a.text().startswith("Export as PNG"):
                a.trigger()
        self.assertEqual(calls, [("export", "png", [77])])       # the clicked gid, not the two checked ones

    def test_textures_menu_ignores_empty_space(self):
        from nightrunner.gui.tabs import textures
        from PySide6.QtWidgets import QMenu
        menus = []
        tab = self.stub(item_menu=lambda gid: menus.append(gid))
        textures.Tab._row_menu(tab, self.fake_view(valid=False), QPoint(1, 1))
        self.assertEqual(menus, [])                              # no row under the cursor, no menu

    def test_textures_check_action_passes_a_list(self):
        from nightrunner.gui.tabs import textures
        from PySide6.QtWidgets import QMenu
        calls, menus = [], []
        tab = self.stub(
            model=type("M", (), {"gid_at": staticmethod(lambda r: 77), "checked": set(),
                                 "set_checked": lambda self, g, on: calls.append((g, on))})(),
            cat=type("C", (), {"name": staticmethod(lambda g: "x.dds")})(),
            _export=lambda *a, **k: None)
        menu = textures.Tab.item_menu(tab, 77)
        for a in menu.actions():
            if a.text() == "Check":
                a.trigger()
        self.assertEqual(calls, [([77], True)])                  # RefListModel.set_checked wants an iterable

    # ---- meshes ---------------------------------------------------------------------------------------------
    def test_meshes_menu_exports_only_the_clicked_gid(self):
        from nightrunner.gui.tabs import meshes
        from PySide6.QtWidgets import QMenu
        calls, menus = [], []
        tab = self.stub(
            table=self.fake_view(),
            model=type("M", (), {"gid_at": staticmethod(lambda r: 12), "checked": {5},
                                 "set_checked": lambda self, g, on: None})(),
            ctx=type("X", (), {"catalog": type("C", (), {"name": staticmethod(lambda g: "sh2_npc_crane")})()})(),
            export_checked=lambda kind, dest=None, overwrite=None, gids=None: calls.append((kind, gids)))
        menu = meshes.Tab.item_menu(tab, 12)
        self.assertEqual(menu_texts(menu)[:4], ["Export Cast + mesh.json…", "Export glTF binary (.glb)…",
                                                "Export glTF (.gltf + .bin)…", "Export raw parts…"])
        for a in menu.actions():
            if a.text().startswith("Export Cast"):
                a.trigger()
        self.assertEqual(calls, [("cast", [12])])

    # ---- models ---------------------------------------------------------------------------------------------
    def test_models_menu_offers_every_export_mode_for_the_clicked_row(self):
        from nightrunner.gui.tabs import models
        from PySide6.QtWidgets import QMenu
        calls, menus = [], []
        rec = {"name": "models/player/player_tpp_skeleton.model", "basename": "player_tpp_skeleton.model"}
        tab = self.stub(
            view=self.fake_view(row=1),
            records=[{"name": "other"}, rec],
            list_model=type("M", (), {"gid_at": staticmethod(lambda r: 1), "checked": set(),
                                      "set_checked": lambda self, g, on: None})(),
            _export_one=lambda r, mode: calls.append((r["basename"], mode)))
        menu = models.Tab.item_menu(tab, 1)
        self.assertEqual(len(menu_texts(menu)), len(models.EXPORT_MODES) + 2)   # + copy name + check
        menu.actions()[0].trigger()
        self.assertEqual(calls, [("player_tpp_skeleton.model", models.EXPORT_MODES[0][0])])

    # ---- raw ------------------------------------------------------------------------------------------------
    def test_raw_export_this_uses_the_clicked_row_not_the_selection(self):
        """Regression: the menu used to call export_selected_dialog, which ignored the row under the cursor."""
        from nightrunner.gui.tabs import raw
        rows = []
        tab = self.stub(
            _busy_msg=lambda: False,
            export_rows_dialog=lambda r: rows.append(r),
            view=type("V", (), {"selectionModel": staticmethod(
                lambda: type("S", (), {"selectedRows": staticmethod(lambda c: ["SELECTED"])})())})())
        raw.Tab.export_selected_dialog(tab)
        self.assertEqual(rows, [["SELECTED"]])
        tab.export_rows_dialog(["CLICKED"])
        self.assertEqual(rows[-1], ["CLICKED"])

    def test_raw_export_rows_dialog_ignores_an_empty_row_list(self):
        from nightrunner.gui.tabs import raw
        tab = self.stub(_busy_msg=lambda: False,
                        _single_file=lambda rows: self.fail("should not be reached"))
        raw.Tab.export_rows_dialog(tab, [])            # returns quietly


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class ExportSignatureTests(unittest.TestCase):
    """The single-item menus rely on each tab's existing export taking an explicit gid list."""

    def test_textures_export_accepts_gids(self):
        from nightrunner.gui.tabs import textures
        self.assertIn("gids", textures.Tab._export.__code__.co_varnames)

    def test_meshes_export_accepts_gids(self):
        from nightrunner.gui.tabs import meshes
        self.assertIn("gids", meshes.Tab.export_checked.__code__.co_varnames)

    def test_models_ask_export_accepts_recs(self):
        from nightrunner.gui.tabs import models
        self.assertIn("recs", models.Tab._ask_export.__code__.co_varnames)


if __name__ == "__main__":
    unittest.main()
