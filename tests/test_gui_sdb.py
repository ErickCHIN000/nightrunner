"""Nightrunner SDB tab: open the real SDB through AppContext, list/search, material detail, presets,
reverse texture lookup, used-by scans (models + meshes), missing-SDB state.

Headless (QT_QPA_PLATFORM=offscreen); skips without PySide6 or without a game install (NIGHTRUNNER_GAME_ROOT or a
Steam install with runtime_dx11.sdb). The reverse-index disk cache is redirected to a temp dir.
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def _game():
    if not HAVE_QT:
        return None
    from nightrunner.gui.game import find_game
    g = find_game(game="dltb")
    return g if g is not None and g.sdb("dx11").is_file() else None


GAME = _game()
TORSO = "player_kc_basic_torso_a_tpp.mat"
SHARED = "zipper_a.mat"          # used by data0.pak models and embedded in a sample mesh


def _wait(app, cond, timeout: float = 60.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.005)
    app.processEvents()
    return bool(cond())


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = tempfile.TemporaryDirectory(prefix="bp_gui_sdb_")
        cls.tmp = Path(cls._tmp.name)
        import nightrunner.gui.services as services
        cls._old_cache = services.CACHE_DIR
        services.CACHE_DIR = cls.tmp / "cache"

    @classmethod
    def tearDownClass(cls):
        import nightrunner.gui.services as services
        services.CACHE_DIR = cls._old_cache
        cls._tmp.cleanup()

    def settings(self, name: str) -> QSettings:
        return QSettings(str(self.tmp / f"{name}.ini"), QSettings.IniFormat)


@unittest.skipUnless(GAME is not None, "no game install with runtime_dx11.sdb")
class TestSdbTab(_Base):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.tabs.sdb import Tab
        cls.ctx = AppContext(QSettings(str(cls.tmp / "s.ini"), QSettings.IniFormat), game=GAME)
        t0 = time.perf_counter()
        cls.tab = Tab(cls.ctx)
        cls.ctor_seconds = time.perf_counter() - t0
        cls.tab.resize(1400, 850)
        assert _wait(cls.app, lambda: cls.tab._lists is not None and bool(cls.tab._lists.presets), 120)

    @classmethod
    def tearDownClass(cls):
        cls.tab.shutdown()
        cls.tab.scanner.wait(30)
        cls.ctx.runner.pool.waitForDone(10000)
        cls.tab.deleteLater()
        cls.ctx.close()
        super().tearDownClass()

    def test_open_is_async_and_lists_materials(self):
        self.assertLess(self.ctor_seconds, 1.0)
        tab = self.tab
        n = self.ctx.sdb.sdb.tables[0xB2].count
        self.assertTrue(_wait(self.app, lambda: tab.mat_model.rowCount() == n))
        self.assertIn(self.ctx.sdb.path.name, tab.header.text())
        tab.set_mode("Materials")
        tab.search.setText("basic_torso_a_tpp")
        self.assertTrue(_wait(self.app, lambda: 0 < tab.mat_model.rowCount() < 50, 10))
        names = [tab._lists.names[i] for i in tab.mat_model.rows]
        self.assertIn(TORSO, names)
        tab.search.setText("")
        self.assertTrue(_wait(self.app, lambda: tab.mat_model.rowCount() == n, 10))
        # the preset column fills lazily
        row = tab._lists.by_name[TORSO.casefold()][0]
        self.assertEqual(tab.mat_model.getters[2](row), "opaque")

    def test_open_material_parameters_and_bindings(self):
        tab = self.tab
        tab.set_mode("Presets")
        self.assertTrue(tab.open_material(TORSO.upper()))
        self.assertEqual(tab._mode(), "Materials")
        self.assertEqual(tab.search.text(), "")
        self.assertTrue(_wait(self.app, lambda: tab._detail is not None and tab._detail["material"]["name"] == TORSO))
        idx = tab._lists.by_name[TORSO.casefold()][0]
        cur = tab.views["Materials"].currentIndex()
        self.assertEqual(tab.mat_model.row_id(cur.row()), idx)
        self.assertGreater(tab.params.rowCount(), 10)
        pnames = {tab.params.item(r, 1).text() for r in range(tab.params.rowCount())}
        self.assertIn("damage_uv_scale", pnames)
        statuses = {tab.params.item(r, 7).text() for r in range(tab.params.rowCount())}
        self.assertIn("overridden", statuses)
        n_set = tab.params.rowCount()
        tab.show_preset_only.setChecked(True)
        self.assertGreaterEqual(tab.params.rowCount(), n_set)
        tab.show_preset_only.setChecked(False)
        self.assertGreaterEqual(tab.variants.topLevelItemCount(), 1)
        textures = {tab.variants.topLevelItem(0).child(k).text(2)
                    for k in range(tab.variants.topLevelItem(0).childCount())}
        self.assertIn("player_kc_basic_torso_a_tpp_dif.png", textures)
        self.assertEqual(tab.used_textures.topLevelItemCount(), len(tab._detail["providers"]))
        self.assertIn("0xB2[", tab.raw.toPlainText())
        self.assertIn("0xA2[", tab.raw.toPlainText())
        self.assertIn("opaque", tab.overview.toPlainText())
        out = self.tmp / "torso.json"
        tab.export_json(out)
        self.assertTrue(_wait(self.app, out.is_file, 10))
        import json
        self.assertEqual(json.loads(out.read_text(encoding="utf-8"))["name"], TORSO)

    def test_unknown_material(self):
        tab = self.tab
        self.assertFalse(tab.open_material("no_such_material_xyz.mat"))
        self.assertIn("no_such_material_xyz.mat", tab.msg.text())

    def test_preset_view(self):
        tab = self.tab
        pi = self.ctx.sdb.sdb.presets_named("opaque")[0]
        tab.open_preset(pi)
        self.assertTrue(_wait(self.app, lambda: tab._preset_cur is not None and tab._preset_cur["index"] == pi))
        self.assertEqual(tab._mode(), "Presets")
        self.assertEqual(tab.preset_params.rowCount(), len(self.ctx.sdb.sdb.preset(pi)["parameters"]))
        users = tab.materials_using_preset("opaque")
        self.assertIn(tab._lists.by_name[TORSO.casefold()][0], users)
        self.assertEqual(tab.preset_users_model.rowCount(), len(users))

    def test_reverse_texture_lookup(self):
        tab = self.tab
        self.assertTrue(_wait(self.app, lambda: tab._reverse is not None, 180))
        tab.set_mode("Textures")
        tab.search.setText("player_kc_basic_torso_a_tpp_dif.png")
        self.assertTrue(_wait(self.app, lambda: tab.tex_model.rowCount() >= 1, 10))
        row = [i for i in range(tab.tex_model.rowCount())
               if tab._tex_names[tab.tex_model.row_id(i)] == "player_kc_basic_torso_a_tpp_dif.png"]
        self.assertTrue(row)
        tab._select_row(tab.views["Textures"], row[0])
        self.assertTrue(_wait(self.app, lambda: tab.tex_providers.rowCount() >= 1, 10))
        users = {tab.tex_users.item(r, 0).text() for r in range(tab.tex_users.rowCount())}
        self.assertIn(TORSO, users)
        tab.search.setText("")
        tab.set_mode("Materials")

    def test_used_by_models_and_meshes(self):
        tab = self.tab
        if not self.ctx.paks.paths:
            self.skipTest("no dataN.pak")
        self.assertTrue(tab.open_material(SHARED))
        self.assertTrue(_wait(self.app, lambda: tab._model_refs is not None, 60))
        refs = tab.model_refs_for(SHARED)
        self.assertTrue(any(r["kind"] == "resource" for r in refs))
        _wait(self.app, lambda: tab._detail is not None and tab._detail["material"]["name"] == SHARED)
        self.assertGreater(tab.models_tree.topLevelItemCount(), 0)
        self.assertTrue(self.ctx.catalog.wait(120))
        _wait(self.app, lambda: self.ctx.catalog.is_ready, 5)
        tab.mat_tabs.setCurrentIndex(tab.usedby_index)
        self.assertTrue(_wait(self.app, lambda: tab.scanner.scanned and not tab.scanner.running()
                              and not tab.scanner.pending_packs(), 300))
        _wait(self.app, lambda: "scan complete" in tab.meshes_label.text(), 5)
        gids = tab.scanner.lookup(SHARED)
        names = {self.ctx.catalog.name(g) for g in gids}
        has_sample = any(e.path.name == "samples_meshes_pc.rpack" for e in self.ctx.catalog.packs)
        if has_sample:
            self.assertIn("npc_b_man_pants_b_holster_bag_c", {n.strip() for n in names})
        self.assertEqual(tab.meshes_table.rowCount(), len(gids))

    def test_stats_page(self):
        tab = self.tab
        tab.show_stats()
        self.assertTrue(_wait(self.app, lambda: tab.stats_tables.rowCount() == 39, 10))
        self.assertIn("materials", tab.stats_text.toPlainText())


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class TestSdbMissing(_Base):
    def test_missing_sdb_is_friendly(self):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        from nightrunner.gui.tabs.sdb import Tab
        root = self.tmp / "fakegame"
        (root / "ph_ft" / "work" / "data_platform" / "pc" / "assets").mkdir(parents=True)
        ctx = AppContext(self.settings("missing"), game=GameInstall(root), autoload=False)
        tab = Tab(ctx)
        self.addCleanup(ctx.close)
        self.assertIn("No SDB loaded", tab.header.text())
        self.assertFalse(tab.open_material(TORSO))
        self.assertIn("no SDB", tab.msg.text())
        tab.show_stats()
        self.assertIn("No SDB", tab.stats_text.toPlainText())
        tab.set_mode("Textures")
        tab.set_mode("Presets")
        self.assertEqual(tab.mat_model.rowCount(), 0)

    def test_corrupt_sdb_shows_error(self):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        from nightrunner.gui.tabs.sdb import Tab
        root = self.tmp / "badgame"
        assets = root / "ph_ft" / "work" / "data_platform" / "pc" / "assets"
        assets.mkdir(parents=True)
        (assets / "runtime_dx11.sdb").write_bytes(b"MDBR" + b"\0" * 100)
        ctx = AppContext(self.settings("bad"), game=GameInstall(root), autoload=False)
        tab = Tab(ctx)
        self.addCleanup(ctx.close)
        self.assertTrue(_wait(self.app, lambda: "cannot open" in tab.header.text(), 10))
        self.assertIn("could not be opened", tab.msg.text())


if __name__ == "__main__":
    unittest.main()
