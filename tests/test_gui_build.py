"""Nightrunner — Build tab (Resources page): drop → match → validate, inspector edits, default output names,
a real build into the output folder + history, and a save/open round trip.

Headless (QT_QPA_PLATFORM=offscreen); skips without PySide6. Uses a fake install made of out/samples/*.rpack and a
synthetic data0.pak.
"""
from __future__ import annotations

import os
import shutil
import time
import unittest
from pathlib import Path

import numpy as np

import tests.synth as S
from tests.paths import SAMPLES

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

MESHES = SAMPLES / "meshes.rpack"
TEXTURES = SAMPLES / "textures.rpack"


def _link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def wait_for(app, cond, timeout: float = 10.0) -> bool:
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if cond():
            return True
        time.sleep(0.01)
    app.processEvents()
    return cond()


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
@unittest.skipUnless(MESHES.is_file() and TEXTURES.is_file(), "out/samples meshes/textures packs not available")
class BuildTabTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PIL import Image

        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        from nightrunner.pak.model_json import write_models_pak
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("guibuild_")
        d = cls.d = Path(cls._tmp.name)
        game = GameInstall(d / "game")
        game.assets.mkdir(parents=True)
        game.source.mkdir(parents=True)
        _link(MESHES, game.assets / "meshes.rpack")
        _link(TEXTURES, game.assets / "textures.rpack")
        write_models_pak(game.source / "data0.pak", {"models/x.model": {"version": 6, "slots": []}})
        (d / "in").mkdir()
        Image.fromarray(np.full((16, 16, 4), (10, 200, 30, 255), np.uint8)).save(d / "in" / "white.png")
        cls.settings = QSettings(str(d / "s.ini"), QSettings.IniFormat)
        cls.ctx = AppContext(cls.settings, game)
        assert cls.ctx.catalog.wait(60)
        cls.app.processEvents()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.runner.pool.waitForDone(10000)
        cls.ctx.close()
        cls._tmp.cleanup()

    def _tab(self):
        from nightrunner.gui.tabs.build import Tab
        tab = Tab(self.ctx)
        self.addCleanup(tab.deleteLater)
        self.addCleanup(tab.shutdown)
        return tab

    def _validated(self, tab):
        got = []
        tab.state.problemsChanged.connect(got.append)
        tab.validate()
        ok = wait_for(self.app, lambda: bool(got))
        tab.state.problemsChanged.disconnect(got.append)
        return ok

    def test_scene_item(self):
        import json
        self.settings.remove("build")
        sd = self.d / "scene"
        sd.mkdir(exist_ok=True)
        (sd / "hero.cast").write_bytes(b"cast")
        (sd / "hero.cast.json").write_text(json.dumps({"format": "nightrunner.model_cast/1", "model": "hero",
                                                       "parts": [{"mesh": "a.msh"}, {"mesh": "b.msh"}]}))
        tab = self._tab()
        it = tab.add_paths([sd])[0]
        self.assertEqual((it.kind, it.target_name), ("scene", "hero"))
        self.assertIn(it.id, tab.tree.rows)
        self.assertEqual(tab.tree.groups["Meshes"].childCount(), 1)
        tbl = tab.insp.scene_tbl
        self.assertEqual([tbl.item(r, 0).text() for r in range(tbl.rowCount())], ["a", "b"])
        tbl.item(1, 0).setCheckState(Qt.Checked)
        tbl.item(1, 1).setText("hero_b")
        self.assertEqual((it.options["meshes"], it.options["renames"]), (["b"], {"b": "hero_b"}))
        self.assertIn("new hero_b from b", tab.insp.diff.toPlainText())
        self.assertEqual(tab.tree.rows[it.id].text(3), "split · 1 new")
        tab.insp.tol.setValue(0.01)
        self.assertEqual(it.options["tol"], 0.01)

    def test_scene_mapping(self):
        import json
        from nightrunner.cast import castlib
        self.settings.remove("build")
        sd = self.d / "n2b"
        sd.mkdir(exist_ok=True)
        c = castlib.Cast()
        mdl = c.CreateRoot().CreateModel()
        for n in ("obj_a", "obj_b.001"):
            m = mdl.CreateMesh()
            m.SetName(n)
            m.SetVertexPositionBuffer([(0, 0, 0), (1, 0, 0), (0, 1, 0)])
            m.SetFaceBuffer([0, 1, 2])
        c.save(str(sd / "N2B.cast"))
        rep = self.d / "exp" / "hero.cast.json"
        rep.parent.mkdir(exist_ok=True)
        rep.write_text(json.dumps({"format": "nightrunner.model_cast/1", "model": "hero",
                                   "parts": [{"mesh": "a.msh", "slot": "HEAD", "logical_name": "a"}],
                                   "mesh_map": [{"name": "HEAD.a.e0.s0", "part": 0, "entry": 0, "submesh": 0,
                                                 "material": "a.mat"},
                                                {"name": "HEAD.a.e0.s1", "part": 0, "entry": 0, "submesh": 1,
                                                 "material": "b.mat"}]}))
        tab = self._tab()
        it = tab.add_paths([sd / "N2B.cast"])[0]
        self.assertEqual(it.kind, "scene")                      # no mesh.json beside it
        self.assertIn("no report", tab.insp.diff.toPlainText())
        it.options["report"] = str(rep)
        tab.insp.bind(it)
        self.assertTrue(wait_for(self.app, lambda: tab.insp.obj_tbl.rowCount() == 2))
        self.assertEqual([tab.insp.obj_tbl.item(r, 0).text() for r in range(2)], ["obj_a", "obj_b.001"])
        combo = tab.insp.obj_tbl.cellWidget(1, 1)
        combo.setCurrentIndex(combo.findData("HEAD.a.e0.s0"))
        self.assertEqual(it.options["assign"], {"obj_b": "HEAD.a.e0.s0"})
        combo = tab.insp.obj_tbl.cellWidget(0, 1)
        combo.setCurrentIndex(combo.findData(""))
        self.assertEqual(it.options["assign"], {"obj_b": "HEAD.a.e0.s0", "obj_a": ""})
        self.assertEqual(tab.insp.tgt_tbl.item(0, 1).text(), "obj_b")
        tab.insp.tgt_tbl.item(1, 2).setCheckState(Qt.Checked)
        self.assertEqual(it.options["hide"], ["HEAD.a.e0.s1"])
        text = tab.insp.diff.toPlainText()
        self.assertIn("obj_b → HEAD.a.e0.s0", text)
        self.assertIn("hide HEAD.a.e0.s1", text)
        tab.insp.obj_tbl.item(1, 2).setCheckState(Qt.Checked)
        self.assertEqual(it.options["double_sided"], ["obj_b"])
        self.assertIn("2-sided obj_b", tab.insp.diff.toPlainText())
        tab.insp.lods.setChecked(False)
        self.assertIs(it.options["lods"], False)
        from nightrunner.project import effective_assign, validate
        combo.setCurrentIndex(0)                                     # obj_a back to unassigned
        msgs = [p.message for p in validate(tab.state.project, tab.state.env) if p.where == it.id]
        self.assertTrue(any("1 objects have no target (obj_a" in m for m in msgs), msgs)
        tab.insp.skip_rest.setChecked(True)
        self.assertEqual(effective_assign(it), {"obj_b": "HEAD.a.e0.s0", "obj_a": ""})
        msgs = [p.message for p in validate(tab.state.project, tab.state.env) if p.where == it.id]
        self.assertFalse(any("have no target" in m for m in msgs), msgs)
        tab._duplicate(it)
        self.assertEqual(len(tab.state.project.items), 2)
        dup = tab.state.project.items[1]
        self.assertEqual(dup.options, it.options)
        self.assertNotEqual(dup.id, it.id)
        # another report on the copy: old targets are dropped, object options kept
        rep2 = self.d / "exp" / "other.cast.json"
        rep2.write_text(rep.read_text().replace('"hero"', '"other"'))
        tab.insp.bind(dup)
        tab.insp._pick_report(str(rep2))
        self.assertEqual((dup.target_name, dup.options.get("assign"), dup.options.get("hide")), ("other", None, None))
        self.assertEqual((dup.options["double_sided"], dup.options["skip_rest"]), (["obj_b"], True))
        self.assertEqual(it.options["assign"], {"obj_b": "HEAD.a.e0.s0"})          # original untouched
        tab.insp.bind(it)
        tab.insp.kind.setCurrentIndex(0)
        self.assertEqual(it.kind, "mesh")
        tab.insp.kind.setCurrentIndex(1)
        self.assertEqual((it.kind, it.target_name), ("scene", "hero"))

    def test_drop_edit_build_history_roundtrip(self):
        self.settings.remove("build")
        tab = self._tab()
        st = tab.state
        # defaults: first free names, existing data0.pak disabled
        self.assertEqual(tab.rpack_edit.placeholderText(), "assets_2_pc.rpack")
        self.assertEqual(tab.pak_combo.lineEdit().placeholderText(), "data2.pak")   # data0/data1 are stock archives (profile)
        self.assertFalse(tab.pak_combo.model().item(0).isEnabled())
        self.assertFalse(tab.pak_combo.model().item(1).isEnabled())               # data1.pak: game archive
        self.assertTrue(tab.pak_combo.model().item(2).isEnabled())
        self.assertEqual(tab.tree.hint.isVisibleTo(tab.tree), True)

        new = tab.add_paths([self.d / "in" / "white.png"])
        self.assertEqual(len(new), 1)
        it = new[0]
        self.assertEqual((it.target_name, it.target_pack, it.new_name), ("white.dds", "textures.rpack", None))
        self.assertEqual(len(tab.tree.rows), 1)
        row = tab.tree.rows[it.id]
        self.assertEqual((row.text(0), row.text(1), row.text(2)), ("white.png", "white.dds", "textures.rpack"))
        self.assertTrue(st.dirty)
        self.assertEqual(tab.pages.tabText(0), "Resources (1)")
        self.assertIs(tab.insp.item, it)
        self.assertIn("exact", tab.insp.match.text())
        self.assertIn("replace white.dds", tab.insp.diff.toPlainText())
        self.assertTrue(self._validated(tab))
        self.assertEqual(row.text(4), "✓")

        # inspector edits write back
        tab.insp.name_mode.setCurrentIndex(1)
        self.assertEqual(it.new_name, "white.png")
        self.assertEqual((row.text(3), row.toolTip(3)), ("white.png", "new"))
        tab.insp.name_mode.setCurrentIndex(0)
        self.assertIsNone(it.new_name)
        tab.insp.fmt.setCurrentIndex(tab.insp.fmt.findData("rgba8"))
        self.assertEqual(it.options.get("format"), "rgba8")
        tab.insp.target.setText("nope.dds")
        tab.insp.target.textEdited.emit("nope.dds")
        self.assertIn("none", tab.insp.match.text())
        self.assertTrue(self._validated(tab))
        self.assertEqual(row.text(4), "✗ no target")
        self.assertIn("Problems (", tab.bottom.tabText(1))
        tab.insp.target.setText("white.dds")
        tab.insp.target.textEdited.emit("white.dds")

        # toggling enabled
        row.setCheckState(0, Qt.Unchecked)
        self.assertFalse(it.enabled)
        self.assertEqual(row.text(4), "off")
        row.setCheckState(0, Qt.Checked)
        self.assertTrue(it.enabled)

        # build (save first: no dialog when the project already has a path)
        self.assertTrue(tab.save_project(path=str(self.d / "proj" / "crane.nrproj")))
        self.assertEqual(st.project.name, "crane")
        self.assertFalse(st.dirty)
        got = []
        st.built.connect(got.append)
        self.assertTrue(tab.build())
        self.assertEqual(tab.build_btn.text(), "Cancel")
        self.assertTrue(wait_for(self.app, lambda: not tab.running, 60))
        self.assertEqual(len(got), 1, tab.log_view.toPlainText())
        out = self.d / "proj" / "out" / "assets_2_pc.rpack"
        self.assertTrue(out.is_file())
        self.assertTrue(tab.open_btn.isEnabled())
        self.assertEqual(tab.build_btn.text(), "Build")
        hist = tab.history()
        self.assertEqual((hist[-1]["project"], hist[-1]["rpack"], hist[-1]["ok"]), ("crane", "assets_2_pc.rpack", True))
        self.assertEqual(tab.hist.item(0, 3).text(), "assets_2_pc.rpack")
        self.assertIn("✓ rpack", tab.log_view.toPlainText())
        self.assertEqual(st.project.rpack_name, "assets_2_pc.rpack")          # auto name pinned after a build

        # a failing build reports in the log, no dialog
        tab.state.project.pak_name = "data0.pak"
        self.assertTrue(tab.build())
        self.assertTrue(wait_for(self.app, lambda: not tab.running, 30))
        self.assertIn("game archive name", tab.log_view.toPlainText())
        self.assertFalse(tab.history()[-1]["ok"])
        tab.state.project.pak_name = ""

        # save / open round trip rebuilds the rows; a new tab reopens the last project
        tab.state.project.rpack_name = "mine.rpack"
        tab.state.touch("settings")
        self.assertTrue(tab.save_project())
        tab.new_project()
        self.assertEqual(len(tab.tree.rows), 0)
        self.assertIsNone(tab.insp.item)
        self.assertTrue(tab.open_project(str(self.d / "proj" / "crane.nrproj")))
        self.assertEqual(len(tab.tree.rows), 1)
        self.assertEqual(tab.rpack_edit.text(), "mine.rpack")
        self.assertIn(str(self.d / "proj" / "crane.nrproj"), tab._recent())
        tab2 = self._tab()
        self.assertEqual(tab2.state.project.name, "crane")
        self.assertEqual(len(tab2.tree.rows), 1)
        self.assertEqual(tab2.proj_name.text(), "<b>crane</b>")

        # removal
        tab2.tree.rows[next(iter(tab2.tree.rows))].setSelected(True)
        tab2._remove_selected()
        self.assertEqual(len(tab2.tree.rows), 0)
        self.assertTrue(tab2.state.dirty)
        self.assertEqual(tab2.proj_name.text(), "<b>crane *</b>")


if __name__ == "__main__":
    unittest.main()
