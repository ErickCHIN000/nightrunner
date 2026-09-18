"""Build tab → Model overrides page (gui/tabs/build_models.py) and Models.open_model_doc, headless.

Fake install: out/samples meshes/textures packs in the assets folder, a synthetic data0.pak with the player TPP / FPP
models and scripts/playerappearances.scr (no SDB). GL is unavailable offscreen: MeshView shows its fallback, so
nothing here looks at pixels.
"""
from __future__ import annotations

import copy
import os
import shutil
import time
import unittest
from pathlib import Path

import tests.synth as S
from tests.paths import SAMPLES
from tests.test_project import SCR, _model, _res

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

MESHES = SAMPLES / "meshes.rpack"
TEXTURES = SAMPLES / "textures.rpack"

TPP = _model("sh2_player_tpp_phx_skeleton.msh", [
    ("HEAD", [_res("sh2_npc_crane.msh")]),
    ("HEAD_PART_1", [_res("sh2_npc_ft_crane_beard_a.msh")]),
    ("HEAD_PART_3", [_res("sh_npc_ft_crane_hair_a.msh", mats=["sh_npc_ft_crane_hair_a.mat"])]),
])
# same slot names, other order, one TPP-only slot
FPP = _model("player_fpp_phx_skeleton.msh", [
    ("HEAD_PART_3", [_res("sh_npc_ft_crane_hair_a.msh", mats=["sh_npc_ft_crane_hair_a.mat"])]),
    ("HEAD", [_res("sh2_npc_crane.msh")]),
])


def _place(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def make_game(root: Path):
    from nightrunner.gui.game import GameInstall
    from nightrunner.pak.model_json import write_models_pak
    gi = GameInstall(root)
    gi.assets.mkdir(parents=True)
    gi.source.mkdir(parents=True)
    _place(MESHES, gi.assets / "meshes.rpack")
    _place(TEXTURES, gi.assets / "textures.rpack")
    write_models_pak(gi.source / "data0.pak", {"models/player/player_tpp_skeleton.model": TPP,
                                               "models/player/player_fpp_skeleton.model": FPP},
                     texts={"scripts/playerappearances.scr": SCR})
    return gi


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
@unittest.skipUnless(MESHES.is_file() and TEXTURES.is_file(), "out/samples packs not available")
class BuildModelsPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("gui_bm_")
        cls.d = Path(cls._tmp.name)
        cls.game = make_game(cls.d / "game")
        cls.contexts = []
        cls.ctx = cls.new_ctx("main")

    @classmethod
    def new_ctx(cls, tag):
        from nightrunner.gui.context import AppContext
        st = QSettings(str(cls.d / f"{tag}.ini"), QSettings.IniFormat)
        ctx = AppContext(settings=st, game=cls.game)
        assert ctx.catalog.wait(60)
        cls.contexts.append(ctx)
        return ctx

    @classmethod
    def tearDownClass(cls):
        for ctx in cls.contexts:
            ctx.runner.pool.waitForDone(10000)
            ctx.close()
        cls._tmp.cleanup()

    def setUp(self):
        self.pages = []

    def tearDown(self):
        for p in self.pages:
            p.shutdown()
            p.deleteLater()
        self.app.processEvents()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def wait(self, cond, timeout=15.0):
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def page(self, ctx=None):
        from nightrunner.gui.buildstate import BuildState
        from nightrunner.gui.tabs.build_models import ModelOverridesPage
        ctx = ctx or self.ctx
        state = BuildState(ctx)
        p = ModelOverridesPage(ctx, state)
        p.resize(1600, 900)
        self.pages.append(p)
        return p, state

    def player(self, page):
        page._pick = lambda title, fetch, free=False, text="": "PlayerMan1 · Default" if title == "Player" else None
        page._on_new_menu("Player…")
        return page.mo

    # ---- tests --------------------------------------------------------------------------------------------------
    def test_new_player_and_grid(self):
        page, state = self.player_page()
        mo = state.project.models[0]
        self.assertEqual(sorted(mo.roles), ["fpp", "lodcc", "tpp", "ui"])
        self.assertEqual(mo.roles["tpp"].member, "models/player/player_tpp_skeleton.model")
        self.assertEqual(page.roles, ["tpp", "fpp"])                      # lodcc / ui share the TPP member
        self.assertEqual(page.grid.columnCount(), 6)
        self.assertEqual(page.slot_names, ["HEAD", "HEAD_PART_1", "HEAD_PART_3"])   # TPP order first
        self.assertFalse(page.cells[("HEAD_PART_1", "fpp")][1].isEnabled())
        self.assertEqual(page.list.count(), 1)
        self.assertEqual(page.list.item(0).data(0x101), "unchanged")
        self.assertEqual(page.source.text(), "data0.pak")
        self.assertEqual(page.pak.currentText(), "data2.pak")   # data0/data1 are stock archives (profile)
        self.assertTrue(state.dirty)
        filters = [b.property("filter") for b in page.filter_group.buttons()]
        self.assertEqual(filters, ["all", "head", "changed"])

    def player_page(self):
        page, state = self.page()
        self.assertIsNotNone(self.player(page))
        return page, state

    def test_link_toggle_and_mesh_combo(self):
        from nightrunner.gui.tabs.build_models import NONE
        from nightrunner.project import slot, slot_mesh
        page, state = self.player_page()
        tpp, fpp = page.doc("tpp"), page.doc("fpp")
        changes = []
        state.modelsChanged.connect(lambda: changes.append(1))
        self.assertFalse(page.link.isChecked())                           # off by default
        page.cells[("HEAD", "fpp")][0].setChecked(False)                 # unlinked: only FPP
        self.assertEqual(slot_mesh(slot(tpp, "HEAD")), "sh2_npc_crane.msh")
        page.cells[("HEAD", "fpp")][0].setChecked(True)
        page.link.setChecked(True)
        page.cells[("HEAD", "tpp")][0].setChecked(False)
        self.assertIsNone(slot_mesh(slot(tpp, "HEAD")))
        self.assertIsNone(slot_mesh(slot(fpp, "HEAD")))                   # Link on: both roles
        self.assertTrue(changes)
        self.assertTrue(page.grid.item(0, 0).text().startswith("●"))
        self.assertFalse(page.cells[("HEAD", "fpp")][0].isChecked())
        self.assertEqual(page.cells[("HEAD", "fpp")][1].currentText(), NONE)
        self.assertIn("- slots[HEAD].meshResources.resources[sh2_npc_crane.msh]", page.diff.toPlainText())
        self.assertEqual(page.list.item(0).data(0x101), "1 slot")
        page.link.setChecked(False)
        page.cells[("HEAD", "tpp")][0].setChecked(True)
        self.assertEqual(slot_mesh(slot(tpp, "HEAD")), "sh2_npc_crane.msh")
        self.assertIsNone(slot_mesh(slot(fpp, "HEAD")))
        page.set_filter("changed")
        self.assertFalse(page.grid.isRowHidden(0))
        self.assertTrue(page.grid.isRowHidden(1))
        page.set_filter("all")
        # mesh combo: typed name -> set_slot_mesh (new entry cloned), linked
        page.link.setChecked(True)
        combo = page.cells[("HEAD_PART_3", "tpp")][1]
        combo.setEditText("sh2_npc_ft_crane_beard_a")
        combo.lineEdit().editingFinished.emit()
        for d in (tpp, fpp):
            s = slot(d, "HEAD_PART_3")
            self.assertEqual(slot_mesh(s), "sh2_npc_ft_crane_beard_a.msh")
            self.assertEqual(len(s["meshResources"]["resources"]), 2)
        self.assertEqual(page.cells[("HEAD_PART_3", "fpp")][1].currentText(), "sh2_npc_ft_crane_beard_a.msh")
        # "—" turns the slot off, picking the old alternative back selects it
        i = combo.findText(NONE)
        combo.activated.emit(i)
        self.assertIsNone(slot_mesh(slot(tpp, "HEAD_PART_3")))
        combo = page.cells[("HEAD_PART_3", "tpp")][1]
        combo.activated.emit(combo.findText("sh_npc_ft_crane_hair_a.msh"))
        self.assertEqual(slot_mesh(slot(fpp, "HEAD_PART_3")), "sh_npc_ft_crane_hair_a.msh")
        page.revert_all()
        self.assertEqual(page.doc("tpp"), TPP)
        self.assertEqual(page.diff.toPlainText().count("—"), 2)

    def test_mirror_tpp_to_fpp(self):
        from nightrunner.project import slot, slot_mesh
        page, state = self.player_page()
        tpp, fpp = page.doc("tpp"), page.doc("fpp")
        page.cells[("HEAD_PART_3", "tpp")][0].setChecked(False)
        page.sel_slot = None
        page._on_slot_menu("TPP → FPP (all)")
        fpp = page.doc("fpp")
        for s in tpp["slots"]:
            self.assertEqual(slot(fpp, s["name"])["meshResources"], s["meshResources"])
        self.assertIn("HEAD_PART_3", page.mo.roles["fpp"].stash)
        self.assertEqual(slot_mesh(slot(fpp, "HEAD")), "sh2_npc_crane.msh")
        page.cells[("HEAD", "fpp")][0].setChecked(False)                   # then drop the head in FPP only
        self.assertIsNone(slot_mesh(slot(fpp, "HEAD")))
        self.assertEqual(slot_mesh(slot(tpp, "HEAD")), "sh2_npc_crane.msh")
        uids = [s["slotUid"] for s in fpp["slots"]]
        self.assertEqual(len(uids), len(set(uids)))

    def test_move_to_and_new_slot(self):
        from nightrunner.project import slot, slot_mesh
        page, state = self.player_page()
        page.move_to("HEAD", "torso_part_9")
        tpp = page.doc("tpp")
        self.assertIsNone(slot_mesh(slot(tpp, "HEAD")))
        self.assertEqual(slot_mesh(slot(tpp, "TORSO_PART_9")), "sh2_npc_crane.msh")
        self.assertIn("TORSO_PART_9", page.slot_names)
        self.assertIsNone(next((s for s in page.doc("fpp")["slots"] if s["name"] == "TORSO_PART_9"), None))
        page.new_slot()
        self.assertTrue(any(s["name"] == "TORSO_PART_1" for s in page.doc("tpp")["slots"]))
        page._on_slot_menu("Revert all")
        self.assertNotIn("TORSO_PART_9", page.slot_names)

    def test_project_meshes_listed(self):
        from nightrunner.gui.tabs.build_models import STAR
        from nightrunner.project import Item
        page, state = self.player_page()
        state.project.items.append(Item(kind="mesh", source="x.glb", target_name="sh2_npc_ft_crane_beard_a",
                                        new_name="crane_beard_copy"))
        state.touch("items")
        combo = page.cells[("HEAD", "tpp")][1]
        self.assertGreaterEqual(combo.findText(STAR + "crane_beard_copy"), 0)
        combo.activated.emit(combo.findText(STAR + "crane_beard_copy"))
        self.assertEqual(combo.currentText(), STAR + "crane_beard_copy.msh")
        self.assertTrue(page.needs_build())
        self.assertTrue(self.wait(lambda: page.preview_out is not None and "Build to preview" in
                                  page.preview_status.text()))

    def test_materials_rtti_and_checks(self):
        page, state = self.player_page()
        page.select_slot("HEAD_PART_3")
        self.assertEqual(page.sel_slot, "HEAD_PART_3")
        self.assertIn("HEAD_PART_3", page.mat_box.title())
        self.assertTrue(self.wait(lambda: page._mat_cache.get("sh_npc_ft_crane_hair_a.msh")))
        mats = [m for m, _ in page._mat_rows]
        self.assertEqual(mats[-1], "sh_npc_ft_crane_hair_a.mat")        # materialsData names follow the mesh's
        mats = mats[::-1]
        page.mat_roles.setCurrentIndex(1)                                  # TPP only
        self.assertEqual(page.mat_targets(), ("tpp", ["tpp"]))
        page.add_param("dif_0_tex", "nope_tex.png", material=mats[0])
        ent = page.doc("tpp")["slots"][2]["meshResources"]["resources"][0]
        vals = [v for g in ent["materialsResources"] for e in g["resources"] for v in e.get("rttiValues", [])]
        self.assertIn({"name": "dif_0_tex", "type": 7, "val_str": "nope_tex.png"}, vals)
        fent = page.doc("fpp")["slots"][0]["meshResources"]["resources"][0]
        self.assertFalse(any(e.get("rttiValues") for g in fent["materialsResources"] for e in g["resources"]))
        self.assertIn("rttiValues[dif_0_tex]", page.diff.toPlainText())
        self.assertIn((mats[0], "dif_0_tex"), page._mat_rows)
        r = page._mat_rows.index((mats[0], "dif_0_tex"))
        self.assertEqual(page.mats.item(r, 3).text(), "game")
        self.assertTrue(page.mats.cellWidget(r, 2).currentText().endswith("nope_tex.png"))
        self.assertEqual(page.list.item(0).data(0x101), "1 slot · 1 tex")
        # a float value, edited in the table
        page.add_param("dif_0_val", material=mats[0])
        r = page._mat_rows.index((mats[0], "dif_0_val"))
        page.mats.item(r, 2).setText("0.5, 0.25, 1")
        vals = [v for g in ent["materialsResources"] for e in g["resources"] for v in e.get("rttiValues", [])]
        self.assertIn({"name": "dif_0_val", "type": 4, "val_vec3": [0.5, 0.25, 1.0]}, vals)
        # base material
        page.set_base(mats[0], "player_kc_basic_torso_a_tpp.mat")
        r = next(i for i, (m, _p) in enumerate(page._mat_rows) if m == mats[0])
        self.assertIn("→ player_kc_basic_torso_a_tpp.mat", page.mats.item(r, 0).text())
        # checks: validate runs in the background
        page._check_deb.flush()
        self.assertTrue(self.wait(lambda: "nope_tex.png not found" in page.checks.toPlainText()))
        self.assertTrue(page.bottom.tabText(1).startswith("Checks ("))
        self.assertIn("✗", page.list.item(0).data(0x101))
        # removal
        page.mats.selectRow(page._mat_rows.index((mats[0], "dif_0_tex")))
        page._delete_param()
        self.assertNotIn((mats[0], "dif_0_tex"), page._mat_rows)
        state.problemsChanged.emit([])
        self.assertEqual(page.checks.toPlainText(), "")
        self.assertEqual(page.bottom.tabText(1), "Checks")

    def test_pair_rename_duplicate_and_reload(self):
        from nightrunner.project import Project
        page, state = self.page()
        page.new_model("player_tpp_skeleton.model")                         # pairs fpp / lodcc / ui
        self.assertEqual(sorted(page.mo.roles), ["fpp", "lodcc", "tpp", "ui"])
        self.assertEqual(page.mo.label, "player_tpp_skeleton")
        page.list.item(0).setText("Crane")
        self.assertEqual(state.project.models[0].label, "Crane")
        page._on_new_menu("Duplicate")
        self.assertEqual([m.label for m in state.project.models], ["Crane", "Crane copy"])
        self.assertEqual(page.list.currentRow(), 1)
        page.role_rows["fpp"][0].setChecked(False)
        self.assertEqual(page.roles, ["tpp"])
        self.assertEqual(page.grid.columnCount(), 4)
        page.role_rows["fpp"][0].setChecked(True)                           # restored from the stash
        self.assertEqual(page.roles, ["tpp", "fpp"])
        page.path_mode.setCurrentIndex(2)
        self.assertEqual(page.mo.path_mode, "both")
        self.assertTrue(page.no_gear.isChecked())                    # player model: on by default
        page.no_gear.setChecked(False)
        self.assertFalse(page.mo.no_gear)
        page.no_gear.setChecked(True)
        self.assertTrue(page.mo.no_gear)
        page.pak.setEditText("data5.pak")
        page.pak.lineEdit().editingFinished.emit()
        self.assertEqual(state.project.pak_name, "data5.pak")
        # a saved project reloads without originals: they are read back from the PAK
        page.cells[("HEAD", "tpp")][0].setChecked(False)
        p = state.save(self.d / "proj" / "t.nrproj")
        state.open(p)
        self.assertEqual(len(state.project.models), 2)
        self.assertIsNotNone(page.mo.roles["tpp"].original)
        self.assertEqual(page.list.item(0).data(0x101), "unchanged")
        self.assertEqual(page.list.item(1).data(0x101), "1 slot")
        self.assertEqual(Project.load(p).models[0].path_mode, "root")

    def test_open_model_doc(self):
        from nightrunner.gui.tabs.models import Tab
        tab = Tab(self.ctx)
        doc = copy.deepcopy(TPP)
        doc["slots"][0]["meshResources"]["resources"][0]["name"] = "sh2_npc_ft_crane_beard_a.msh"
        self.assertTrue(tab.open_model_doc("player_tpp_skeleton.model", doc, []))
        self.assertTrue(self.wait(lambda: tab.result is not None and tab.result.get("resolution")))
        res = tab.result["resolution"]
        self.assertEqual([s["name"] for s in res["slots"]], ["HEAD", "HEAD_PART_1", "HEAD_PART_3"])
        self.assertEqual(res["slots"][0]["meshes"][0]["name"], "sh2_npc_ft_crane_beard_a.msh")
        self.assertTrue(res["slots"][0]["meshes"][0]["gids"])
        self.assertIn("player_tpp_skeleton.model", tab.header.text())
        self.assertIn("sh2_npc_ft_crane_beard_a", tab.raw.toPlainText())
        # the page routes "Open in Models" through ctx.openModelDoc
        page, _state = self.player_page()
        got = []
        self.ctx.openModelDoc.connect(lambda n, d, p: got.append((n, d, p)))
        page.open_in_models()
        self.assertEqual(got[-1][0], "player_tpp_skeleton.model")
        self.assertEqual(got[-1][1], page.doc("tpp"))
        self.assertIsNot(got[-1][1], page.doc("tpp"))
        tab.shutdown()
        tab.deleteLater()

    def test_preview_resolution(self):
        page, _state = self.player_page()
        page._preview_deb.flush()
        ok = self.wait(lambda: page.preview_out is not None)
        self.assertTrue(ok, page.preview_status.text())
        vw = page.preview_out["views"]
        self.assertEqual([v["tag"] for v in vw], ["player_tpp_skeleton.model"])
        self.assertTrue(vw[0]["resolution"]["slots"][0]["meshes"][0]["gids"])
        self.assertTrue(vw[0]["preview"]["items"])
        page.cells[("HEAD", "tpp")][0].setChecked(False)
        page.view_btns["side"].click()
        page.preview_out = None
        page._preview_deb.flush()
        self.assertTrue(self.wait(lambda: page.preview_out is not None))
        vw = page.preview_out["views"]
        self.assertEqual(len(vw), 2)
        self.assertNotIn("HEAD", [s["name"] for s in vw[0]["resolution"]["slots"]])   # off slots hidden
        page.ghost.setChecked(True)
        page.preview_out = None
        page._preview_deb.flush()
        self.assertTrue(self.wait(lambda: page.preview_out is not None))
        self.assertIn("HEAD", [s["name"] for s in page.preview_out["views"][0]["resolution"]["slots"]])
        self.assertEqual(page.preview_out["views"][0]["off"], {"HEAD"})

    def test_built_pack_preferred(self):
        from nightrunner.container.rp6l import Pack
        from nightrunner.cast.export import export_cast
        from nightrunner.mesh.decode import decode_resource
        from nightrunner.project import GameEnv, Item, Project, build_project
        ctx = self.new_ctx("built")
        with Pack.open(MESHES) as pk:
            beard = decode_resource(pk.resource(pk.find("sh2_npc_ft_crane_beard_a")[0]))
        src = self.d / "beard.glb"
        export_cast(beard, src)
        env = GameEnv(self.d, self.d, packs={"meshes.rpack": MESHES})
        pr = Project(name="bm")
        for new in ("crane_beard_copy", None):
            pr.items.append(Item(kind="mesh", source=str(src), target_name="sh2_npc_ft_crane_beard_a",
                                 target_pack="meshes.rpack", new_name=new))
        rep = build_project(pr, env, out_dir=self.d / "build")
        page, state = self.page(ctx)
        state.project.items = list(pr.items)
        self.player(page)
        page.set_mesh("HEAD", "tpp", "crane_beard_copy")
        self.assertTrue(page.needs_build())
        self.assertFalse(ctx.catalog.lookup("crane_beard_copy", 0x10))
        state.built.emit(rep)
        self.assertEqual(len(page.preview_packs), 1)
        copy_path, pid = page.preview_packs[0]
        self.assertTrue(copy_path.is_file())
        self.assertNotEqual(copy_path, Path(rep["outputs"]["rpack"]["path"]))
        self.assertTrue(self.wait(lambda: ctx.catalog.packs[pid].pack is not None))
        self.assertFalse(page.needs_build())
        view = page.view()
        hits = view.catalog.lookup("crane_beard_copy", 0x10)
        self.assertEqual([ctx.catalog.entry(g).id for g in hits], [pid])
        same = view.catalog.lookup("sh2_npc_ft_crane_beard_a", 0x10)
        self.assertEqual(len(same), 2)
        self.assertEqual(ctx.catalog.entry(same[0]).id, pid)               # the build wins in the preview
        self.assertNotEqual(ctx.catalog.entry(ctx.catalog.lookup("sh2_npc_ft_crane_beard_a", 0x10)[0]).id, pid)
        page.preview_out = None
        page._preview_deb.flush()
        self.assertTrue(self.wait(lambda: page.preview_out is not None))
        head = page.preview_out["views"][0]["resolution"]["slots"][0]["meshes"]
        sel = next(m for m in head if m["chosen"])
        self.assertEqual(sel["name"], "crane_beard_copy.msh")
        self.assertEqual([ctx.catalog.entry(g).id for g in sel["gids"]], [pid])
        self.assertNotIn("Build to preview", page.preview_status.text())
        # a second build: new copy, newest preferred
        state.built.emit(rep)
        self.assertEqual(len(page.preview_packs), 2)
        pid2 = page.preview_packs[1][1]
        self.assertTrue(self.wait(lambda: ctx.catalog.packs[pid2].pack is not None))
        self.assertEqual(ctx.catalog.entry(page.view().catalog.lookup("crane_beard_copy", 0x10)[0]).id, pid2)
        paths = [p for p, _ in page.preview_packs]
        page.shutdown()
        self.assertFalse(any(p.exists() for p in paths))


if __name__ == "__main__":
    unittest.main()


class WheelGuardTests(unittest.TestCase):
    def test_wheel_scrolls_table_not_combo(self):
        try:
            from PySide6.QtCore import QPoint, QPointF, Qt
            from PySide6.QtGui import QWheelEvent
            from PySide6.QtWidgets import QApplication, QComboBox, QTableWidget
        except ImportError:
            self.skipTest("PySide6 not installed")
        from nightrunner.gui.widgets import WheelGuard
        QApplication.instance() or QApplication([])
        t = QTableWidget(60, 1)
        t.resize(200, 150)
        combos = []
        for r in range(60):
            c = QComboBox()
            c.addItems(["a", "b", "c"])
            t.setCellWidget(r, 0, c)
            combos.append(c)
        g = WheelGuard(t)
        g.guard(*combos)
        t.show()
        QApplication.processEvents()
        c = combos[0]
        ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0), QPoint(0, -120), Qt.NoButton, Qt.NoModifier,
                         Qt.NoScrollPhase, False)
        QApplication.sendEvent(c, ev)
        QApplication.processEvents()
        self.assertEqual(c.currentIndex(), 0)
        self.assertGreater(t.verticalScrollBar().value(), 0)
