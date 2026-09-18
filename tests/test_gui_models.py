"""Models tab: `.model` resolution (gui/modelresolve.py), tab population / navigation, export.

Needs the game data (NIGHTRUNNER_GAME_ROOT or the real install): data0.pak + runtime_dx11.sdb for the player model,
the dlc_samples packs for the found-mesh cases. Skips otherwise; the variant test is fully synthetic.
"""
from __future__ import annotations

import json
import os
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import tests.synth as S

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

PLAYER = "models/player/player_tpp_skeleton.model"


def _game():
    if not HAVE_QT:
        return None
    from nightrunner.gui.game import find_game
    g = find_game(os.environ.get("NIGHTRUNNER_GAME_ROOT") or None, game="dltb")
    if g is None or not g.paks() or g.sdb("dx11") is None or not g.sdb("dx11").is_file():
        return None
    return g


class StubCatalog:
    """lookup() over a {(name.lower(), type): [gid]} table; everything else missing."""

    def __init__(self, table=None):
        self.table = table or {}

    def lookup(self, name, type_id):
        return list(self.table.get((name.lower(), type_id), []))

    def entry(self, gid):
        return SimpleNamespace(label=f"stub{gid}")

    def split(self, gid):
        raise KeyError(gid)


class StubSdb:
    def __init__(self, mats):
        self.mats = {k.lower(): v for k, v in mats.items()}

    def material_info(self, name):
        return self.mats.get(name.lower())


def sdb_mat(name, bindings, params=()):
    return {"index": 1, "name": name, "non_rendering": False, "routes": [{
        "preset": "opaque", "tokens": "opaque;",
        "parameters": [{"name": n, "id": i, "offset": i * 4, "type": t, "value": v, "string": None}
                       for i, (n, t, v) in enumerate(params)],
        "variants": [{"texture_bindings": [{"binding": i, "param_id": 50 + i, "param": p, "texture": t,
                                            "source": "override"} for i, (p, t) in enumerate(bindings)]}]}]}


def synthetic_model(mesh="sh2_npc_crane.msh", material="sh2_npc_crane.mat", rtti=None, base=None):
    return {"version": 6, "preset": {"skeletonName": "sh2_player_tpp_phx_skeleton.msh"},
            "slots": [{"slotUid": 1, "name": "HEAD", "filterText": "head", "meshResources": {"resources": [{
                "name": mesh, "selected": True, "layoutId": 4,
                "materialsData": [{"number": 1, "name": material, "layoutId": 4, "loadFlags": "S"}],
                "materialsResources": [{"number": 1, "resources": [
                    {"name": "unused_alt.mat", "selected": False, "rttiValues": []},
                    {"name": base or material, "selected": True, "layoutId": 4, "loadFlags": "S",
                     "rttiValues": rtti or []}]}]}]}}]}


def all_subs(res):
    return [s for sl in res["slots"] for m in sl["meshes"] for s in m["submeshes"]]


# ---- pure resolution ------------------------------------------------------------------------------------------------

class TestResolveSynthetic(unittest.TestCase):
    def test_variant_remap_before_join(self):
        from nightrunner.gui.modelresolve import resolve_model
        doc = synthetic_model(mesh="car.msh", material="car_olive.mat", rtti=[
            {"name": "dif_0_tex", "type": 7, "val_str": "car_olive_decal_dif.png"},
            {"name": "gloss", "type": 2, "val_float": 0.25},
            {"name": "tint", "type": 4, "val_vec3": [1, 0, 0]}])
        cat = StubCatalog({("car.msh", 0x10): [7], ("car_olive_decal_dif.png", 0x20): [9]})
        sdb = StubSdb({"car.mat": sdb_mat("car.mat", [("dif_0_tex", "car_dif.png")]),
                       "car_olive.mat": sdb_mat("car_olive.mat", [("dif_0_tex", "car_olive_dif.png"),
                                                                  ("nrm_0_tex", "car_nrm.png")],
                                                [("gloss", "float", 0.5)])})

        def reader(_cat, gid):
            self.assertEqual(gid, 7)
            return {"submeshes": [{"entry": 0, "submesh": 0, "slot": 0, "material": "car.mat"}],
                    "materials": ["car.mat"],
                    "variants": [{"name": "Default", "map": {}, "material_map": {}},
                                 {"name": "olive", "map": {"car.mat": "car_olive.mat"},
                                  "material_map": {"0": "car_olive.mat"}}]}

        ctx = SimpleNamespace(catalog=cat, sdb=sdb)
        base = resolve_model(ctx, doc, None, mesh_reader=reader)
        self.assertEqual(base["variants"], ["Default", "olive"])
        s = all_subs(base)[0]
        self.assertEqual((s["base_material"], s["material_source"], s["number"]), ("car.mat", "mesh default", None))
        self.assertEqual(s["textures"][0]["texture"], "car_dif.png")

        v = resolve_model(ctx, doc, "olive", mesh_reader=reader)
        s = all_subs(v)[0]
        self.assertEqual(s["variant_material"], "car_olive.mat")
        self.assertEqual((s["base_material"], s["number"], s["material_source"]), ("car_olive.mat", 1, "variant"))
        self.assertEqual(s["alternatives"], ["unused_alt.mat"])
        tex = {t["param"]: t for t in s["textures"]}
        self.assertEqual(tex["dif_0_tex"]["texture"], "car_olive_decal_dif.png")
        self.assertEqual(tex["dif_0_tex"]["original"], "car_olive_dif.png")
        self.assertEqual(tex["dif_0_tex"]["source"], "model override")
        self.assertEqual(tex["dif_0_tex"]["gids"], [9])
        self.assertEqual(tex["nrm_0_tex"]["status"], "missing")
        ov = {o["name"]: o for o in s["overrides"]}
        self.assertTrue(ov["gloss"]["applied"])
        self.assertEqual(ov["gloss"]["original"], 0.5)
        self.assertFalse(ov["tint"]["applied"])
        self.assertEqual(v["summary"]["overrides"], 3)
        self.assertEqual(v["summary"]["texture_overrides"], 1)
        self.assertEqual(v["summary"]["textures_found"], 1)
        names = [t["name"] for t in v["unique_textures"]]
        self.assertIn("car_olive_dif.png", names)          # the overridden original is still listed
        # unknown variant: note + defaults
        u = resolve_model(ctx, doc, "chrome", mesh_reader=reader)
        self.assertTrue(any("chrome" in n for n in u["notes"]))

    def test_normalize_variants_shapes(self):
        from nightrunner.gui.modelresolve import normalize_variants
        mats = ["a.mat", "b.mat"]
        out = normalize_variants({"variants": [
            {"name": "x", "material_map": {0: 1}},
            {"name": "y", "material_map": ["a_y.mat", None]},
            {"name": "z", "material_map": [{"from": "b.mat", "to": "b_z.mat"}]}]}, mats)
        self.assertEqual(out[0]["map"], {"a.mat": "b.mat"})
        self.assertEqual(out[1]["map"], {"a.mat": "a_y.mat"})
        self.assertEqual(out[2]["map"], {"b.mat": "b_z.mat"})

    def test_missing_mesh_uses_materials_data_and_join_errors(self):
        from nightrunner.gui.modelresolve import resolve_model
        doc = synthetic_model(mesh="nothere", material="m.mat", base="m2.mat")
        doc["slots"][0]["meshResources"]["resources"][0]["materialsData"].append(
            {"number": 1, "name": "m.mat"})
        res = resolve_model(SimpleNamespace(catalog=StubCatalog(), sdb=None), doc)
        subs = all_subs(res)
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0]["embedded_source"], "model materialsData")
        self.assertIn("listed 2 times", subs[0]["join_error"])
        self.assertFalse(subs[0]["sdb"]["found"])
        self.assertEqual(res["summary"]["meshes_found"], 0)


@unittest.skipUnless(_game() is not None, "game data (NIGHTRUNNER_GAME_ROOT) not available")
class TestResolveReal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.gui.catalog import Catalog
        from nightrunner.gui.services import PakService, SdbService
        cls.game = _game()
        cls.sdb = SdbService(cls.game.sdb("dx11"))
        cls.paks = PakService(cls.game.paks())
        cls.cat = Catalog()
        samples = [p for p in cls.game.rpacks() if "samples_" in p.name]
        cls.cat.load(samples, assets_root=cls.game.assets)
        cls.cat.wait()
        cls.have_crane = bool(cls.cat.lookup("sh2_npc_crane", 0x10))

    @classmethod
    def tearDownClass(cls):
        cls.sdb.close()
        cls.paks.close()
        cls.cat.close()

    def _load(self, name):
        rec = next(r for r in self.paks.models() if r["name"] == name)
        return self.paks.load_model(rec)

    def test_player_stub_catalog(self):
        from nightrunner.gui.modelresolve import json_safe, resolve_model
        import json
        doc = self._load(PLAYER)
        res = resolve_model(SimpleNamespace(catalog=StubCatalog(), sdb=self.sdb), doc, name=PLAYER)
        sm = res["summary"]
        self.assertEqual(sm["slots"], len(doc["slots"]))
        self.assertEqual(sm["meshes_found"], 0)
        self.assertGreaterEqual(sm["meshes"], 10)
        self.assertEqual(sm["textures_found"], 0)
        self.assertGreater(sm["textures"], 50)
        self.assertEqual(sm["materials_in_sdb"], sm["materials"])      # every player material is in the SDB
        self.assertGreater(sm["material_remaps"], 0)                   # e.g. sh2_eye_green → sh2_eye_kyle_crane
        self.assertEqual(res["skeleton"]["status"], "missing")
        subs = all_subs(res)
        remap = [s for s in subs if s["embedded_material"] == "sh2_eye_green.mat"]
        self.assertEqual(remap[0]["base_material"], "sh2_eye_kyle_crane.mat")
        self.assertEqual(remap[0]["material_source"], "model (remap)")
        crane = next(s for s in subs if s["base_material"] == "sh2_npc_crane.mat")
        self.assertTrue(crane["sdb"]["found"])
        self.assertEqual(crane["sdb"]["preset"], "opaque")
        params = {t["param"]: t for t in crane["textures"]}
        self.assertEqual(params["rgh_0_tex"]["texture"], "sh_npc_crane_rgh.png")
        self.assertEqual(params["rgh_0_tex"]["source"], "sdb override")
        self.assertTrue(all(t["status"] == "missing" for t in crane["textures"]))
        json.dumps(json_safe(res))

    def test_trailer_overrides_and_found_meshes(self):
        from nightrunner.gui.modelresolve import resolve_model
        if not self.have_crane:
            self.skipTest("sample crane mesh not present")
        name = "models/ft/player_kyle_crane_trailer.model"
        res = resolve_model(SimpleNamespace(catalog=self.cat, sdb=self.sdb), self._load(name), name=name)
        self.assertEqual(res["summary"]["texture_overrides"], 9)
        self.assertEqual(res["skeleton"]["providers"], ["dlc_samples/samples_meshes_pc.rpack"])
        head = next(m for sl in res["slots"] for m in sl["meshes"] if m["name"] == "sh2_npc_crane.msh")
        self.assertTrue(head["gids"])
        self.assertEqual(len(head["submeshes"]), 7)                    # decoded from the real mesh
        self.assertTrue(all(s["embedded_source"] == "mesh" for s in head["submeshes"]))
        face = next(s for s in head["submeshes"] if s["base_material"] == "sh2_npc_crane.mat")
        tex = {t["param"]: t for t in face["textures"]}
        self.assertEqual(tex["dif_0_tex"]["texture"], "sh_npc_crane_bloody_dif.png")
        self.assertEqual(tex["dif_0_tex"]["original"], "sh_npc_crane_dif.png")
        self.assertEqual(tex["dif_0_tex"]["source"], "model override")
        missing = next(m for sl in res["slots"] for m in sl["meshes"] if m["name"] == "player_kc_torso_a_tpp.msh")
        self.assertEqual(missing["status"], "missing")
        self.assertEqual([s["base_material"] for s in missing["submeshes"]][:2],
                         ["player_kc_torso_a_bloody_tpp.mat", "player_kc_torso_a_sleeves_a_bloody_tpp.mat"])

    def test_synthetic_found_mesh_found_textures(self):
        from nightrunner.gui.modelresolve import resolve_model
        if not self.have_crane or not self.cat.lookup("white.dds", 0x20):
            self.skipTest("sample crane mesh / white.dds not present")
        doc = synthetic_model(material="sh2_npc_crane.mat", base="custom.mat",
                              rtti=[{"name": "dif_0_tex", "type": 7, "val_str": "white.dds"}])
        sdb = StubSdb({"custom.mat": sdb_mat("custom.mat", [("dif_0_tex", "black_dif.dds"),
                                                            ("nrm_0_tex", "nope_nrm.png")])})
        res = resolve_model(SimpleNamespace(catalog=self.cat, sdb=sdb), doc)
        subs = all_subs(res)
        self.assertEqual(len(subs), 7)
        face = next(s for s in subs if s["embedded_material"] == "sh2_npc_crane.mat")
        self.assertEqual((face["base_material"], face["material_source"], face["number"]),
                         ("custom.mat", "model (remap)", 1))
        others = [s for s in subs if s is not face]
        self.assertTrue(all(s["material_source"] == "mesh default" for s in others))
        tex = {t["param"]: t for t in face["textures"]}
        self.assertTrue(tex["dif_0_tex"]["gids"])
        self.assertTrue(tex["dif_0_tex"]["original_gids"])           # black_dif.dds is in the samples too
        self.assertEqual(res["summary"]["meshes_found"], 1)
        self.assertEqual(res["summary"]["textures_found"], 2)
        self.assertEqual(res["summary"]["textures"], 3)


# ---- tab + export ---------------------------------------------------------------------------------------------------

@unittest.skipUnless(_game() is not None, "game data (NIGHTRUNNER_GAME_ROOT) not available")
class TestModelsTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.gui.context import AppContext
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("gui_models_")
        cls.d = Path(cls._tmp.name)
        st = QSettings(str(cls.d / "s.ini"), QSettings.IniFormat)
        cls.ctx = AppContext(settings=st, game=_game())
        cls.ctx.catalog.wait(120)

    @classmethod
    def tearDownClass(cls):
        cls.ctx.runner.pool.waitForDone(10000)
        cls.ctx.close()
        cls._tmp.cleanup()

    def pump(self, cond, timeout=30.0):
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.01)
        return False

    def make_tab(self):
        from nightrunner.gui.tabs.models import TITLE, Tab
        t = Tab(self.ctx)
        self.assertEqual(TITLE, "Models")
        self.assertTrue(self.pump(lambda: len(t.records) > 0 and len(t.list_model.gids) == len(t.records)))
        return t

    def test_populate_open_filter(self):
        t = self.make_tab()
        self.assertGreaterEqual(len(t.records), 1)
        self.assertTrue(t.open_model("PLAYER_TPP_SKELETON"))          # basename, no suffix, any case
        self.assertTrue(self.pump(lambda: t.result is not None
                                  and t.result["rec"]["basename"] == "player_tpp_skeleton.model"))
        # a basename hit takes the overriding provider, so an installed mod pak with a bare root-level member
        # shadows the stock model; the assertions below need the stock one, reached by its full member path.
        self.assertTrue(t.open_model(PLAYER))
        self.assertTrue(self.pump(lambda: t.result is not None and t.result["rec"]["name"] == PLAYER))
        self.assertIsNone(t.result["error"])
        self.assertIn("meshes", t.header.text())
        self.assertGreater(t.tree.topLevelItemCount(), 10)
        self.assertIn('"skeletonName"', t.raw.toPlainText())
        self.assertIn("sh_npc_crane_rgh.png", t.tex_text.toPlainText())
        self.assertIn("sh2_eye_kyle_crane.mat", t.mat_text.toPlainText())
        self.assertEqual(t.variant.itemText(0), "Default")
        # filters hide rows, but never all of them for the player (it has missing meshes)
        t.only_missing.setChecked(True)
        visible = [t.tree.topLevelItem(i) for i in range(t.tree.topLevelItemCount())
                   if not t.tree.topLevelItem(i).isHidden()]
        self.assertLess(len(visible), t.tree.topLevelItemCount())   # empty when the install has every mesh
        t.only_missing.setChecked(False)
        # full path with backslashes selects the other document; search narrows the list
        t.search.edit.setText("crane_trailer")
        self.assertTrue(self.pump(lambda: len(t.list_model.gids) < len(t.records)))
        self.assertTrue(t.open_model("models\\ft\\player_kyle_crane_trailer.model"))
        self.assertTrue(self.pump(lambda: t.result["rec"]["name"].endswith("player_kyle_crane_trailer.model")))
        self.assertFalse(t.open_model("no_such_model"))
        # navigation: double-clicking a material row emits openMaterial
        got = []
        self.ctx.openMaterial.connect(got.append)
        it = self._find(t.tree, lambda i: i.data(0, t_role()) == "material")
        t._on_double(it, 0)
        self.assertTrue(got)
        # a slot with a found mesh gets a checkable entry in the 3D slot list
        names = [t.slot_list.item(i).text() for i in range(t.slot_list.count())]
        self.assertIn("HEAD", names)
        t.shutdown()
        t.deleteLater()

    def _find(self, tree, pred):
        stack = [tree.topLevelItem(i) for i in range(tree.topLevelItemCount())]
        while stack:
            it = stack.pop(0)
            if pred(it):
                return it
            stack.extend(it.child(k) for k in range(it.childCount()))
        self.fail("no matching tree item")

    def test_pending_open_before_list(self):
        from nightrunner.gui.tabs.models import Tab
        t = Tab(self.ctx)
        t.records = []
        self.assertFalse(t.open_model(PLAYER))
        self.assertTrue(self.pump(lambda: t.result is not None and t.result["rec"]["name"] == PLAYER))
        t.deleteLater()

    def test_export(self):
        if not self.ctx.catalog.lookup("sh2_npc_crane", 0x10):
            self.skipTest("sample crane mesh not present")
        from nightrunner.gui.tabs.models import export_models
        rec = next(r for r in self.ctx.paks.models() if r["name"] == "models/ft/player_kyle_crane_trailer.model")
        out = self.d / "export"
        res = export_models(self.ctx, [rec], out)
        self.assertFalse(res["errors"])
        m = res["models"][0]
        md = Path(m["dir"])
        self.assertEqual(md.name, "player_kyle_crane_trailer")
        self.assertTrue((md / "player_kyle_crane_trailer.model").is_file())
        self.assertTrue((md / "mapping.json").is_file())
        self.assertGreaterEqual(m["meshes"], 3)
        self.assertTrue((md / "meshes" / "sh2_npc_crane" / "sh2_npc_crane.cast").is_file())
        self.assertTrue((md / "meshes" / "sh2_npc_crane" / "raw" / "image.bin").is_file())
        log = (md / "export_log.txt").read_text()
        if not self.ctx.catalog.lookup("player_kc_torso_a_tpp", 0x10):   # sample installs lack it
            self.assertIn("MISSING mesh player_kc_torso_a_tpp.msh", log)
        self.assertGreater(m["missing"], 0)
        if m["missing"] > 100:                                              # sample install
            self.assertIn("MISSING texture", log)
        self.assertEqual(m["failed"], 0)
        # second export never overwrites: a new folder is chosen
        res2 = export_models(self.ctx, [rec], out)
        self.assertNotEqual(res2["models"][0]["dir"], m["dir"])

    def test_export_single_cast(self):
        if not (self.ctx.catalog.lookup("sh2_npc_crane", 0x10)
                and self.ctx.catalog.lookup("sh2_player_tpp_phx_skeleton", 0x10)):
            self.skipTest("sample crane / player skeleton meshes not present")
        import numpy as np
        from nightrunner.cast import castlib
        from nightrunner.gui.tabs.models import export_models
        rec = next(r for r in self.ctx.paks.models() if r["name"] == "models/ft/player_kyle_crane_trailer.model")
        res = export_models(self.ctx, [rec], self.d / "single", mode="cast")
        self.assertFalse(res["errors"])
        m = res["models"][0]
        md = Path(m["dir"])
        cast_path = md / "player_kyle_crane_trailer.cast"
        self.assertEqual(m["cast"], str(cast_path))
        self.assertFalse((md / "meshes").exists())                  # cast-only mode: no split files
        self.assertTrue((md / "nightrunner_blender_materials.py").is_file())
        rep = json.loads((md / "player_kyle_crane_trailer.cast.json").read_text())
        self.assertEqual(rep["skeleton"]["name"], "sh2_player_tpp_phx_skeleton.msh")
        c = castlib.Cast.load(str(cast_path))
        mdl = c.Roots()[0].ChildOfType(castlib.Model)
        bones = mdl.Skeleton().Bones()
        self.assertEqual(len(bones), rep["skeleton"]["merged_bones"])
        self.assertEqual(bones[0].Name(), "pelvis")                  # preset skeleton order first
        names = [b.Name() for b in bones]
        self.assertIn("sh_npc_ft_crane_hair_a", names)             # part-only bones appended
        meshes = mdl.Meshes()
        self.assertEqual(len(meshes), rep["meshes"])
        self.assertTrue(any(x.Name().startswith("HEAD_PART_1.sh_npc_ft_crane_hair_a") for x in meshes))
        for x in meshes:
            wb = np.asarray(x.VertexWeightBoneBuffer())
            self.assertLess(int(wb.max()), len(bones))
            self.assertIsNotNone(x.Material())
        # the hair's own rig equals the skeleton around its bones: rebind is (almost) the identity
        hair = next(p for p in rep["parts"] if p["mesh"] == "sh_npc_ft_crane_hair_a.msh")
        self.assertEqual(hair["submeshes"], 1)
        if not self.ctx.catalog.lookup("sh2_npc_ft_crane_eyebrows", 0x10):
            self.assertIn("mesh sh2_npc_ft_crane_eyebrows.msh missing", " ".join(rep["problems"]))

    def test_export_texture_and_async(self):
        import threading
        from nightrunner.gui.tabs.models import Tab, export_models
        if not self.ctx.catalog.lookup("white.dds", 0x20):
            self.skipTest("white.dds not present")
        doc = synthetic_model(mesh="dummybox_025m", material="DUMMYBOX.MAT",
                              rtti=[{"name": "dif_0_tex", "type": 7, "val_str": "white.dds"}])
        rec = {"name": "synthetic/test_box.model", "basename": "test_box.model", "pak": self.d / "none.pak",
               "member": None, "overridden_by": []}
        paks = SimpleNamespace(load_model=lambda r: doc, index=lambda p: None)
        ctx = SimpleNamespace(catalog=self.ctx.catalog, sdb=self.ctx.sdb, paks=paks)
        res = export_models(ctx, [rec], self.d / "exp_tex")
        m = res["models"][0]
        self.assertEqual(m["textures"], 1)
        self.assertTrue((Path(m["dir"]) / "textures" / "white.dds").is_file())
        self.assertIn("re-serialised", (Path(m["dir"]) / "export_log.txt").read_text())
        cancel = threading.Event()
        cancel.set()
        res = export_models(ctx, [rec, rec], self.d / "exp_cancel", cancel=cancel)
        self.assertTrue(res["cancelled"])
        self.assertEqual(len(res["models"]), 1)
        # through the tab (worker + progress signals)
        t = Tab(self.ctx)
        t.ctx = SimpleNamespace(**{k: getattr(self.ctx, k) for k in ("catalog", "sdb", "runner", "status")},
                                paks=paks)
        done = []
        t.exportFinished.connect(done.append)
        self.assertTrue(t.start_export([rec], self.d / "exp_tab"))
        self.assertFalse(t.start_export([rec], self.d / "exp_tab"))
        self.assertTrue(self.pump(lambda: bool(done)))
        self.assertEqual(done[0]["models"][0]["textures"], 1)
        self.assertTrue(t.btn_export_checked.isEnabled())
        t.deleteLater()


def t_role():
    from nightrunner.gui.tabs.models import ROLE_KIND
    return ROLE_KIND


if __name__ == "__main__":
    unittest.main()
