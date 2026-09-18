"""Build projects (nightrunner/project.py): default output names, .model editing helpers, appearance parsing,
validation, and a full build (textures new + same-name, a split single-model scene with a rename, a per-mesh GLB
clone, a player model override) against out/samples/{meshes,textures}.rpack and a synthetic data0.pak."""
from __future__ import annotations

import copy
import json
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tests.synth as S
from tests.paths import SAMPLES
from tests.test_model_split import _Catalog, _blenderize, _resolution

MESHES = SAMPLES / "meshes.rpack"
TEXTURES = SAMPLES / "textures.rpack"

OUTFIT = '''
// mapping of outfit part slots to model slots.
sub main()	
{
	Torso( "TORSO", "ARMS" );
	Head( "HEAD", "HEAD_PART_1" );
}
'''

SCR = '''sub appearances()
{
    Character("PlayerMan1")
    {
        Appearance("Default")
        {
            ModelFpp("player_fpp_skeleton.model");
            ModelTpp("player_tpp_skeleton.model");
            ModelTppLodCC("player_tpp_skeleton.model");
            ModelUI("player_tpp_skeleton.model"); // ui
        }
        Appearance("Night")
        {
            ModelTpp("player_night_tpp.model");
        }
    }
}
'''


def _res(name, selected=True, mats=()):
    return {"name": name, "selected": selected, "layoutId": 4, "userData": [0, 0, 0, 0],
            "materialsData": [{"number": i, "name": m, "layoutId": 4, "loadFlags": "S"} for i, m in enumerate(mats)],
            "materialsResources": [{"number": i, "resources": [{"name": m, "selected": True, "layoutId": 4,
                                                                "loadFlags": "S", "rttiValues": []}]}
                                   for i, m in enumerate(mats)]}


def _model(skel, slots):
    return {"version": 6, "preset": {"skeletonName": skel}, "data": {"meshAttribute": [0, 0, 0, 0], "properties": []},
            "slots": [{"slotUid": i, "name": n, "filterText": n.split("_")[0].lower(), "tagsBits": 0,
                       "meshResources": {"resources": r}} for i, (n, r) in enumerate(slots)]}


TPP = _model("sh2_player_tpp_phx_skeleton.msh", [
    ("HEAD", [_res("sh2_npc_crane.msh")]),
    ("HEAD_PART_1", [_res("sh2_npc_ft_crane_beard_a.msh")]),
    ("HEAD_PART_3", [_res("sh_npc_ft_crane_hair_a.msh", mats=["sh_npc_ft_crane_hair_a.mat"])]),
])
FPP = _model("player_fpp_phx_skeleton.msh", [("HEAD", []), ("HEAD_PART_3", [])])


@unittest.skipUnless(MESHES.is_file() and TEXTURES.is_file(), "out/samples meshes/textures packs not available")
class ProjectTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.container.rp6l import Pack
        from nightrunner.gui.modelcast import export_model_cast
        from nightrunner.pak.model_json import write_models_pak
        from nightrunner.project import GameEnv
        cls._tmp = S.tmpdir("proj_")
        d = cls.d = Path(cls._tmp.name)
        # fake install: assets with an existing assets_2, source with data0/data1/data3
        (d / "assets").mkdir()
        (d / "assets" / "assets_2_pc.rpack").write_bytes(b"")
        (d / "source").mkdir()
        write_models_pak(d / "source" / "data0.pak", {"models/player/player_tpp_skeleton.model": TPP,
                                                      "models/player/player_fpp_skeleton.model": FPP},
                         texts={"scripts/playerappearances.scr": SCR, "scripts/player_outfit_slots.scr": OUTFIT})
        for n in ("data1.pak", "data3.pak"):
            (d / "source" / n).write_bytes(b"")
        cls.env = GameEnv(d / "assets", d / "source", packs={"meshes.rpack": MESHES, "textures.rpack": TEXTURES})
        # inputs
        from PIL import Image
        (d / "in").mkdir()
        Image.fromarray(np.full((64, 32, 4), (200, 30, 30, 255), np.uint8)).save(d / "in" / "crane_retex_dif.png")
        Image.fromarray(np.full((8, 8, 4), 128, np.uint8)).save(d / "in" / "white.png")
        cls.pack = Pack.open(MESHES)
        cat = _Catalog(cls.pack)
        rep = export_model_cast(SimpleNamespace(catalog=cat, sdb=None), _resolution(cat), d / "exp", "crane_head",
                                textures=False)
        cls.report = d / "exp" / "crane_head.cast.json"

        def edit(m, mdl):
            if m.Name() == "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0":
                vp = np.asarray(m.VertexPositionBuffer()).reshape(-1, 3)
                vp[:5, 1] += 0.02
                m.SetVertexPositionBuffer(vp.tolist())
        cls.scene = d / "in" / "crane_head.cast"
        _blenderize(Path(rep["cast"]), cls.scene, edit)
        # the report must sit next to the scene for add_files to call it a scene
        (d / "in" / "crane_head.cast.json").write_text(cls.report.read_text())
        from nightrunner.cast.export import export_cast
        from nightrunner.mesh.decode import decode_resource
        beard = decode_resource(cls.pack.resource(cls.pack.find("sh2_npc_ft_crane_beard_a")[0]))
        (d / "in" / "beard").mkdir()
        export_cast(beard, d / "in" / "beard" / "beard_copy.glb")

    @classmethod
    def tearDownClass(cls):
        cls.pack.close()
        cls._tmp.cleanup()

    # ---- helpers -----------------------------------------------------------------------------------------------
    def _project(self):
        from nightrunner.project import Project, add_files, load_override, set_rtti, set_slot_enabled, \
            set_slot_mesh, material_entry, parse_appearances
        pr = Project(name="crane_retex", output_dir=str(self.d / "out"))
        add_files(pr, [self.d / "in" / "crane_retex_dif.png", self.d / "in" / "white.png", self.scene,
                       self.d / "in" / "beard" / "beard_copy.glb"], self.env)
        tex_new, tex_rep, scene, beard = pr.items
        tex_new.target_name, tex_new.target_pack = "npc_ft_hunter_pants_a.png", "textures.rpack"
        self.assertEqual((tex_rep.target_name, tex_rep.target_pack, tex_rep.new_name),
                         ("white.dds", "textures.rpack", None))
        scene.options["renames"] = {"sh_npc_ft_crane_hair_a": "crane_hair_long"}
        beard.kind = "mesh"
        beard.target_name, beard.target_pack, beard.new_name = "sh2_npc_ft_crane_beard_a", "meshes.rpack", "crane_beard_copy"
        with __import__("zipfile").ZipFile(self.d / "source" / "data0.pak") as z:
            app = parse_appearances(z.read("scripts/playerappearances.scr").decode())[0]
        mo = load_override("Player · Default", {k: app[k] for k in ("tpp", "fpp", "lodcc", "ui")},
                           self.d / "source" / "data0.pak")
        tpp = mo.roles["tpp"].doc
        hit = set_slot_mesh(tpp, "HEAD_PART_3", "crane_hair_long")
        set_rtti(material_entry(hit, "sh_npc_ft_crane_hair_a.mat"), "dif_0_tex", "crane_retex_dif.png")
        set_slot_enabled(tpp, "HEAD_PART_1", False, mo.roles["tpp"].stash)
        set_slot_mesh(tpp, "HEAD", "crane_beard_copy")
        pr.models.append(mo)
        return pr

    # ---- tests -------------------------------------------------------------------------------------------------
    def test_default_names(self):
        from nightrunner.project import Project, next_free_pak, next_free_rpack, output_names
        self.assertEqual(next_free_pak(self.d / "source"), "data2.pak")
        self.assertEqual(next_free_rpack(self.d / "assets"), "assets_3_pc.rpack")
        self.assertEqual(next_free_pak(None), "data0.pak")
        self.assertEqual(output_names(Project(pak_name="mine.pak"), self.env), ("assets_3_pc.rpack", "mine.pak"))

    def test_appearances(self):
        from nightrunner.project import parse_appearances
        a = parse_appearances(SCR)
        self.assertEqual(a[0], {"character": "PlayerMan1", "appearance": "Default", "fpp": "player_fpp_skeleton.model",
                                "tpp": "player_tpp_skeleton.model", "lodcc": "player_tpp_skeleton.model",
                                "ui": "player_tpp_skeleton.model"})
        self.assertEqual(a[1], {"character": "PlayerMan1", "appearance": "Night", "tpp": "player_night_tpp.model"})

    def test_model_helpers_and_diff(self):
        from nightrunner.project import diff_docs, material_entry, set_rtti, set_slot_enabled, set_slot_mesh, slot, \
            slot_mesh
        doc = copy.deepcopy(TPP)
        stash = {}
        set_slot_enabled(doc, "HEAD", False, stash)
        self.assertIsNone(slot_mesh(slot(doc, "HEAD")))
        self.assertEqual(slot(doc, "HEAD")["meshResources"]["resources"], [])     # off = empty (stock convention)
        self.assertEqual(list(stash), ["HEAD"])
        set_slot_enabled(doc, "HEAD", True, stash)
        self.assertEqual(stash, {})
        self.assertEqual(slot_mesh(slot(doc, "HEAD")), "sh2_npc_crane.msh")
        new = set_slot_mesh(doc, "HEAD_PART_3", "my_hair")
        self.assertEqual(new["name"], "my_hair.msh")
        self.assertEqual(new["materialsData"], TPP["slots"][2]["meshResources"]["resources"][0]["materialsData"])
        self.assertEqual([r["selected"] for r in slot(doc, "HEAD_PART_3")["meshResources"]["resources"]], [False, True])
        d = diff_docs(TPP, doc)
        self.assertIn("~ slots[HEAD_PART_3].meshResources.resources[sh_npc_ft_crane_hair_a.msh].selected: true → false", d)
        set_slot_mesh(doc, "HEAD_PART_3", "sh_npc_ft_crane_hair_a")          # existing entry reselected, no dup
        self.assertEqual(len(slot(doc, "HEAD_PART_3")["meshResources"]["resources"]), 2)
        ent = material_entry(new, "other.mat", base="player_kc_basic_torso_a_tpp.mat")
        set_rtti(ent, "dif_0_tex", "x.png")
        set_rtti(ent, "dif_0_val", [1, 0.5, 0.25])
        set_rtti(ent, "dif_0_tex", "y.png")
        self.assertEqual(ent["rttiValues"], [{"name": "dif_0_val", "type": 4, "val_vec3": [1.0, 0.5, 0.25]},
                                             {"name": "dif_0_tex", "type": 7, "val_str": "y.png"}])
        self.assertEqual(new["materialsData"][-1]["number"], 1)
        d = diff_docs(TPP, doc)
        self.assertIn("+ slots[HEAD_PART_3].meshResources.resources[my_hair.msh]", d)
        self.assertFalse(any("crane_hair_a.msh].selected" in x for x in d))

    def test_save_load_round_trip(self):
        from nightrunner.project import Project
        pr = self._project()
        p = pr.save(self.d / "proj" / "crane_retex.nrproj")
        raw = json.loads(p.read_text())
        self.assertTrue(Path(raw["items"][0]["source"]).is_absolute())      # outside the project folder
        pr2 = self._project()
        p2 = pr2.save(self.d / "in" / "local.nrproj")
        self.assertEqual(json.loads(p2.read_text())["items"][0]["source"], "crane_retex_dif.png")
        self.assertEqual(Project.load(p2).items[0].source, pr2.items[0].source)
        back = Project.load(p)
        self.assertEqual([i.source for i in back.items], [i.source for i in pr.items])
        self.assertEqual(back.models[0].roles["tpp"].doc, pr.models[0].roles["tpp"].doc)
        self.assertIn("HEAD_PART_1", back.models[0].roles["tpp"].stash)
        self.assertEqual(back.to_json(), pr.to_json())

    def test_move_and_add_slot(self):
        from nightrunner.project import (add_slot, free_slot_uid, material_entry, move_slot_mesh, next_torso_slot,
                                       set_rtti, set_slot_mesh, slot, slot_mesh, validate)
        doc = copy.deepcopy(TPP)
        self.assertEqual(free_slot_uid(doc), 100)                  # synthetic uids are 0..2; new ones start at 100
        stash = {}
        hit = set_slot_mesh(doc, "HEAD", "my_head", stash)
        set_rtti(material_entry(hit, "a.mat"), "dif_0_tex", "x.png")
        self.assertEqual(next_torso_slot(doc), "TORSO_PART_1")
        ent = move_slot_mesh(doc, "HEAD", "TORSO_PART_1", stash)
        self.assertIs(ent, hit)
        s = slot(doc, "TORSO_PART_1")
        self.assertEqual((s["filterText"], s["shadowMaps"], s["slotUid"]), ("torso", 15, 100))
        self.assertEqual(s["meshResources"]["resources"], [hit])
        self.assertIsNone(slot_mesh(slot(doc, "HEAD")))
        self.assertEqual([r["name"] for r in stash["HEAD"]], ["sh2_npc_crane.msh"])
        self.assertIs(add_slot(doc, "TORSO_PART_1"), s)
        self.assertEqual(add_slot(doc, "TORSO_PART_2")["slotUid"], 101)
        self.assertIsNone(move_slot_mesh(doc, "HEAD", "TORSO_PART_2"))
        # head/legs warning for project meshes
        pr = self._project()
        msgs = [p.message for p in validate(pr, self.env)]
        self.assertTrue(any("HEAD_PART_3: new meshes in head/legs slots" in m for m in msgs), msgs)
        tpp = pr.models[0].roles["tpp"]
        move_slot_mesh(tpp.doc, "HEAD_PART_3", "TORSO_PART_1", tpp.stash)
        move_slot_mesh(tpp.doc, "HEAD", "TORSO_PART_2", tpp.stash)
        msgs = [p.message for p in validate(pr, self.env)]
        self.assertFalse(any("head/legs slots" in m for m in msgs), msgs)

    def test_no_gear(self):
        from nightrunner.project import OUTFIT_TEMPLATE, Project, empty_outfit_script, validate
        pr = self._project()
        mo = pr.models[0]
        self.assertTrue(mo.is_player() and mo.no_gear)             # on by default for player models
        self.assertEqual(empty_outfit_script(None), OUTFIT_TEMPLATE)
        self.assertEqual(empty_outfit_script("no main here"), OUTFIT_TEMPLATE)
        self.assertEqual(empty_outfit_script("x\nsub main() { A(); { B(); } }\ny"), "x\nsub main() {\n\n}\ny")
        mo.no_gear = False
        self.assertIn("gear still replaces slots (No gear off)", [p.message for p in validate(pr, self.env)])
        back = Project.from_json(pr.to_json())
        self.assertFalse(back.models[0].no_gear)
        mo.no_gear = True
        self.assertTrue(Project.from_json(pr.to_json()).models[0].no_gear)

    def test_validate_catches_missing(self):
        from nightrunner.project import material_entry, set_rtti, set_slot_mesh, validate
        pr = self._project()
        self.assertEqual([p for p in validate(pr, self.env) if p.level == "error"], [])
        tpp = pr.models[0].roles["tpp"].doc
        hit = set_slot_mesh(tpp, "HEAD_PART_1", "nope_mesh")
        set_rtti(material_entry(hit, "a.mat"), "nrm_0_tex", "nope.png")
        pr.items[0].target_name = "missing.png"
        msgs = [p.message for p in validate(pr, self.env) if p.level == "error"]
        self.assertTrue(any("nope_mesh.msh not found" in m for m in msgs), msgs)
        self.assertTrue(any("nope.png not found" in m for m in msgs), msgs)
        self.assertTrue(any("no target" in m for m in msgs), msgs)
        # crane_retex_dif.png is no longer provided (its item lost its target) -> not an error by itself

    def test_build(self):
        from nightrunner.container.rp6l import Pack
        from nightrunner.container.validate import validate as pack_validate
        from nightrunner.mesh.decode import decode_resource
        from nightrunner.project import build_project
        pr = self._project()
        rep = build_project(pr, self.env, out_dir=self.d / "build")
        rp = Path(rep["outputs"]["rpack"]["path"])
        self.assertEqual(rp.name, "assets_3_pc.rpack")
        self.assertTrue(rep["outputs"]["rpack"]["validation_ok"])
        with Pack.open(rp) as pk:
            self.assertTrue(pack_validate(pk).ok)
            names = sorted((r.name, r.type) for r in pk)
            self.assertEqual(names, [("crane_beard_copy", 0x10), ("crane_hair_long", 0x10),
                                     ("crane_retex_dif.png", 0x20), ("white.dds", 0x20)])
            self.assertEqual(pk.header.field08, 0x1000)
            def vertex(pack, name):
                res = pack.resource(pack.find(name)[0])
                return next(bytes(pack.read_part(i)) for i in res.part_indices if pack.part_type(i) == 0xF0)
            self.assertNotEqual(vertex(pk, "crane_hair_long"), vertex(self.pack, "sh_npc_ft_crane_hair_a"))
            self.assertEqual(vertex(pk, "crane_beard_copy"), vertex(self.pack, "sh2_npc_ft_crane_beard_a"))
            self.assertEqual(decode_resource(pk.resource(pk.find("crane_hair_long")[0])).name, "crane_hair_long")
        pak = Path(rep["outputs"]["pak"]["path"])
        self.assertEqual(pak.name, "data2.pak")
        with zipfile.ZipFile(pak) as z:
            self.assertEqual(sorted(z.namelist()), ["player_fpp_skeleton.model", "player_outfit_slots.scr",
                                                    "player_tpp_skeleton.model"])
            doc = json.loads(z.read("player_tpp_skeleton.model"))
            outfit = z.read("player_outfit_slots.scr").decode()
        self.assertIn("// mapping of outfit part slots", outfit)
        self.assertNotIn("Torso(", outfit)
        self.assertRegex(outfit, r"sub main\(\)\s*\{\s*\}")
        from nightrunner.project import game_doc
        self.assertEqual(doc, game_doc(pr.models[0].roles["tpp"].doc))
        self.assertTrue(all(len(s["meshResources"]["resources"]) <= 1 for s in doc["slots"]))
        self.assertEqual(rep["install"], {"rpack": "ph_ft/work/data_platform/pc/assets/assets_3_pc.rpack",
                                          "pak": "ph_ft/source/data2.pak"})
        self.assertTrue((self.d / "build" / "crane_retex.build.json").is_file())
        self.assertFalse((self.d / "build" / ".crane_retex.work").exists())
        split = next(i for i in rep["items"] if i["kind"] == "scene")
        hair_split = next(s for s in split["split"] if s["mesh"] == "sh_npc_ft_crane_hair_a")
        self.assertEqual(hair_split["applied"], 1)


class HashCacheTests(unittest.TestCase):
    def test_cached_until_file_changes(self):
        import os, tempfile
        from nightrunner.util import hashing
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "a.bin"
            f.write_bytes(b"one")
            h1 = hashing.sha256_file(f)
            self.assertEqual(h1, hashing.sha256_bytes(b"one"))
            f.write_bytes(b"two!")
            os.utime(f, ns=(1, 10**18))
            self.assertEqual(hashing.sha256_file(f), hashing.sha256_bytes(b"two!"))


class GameDocTests(unittest.TestCase):
    def test_one_entry_per_slot(self):
        from nightrunner.project import game_doc
        doc = {"slots": [
            {"name": "A", "meshResources": {"resources": [{"name": "stock.msh", "selected": False},
                                                          {"name": "new.msh", "selected": True}]}},
            {"name": "B", "meshResources": {"resources": [{"name": "x.msh"}, {"name": "y.msh"}]}},
            {"name": "C", "meshResources": {"resources": []}},
            {"name": "D", "meshResources": {"resources": [{"name": "only.msh", "selected": True}]}}]}
        g = game_doc(doc)
        self.assertEqual([[r["name"] for r in s["meshResources"]["resources"]] for s in g["slots"]],
                         [["new.msh"], ["x.msh"], [], ["only.msh"]])
        self.assertTrue(g["slots"][1]["meshResources"]["resources"][0]["selected"])
        self.assertEqual(len(doc["slots"][0]["meshResources"]["resources"]), 2)      # project copy untouched
