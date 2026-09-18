"""Single-file model Cast → per-mesh model.cast (nightrunner/cast/split.py), without Blender.

A single Cast is exported (gui/modelcast.py, textures off, stub catalog) for the Crane head + hair + beard on the
player skeleton from out/samples/meshes.rpack. "Blender" is simulated the way the official exporter behaves:
bp_* properties gone, triangles rotated, weight lanes in another order. Skips without the sample pack.
"""
from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tests.synth as S
from tests.paths import SAMPLES

PACK = SAMPLES / "meshes.rpack"
MESHES = [("HEAD", "sh2_npc_crane.msh"), ("HEAD_PART_1", "sh_npc_ft_crane_hair_a.msh"),
          ("HEAD_PART_3", "sh2_npc_ft_crane_beard_a.msh")]


class _Catalog:
    def __init__(self, pack):
        self.pack = pack
        self.entry = SimpleNamespace(label="samples", path=PACK, pack=pack)

    def lookup(self, name, type_id):
        return list(self.pack.find(name.removesuffix(".msh"), type_id))

    def split(self, gid):
        return self.entry, gid


def _resolution(cat) -> dict:
    return {"name": "synthetic/crane_head.model", "variant": "Default",
            "skeleton": {"name": "sh2_player_tpp_phx_skeleton.msh",
                         "gids": cat.lookup("sh2_player_tpp_phx_skeleton", 0x10)},
            "slots": [{"name": slot, "meshes": [{"name": m, "chosen": True, "gids": cat.lookup(m, 0x10),
                                                 "submeshes": []}]} for slot, m in MESHES]}


def _blenderize(src: Path, dst: Path, edit=None) -> None:
    """Re-save like the Blender exporter: no custom properties, rotated triangles, reversed weight lanes."""
    from nightrunner.cast import castlib
    c = castlib.Cast.load(str(src))
    mdl = c.Roots()[0].ChildOfType(castlib.Model)
    for m in mdl.Meshes():
        for k in [k for k in m.properties if k.startswith("bp_")] + ["vt"]:
            m.properties.pop(k, None)
        f = np.asarray(m.FaceBuffer()).reshape(-1, 3)[:, [2, 0, 1]]
        m.SetFaceBuffer(f.reshape(-1).tolist())
        mi = m.MaximumWeightInfluence()
        wb = np.asarray(m.VertexWeightBoneBuffer()).reshape(-1, mi)[:, ::-1]
        wv = np.asarray(m.VertexWeightValueBuffer()).reshape(-1, mi)[:, ::-1]
        m.SetVertexWeightBoneBuffer(wb.reshape(-1).tolist())
        m.SetVertexWeightValueBuffer(wv.reshape(-1).tolist())
        if edit:
            edit(m, mdl)
    c.save(str(dst))


@unittest.skipUnless(PACK.is_file(), "out/samples/meshes.rpack not available")
class SplitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.container.rp6l import Pack
        from nightrunner.gui.modelcast import export_model_cast
        cls._tmp = S.tmpdir("split_")
        cls.d = Path(cls._tmp.name)
        cls.pack = Pack.open(PACK)
        cat = _Catalog(cls.pack)
        cls.rep = export_model_cast(SimpleNamespace(catalog=cat, sdb=None), _resolution(cat), cls.d / "exp",
                                    "crane_head", textures=False)
        cls.cast = Path(cls.rep["cast"])
        cls.report = cls.d / "exp" / "crane_head.cast.json"

    @classmethod
    def tearDownClass(cls):
        cls.pack.close()
        cls._tmp.cleanup()

    def split(self, edited, name, **kw):
        from nightrunner.cast.split import split_model_cast
        return split_model_cast(edited, self.report, self.d / name, **kw)

    def test_export_report(self):
        self.assertEqual(self.rep["format"], "nightrunner.model_cast/1")
        self.assertEqual(len(self.rep["parts"]), 3)
        self.assertEqual(self.rep["mesh_map"][0]["name"], "HEAD.sh2_npc_crane.e0.s0")
        self.assertEqual(len(self.rep["bones"]), self.rep["skeleton"]["merged_bones"])

    def test_unedited_round_trip_is_byte_identical(self):
        edited = self.d / "plain.cast"
        _blenderize(self.cast, edited)
        res = self.split(edited, "plain")
        self.assertEqual(res["problems"], [])
        for m in res["meshes"]:
            self.assertTrue(m["check"]["ok"], m)
            self.assertTrue(m["check"]["unchanged"], m["mesh"])
            self.assertTrue(all(s["in_place"] for s in m["submeshes"]))
        self.assertTrue((self.d / "plain" / "split_report.json").is_file())
        self.assertTrue((self.d / "plain" / "sh2_npc_crane" / "mesh.json").is_file())

    def test_moved_vertices_are_unbound_back(self):
        from nightrunner.cast import castlib

        def edit(m, mdl):
            if m.Name() == "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0":
                vp = np.asarray(m.VertexPositionBuffer()).reshape(-1, 3)
                vp[:10, 1] += 0.01
                m.SetVertexPositionBuffer(vp.tolist())

        edited = self.d / "moved.cast"
        _blenderize(self.cast, edited, edit)
        res = self.split(edited, "moved")
        hair = next(m for m in res["meshes"] if m["mesh"] == "sh_npc_ft_crane_hair_a.msh")
        sub = hair["submeshes"][0]
        self.assertEqual((sub["moved"], sub["new"], sub["in_place"]), (10, 0, True))
        self.assertTrue(hair["check"]["ok"])
        self.assertFalse(hair["check"]["unchanged"])
        # the hair's rebind is ~identity: the Cast written for the mesh moved by the same 0.01
        from nightrunner.cast.export import build_cast
        from nightrunner.mesh.decode import decode_resource
        orig, _ = build_cast(decode_resource(self.pack.resource(self.pack.find("sh_npc_ft_crane_hair_a")[0])))
        new = castlib.Cast.load(str(Path(hair["dir"]) / "model.cast"))

        def e0s0(c):
            node = next(x for x in c.Roots()[0].ChildOfType(castlib.Model).Meshes() if x.Name().endswith("e0.s0"))
            return np.asarray(node.VertexPositionBuffer()).reshape(-1, 3)

        d = e0s0(new) - e0s0(orig)
        self.assertTrue(np.allclose(d[:10, 1], 0.01, atol=1e-4))
        self.assertTrue(np.allclose(d[10:], 0))
        others = [m for m in res["meshes"] if m["mesh"] != "sh_npc_ft_crane_hair_a.msh"]
        self.assertTrue(all(m["check"]["unchanged"] for m in others))

    def test_topology_change_and_loose_vertices(self):
        def edit(m, mdl):
            if m.Name() == "HEAD_PART_3.sh2_npc_ft_crane_beard_a.e0.s0":
                f = np.asarray(m.FaceBuffer()).reshape(-1, 3)
                m.SetFaceBuffer(f[20:].reshape(-1).tolist())          # 20 triangles gone, vertices left loose

        edited = self.d / "topo.cast"
        _blenderize(self.cast, edited, edit)
        res = self.split(edited, "topo")
        beard = next(m for m in res["meshes"] if m["mesh"] == "sh2_npc_ft_crane_beard_a.msh")
        sub = beard["submeshes"][0]
        self.assertTrue(beard["check"]["ok"], beard)
        self.assertGreater(sub["loose_dropped"], 0)
        self.assertEqual(sub["new"], 0)
        self.assertEqual(sub["faces"], sub["faces"])

    def test_weight_on_foreign_bone_is_refused(self):
        def edit(m, mdl):
            if m.Name() == "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0":
                names = [b.Name() for b in mdl.Skeleton().Bones()]
                mi = m.MaximumWeightInfluence()
                wb = np.asarray(m.VertexWeightBoneBuffer()).reshape(-1, mi)
                wv = np.asarray(m.VertexWeightValueBuffer()).reshape(-1, mi)
                wb[0, int(np.argmax(wv[0]))] = names.index("sh2_npc_ft_crane_beard_a")  # a beard-only bone
                m.SetVertexWeightBoneBuffer(wb.reshape(-1).tolist())

        edited = self.d / "foreign.cast"
        _blenderize(self.cast, edited, edit)
        res = self.split(edited, "foreign")
        hair = next(m for m in res["meshes"] if m["mesh"] == "sh_npc_ft_crane_hair_a.msh")
        self.assertTrue(any("sh2_npc_ft_crane_beard_a" in p for p in hair["problems"]))
        self.assertTrue(hair["submeshes"][0]["status"].startswith("REFUSED"))

    def test_install_into_rpx_and_build(self):
        from nightrunner.build import build
        from nightrunner.container.rp6l import Pack
        from nightrunner.extract import extract

        def edit(m, mdl):
            if m.Name() == "HEAD.sh2_npc_crane.e0.s4":
                vp = np.asarray(m.VertexPositionBuffer()).reshape(-1, 3)
                vp[:5, 2] += 0.005
                m.SetVertexPositionBuffer(vp.tolist())

        edited = self.d / "rpx_edit.cast"
        _blenderize(self.cast, edited, edit)
        tree = self.d / "tree.rpx"
        extract(self.pack, tree, progress=False)
        res = self.split(edited, "rpx_split", rpx=tree)
        head = next(m for m in res["meshes"] if m["mesh"] == "sh2_npc_crane.msh")
        self.assertTrue(head["rpx"].endswith("model.cast"))
        self.assertTrue(Path(head["rpx"]).with_name("model.cast.orig").is_file())
        out = self.d / "built.rpack"
        rep = build(tree, out)
        self.assertTrue(rep["validation"]["ok"])
        self.assertIn("sh2_npc_crane", [r["name"] for r in rep["regenerated"]])
        with Pack.open(out) as pk:
            self.assertEqual(len(pk), len(self.pack))

    def test_stale_report_is_refused(self):
        from nightrunner.cast.split import SplitError, split_model_cast
        rep = json.loads(self.report.read_text())
        rep["bones"] = rep["bones"][:-1]
        with self.assertRaises(SplitError):
            split_model_cast(self.cast, rep, self.d / "stale")
        rep["format"] = "x"
        with self.assertRaises(SplitError):
            split_model_cast(self.cast, rep, self.d / "stale")
        shutil.rmtree(self.d / "stale", ignore_errors=True)


if __name__ == "__main__":
    unittest.main()


@unittest.skipUnless(PACK.is_file(), "out/samples/meshes.rpack not available")
class AssignHideTests(unittest.TestCase):
    """New objects mapped onto export submeshes (whole replacement), hidden submeshes, LOD follow."""

    @classmethod
    def setUpClass(cls):
        from nightrunner.container.rp6l import Pack
        from nightrunner.gui.modelcast import export_model_cast
        cls._tmp = S.tmpdir("assign_")
        cls.d = Path(cls._tmp.name)
        cls.pack = Pack.open(PACK)
        cat = _Catalog(cls.pack)
        cls.rep = export_model_cast(SimpleNamespace(catalog=cat, sdb=None), _resolution(cat), cls.d / "exp",
                                    "crane_head", textures=False)
        cls.report = cls.d / "exp" / "crane_head.cast.json"

    @classmethod
    def tearDownClass(cls):
        cls.pack.close()
        cls._tmp.cleanup()

    @staticmethod
    def sphere(mdl, name, bone, center, radius=0.1):
        U, V = 12, 8
        pts, nrm, uvs = [], [], []
        for j in range(V + 1):
            for i in range(U + 1):
                th, ph = np.pi * j / V, 2 * np.pi * i / U
                n = (np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph))
                pts.append(tuple(np.asarray(center) + radius * np.asarray(n)))
                nrm.append(n)
                uvs.append((i / U, j / V))
        faces = []
        for j in range(V):
            for i in range(U):
                a = j * (U + 1) + i
                b = a + U + 1
                faces += [a, b, a + 1, a + 1, b, b + 1]
        m = mdl.CreateMesh()
        m.SetName(name)
        m.SetVertexPositionBuffer(pts)
        m.SetVertexNormalBuffer(nrm)
        m.SetUVLayerCount(1)
        m.SetVertexUVLayerBuffer(0, uvs)
        m.SetFaceBuffer(faces)
        m.SetMaximumWeightInfluence(1)
        mat = mdl.CreateMaterial()
        mat.SetName("Blender_" + name.split(".")[0])
        m.SetMaterial(mat.Hash())
        m.SetVertexWeightBoneBuffer([bone] * len(pts))
        m.SetVertexWeightValueBuffer([1.0] * len(pts))
        return m

    def edited(self, name, drop_originals=True):
        from nightrunner.cast import castlib
        c = castlib.Cast.load(self.rep["cast"])
        mdl = c.Roots()[0].ChildOfType(castlib.Model)
        bones = [b.Name() for b in mdl.Skeleton().Bones()]
        if drop_originals:            # a new character: the artist deleted the exported meshes
            for m in list(mdl.Meshes()):
                mdl.childNodes.remove(m)
        self.sphere(mdl, "ball.001", bones.index("head"), (0, 1.7, 0))
        self.sphere(mdl, "ball_b", bones.index("head"), (0, 1.8, 0), 0.05)
        self.sphere(mdl, "junk", bones.index("head"), (0, 0, 0), 0.01)
        out = self.d / f"{name}.cast"
        c.save(str(out))
        return out

    def test_assign_merge_hide_and_lods(self):
        from nightrunner.cast.split import split_model_cast
        hair = "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0"
        res = split_model_cast(self.edited("a"), self.report, self.d / "a",
                               assign={"ball": hair, "ball_b": hair, "junk": ""},
                               hide=["HEAD.sh2_npc_crane.e0.s1"])
        self.assertEqual(res["problems"], [])
        self.assertEqual(res["unmatched_meshes"], [])
        h = next(m for m in res["meshes"] if m["logical_name"] == "sh_npc_ft_crane_hair_a")
        subs = {s["name"]: s for s in h["submeshes"]}
        self.assertEqual(subs[hair]["new"], 117 * 2)                 # two merged spheres
        self.assertTrue(subs[hair]["replaced"])
        lod = [s for s in h["submeshes"] if s.get("lod_of") == hair]
        self.assertEqual(len(lod), 1)                                  # e1 follows
        self.assertEqual(lod[0]["status"], "applied")
        self.assertTrue(h["check"]["ok"], h["check"])
        head = next(m for m in res["meshes"] if m["logical_name"] == "sh2_npc_crane")
        hid = next(s for s in head["submeshes"] if s["name"] == "HEAD.sh2_npc_crane.e0.s1")
        self.assertTrue(hid["hidden"])
        self.assertEqual(hid["vertices"], 1)
        self.assertTrue(head["check"]["ok"], head["check"])
        self.assertTrue(all(s["status"].startswith("not in") for s in head["submeshes"] if s["name"] != hid["name"]))

    def test_double_sided(self):
        from nightrunner.cast.split import split_model_cast
        hair = "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0"
        res = split_model_cast(self.edited("ds"), self.report, self.d / "ds", assign={"ball": hair},
                               double_sided=["ball"], lods=False, check=True)
        h = next(m for m in res["meshes"] if m["logical_name"] == "sh_npc_ft_crane_hair_a")
        sub = next(s for s in h["submeshes"] if s["name"] == hair)
        self.assertEqual((sub["new"], sub["faces"]), (117 * 2, 192 * 2))
        self.assertTrue(h["check"]["ok"], h["check"])

    def test_no_lods_and_unassigned_objects(self):
        from nightrunner.cast.split import SplitError, split_model_cast
        hair = "HEAD_PART_1.sh_npc_ft_crane_hair_a.e0.s0"
        res = split_model_cast(self.edited("b"), self.report, self.d / "b", assign={"ball": hair}, lods=False)
        h = next(m for m in res["meshes"] if m["logical_name"] == "sh_npc_ft_crane_hair_a")
        self.assertFalse(any(s.get("lod_of") for s in h["submeshes"]))
        self.assertEqual(sorted(res["unmatched_meshes"]), ["ball_b", "junk"])
        with self.assertRaises(SplitError):
            split_model_cast(self.edited("c"), self.report, self.d / "c", assign={"ball": "NOPE.e0.s0"})
