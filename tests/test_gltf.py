"""glTF / GLB (nightrunner/cast/gltf.py): writer validity, lossless per-mesh round trip through the mesh encoder,
single-model GLB split, and the SDB dx11/dx12 switch of the GUI context. Uses out/samples/meshes.rpx."""
from __future__ import annotations

import json
import os
import struct
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tests.synth as S
from tests.paths import SAMPLES

RPX = SAMPLES / "meshes.rpx"
PACK = SAMPLES / "meshes.rpack"
PART_FILES = {0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin",
              0xF3: "cloth.bin"}


def _glb_json(path: Path) -> dict:
    raw = path.read_bytes()
    magic, version, total = struct.unpack_from("<III", raw)
    assert magic == 0x46546C67 and version == 2 and total == len(raw)
    ln, kind = struct.unpack_from("<II", raw, 12)
    assert kind == 0x4E4F534A
    return json.loads(raw[20:20 + ln])


@unittest.skipUnless((RPX / "pack.json").is_file(), "out/samples/meshes.rpx not available")
class PerMeshTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = S.tmpdir("gltf_")
        cls.d = Path(cls._tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _dirs(self):
        return sorted(p for p in (RPX / "mesh").iterdir() if (p / "mesh.json").is_file())

    def test_every_sample_round_trips_byte_identical(self):
        from nightrunner.cast.export import export_cast
        from nightrunner.mesh.codec import rebuild_from_files
        from nightrunner.mesh.decode import decode_parts
        n = 0
        for d in self._dirs():
            parts = {t: (d / f).read_bytes() for t, f in PART_FILES.items() if (d / f).exists()}
            side = json.loads((d / "mesh.json").read_text())
            m = decode_parts(side["name"], parts[0x10], parts[0x11], vertex=parts.get(0xF0), index=parts.get(0xF1),
                             skin=parts.get(0x12), cloth=parts.get(0xF3))
            for ext in ("glb", "gltf"):
                f = self.d / d.name / f"model.{ext}"
                export_cast(m, f)
                r = rebuild_from_files(f, d / "mesh.json", parts, side["name"])
                self.assertEqual(r.image, parts[0x10], (d.name, ext))
                self.assertEqual(r.fixups, parts[0x11], (d.name, ext))
                self.assertEqual(r.vertex, parts.get(0xF0), (d.name, ext))
                self.assertEqual(r.index, parts.get(0xF1), (d.name, ext))
                n += 1
        self.assertGreaterEqual(n, 30)

    def test_document_shape(self):
        from nightrunner.cast.export import export_cast
        from nightrunner.mesh.decode import decode_parts
        d = next(p for p in self._dirs() if p.name.endswith("wn_pistol_b_b"))
        parts = {t: (d / f).read_bytes() for t, f in PART_FILES.items() if (d / f).exists()}
        m = decode_parts("wn_pistol_b_b", parts[0x10], parts[0x11], vertex=parts[0xF0], index=parts[0xF1])
        f = self.d / "pistol.glb"
        export_cast(m, f)
        doc = _glb_json(f)
        self.assertEqual(doc["asset"]["version"], "2.0")
        skin = doc["skins"][0]
        arm = doc["nodes"][skin["skeleton"]]
        self.assertTrue(arm["name"].endswith("_armature"))
        self.assertNotIn(skin["skeleton"], skin["joints"])
        names = [n["name"] for n in doc["nodes"] if "mesh" in n]
        self.assertIn("wn_pistol_b_b.e0.s0", names)
        prim = doc["meshes"][0]["primitives"][0]
        for a in ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0", "_BP_VERTEX_ID"):
            self.assertIn(a, prim["attributes"])
        # the pistol has inf UVs: 0 in the accessor, exact values in extras
        self.assertTrue(any("bp_nonfinite_uv" in (g.get("extras") or {}) for g in doc["meshes"]))
        for a in doc["accessors"]:
            for k in ("min", "max"):
                self.assertTrue(all(np.isfinite(a.get(k, [0]))))
        # the sidecar sits next to a .gltf export through the GUI helper
        from nightrunner.gui.meshdata import export_cast_files
        from nightrunner.container.rp6l import Pack
        if PACK.is_file():
            with Pack.open(PACK) as pk:
                out = export_cast_files(pk, pk.find("wn_pistol_b_b")[0], self.d / "gui" / "wn_pistol_b_b.gltf")
            self.assertTrue(Path(out["sidecar"]).is_file())
            self.assertTrue((self.d / "gui" / "wn_pistol_b_b.bin").is_file())

    def test_blender_like_edit_through_rpx_build(self):
        """A GLB dropped into an extracted mesh folder (model.glb) wins over model.cast and builds."""
        from nightrunner.build import build
        from nightrunner.cast.gltf import cast_to_gltf
        from nightrunner.cast import castlib
        from nightrunner.container.rp6l import Pack
        from nightrunner.extract import extract
        if not PACK.is_file():
            self.skipTest("meshes.rpack not available")
        tree = self.d / "tree.rpx"
        with Pack.open(PACK) as pk:
            extract(pk, tree, progress=False)
        spec = json.loads((tree / "pack.json").read_text())
        res = next(r for r in spec["resources"] if r["name"] == "sh_npc_ft_crane_hair_a")
        rd = tree / res["dir"]
        c = castlib.Cast.load(str(rd / "model.cast"))
        mdl = c.Roots()[0].ChildOfType(castlib.Model)
        for m in mdl.Meshes():                       # what Blender does: custom props gone, float noise
            for k in [k for k in m.properties if k.startswith("bp_")]:
                m.properties.pop(k)
            vp = np.asarray(m.VertexPositionBuffer()).reshape(-1, 3) + 2e-7
            if m.Name().endswith("e0.s0"):
                vp[:4, 1] += 0.02
            m.SetVertexPositionBuffer(vp.tolist())
        cast_to_gltf(c, rd / "model.glb")
        out = self.d / "built.rpack"
        rep = build(tree, out)
        self.assertTrue(rep["validation"]["ok"])
        regen = next(r for r in rep["regenerated"] if r["name"] == "sh_npc_ft_crane_hair_a")
        self.assertIn("model.glb", regen["changed"])
        from nightrunner.mesh.decode import decode_resource
        with Pack.open(PACK) as a, Pack.open(out) as b:
            ma = decode_resource(a.resource(a.find("sh_npc_ft_crane_hair_a")[0]))
            mb = decode_resource(b.resource(b.find("sh_npc_ft_crane_hair_a")[0]))
            pa = ma.geometry_entries[0].vertices.positions
            pb = mb.geometry_entries[0].vertices.positions
            moved = np.linalg.norm(pb - pa, axis=1) > 1e-6
            self.assertEqual(int(moved.sum()), 4)
            self.assertEqual(ma.geometry_entries[1].vertices.raw.tobytes(), mb.geometry_entries[1].vertices.raw.tobytes())


@unittest.skipUnless(PACK.is_file(), "out/samples/meshes.rpack not available")
class SingleModelGlbTests(unittest.TestCase):
    def test_glb_export_and_split(self):
        from nightrunner.cast.gltf import cast_to_gltf, load_scene
        from nightrunner.cast.split import split_model_cast
        from nightrunner.container.rp6l import Pack
        from nightrunner.gui.modelcast import export_model_cast
        from tests.test_model_split import _Catalog, _resolution
        with S.tmpdir("gltf_model_") as d, Pack.open(PACK) as pk:
            d = Path(d)
            cat = _Catalog(pk)
            rep = export_model_cast(SimpleNamespace(catalog=cat, sdb=None), _resolution(cat), d / "exp",
                                    "crane_head", textures=False, formats=("glb", "gltf"))
            self.assertEqual(sorted(rep["files"]), ["glb", "gltf"])
            self.assertIsNone(rep["blender_script"])
            glb = Path(rep["files"]["glb"])
            doc = _glb_json(glb)
            self.assertEqual(len(doc["skins"][0]["joints"]), rep["skeleton"]["merged_bones"])
            # unedited → every mesh unchanged
            res = split_model_cast(glb, d / "exp" / "crane_head.cast.json", d / "plain")
            self.assertTrue(all(m["check"]["unchanged"] for m in res["meshes"]), res)
            # moved vertices + props stripped (as Blender's glTF exporter does by default)
            c = load_scene(glb)
            for m in c.Roots()[0].ChildOfType(__import__("nightrunner.cast.castlib", fromlist=["Model"]).Model).Meshes():
                for k in [k for k in m.properties if k.startswith("bp_")]:
                    m.properties.pop(k)
                if m.Name() == "HEAD_PART_3.sh2_npc_ft_crane_beard_a.e0.s0":
                    vp = np.asarray(m.VertexPositionBuffer()).reshape(-1, 3)
                    vp[:7, 0] += 0.01
                    m.SetVertexPositionBuffer(vp.tolist())
            edited = d / "edited.glb"
            cast_to_gltf(c, edited)
            res = split_model_cast(edited, d / "exp" / "crane_head.cast.json", d / "edit")
            beard = next(m for m in res["meshes"] if m["mesh"] == "sh2_npc_ft_crane_beard_a.msh")
            self.assertEqual(beard["submeshes"][0]["moved"], 7)
            self.assertTrue(beard["check"]["ok"])


class SdbSwitchTests(unittest.TestCase):
    def test_context_switches_api(self):
        try:
            os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
            from PySide6.QtCore import QSettings
            from PySide6.QtWidgets import QApplication
        except ImportError:
            self.skipTest("PySide6 not installed")
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        QApplication.instance() or QApplication([])
        with S.tmpdir("sdbapi_") as d:
            d = Path(d)
            assets = d / "ph_ft" / "work" / "data_platform" / "pc" / "assets"
            assets.mkdir(parents=True)
            ctx = AppContext(QSettings(str(d / "s.ini"), QSettings.IniFormat), game=GameInstall(d), autoload=False)
            got = []
            ctx.sdbChanged.connect(got.append)
            self.assertEqual(ctx.sdb_api(), "dx11")
            self.assertEqual(ctx.sdb.path.name, "runtime_dx11.sdb")
            ctx.set_sdb_api("DX12")
            self.assertEqual((ctx.sdb_api(), ctx.sdb.path.name, got), ("dx12", "runtime_dx12.sdb", ["dx12"]))
            ctx.set_sdb_api("dx12")                       # no-op
            self.assertEqual(got, ["dx12"])
            ctx.close()


if __name__ == "__main__":
    unittest.main()
