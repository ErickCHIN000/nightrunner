"""Nightrunner — Meshes tab: meshdata on the real sample meshes, material names, garbage handling, list
population, open_gid, MeshView smoke test, Cast export parity with `nr mesh export`.

Headless (QT_QPA_PLATFORM=offscreen, where the 3D view shows its fallback message; GL-dependent asserts are
skipped unless the view is live); skips without PySide6. Game data: $NIGHTRUNNER_GAME_ROOT, else /tmp/game, else the
real install; skipped when none exists.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import unittest
from pathlib import Path

import numpy as np

import tests.synth as S
from tests import paths

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def game_root() -> Path | None:
    for cand in (os.environ.get("NIGHTRUNNER_GAME_ROOT"), "/tmp/game", str(paths.GAME)):
        if cand and (Path(cand) / "ph_ft" / "work" / "data_platform" / "pc" / "assets").is_dir():
            return Path(cand)
    return None


GAME_ROOT = game_root() if HAVE_QT else None
SAMPLE_PACK = paths.SAMPLES / "meshes.rpack"


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
@unittest.skipUnless(SAMPLE_PACK.is_file(), "out/samples/meshes.rpack missing")
class TestMeshData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.container.rp6l import Pack
        cls.pk = Pack.open(SAMPLE_PACK)

    @classmethod
    def tearDownClass(cls):
        cls.pk.close()

    def index_of(self, name: str) -> int:
        hits = self.pk.find(name, type_id=0x10)
        self.assertTrue(hits, name)
        return hits[0]

    def test_counts_match_decoder_and_cast(self):
        from nightrunner.cast.export import build_cast
        from nightrunner.gui.meshdata import load_mesh_geometry
        from nightrunner.mesh.decode import decode_resource
        for i in range(len(self.pk)):
            with self.subTest(i=i):
                geoms, info = load_mesh_geometry(self.pk, i)
                self.assertIsNone(info["error"])
                model = decode_resource(self.pk.resource(i))
                s = model.summary()
                self.assertEqual(info["vertices"], s["vertices"])
                self.assertEqual(info["triangles"], s["triangles"])
                self.assertEqual(info["materials"], s["materials"])
                self.assertEqual(len(info["entries"]), s["geometry_entries"])
                _, rep = build_cast(model)
                self.assertEqual([(m.entry, m.submesh, m.vertex_count, m.face_count * 3) for m in rep.meshes],
                                 [(g.entry, g.submesh, len(g.positions), len(g.indices)) for g in geoms])
                for g in geoms:
                    e = model.geometry_entries[g.entry]
                    sm = e.submeshes[g.submesh]
                    self.assertEqual(len(g.indices), sm.index_count)
                    self.assertEqual(g.positions.dtype, np.float32)
                    self.assertEqual(g.indices.dtype, np.uint32)
                    self.assertEqual(g.positions.shape[1], 3)
                    self.assertEqual(g.uv.shape, (len(g.positions), 2))
                    self.assertLess(int(g.indices.max()), len(g.positions))
                    np.testing.assert_array_equal(g.positions[g.indices],
                                                  e.vertices.positions[sm.indices.astype(np.int64)])
                    self.assertEqual(g.material, model.material_name(sm.material_slot))
                self.assertEqual(len({g.key for g in geoms}), len(geoms))

    def test_rigged_npc(self):
        from nightrunner.gui.meshdata import load_mesh_geometry
        geoms, info = load_mesh_geometry(self.pk, self.index_of("sh2_npc_crane"))
        self.assertEqual(len(info["bones"]), 308)
        self.assertTrue(info["skinned"])
        self.assertEqual(info["vertices"], 15376)
        self.assertEqual(len(geoms), 7)
        self.assertIsNotNone(info["bounds"])

    def test_lod_filter(self):
        from nightrunner.gui.meshdata import load_mesh_geometry
        i = self.index_of("sh_npc_ft_crane_hair_a")
        all_g, info = load_mesh_geometry(self.pk, i)
        self.assertEqual(info["lod_count"], 2)
        lod0, _ = load_mesh_geometry(self.pk, i, lods=[0])
        lod1, _ = load_mesh_geometry(self.pk, i, lods=[1])
        self.assertEqual(len(lod0) + len(lod1), len(all_g))
        self.assertTrue(lod0 and lod1)
        self.assertTrue(all(g.lod == 0 for g in lod0) and all(g.lod == 1 for g in lod1))
        self.assertGreater(sum(len(g.indices) for g in lod0), sum(len(g.indices) for g in lod1))

    def test_skeleton_only(self):
        from nightrunner.gui.meshdata import load_mesh_geometry
        geoms, info = load_mesh_geometry(self.pk, self.index_of("man_basic_skeleton"))
        self.assertEqual(geoms, [])
        self.assertIsNone(info["error"])
        self.assertEqual(len(info["bones"]), 287)
        self.assertEqual(info["entries"], [])
        self.assertIsNone(info["bounds"])

    def test_mesh_materials(self):
        from nightrunner.gui.meshdata import load_mesh_geometry, mesh_materials
        for i in range(len(self.pk)):
            with self.subTest(i=i):
                self.assertEqual(mesh_materials(self.pk, i), load_mesh_geometry(self.pk, i)[1]["materials"])
        self.assertIn("wn_pistol_b_frame.mat", mesh_materials(self.pk, self.index_of("wn_pistol_b_b")))

    def test_variants_or_raw(self):
        from nightrunner.gui.meshdata import load_mesh_geometry, skin_strings
        _, info = load_mesh_geometry(self.pk, self.index_of("wn_pistol_b_b"))
        self.assertIsNotNone(info["skin_raw"])
        self.assertEqual(len(info["skin_raw"]), info["skin_size"])
        self.assertTrue(info["variants"] is not None or info["variants_error"])
        self.assertIsInstance(skin_strings(info["skin_raw"]), list)

    def test_garbage_is_error_dict(self):
        from nightrunner.container.rp6l import Pack
        from nightrunner.gui.meshdata import load_mesh_geometry, mesh_materials
        with S.tmpdir("guimesh_") as d:
            p = Path(d) / "bad.rpack"
            S.write_pack([S.single(b"bad_mesh", 0x10, 96, seed=3)], p)
            with Pack.open(p) as pk:
                geoms, info = load_mesh_geometry(pk, 0)
                self.assertEqual(geoms, [])
                self.assertTrue(info["error"])
                self.assertEqual(mesh_materials(pk, 0), [])
                geoms, info = load_mesh_geometry(pk, 99)       # out of range → error too
                self.assertTrue(info["error"])

    def test_export_cast_matches_cli(self):
        from nightrunner.gui.meshdata import export_cast_files
        i = self.index_of("wn_pistol_b_b")
        with S.tmpdir("guimesh_") as d:
            d = Path(d)
            export_cast_files(self.pk, i, d / "gui" / "m.cast")
            r = subprocess.run([sys.executable, "-m", "nightrunner", "mesh", "export", str(SAMPLE_PACK), str(i),
                                str(d / "cli" / "m.cast")], cwd=paths.ROOT, capture_output=True, text=True)
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertEqual((d / "gui" / "m.cast").read_bytes(), (d / "cli" / "m.cast").read_bytes())
            self.assertEqual((d / "gui" / "m.mesh.json").read_bytes(), (d / "cli" / "m.mesh.json").read_bytes())


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class TestMeshView(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_add_clear_frame(self):
        from PySide6.QtGui import QImage
        from nightrunner.gui.meshview import MeshView, smooth_normals
        v = MeshView()
        v.resize(320, 240)
        v.show()
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]], np.float32)
        idx = np.array([0, 1, 2, 0, 2, 3], np.uint32)
        uv = np.zeros((4, 2), np.float32)
        tex = QImage(8, 8, QImage.Format_RGBA8888)
        tex.fill(0xFF00FF00)
        v.add_mesh("a", pos, idx, uv=uv, color=(1, 0, 0), texture=tex)
        v.add_mesh("b", pos + 10, idx)
        v.add_mesh("empty", np.zeros((0, 3)), np.zeros(0))
        v.add_mesh("bad", pos, np.array([0, 1, 99], np.uint32))      # out-of-range triangle dropped, no crash
        self.assertEqual(sorted(v.keys()), ["a", "b", "bad", "empty"])
        v.set_visible("b", False)
        self.assertFalse(v.is_visible("b"))
        v.set_visible("missing", True)
        v.remove("empty")
        v.remove("bad")
        v.frame_all()
        lo, hi = v.bounds()
        np.testing.assert_allclose(lo, [0, 0, 0])
        np.testing.assert_allclose(hi, [1, 2, 3])
        np.testing.assert_allclose(v.camera()["target"], [0.5, 1, 1.5])
        v.orbit(10, 5)
        v.pan(3, 4)
        v.zoom(2)
        v.set_wireframe(True)
        v.set_shading(1)
        v.set_texture("a", None)
        v.set_color("a", (0, 0, 1))
        self.assertFalse(v.screenshot().isNull())
        v.clear()
        self.assertEqual(v.keys(), [])
        v.frame_all()
        n = smooth_normals(pos, idx)
        self.assertEqual(n.shape, (4, 3))
        np.testing.assert_allclose(np.linalg.norm(n, axis=1), 1, rtol=1e-5)
        if not v.available:
            self.assertTrue(v.error)
        v.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
@unittest.skipUnless(GAME_ROOT is not None, "no game install (NIGHTRUNNER_GAME_ROOT, /tmp/game or the real install)")
class TestMeshesTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        from nightrunner.gui.tabs.meshes import Tab
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("guimeshes_")
        cls.d = Path(cls._tmp.name)
        settings = QSettings(str(cls.d / "settings.ini"), QSettings.IniFormat)
        cls.ctx = AppContext(settings, GameInstall(GAME_ROOT))
        assert cls.ctx.catalog.wait(600)
        cls.cat = cls.ctx.catalog
        cls.tab = Tab(cls.ctx)
        cls.tab.resize(1200, 800)

    @classmethod
    def tearDownClass(cls):
        cls.tab.shutdown()
        cls.ctx.runner.pool.waitForDone(10000)
        cls.tab.close()
        cls.ctx.close()
        cls._tmp.cleanup()

    def pump(self, cond, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.005)
        return False

    def mesh_gid(self, name: str) -> int:
        hits = self.cat.lookup(name, 0x10)
        if not hits:
            self.skipTest(f"{name} not in the install")
        return hits[0]

    def search(self, text: str) -> None:
        self.tab.search.edit.setText(text)
        self.tab._search_deb.flush()
        self.assertTrue(self.pump(lambda: self.ctx.runner._jobs.get("mesh-search") is None))
        self.app.processEvents()

    def test_list_population(self):
        self.search("")
        total = self.cat.type_counts().get(0x10, 0)
        self.assertGreater(total, 0)
        self.assertEqual(self.tab.model.rowCount(), total)
        self.assertTrue(all(self.cat.type(int(g)) == 0x10 for g in self.tab.model.gids[:500]))
        self.search("pistol")
        names = [self.cat.name(int(g)) for g in self.tab.model.gids]
        self.assertTrue(names and all("pistol" in n.lower() for n in names))
        self.assertTrue(self.tab.search.pack_combo.count() > 1)
        m = self.tab.model
        self.assertTrue(m.data(m.index(0, 0)))
        self.assertTrue(m.data(m.index(0, 3)))
        self.search("")

    def test_open_gid_loads(self):
        g = self.mesh_gid("sh2_npc_crane")
        self.search("zzz_no_such_mesh")
        self.assertEqual(self.tab.model.rowCount(), 0)
        self.tab.open_gid(g)
        self.assertTrue(self.pump(lambda: self.tab._result is not None and self.tab._result["gid"] == g))
        self.assertTrue(self.pump(lambda: self.tab.current_list_gid() == g))
        self.assertEqual(self.tab.search.text(), "")
        res = self.tab._result
        self.assertEqual(len(res["geoms"]), 7)
        self.assertEqual(sorted(self.tab.view.keys()), sorted(x.key for x in res["geoms"]))
        self.assertEqual(self.tab.skel_tree.topLevelItemCount() >= 1, True)
        self.assertIn("sh2_npc_crane", self.tab.overview.toPlainText())
        self.assertEqual(self.tab.mat_tree.topLevelItemCount(), len(res["info"]["materials"]))
        if self.ctx.sdb.available():
            self.assertTrue(any(m["in_sdb"] for m in res["materials"]))
        self.assertEqual(self.tab.parts_tree.topLevelItemCount(), 5)
        # submesh visibility follows the tree
        top = self.tab.sub_tree.topLevelItem(0)
        from PySide6.QtCore import Qt
        ch = top.child(0)
        ch.setCheckState(0, Qt.Unchecked)
        self.assertFalse(self.tab.view.is_visible(ch.data(0, Qt.UserRole)))
        ch.setCheckState(0, Qt.Checked)
        self.assertTrue(self.tab.view.is_visible(ch.data(0, Qt.UserRole)))

    def test_lod_selector(self):
        g = self.mesh_gid("sh_npc_ft_crane_hair_a")
        self.tab.open_gid(g)
        self.assertTrue(self.pump(lambda: self.tab._result is not None and self.tab._result["gid"] == g))
        self.assertEqual(self.tab.lod_combo.count(), 3)
        vis = {x.key: self.tab.view.is_visible(x.key) for x in self.tab._result["geoms"]}
        self.assertEqual(vis, {x.key: x.lod == 0 for x in self.tab._result["geoms"]})
        self.tab.lod_combo.setCurrentIndex(1)
        self.assertTrue(all(self.tab.view.is_visible(x.key) == (x.lod == 1) for x in self.tab._result["geoms"]))
        self.tab.lod_combo.setCurrentIndex(2)
        self.assertTrue(all(self.tab.view.is_visible(x.key) for x in self.tab._result["geoms"]))

    def test_variant_selector(self):
        from nightrunner.gui.tabs.meshes import variant_overrides
        g = self.mesh_gid("wn_pistol_b_b")
        self.tab.open_gid(g)
        self.assertTrue(self.pump(lambda: self.tab._result is not None and self.tab._result["gid"] == g))
        v = self.tab._result["info"]["variants"]
        if not isinstance(v, dict) or not v.get("variants"):
            self.skipTest("variants decoder unavailable")
        texts = [self.tab.variant_combo.itemText(i) for i in range(self.tab.variant_combo.count())]
        self.assertEqual(len(texts), len(v["variants"]) + 1)
        k = texts.index("olive_plastic")
        self.tab.variant_combo.setCurrentIndex(k)
        self.assertIn("olive", self.tab._effective["e0/s0"][0])
        self.assertEqual(variant_overrides(v, k - 1)["e0/s0"], self.tab._effective["e0/s0"])
        names = [m["name"] for m in self.tab._result["materials"]]
        self.assertIn("wn_pistol_b_frame_olive_plastic.mat", names)
        self.tab.variant_combo.setCurrentIndex(0)
        self.assertEqual(self.tab._effective["e0/s0"][0], "wn_pistol_b_frame.mat")
        self.assertEqual(variant_overrides(None, 0), {})
        self.assertEqual(variant_overrides({"error": "x"}, 0), {})

    def test_garbage_and_skeleton(self):
        stress = self.cat.search("stress_mesh_0000", types=(0x10,))
        if len(stress):
            g = int(stress[0])
            self.tab.open_gid(g)
            self.assertTrue(self.pump(lambda: self.tab._result is not None and self.tab._result["gid"] == g))
            self.assertTrue(self.tab._result["info"]["error"])
            self.assertEqual(self.tab.view.keys(), [])
            self.assertIn("ERROR", self.tab.overview.toPlainText())
        g = self.mesh_gid("man_basic_skeleton")
        self.tab.open_gid(g)
        self.assertTrue(self.pump(lambda: self.tab._result is not None and self.tab._result["gid"] == g))
        self.assertEqual(self.tab.view.keys(), [])
        self.assertEqual(self.tab.skel_tree.topLevelItemCount(), 1)

    def test_material_report_and_textures(self):
        from nightrunner.gui.tabs.meshes import load_textures, material_report
        if not self.ctx.sdb.available():
            self.skipTest("no SDB")
        rep = material_report(self.ctx, ["mat_bg_the_beast_layer.mat", "no_such_material.mat"])
        self.assertTrue(rep[0]["in_sdb"])
        self.assertEqual(rep[0]["diffuse"], "ui_bg_the_beast_texture_dif.png")
        self.assertFalse(rep[1]["in_sdb"])
        if rep[0]["diffuse_gids"]:
            # keys are material names; each value is (QImage | None, summary, alpha mode, cutoff, hidden)
            tr = load_textures(self.ctx, 0, ["mat_bg_the_beast_layer.mat", "no_such_material.mat"])
            if tr["error"] is None:
                img, _msg, mode, _cut, hidden = tr["images"]["mat_bg_the_beast_layer.mat"]
                self.assertIsNotNone(img)
                self.assertFalse(hidden)
                self.assertLessEqual(max(img.width(), img.height()), 2048)
                self.assertIsNone(tr["images"]["no_such_material.mat"][0])

    def test_export_checked_cast(self):
        from nightrunner.gui.tabs.meshes import export_targets, run_export
        g = self.mesh_gid("wn_pistol_b_b")
        bad = self.cat.search("stress_mesh_00001", types=(0x10,))
        out = self.d / "export"
        self.tab.model.clear_checks()
        self.tab.model.set_checked([g] + [int(x) for x in bad[:1]], True)
        self.assertTrue(self.tab.export_checked("cast", str(out), overwrite=False))
        self.assertTrue(self.pump(lambda: self.tab._export_cancel is None, 120))
        r = self.tab.last_export
        self.assertEqual(len(r["done"]), 1)
        self.assertEqual(len(r["failed"]), len(bad[:1]))
        cast = Path(r["done"][0])
        self.assertEqual(cast.suffix, ".cast")
        e, idx = self.cat.split(g)
        cli = self.d / "cli" / "x.cast"
        p = subprocess.run([sys.executable, "-m", "nightrunner", "mesh", "export", str(e.path), str(idx), str(cli)],
                           cwd=paths.ROOT, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertEqual(cast.read_bytes(), cli.read_bytes())
        # no silent overwrite: a second run keeps both
        before = cast.read_bytes()
        self.tab.model.set_checked([int(x) for x in bad[:1]], False)
        self.assertTrue(self.tab.export_checked("cast", str(out), overwrite=False))
        self.assertTrue(self.pump(lambda: self.tab._export_cancel is None, 120))
        second = Path(self.tab.last_export["done"][0])
        self.assertNotEqual(second, cast)
        self.assertEqual(cast.read_bytes(), before)
        self.assertTrue(second.with_name(second.stem + ".mesh.json").is_file())
        # raw parts + cancel
        jobs = export_targets(self.cat, [g], self.d / "raw", "raw")
        ev = threading.Event()
        rr = run_export(self.cat, jobs, "raw", False, ev)
        self.assertEqual(len(rr["done"]), 1)
        self.assertEqual(sorted(x.name for x in Path(rr["done"][0]).iterdir()),
                         ["fixups.bin", "image.bin", "index.bin", "skin.bin", "vertex.bin"])
        ev.set()
        rc = run_export(self.cat, jobs, "raw", False, ev)
        self.assertTrue(rc["cancelled"])
        self.assertEqual(rc["skipped"], [g])
        self.tab.model.clear_checks()


if __name__ == "__main__":
    unittest.main()
