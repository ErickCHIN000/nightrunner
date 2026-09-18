"""Cast import (nightrunner.cast.import_): parsing, matching against the sidecar, Blender-round-trip tolerance,
refusals. Runs on the extracted sample tree out/samples/meshes.rpx (skips without it)."""
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.cast import castlib  # noqa: E402
from nightrunner.cast.import_ import read_cast, resolve_import, load_sidecar, MESH_NAME_RE  # noqa: E402
from nightrunner.errors import BuildError, UnsupportedError  # noqa: E402
from nightrunner.mesh.vertex import tangents_from_uv  # noqa: E402
from tests.paths import SAMPLES  # noqa: E402
from tests.synth import tmpdir  # noqa: E402

SAMPLE_RPX = SAMPLES / "meshes.rpx"
PANTS = "000000_npc_b_man_pants_b_holster_bag_c"
BOX = "000003_dummybox_025m"


def _dir(name: str) -> Path:
    d = SAMPLE_RPX / "mesh" / name
    if not (d / "model.cast").exists():
        raise unittest.SkipTest(f"{d} missing (run tools/make_samples_mesh.py + nr extract)")
    return d


def _cast(d: Path):
    c = castlib.Cast.load(str(d / "model.cast"))
    return c, c.Roots()[0].ChildOfType(castlib.Model)


def strip_blender_like(mdl) -> None:
    """Simulate the official Blender plugin round trip: custom properties gone, no tangents, `.001` suffixes."""
    for mesh in mdl.Meshes():
        for k in [k for k in mesh.properties if k.startswith("bp_")]:
            del mesh.properties[k]
        mesh.properties.pop("vt", None)
        mesh.SetName(mesh.Name() + ".001")
    skel = mdl.Skeleton()
    if skel is not None:
        for b in skel.Bones():
            for k in [k for k in b.properties if k.startswith("bp_")]:
                del b.properties[k]
    for m in mdl.Materials():
        for k in [k for k in m.properties if k.startswith("bp_")]:
            del m.properties[k]


class NameRuleTests(unittest.TestCase):
    def test_mesh_name_pattern(self):
        self.assertEqual(MESH_NAME_RE.search("x.e0.s1").groups(), ("0", "1"))
        self.assertEqual(MESH_NAME_RE.search("a.b.e12.s3.001").groups(), ("12", "3"))
        self.assertIsNone(MESH_NAME_RE.search("a.e1s2"))
        self.assertIsNone(MESH_NAME_RE.search("a.e1.s2.abc"))


class ReadResolveTests(unittest.TestCase):
    def test_every_sample_resolves_by_bp_properties(self):
        if not SAMPLE_RPX.exists():
            raise unittest.SkipTest("meshes.rpx missing")
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        for entry in spec["resources"]:
            d = SAMPLE_RPX / entry["dir"]
            side = load_sidecar(d / "mesh.json")
            scene = read_cast(d / "model.cast")
            self.assertEqual(scene.up_axis, "y")
            self.assertTrue(scene.software.startswith(("nightrunner", "beastpack")))
            self.assertEqual(len(scene.bones), len(side["entities"]))
            self.assertEqual(len(scene.meshes), len(side["cast"]["meshes"]))
            imp = resolve_import(scene, side)
            self.assertEqual(imp.warnings, [], entry["name"])
            self.assertEqual(imp.bone_changes, [], entry["name"])
            self.assertEqual(imp.new_materials, [])
            self.assertEqual(imp.bones_matched_by, "identity")
            for m in imp.meshes:
                self.assertEqual(m.matched_by, "bp")
                self.assertEqual(m.material_matched_by, "bp")
                self.assertIsNotNone(m.vertex_ids)
                self.assertEqual(m.derived, [])
                se = side["geometry_entries"][m.entry]
                self.assertEqual(m.format, se["format"])
                self.assertEqual(se["submeshes"][m.submesh]["material_slot"], m.material_slot)
                self.assertEqual(len(m.faces) * 3, se["submeshes"][m.submesh]["index_count"])
                if m.weights is not None:
                    np.testing.assert_allclose(m.weights.sum(axis=1), 1.0, atol=1e-5)
                    self.assertTrue((m.bones < len(side["entities"])).all())

    def test_blender_like_cast_matches_by_name(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        strip_blender_like(mdl)
        with tmpdir("imp_") as t:
            p = Path(t) / "stripped.cast"
            c.save(str(p))
            scene = read_cast(p)
            imp = resolve_import(scene, load_sidecar(d / "mesh.json"))
        self.assertEqual([m.matched_by for m in imp.meshes], ["name", "name"])
        self.assertEqual([m.material_matched_by for m in imp.meshes], ["exact", "exact"])
        self.assertTrue(all(m.vertex_ids is None for m in imp.meshes))
        self.assertTrue(all("tangents (from UV derivatives)" in m.derived for m in imp.meshes))
        self.assertEqual(imp.bones_matched_by, "identity")
        self.assertEqual(imp.bone_changes, [])

    def test_unmatchable_name_refused(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        strip_blender_like(mdl)
        mdl.Meshes()[0].SetName("Cube")
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            with self.assertRaises(BuildError) as cm:
                resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertIn("'Cube'", str(cm.exception))
        self.assertIn(".e<entry>.s<submesh>", str(cm.exception))

    def test_duplicate_target_refused(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        mdl.Meshes()[1].properties["bp_submesh"].values = [0]
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            with self.assertRaises(BuildError) as cm:
                resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertIn("both map to entry 0 submesh 0", str(cm.exception))

    def test_bone_count_mismatch_refused(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        skel = mdl.Skeleton()
        skel.childNodes.pop()
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            with self.assertRaises(UnsupportedError) as cm:
                resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertIn("85 bones", str(cm.exception))
        self.assertIn("86 entities", str(cm.exception))

    def test_bone_reorder_matched_by_name(self):
        d = _dir(BOX)
        # the box has one bone; use the pants (86) and swap two leaf bones without children
        d = _dir(PANTS)
        c, mdl = _cast(d)
        skel = mdl.Skeleton()
        bones = skel.Bones()
        names = [b.Name() for b in bones]
        # two bones that are nobody's parent
        parents = {b.ParentIndex() for b in bones}
        leaves = [i for i in range(len(bones)) if i not in parents]
        i, j = leaves[0], leaves[1]
        skel.childNodes[i], skel.childNodes[j] = skel.childNodes[j], skel.childNodes[i]
        for b in skel.Bones():
            p = b.ParentIndex()
            b.SetParentIndex(j if p == i else i if p == j else p)
            for k in [k for k in b.properties if k.startswith("bp_")]:
                del b.properties[k]
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            imp = resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertEqual(imp.bones_matched_by, "name")
        self.assertEqual(imp.entity_map[i], j)
        self.assertEqual(imp.entity_map[j], i)
        self.assertEqual(imp.bone_changes, [])
        # weights still point at the same entities
        side = load_sidecar(d / "mesh.json")
        for m in imp.meshes:
            pal = np.asarray(side["geometry_entries"][m.entry]["submeshes"][m.submesh]["palette"])
            active = m.weights > 0
            self.assertTrue(np.isin(m.bones[active], pal).all())

    def test_bone_transform_change_detected(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        b = mdl.Skeleton().Bones()[3]
        lp = list(b.LocalPosition())
        lp[1] += 0.02
        b.SetLocalPosition(lp)
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            imp = resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertEqual(len(imp.bone_changes), 1)
        self.assertIn(b.Name(), imp.bone_changes[0])

    def test_material_new_and_case_fold(self):
        d = _dir(BOX)
        c, mdl = _cast(d)
        mat = mdl.Materials()[0]
        mat.SetName("dummybox.mat")        # sidecar has 'DUMMYBOX.MAT'
        del mat.properties["bp_material_slot"]
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            imp = resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
            self.assertEqual(imp.meshes[0].material_matched_by, "fold")
            self.assertEqual(imp.meshes[0].material_slot, 0)
            self.assertEqual(imp.new_materials, [])
            mat.SetName("brand_new.mat")
            c.save(str(p))
            imp = resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertEqual(imp.meshes[0].material_matched_by, "new")
        self.assertIsNone(imp.meshes[0].material_slot)
        self.assertEqual(imp.new_materials, ["brand_new.mat"])

    def test_weight_refusals(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        n = mesh.VertexCount()
        wv = list(mesh.properties["wv"].values)
        wv[0:4] = [0.0, 0.0, 0.0, 0.0]
        mesh.properties["wv"].values = wv
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            with self.assertRaises(BuildError) as cm:
                resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
            self.assertIn("vertex 0 has no bone weights", str(cm.exception))
            # five influences
            wv5, wb5 = [], []
            wb = list(mesh.properties["wb"].values)
            for i in range(n):
                wv5 += [0.2, 0.2, 0.2, 0.2, 0.2] if i == 0 else list(mesh.properties["wv"].values[i * 4:i * 4 + 4]) + [0.0]
                wb5 += [0, 1, 2, 3, 4] if i == 0 else wb[i * 4:i * 4 + 4] + [0]
            mesh.properties["wv"].values = wv5
            mesh.properties["wb"].values = wb5
            mesh.SetMaximumWeightInfluence(5)
            c.save(str(p))
            with self.assertRaises(BuildError) as cm:
                resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        self.assertIn("5 non-zero weights", str(cm.exception))

    def test_more_than_four_lanes_but_four_active_is_fine(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[1]
        n = mesh.VertexCount()
        wv = list(mesh.properties["wv"].values)
        wb = list(mesh.properties["wb"].values)
        mesh.properties["wv"].values = [x for i in range(n) for x in wv[i * 4:i * 4 + 4] + [0.0, 0.0]]
        mesh.properties["wb"].values = [x for i in range(n) for x in wb[i * 4:i * 4 + 4] + [0, 0]]
        mesh.SetMaximumWeightInfluence(6)
        with tmpdir("imp_") as t:
            p = Path(t) / "x.cast"
            c.save(str(p))
            imp = resolve_import(read_cast(p), load_sidecar(d / "mesh.json"))
        m = imp.meshes[1]
        self.assertEqual(m.weights.shape, (n, 4))
        np.testing.assert_allclose(m.weights.sum(axis=1), 1.0, atol=1e-5)


class TangentDerivationTests(unittest.TestCase):
    def test_uv_tangents_agree_with_stored_frames_on_outfit_mesh(self):
        d = _dir(PANTS)
        scene = read_cast(d / "model.cast")
        for m in scene.meshes:
            faces = m.faces.astype(np.int64)
            t, s = tangents_from_uv(m.positions, m.normals, m.uv0, faces)
            dot = np.abs(np.sum(t * m.tangents, axis=1))
            self.assertGreater(float(np.mean(dot > 0.9)), 0.99, m.name)
            self.assertGreater(float(np.mean(s == m.tangent_sign)), 0.99, m.name)

    def test_degenerate_input(self):
        t, s = tangents_from_uv(np.zeros((3, 3)), np.array([[0, 0, 1]] * 3, dtype=float), np.zeros((3, 2)),
                                np.array([[0, 1, 2]]))
        self.assertEqual(t.shape, (3, 3))
        np.testing.assert_allclose(np.linalg.norm(t, axis=1), 1.0, atol=1e-6)
        self.assertEqual(s.tolist(), [1, 1, 1])


if __name__ == "__main__":
    unittest.main()
