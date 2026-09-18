"""Cast export tests: every sample exports, re-parses with castlib, and matches the decoded model."""
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.cast import castlib  # noqa: E402
from nightrunner.cast.export import build_cast, cast_node_counts, export_cast, load_cast, quaternion_xyzw, polar_rotation  # noqa: E402
from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.mesh.decode import decode_resource  # noqa: E402
from tests.paths import OUT, SAMPLES  # noqa: E402

SAMPLE_PACK = SAMPLES / "meshes.rpack"
SPEC_IDS = {"root": 0x746F6F72, "modl": 0x6C646F6D, "skel": 0x6C656B73, "bone": 0x656E6F62, "mesh": 0x6873656D,
            "matl": 0x6C74616D, "file": 0x656C6966, "meta": 0x6174656D}


def _samples():
    if not SAMPLE_PACK.exists():
        raise unittest.SkipTest(f"{SAMPLE_PACK} missing (run tools/make_samples_mesh.py)")
    return Pack.open(SAMPLE_PACK)


class CastLibTests(unittest.TestCase):
    def test_identifiers_per_spec(self):
        for name, ident in SPEC_IDS.items():
            self.assertEqual(ident.to_bytes(4, "little").decode("ascii"), name)
            self.assertIn(ident, castlib.typeSwitcher)
            self.assertEqual(castlib.typeSwitcher[ident]().identifier, ident)

    def test_quaternion_helpers(self):
        r = np.eye(3)
        self.assertEqual(quaternion_xyzw(r), [0.0, 0.0, 0.0, 1.0])
        # 90° about Z
        rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        q = quaternion_xyzw(rz)
        np.testing.assert_allclose(q, [0, 0, np.sqrt(0.5), np.sqrt(0.5)], atol=1e-9)
        # polar rotation strips scale
        np.testing.assert_allclose(polar_rotation(rz * 2.5), rz, atol=1e-12)


class ExportTests(unittest.TestCase):
    def test_export_all_samples(self):
        out_dir = OUT / "cast"
        out_dir.mkdir(parents=True, exist_ok=True)
        with _samples() as pk:
            for r in pk.resources_of_type(0x10):
                m = decode_resource(r)
                path = out_dir / f"{r.index:03d}.cast"
                rep = export_cast(m, path)
                self.assertTrue(path.exists())
                c = load_cast(path)
                counts = cast_node_counts(c)
                self.assertEqual(counts, {k: v for k, v in rep.nodes.items() if v}, r.name)
                self.assertEqual(counts["bone"], len(m.entities))
                self.assertEqual(counts.get("mesh", 0), sum(1 for e in m.geometry_entries for s in e.submeshes
                                                            if e.vertices is not None and s.index_count))
                self.assertEqual(counts.get("matl", 0), len({s.material_slot for e in m.geometry_entries for s in e.submeshes}))
                root = c.Roots()[0]
                meta = root.ChildOfType(castlib.Metadata)
                self.assertEqual(meta.UpAxis(), "y")
                mdl = root.ChildOfType(castlib.Model)
                self.assertEqual(mdl.Name(), r.name)
                bones = mdl.Skeleton().Bones()
                for i, b in enumerate(bones):
                    self.assertEqual(b.Name(), m.entities[i].name_str)
                    self.assertEqual(b.ParentIndex(), m.entities[i].parent)
                    self.assertEqual(b.properties["bp_entity"].values[0], i)
                for mesh in mdl.Meshes():
                    entry = mesh.properties["bp_entry"].values[0]
                    sub = mesh.properties["bp_submesh"].values[0]
                    e = m.geometry_entries[entry]
                    s = e.submeshes[sub]
                    self.assertEqual(mesh.Name(), f"{r.name}.e{entry}.s{sub}")
                    self.assertEqual(mesh.properties["bp_format"].values[0], e.format)
                    ids = np.array(mesh.properties["bp_vertex_id"].values)
                    used = np.unique(s.indices)
                    self.assertTrue(np.array_equal(ids, used))
                    self.assertEqual(mesh.VertexCount(), len(used))
                    self.assertEqual(mesh.FaceCount(), s.index_count // 3)
                    vp = np.array(mesh.VertexPositionBuffer(), dtype=np.float32).reshape(-1, 3)
                    pos = e.vertices.positions[used]
                    fin = np.isfinite(pos)
                    self.assertTrue(np.array_equal(vp[fin], pos[fin]))
                    faces = np.array(mesh.FaceBuffer())
                    self.assertTrue(np.array_equal(used[faces], s.indices.astype(np.uint32)))
                    self.assertEqual(mesh.Material().Name(), m.material_name(s.material_slot))
                    self.assertEqual(mesh.UVLayerCount(), 2 if e.vertices.uv1 is not None else 1)
                    if e.vertices.skinned:
                        self.assertEqual(mesh.MaximumWeightInfluence(), 4)
                        wb = np.array(mesh.VertexWeightBoneBuffer()).reshape(-1, 4)
                        wv = np.array(mesh.VertexWeightValueBuffer()).reshape(-1, 4)
                        self.assertEqual(len(wb), len(used))
                        self.assertTrue((wb < len(m.entities)).all())
                        np.testing.assert_allclose(wv.sum(axis=1), 1.0, atol=1e-5)
                        joints = e.vertices.joints[used]
                        active = wv > 0
                        self.assertTrue(np.array_equal(wb[active], s.palette[joints[active]]))
                    else:
                        self.assertIsNone(mesh.VertexWeightBoneBuffer())
                    sign = np.array(mesh.properties["bp_tangent_sign"].values)
                    self.assertTrue(set(sign.tolist()) <= {1.0, -1.0})

    def test_build_cast_in_memory(self):
        with _samples() as pk:
            idx = pk.find("sh_npc_ft_crane_hair_a", type_id=0x10, fold=False)
            if not idx:
                raise unittest.SkipTest("hair sample missing")
            m = decode_resource(pk.resource(idx[0]))
            cast, rep = build_cast(m)
            self.assertEqual(rep.nodes["mesh"], 2)
            names = [x.name for x in rep.meshes]
            self.assertEqual(names, ["sh_npc_ft_crane_hair_a.e0.s0", "sh_npc_ft_crane_hair_a.e1.s0"])
            self.assertLess(rep.bone_frame_residual, 1e-3)


if __name__ == "__main__":
    unittest.main()
