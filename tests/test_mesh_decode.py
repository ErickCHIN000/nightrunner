"""Mesh decode / re-encode tests on the sample pack + corpus slices."""
import base64
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.mesh import vertex as V  # noqa: E402
from nightrunner.mesh.codec import MeshCodec  # noqa: E402
from nightrunner.mesh.decode import decode_resource  # noqa: E402
from nightrunner.mesh.encode import encode_index_buffer, encode_vertex_buffer, plan_new_layout  # noqa: E402
from tests.paths import ASSETS, SAMPLES, have_game  # noqa: E402

SAMPLE_PACK = SAMPLES / "meshes.rpack"
SAMPLE_RPX = SAMPLES / "meshes.rpx"


def _samples():
    if not SAMPLE_PACK.exists():
        raise unittest.SkipTest(f"{SAMPLE_PACK} missing (run tools/make_samples_mesh.py)")
    return Pack.open(SAMPLE_PACK)


def _by_name(pk, name):
    idx = pk.find(name, type_id=0x10, fold=False)
    if not idx:
        raise unittest.SkipTest(f"sample {name!r} missing")
    return pk.resource(idx[0])


class VertexMathTests(unittest.TestCase):
    def test_dtypes(self):
        self.assertEqual(V.STRIDES, {0: 16, 3: 32, 6: 40, 8: 80})
        self.assertEqual(V.VERTEX_DTYPES[6].fields["qtan"][1], 0x14)
        self.assertEqual(V.VERTEX_DTYPES[6].fields["uv0"][1], 0x1C)
        self.assertEqual(V.VERTEX_DTYPES[8].fields["raw_ext"][1], 0x28)
        self.assertEqual(V.VERTEX_DTYPES[0].fields["qtan"][1], 0x08)
        self.assertEqual(V.VERTEX_DTYPES[3].fields["uv1"][1], 0x18)

    def test_qtangent_sign_rule(self):
        # largest |component| picks the sign; ties resolve z > y > x > w
        q = np.array([[0, 0, 0, 32767], [0, 0, 0, -32767], [-100, 5, 5, 5], [7, 7, -7, 7], [7, -7, 7, 7], [-7, 7, 7, 7], [7, 7, 7, -7]])
        self.assertEqual(V.qtangent_sign(q).tolist(), [1, -1, -1, -1, 1, 1, 1])

    def test_qtangent_roundtrip_identity(self):
        q = np.array([[0, 0, 0, 32767]], dtype=np.int16)
        t, n, s = V.decode_qtangent(q, 32767.0)
        np.testing.assert_allclose(t, [[1, 0, 0]], atol=1e-6)
        np.testing.assert_allclose(n, [[0, 0, 1]], atol=1e-6)
        self.assertEqual(s.tolist(), [1])
        enc = V.encode_qtangent(t, n, s, 6)
        self.assertEqual(enc.tolist(), [[0, 0, 0, 32767]])
        # forcing the other handedness negates the quad and keeps the frame
        enc2 = V.encode_qtangent(t, n, np.array([-1]), 6)
        self.assertEqual(V.qtangent_sign(enc2).tolist(), [-1])
        t2, n2, _ = V.decode_qtangent(enc2, 32767.0)
        np.testing.assert_allclose(t2, t, atol=1e-4)
        np.testing.assert_allclose(n2, n, atol=1e-4)

    def test_quantize_weights(self):
        w = np.array([[0.5, 0.5, 0, 0], [1, 0, 0, 0], [0.333, 0.333, 0.334, 0], [0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4]])
        q = V.quantize_weights(w)
        self.assertEqual(q.dtype, np.uint8)
        self.assertEqual(q.sum(axis=1).tolist(), [255, 255, 255, 0, 255])
        self.assertEqual(q[1].tolist(), [255, 0, 0, 0])
        self.assertEqual(q[0].tolist(), [128, 127, 0, 0])

    def test_align_to(self):
        self.assertEqual(V.align_to(39720, 160), 39840)
        self.assertEqual(V.align_to(960, 160), 960)
        self.assertEqual(V.align_to(0, 160), 0)


class DecodeTests(unittest.TestCase):
    def test_all_samples_decode_and_reencode(self):
        codec = MeshCodec()
        with _samples() as pk:
            formats = set()
            for r in pk.resources_of_type(0x10):
                m = decode_resource(r)
                self.assertEqual(m.embedded_name, r.name_raw + b".msh")
                self.assertEqual(m.warnings, [], r.name)
                for e in m.geometry_entries:
                    formats.add(e.format)
                    self.assertIsNotNone(e.owner_entity)
                    self.assertIn(e.index, m.entities[e.owner_entity].geometry_entries)
                    for s in e.submeshes:
                        self.assertEqual(s.index_count % 3, 0)
                        self.assertEqual(len(s.indices), s.index_count)
                        if e.vertices is not None and e.vertices.skinned:
                            self.assertGreater(len(s.palette), 0)
                            self.assertTrue((s.palette < len(m.entities)).all())
                produced = codec.reencode(r, m)
                for k, i in enumerate(r.part_indices):
                    if k in produced:
                        self.assertEqual(produced[k], bytes(pk.read_part(i)), f"{r.name} part 0x{pk.part_type(i):02X}")
            self.assertEqual(formats, {0, 3, 6, 8})

    def test_multi_entry_lod(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "sh_npc_ft_crane_hair_a"))
            self.assertEqual(len(m.geometry_entries), 2)
            e0, e1 = m.geometry_entries
            self.assertEqual(e0.owner_entity, e1.owner_entity)
            self.assertEqual(m.entities[e0.owner_entity].geometry_count, 2)
            self.assertEqual((e0.vertex_count, e0.vertex_base, e0.index_base), (23291, 0, 0))
            self.assertEqual((e1.vertex_count, e1.vertex_base, e1.index_base), (9821, 931680, 91764))
            self.assertLess(e1.vertex_count, e0.vertex_count)
            bases, vsize, isize = plan_new_layout(m.geometry_entries)
            self.assertEqual(bases, [(0, 0), (931680, 91764)])
            self.assertEqual((vsize, isize), (m.vertex_buffer_size, m.index_buffer_size))

    def test_skeleton_only(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "sh2_player_tpp_phx_skeleton"))
            self.assertEqual(len(m.entities), 475)
            self.assertEqual(m.geometry_entries, [])
            self.assertIsNone(m.vertex_buffer)
            g = m.entity_globals()
            self.assertEqual(g.shape, (475, 4, 4))
            self.assertTrue(np.isfinite(g).all())

    def test_cloth_and_skin_parts_carried(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "dlc_ft_freak_banshee_clothes_matriarch"))
            self.assertIsNotNone(m.cloth_raw)
            self.assertIsNotNone(m.skin_raw)
            self.assertTrue(m.skinned)

    def test_nonfinite_meshes_decode(self):
        with _samples() as pk:
            for name in ("gas_tank_pistol_anm", "wn_pistol_b_b"):
                m = decode_resource(_by_name(pk, name))
                v = m.geometry_entries[0].vertices
                self.assertFalse(np.isfinite(v.positions).all() and np.isfinite(v.uv0).all(), name)
                self.assertEqual(encode_vertex_buffer(m), bytes(m.vertex_buffer))

    def test_format_fields(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "dlc_ft_safe_zone_cable_e"))
            v = m.geometry_entries[0].vertices
            self.assertEqual(v.format, 0)
            self.assertTrue((v.raw["raw_06"] == 0x3C00).all())
            self.assertIsNone(v.uv1)
            m = decode_resource(_by_name(pk, "sh2_npc_aiden_beast"))
            v = m.geometry_entries[0].vertices
            self.assertEqual(v.format, 8)
            self.assertEqual(v.raw["raw_ext"].shape, (15376, 40))
            self.assertEqual(int(v.weight_sums().min()), 255)
            self.assertEqual(int(v.weight_sums().max()), 255)

    def test_edit_position_changes_only_that_vertex(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "sh2_npc_crane"))
            v = m.geometry_entries[0].vertices
            orig = bytes(m.vertex_buffer)
            keep = v.positions[10].copy()
            v.positions[10] += np.float32(0.25)
            out = encode_vertex_buffer(m)
            self.assertNotEqual(out, orig)
            stride = V.STRIDES[8]
            diff = [i for i in range(len(orig)) if orig[i] != out[i]]
            self.assertTrue(all(10 * stride <= i < 10 * stride + 12 for i in diff))
            # an edited normal re-quantises the qtangent of that vertex only
            v.positions[10] = keep
            self.assertEqual(encode_vertex_buffer(m), orig)
            v.normals[20] = np.array([0, 1, 0], dtype=np.float32)
            v.tangents[20] = np.array([1, 0, 0], dtype=np.float32)
            out = encode_vertex_buffer(m)
            diff = [i for i in range(len(orig)) if orig[i] != out[i]]
            self.assertTrue(diff)
            self.assertTrue(all(20 * stride + 0x14 <= i < 20 * stride + 0x1C for i in diff))
            enc = V.decode_qtangent(np.frombuffer(out, dtype=V.VERTEX_DTYPES[8])["qtan"][20:21], 32767.0)
            np.testing.assert_allclose(enc[1], [[0, 1, 0]], atol=2e-4)
            np.testing.assert_allclose(enc[0], [[1, 0, 0]], atol=2e-4)

    def test_index_buffer_reencode_after_topology_edit(self):
        with _samples() as pk:
            m = decode_resource(_by_name(pk, "dummybox_025m"))
            s = m.geometry_entries[0].submeshes[0]
            idx = s.indices.copy()
            idx[0:3] = idx[3:6]
            s.indices = idx
            out = encode_index_buffer(m)
            self.assertEqual(np.frombuffer(out, dtype="<u2")[:3].tolist(), idx[3:6].tolist())
            self.assertEqual(len(out), m.index_buffer_size)


class ExtractedTreeTests(unittest.TestCase):
    def test_sidecar_matches_decode(self):
        if not SAMPLE_RPX.exists():
            raise unittest.SkipTest("meshes.rpx missing")
        spec = json.load(open(SAMPLE_RPX / "pack.json", encoding="utf-8"))
        with _samples() as pk:
            for entry in spec["resources"]:
                self.assertEqual(entry["editable"]["kind"], "cast", entry["name"])
                d = SAMPLE_RPX / entry["dir"]
                self.assertTrue((d / "model.cast").exists())
                side = json.load(open(d / "mesh.json", encoding="utf-8"))
                m = decode_resource(pk.resource(entry["index"]))
                self.assertEqual(side["embedded_name"], m.embedded_name.decode())
                self.assertEqual(len(side["entities"]), len(m.entities))
                self.assertEqual(len(side["geometry_entries"]), len(m.geometry_entries))
                self.assertEqual(side["materials"]["count"], len(m.materials))
                for se, ge in zip(side["geometry_entries"], m.geometry_entries):
                    if ge.vertices is None:
                        continue
                    raw = base64.b64decode(se["vertex_raw_b64"])
                    self.assertEqual(raw, ge.vertices.raw.tobytes())
                    self.assertEqual(se["vertex_stride"], V.STRIDES[ge.format])
                    self.assertEqual([s["palette"] for s in se["submeshes"]], [s.palette.tolist() for s in ge.submeshes])


@unittest.skipUnless(have_game(), "game not installed")
class CorpusTests(unittest.TestCase):
    def test_roundtrip_slice(self):
        codec = MeshCodec()
        with Pack.open(ASSETS / "common_meshes_pc.rpack") as pk:
            n = 0
            for r in pk.resources_of_type(0x10):
                if n >= 300:
                    break
                n += 1
                produced = codec.roundtrip(r)
                for k, i in enumerate(r.part_indices):
                    if k in produced:
                        self.assertEqual(produced[k], bytes(pk.read_part(i)), f"#{r.index} {r.name}")

    def test_engine_pc_all(self):
        codec = MeshCodec()
        with Pack.open(ASSETS / "engine_pc.rpack") as pk:
            for r in pk.resources_of_type(0x10):
                produced = codec.roundtrip(r)
                for k, i in enumerate(r.part_indices):
                    if k in produced:
                        self.assertEqual(produced[k], bytes(pk.read_part(i)), f"#{r.index} {r.name}")


if __name__ == "__main__":
    unittest.main()
