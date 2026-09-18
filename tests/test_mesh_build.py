"""Phase-2 mesh build (nightrunner.mesh.rebuild / MeshCodec.build):
(a) export → import → build with no edits is byte-identical for every sample,
(b) one moved vertex changes only its position bytes,
(c) one deleted triangle re-lays out the buffers per the census rule and the rebuilt pack validates,
(d) a Blender-like Cast (no bp_* properties, no tangents, `.001` names) still builds,
(e) refusals carry the documented messages,
plus palette growth, new material within capacity, added submesh, identity rename, format-8 hole policy.
Runs on out/samples/meshes.rpx (skips without it); temp files under tests.synth.tmpdir."""
import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.cast import castlib  # noqa: E402
from nightrunner.cast.import_ import read_cast, resolve_import, load_sidecar  # noqa: E402
from nightrunner.classreader.fixups import Fixups  # noqa: E402
from nightrunner.classreader.image import embedded_mesh_name  # noqa: E402
from nightrunner.codecs import BuildContext  # noqa: E402
from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.container.validate import validate  # noqa: E402
from nightrunner.errors import BuildError, UnsupportedError  # noqa: E402
from nightrunner.mesh import vertex as V  # noqa: E402
from nightrunner.mesh.codec import MeshCodec, rebuild_from_files  # noqa: E402
from nightrunner.mesh.decode import decode_parts, decode_resource  # noqa: E402
from nightrunner.mesh.encode import plan_new_layout  # noqa: E402
from nightrunner.mesh.rebuild import make_plan  # noqa: E402
from tests.paths import SAMPLES  # noqa: E402
from tests.synth import tmpdir  # noqa: E402
from tests.test_mesh_import import strip_blender_like  # noqa: E402

SAMPLE_RPX = SAMPLES / "meshes.rpx"
SAMPLE_PACK = SAMPLES / "meshes.rpack"
PANTS = "000000_npc_b_man_pants_b_holster_bag_c"
BOX = "000003_dummybox_025m"
HAIR = "000006_sh_npc_ft_crane_hair_a"
AIDEN = "000004_sh2_npc_aiden_beast"
CRANE = "000005_sh2_npc_crane"
PART_FILES = {0x10: "image.bin", 0x11: "fixups.bin", 0x12: "skin.bin", 0xF0: "vertex.bin", 0xF1: "index.bin", 0xF3: "cloth.bin"}


def _dir(name: str) -> Path:
    d = SAMPLE_RPX / "mesh" / name
    if not (d / "model.cast").exists():
        raise unittest.SkipTest(f"{d} missing")
    return d


def _parts(d: Path) -> dict[int, bytes]:
    return {t: (d / n).read_bytes() for t, n in PART_FILES.items() if (d / n).exists()}


def _cast(d: Path):
    c = castlib.Cast.load(str(d / "model.cast"))
    return c, c.Roots()[0].ChildOfType(castlib.Model)


def _name(d: Path) -> bytes:
    side = json.loads((d / "mesh.json").read_text(encoding="utf-8"))
    return side["name"].encode("utf-8", "surrogateescape")


def _rebuild(d: Path, cast, tmp: Path, name: bytes | None = None, **kw):
    p = tmp / "edited.cast"
    cast.save(str(p))
    return rebuild_from_files(p, d / "mesh.json", _parts(d), name or _name(d), **kw)


def bp(*argv) -> tuple[int, str]:
    from nightrunner.cli import main
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = main([str(a) for a in argv])
    return rc, out.getvalue()


class UneditedRoundTripTests(unittest.TestCase):
    def test_all_samples_byte_identical(self):
        if not SAMPLE_RPX.exists():
            raise unittest.SkipTest("meshes.rpx missing")
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        n = 0
        for entry in spec["resources"]:
            d = SAMPLE_RPX / entry["dir"]
            parts = _parts(d)
            res = rebuild_from_files(d / "model.cast", d / "mesh.json", parts, bytes.fromhex(entry["name_hex"]))
            self.assertEqual(res.image, parts[0x10], entry["name"] + " image")
            self.assertEqual(res.fixups, parts[0x11], entry["name"] + " fixups")
            self.assertEqual(res.vertex, parts.get(0xF0), entry["name"] + " vertex")
            self.assertEqual(res.index, parts.get(0xF1), entry["name"] + " index")
            self.assertEqual(res.report["warnings"], [], entry["name"])
            self.assertFalse(res.report["layout_changed"])
            self.assertTrue(all(e["path"] in ("in-place", "untouched") for e in res.report["entries"]))
            n += 1
        self.assertEqual(n, 17)

    def test_codec_build_entry_point(self):
        if not SAMPLE_RPX.exists():
            raise unittest.SkipTest("meshes.rpx missing")
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        entry = next(r for r in spec["resources"] if r["dir"].endswith(HAIR))
        ctx = BuildContext(SAMPLE_RPX, spec, {})
        out = MeshCodec().build(ctx, entry, SAMPLE_RPX / entry["dir"])
        self.assertEqual(sorted(out), [k for k, p in enumerate(entry["parts"]) if int(p["type"], 0) in (0x10, 0x11, 0xF0, 0xF1)])
        for k, src in out.items():
            self.assertEqual(bytes(src.data), (SAMPLE_RPX / entry["parts"][k]["raw"]).read_bytes())
        self.assertEqual(ctx.warnings, [])


class EditTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tmpdir("mb_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_move_one_vertex(self):
        d = _dir(PANTS)
        parts = _parts(d)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        vp = list(mesh.properties["vp"].values)
        vp[0] += 0.01
        mesh.properties["vp"].values = vp
        vid = int(mesh.properties["bp_vertex_id"].values[0])
        res = _rebuild(d, c, self.tmp)
        self.assertEqual(res.image, parts[0x10])
        self.assertEqual(res.fixups, parts[0x11])
        self.assertEqual(res.index, parts[0xF1])
        stride = V.STRIDES[6]
        diff = [i for i in range(len(parts[0xF0])) if parts[0xF0][i] != res.vertex[i]]
        self.assertTrue(diff)
        self.assertTrue(all(vid * stride <= i < vid * stride + 12 for i in diff), diff)
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        m1 = decode_parts("x", parts[0x10], parts[0x11], vertex=parts[0xF0], index=parts[0xF1])
        np.testing.assert_allclose(m2.geometry_entries[0].vertices.positions[vid],
                                   m1.geometry_entries[0].vertices.positions[vid] + np.array([0.01, 0, 0], dtype=np.float32), atol=1e-6)
        rep = res.report["entries"][0]["submeshes"][0]
        self.assertEqual((rep["moved"], rep["edited"], rep["new_vertices"]), (1, 1, 0))
        self.assertEqual(res.report["entries"][0]["path"], "in-place")
        # skinned entity bounds are the exact AABB: a vertex moved inside the box leaves the image untouched
        self.assertEqual(res.report["bounds_rewritten"], [0])

    def test_delete_triangle_relayout_and_pack_build(self):
        d = _dir(PANTS)
        parts = _parts(d)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        f = list(mesh.properties["f"].values)
        mesh.properties["f"].values = f[3:]
        res = _rebuild(d, c, self.tmp)
        rep = res.report
        self.assertTrue(rep["layout_changed"])
        self.assertEqual(rep["entries"][0]["index_count"], [3840, 3837])
        self.assertEqual(rep["entries"][0]["vertex_count"], [993, 993])
        self.assertEqual(res.vertex, parts[0xF0])              # window untouched
        self.assertEqual(len(res.index) % 16, 0)               # census: index part padded to 16 in bit-12 packs
        self.assertEqual(len(res.index), V.align_to(3837 * 2, 16))
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        self.assertEqual(m2.warnings, [])
        e = m2.geometry_entries[0]
        side = load_sidecar(d / "mesh.json")
        c0, c1 = [s["index_count"] for s in side["geometry_entries"][0]["submeshes"]]
        self.assertEqual([s.index_count for s in e.submeshes], [c0 - 3, c1])
        self.assertEqual([s.index_base for s in e.submeshes], [0, (c0 - 3) * 2])
        ids = np.asarray(mesh.properties["bp_vertex_id"].values, dtype=np.int64)
        self.assertEqual(e.submeshes[0].indices.tolist(), ids[np.asarray(f[3:], dtype=np.int64)].tolist())
        # the whole tree builds into a valid pack with the census layout rule
        tree = self.tmp / "m.rpx"
        shutil.copytree(SAMPLE_RPX, tree)
        c.save(str(tree / "mesh" / PANTS / "model.cast"))
        rc, out = bp("build", tree, self.tmp / "m.rpack", "--layout", "auto")
        self.assertEqual(rc, 0, out)
        j = json.loads(out.split("\nvalidation")[0])
        self.assertEqual([r["name"] for r in j["regenerated"]], [" npc_b_man_pants_b_holster_bag_c"])
        rc, out = bp("validate", self.tmp / "m.rpack")
        self.assertEqual(rc, 0, out)
        with Pack.open(self.tmp / "m.rpack") as pk:
            self.assertTrue(validate(pk).ok)
            r = pk.resource(0)
            m3 = decode_resource(r)
            self.assertEqual(m3.geometry_entries[0].index_count, 3837)
            self.assertEqual(bytes(r.read_part_by_type(0xF1)), res.index)
            # phase-1 re-encode of the rebuilt pack reproduces it (layout rule consistency)
            for k, i in enumerate(r.part_indices):
                produced = MeshCodec().roundtrip(r)
                if k in produced:
                    self.assertEqual(produced[k], bytes(pk.read_part(i)))

    def test_blender_like_cast_rebuilds(self):
        d = _dir(HAIR)          # two LOD entries
        parts = _parts(d)
        c, mdl = _cast(d)
        strip_blender_like(mdl)
        res = _rebuild(d, c, self.tmp)
        rep = res.report
        self.assertEqual([e["path"] for e in rep["entries"]], ["rebuild", "rebuild"])
        self.assertEqual([e["vertex_count"] for e in rep["entries"]], [[23291, 23291], [9821, 9821]])
        self.assertEqual([e["index_count"] for e in rep["entries"]], [[45882, 45882], [13761, 13761]])
        self.assertTrue(any("recovered by exact position match" in w for w in rep["warnings"]))
        subs = [s for e in rep["entries"] for s in e["submeshes"]]
        self.assertEqual([s["new_vertices"] for s in subs], [0, 0])
        self.assertEqual([s["matched_by"] for s in subs], ["name", "name"])
        self.assertEqual(res.index, parts[0xF1])          # compaction order == window order for this mesh
        self.assertEqual(len(res.vertex), len(parts[0xF0]))
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        self.assertEqual(m2.warnings, [])
        m1 = decode_parts("x", parts[0x10], parts[0x11], vertex=parts[0xF0], index=parts[0xF1])
        for a, b in zip(m1.geometry_entries, m2.geometry_entries):
            np.testing.assert_array_equal(a.vertices.positions, b.vertices.positions)
            np.testing.assert_array_equal(a.vertices.raw["weights"], b.vertices.raw["weights"])
            np.testing.assert_array_equal(a.vertices.raw["joints"], b.vertices.raw["joints"])
            np.testing.assert_array_equal(a.vertices.raw["raw_tail"], b.vertices.raw["raw_tail"])
            # frames re-quantised from UV-derived tangents: normals preserved closely
            self.assertGreater(float(np.mean(np.sum(a.vertices.normals * b.vertices.normals, axis=1) > 0.999)), 0.99)

    def test_added_vertex_uses_hole_policy_and_relayouts(self):
        d = _dir(BOX)                       # format 0 (raw_06 hole), static
        parts = _parts(d)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        n = mesh.VertexCount()
        for key, cols in (("vp", 3), ("vn", 3), ("vt", 3), ("u0", 2)):
            vals = list(mesh.properties[key].values)
            mesh.properties[key].values = vals + vals[:cols]
        mesh.properties["bp_vertex_id"].values = list(mesh.properties["bp_vertex_id"].values) + [0]   # duplicate id ⇒ rebuild path
        mesh.properties["bp_tangent_sign"].values = list(mesh.properties["bp_tangent_sign"].values) + [1.0]
        f = list(mesh.properties["f"].values)
        mesh.properties["f"].values = f + [n, 1, 2]
        del mesh.properties["bp_vertex_id"]      # no ids at all → recovery by position; the copy of vertex 0 recovers id 0
        res = _rebuild(d, c, self.tmp)
        rep = res.report["entries"][0]
        self.assertEqual(rep["path"], "rebuild")
        self.assertEqual(rep["vertex_count"], [24, 25])
        self.assertEqual(rep["index_count"], [36, 39])
        self.assertTrue(res.report["layout_changed"])
        self.assertEqual(len(res.vertex), V.align_to(25 * 16, 160))
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        self.assertEqual(m2.warnings, [])
        v = m2.geometry_entries[0].vertices
        self.assertTrue((v.raw["raw_06"] == 0x3C00).all())
        np.testing.assert_array_equal(v.positions[24], v.positions[0])
        self.assertEqual(m2.entities[0].bounds_half.tolist(), [0.125, 0.125, 0.125])   # grow-only: unchanged box

    def test_format8_new_vertex_gets_zero_ext(self):
        d = _dir(AIDEN)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        n = mesh.VertexCount()
        for key, cols in (("vp", 3), ("vn", 3), ("vt", 3), ("u0", 2), ("u1", 2), ("wv", 4), ("wb", 4)):
            vals = list(mesh.properties[key].values)
            mesh.properties[key].values = vals + vals[:cols]
        vp = list(mesh.properties["vp"].values); vp[-3] += 0.001; mesh.properties["vp"].values = vp   # genuinely new position
        mesh.properties["bp_tangent_sign"].values = list(mesh.properties["bp_tangent_sign"].values) + [1.0]
        del mesh.properties["bp_vertex_id"]
        f = list(mesh.properties["f"].values)
        mesh.properties["f"].values = f + [n, f[1], f[2]]
        res = _rebuild(d, c, self.tmp)
        self.assertTrue(any("format 8" in w and "all zero" in w for w in res.report["warnings"]), res.report["warnings"])
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        e = m2.geometry_entries[0]
        self.assertEqual(e.vertex_count, 15377)
        k = n                                          # submesh 0 is first in the rebuilt window; its new vertex is last
        np.testing.assert_allclose(e.vertices.positions[k], vp[-3:], atol=1e-6)
        self.assertTrue((e.vertices.raw["raw_ext"][k] == 0).all())
        self.assertEqual(int(e.vertices.raw["raw_tail"][k]), 0)
        self.assertEqual(int(e.vertices.weight_sums()[k]), 255)
        self.assertEqual(res.report["entries"][0]["submeshes"][0]["new_vertices"], 1)

    def test_new_material_within_capacity(self):
        d = _dir(BOX)                       # count 2, capacity 3
        parts = _parts(d)
        c, mdl = _cast(d)
        mdl.Meshes()[0].Material().SetName("my_box.mat")
        res = _rebuild(d, c, self.tmp)
        self.assertEqual(res.report["materials_added"], [{"name": "my_box.mat", "slot": 2}])
        m2 = decode_parts("dummybox_025m", res.image, res.fixups, vertex=res.vertex, index=res.index)
        self.assertEqual([m.name_str for m in m2.materials], ["DUMMYBOX.MAT", "auto_shadow_caster.mat", "my_box.mat"])
        self.assertEqual(m2.materials[2].name_tag, 0x2100)
        self.assertEqual(m2.geometry_entries[0].submeshes[0].material_slot, 2)
        self.assertEqual(m2.material_capacity, 3)
        self.assertEqual(m2.warnings, [])
        self.assertEqual(res.vertex, parts[0xF0])
        self.assertEqual(res.index, parts[0xF1])
        self.assertEqual(len(res.image) % 16, 0)
        # {len, cap} prefix in the shipped shape
        t = m2.image.pointer(m2.materials[2].offset + 8).target
        self.assertEqual(m2.image.unpack("<II", t - 8), (10, 10))

    def test_new_material_refused_when_full(self):
        d = _dir(PANTS)                     # count 3 == capacity 3
        c, mdl = _cast(d)
        mdl.Meshes()[1].Material().SetName("my_new_material.mat")
        with self.assertRaises(BuildError) as cm:
            _rebuild(d, c, self.tmp)
        self.assertIn("count 3 == capacity 3", str(cm.exception))

    def test_palette_growth(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        side = load_sidecar(d / "mesh.json")
        mesh = mdl.Meshes()[0]
        pal = side["geometry_entries"][0]["submeshes"][0]["palette"]
        newbone = next(i for i in range(len(side["entities"])) if i not in pal)
        wb = list(mesh.properties["wb"].values)
        wv = list(mesh.properties["wv"].values)
        wb[0], wv[0:4] = newbone, [1.0, 0.0, 0.0, 0.0]
        mesh.properties["wb"].values, mesh.properties["wv"].values = wb, wv
        vid = int(mesh.properties["bp_vertex_id"].values[0])
        res = _rebuild(d, c, self.tmp)
        self.assertEqual(res.report["arrays_appended"], 1)
        m2 = decode_parts("x", res.image, res.fixups, vertex=res.vertex, index=res.index)
        s0 = m2.geometry_entries[0].submeshes[0]
        self.assertEqual(s0.palette.tolist(), pal + [newbone])
        self.assertEqual(m2.geometry_entries[0].vertices.raw["joints"][vid][0], len(pal))
        self.assertEqual(m2.geometry_entries[0].vertices.raw["weights"][vid].tolist(), [255, 0, 0, 0])
        self.assertEqual(m2.warnings, [])
        fx = Fixups.parse(res.fixups)
        self.assertEqual(len(fx.slots), len(m2.fixups.slots))

    def test_palette_overflow_refused(self):
        d = _dir(CRANE)                     # 308 entities
        c, mdl = _cast(d)
        mesh = max(mdl.Meshes(), key=lambda m: m.VertexCount())
        n = mesh.VertexCount()
        mesh.properties["wb"].values = [(i % 300) if k == 0 else 0 for i in range(n) for k in range(4)]
        mesh.properties["wv"].values = [1.0 if k == 0 else 0.0 for i in range(n) for k in range(4)]
        with self.assertRaises(BuildError) as cm:
            _rebuild(d, c, self.tmp)
        self.assertIn("bones in one submesh", str(cm.exception))
        self.assertIn("256", str(cm.exception))

    def test_bone_change_refused_unless_ignored(self):
        d = _dir(PANTS)
        parts = _parts(d)
        c, mdl = _cast(d)
        b = mdl.Skeleton().Bones()[5]
        lp = list(b.LocalPosition())
        lp[0] += 0.05
        b.SetLocalPosition(lp)
        with self.assertRaises(UnsupportedError) as cm:
            _rebuild(d, c, self.tmp)
        self.assertIn("bone transforms differ", str(cm.exception))
        self.assertIn("'l_calf'", str(cm.exception))
        res = _rebuild(d, c, self.tmp, ignore_bone_changes=True)
        self.assertEqual(res.image, parts[0x10])
        self.assertTrue(any("native skeleton kept" in w for w in res.report["warnings"]))

    def test_lod_entry_removal_refused(self):
        d = _dir(HAIR)
        c, mdl = _cast(d)
        mdl.childNodes.remove(mdl.Meshes()[1])
        with self.assertRaises(UnsupportedError) as cm:
            _rebuild(d, c, self.tmp)
        self.assertIn("entry 1", str(cm.exception))
        self.assertIn("removing a geometry entry is not supported", str(cm.exception))

    def test_submesh_gap_refused(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        mdl.Meshes()[1].properties["bp_submesh"].values = [5]
        with self.assertRaises(BuildError) as cm:
            _rebuild(d, c, self.tmp)
        self.assertIn("[0, 5]", str(cm.exception))

    def test_added_submesh(self):
        d = _dir(BOX)
        parts = _parts(d)
        c, mdl = _cast(d)
        m0 = mdl.Meshes()[0]
        m1 = mdl.CreateMesh()
        for k, p in m0.properties.items():
            m1.CreateProperty(k, p.type.identifier).values = list(p.values)
        m1.SetName("dummybox_025m.e0.s1")
        m1.properties["bp_submesh"].values = [1]
        m1.SetMaterial(mdl.Materials()[0].Hash())
        res = _rebuild(d, c, self.tmp)
        self.assertEqual(res.report["records_relocated"], 1)
        m2 = decode_parts("dummybox_025m", res.image, res.fixups, vertex=res.vertex, index=res.index)
        self.assertEqual(m2.warnings, [])
        e = m2.geometry_entries[0]
        self.assertEqual(e.submesh_count, 2)
        self.assertEqual([(s.index_count, s.material_slot) for s in e.submeshes], [(36, 0), (36, 0)])
        self.assertEqual(e.vertex_count, 24)          # ids present ⇒ in-place window, shared by both submeshes
        self.assertEqual(len(res.index), V.align_to(72 * 2, 16))
        fx = Fixups.parse(res.fixups)
        offs = [r.offset for r in fx.records]
        self.assertEqual(offs, sorted(offs))
        self.assertEqual(len(fx.records), len(Fixups.parse(parts[0x11]).records))
        rec7 = [r for r in fx.records if r.class_id == 7]
        self.assertEqual(len(rec7), 1)
        self.assertEqual(rec7[0].count, 2)
        self.assertEqual(rec7[0].offset, e.submeshes[0].palette_desc_offset)

    def test_identity_rename_in_codec(self):
        d = _dir(BOX)
        parts = _parts(d)
        c, _ = _cast(d)
        res = _rebuild(d, c, self.tmp, name=b"my_box")
        self.assertTrue(res.report["identity_renamed"])
        self.assertEqual(embedded_mesh_name(res.image, res.fixups), b"my_box.msh")
        self.assertEqual(res.vertex, parts[0xF0])
        self.assertEqual(res.index, parts[0xF1])
        self.assertEqual(res.image[8:len(parts[0x10]) - 14], parts[0x10][8:len(parts[0x10]) - 14])

    def test_cli_import_and_diff(self):
        d = _dir(PANTS)
        c, mdl = _cast(d)
        mesh = mdl.Meshes()[0]
        f = list(mesh.properties["f"].values)
        mesh.properties["f"].values = f[3:]
        p = self.tmp / "e.cast"
        c.save(str(p))
        rc, out = bp("mesh", "import", p, d / "mesh.json", "--out-dir", self.tmp / "imp")
        self.assertEqual(rc, 0, out)
        rep = json.loads(out)
        self.assertEqual(rep["identical"], {"image.bin": False, "fixups.bin": True, "vertex.bin": True, "index.bin": False})
        self.assertTrue((self.tmp / "imp" / "import_report.json").exists())
        self.assertTrue((self.tmp / "imp" / "skin.bin").exists())
        tree = self.tmp / "res"
        shutil.copytree(d, tree)
        c.save(str(tree / "model.cast"))
        rc, out = bp("mesh", "diff", tree)
        self.assertEqual(rc, 0, out)
        self.assertIn("indices 3840 -> 3837", out)
        self.assertIn("in-place", out)
        rc, out = bp("mesh", "diff", tree, "--json")
        self.assertEqual(json.loads(out)["entries"][0]["index_count"], [3840, 3837])


class PlanTests(unittest.TestCase):
    def test_plan_is_noop_for_unedited_cast(self):
        d = _dir(HAIR)
        parts = _parts(d)
        model = decode_parts("sh_npc_ft_crane_hair_a", parts[0x10], parts[0x11], vertex=parts[0xF0], index=parts[0xF1])
        imp = resolve_import(read_cast(d / "model.cast"), load_sidecar(d / "mesh.json"))
        plan = make_plan(model, imp, b"sh_npc_ft_crane_hair_a")
        self.assertFalse(plan.layout_changed)
        self.assertTrue(all(not p.changed for p in plan.entries))
        self.assertIsNone(plan.identity_rename)
        bases, vs, is_ = plan_new_layout(model.geometry_entries)
        self.assertEqual(bases, [(0, 0), (931680, 91764)])


if __name__ == "__main__":
    unittest.main()
