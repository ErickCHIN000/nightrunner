"""Dying Light 2 mesh layout (notes/FORMATS/mesh-dl2.md): detection, decode, byte-exact re-encode, GUI data path,
Cast export and the read-only guard.

Fixtures: three tiny shipped DL2 meshes under tests/data/dl2_mesh (≈14 KB, parts named <mesh>.<part type hex>)
and a synthetic skinned DL2 image built here from the documented layout."""
import struct
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.classreader.fixups import Fixups, Record, Slot  # noqa: E402
from nightrunner.classreader.graph import LAYOUT_DL2, LAYOUT_DLTB, MeshGraph, detect_layout  # noqa: E402
from nightrunner.classreader.image import Image  # noqa: E402
from nightrunner.errors import FormatError, UnsupportedError  # noqa: E402
from nightrunner.mesh import variants  # noqa: E402
from nightrunner.mesh.codec import MeshCodec, rebuild_from_files  # noqa: E402
from nightrunner.mesh.decode import decode_parts, decode_resource  # noqa: E402
from nightrunner.mesh.encode import encode_index_buffer, encode_vertex_buffer  # noqa: E402
from nightrunner.mesh.rebuild import refuse_layout  # noqa: E402
from nightrunner.mesh.sidecar import build_sidecar  # noqa: E402
from tests import synth  # noqa: E402
from tests.paths import SAMPLES  # noqa: E402

DATA = ROOT / "tests" / "data" / "dl2_mesh"
PART_TYPES = (0x10, 0x12, 0x11, 0xF0, 0xF1)          # stock logical order (image, skin, fixups, vertex, index)
SAMPLE_NAMES = ("prp_hotel_number_10_a", "alarm_lamp_a", "air_conditioner_wall_dest_a")


def sample_parts(name: str) -> dict[int, bytes]:
    out = {}
    for t in PART_TYPES:
        p = DATA / f"{name}.{t:02x}"
        if not p.exists():
            raise unittest.SkipTest(f"{p} missing")
        out[t] = p.read_bytes()
    return out


def decode_sample(name: str):
    p = sample_parts(name)
    return decode_parts(name, p[0x10], p[0x11], vertex=p[0xF0], index=p[0xF1], skin=p[0x12]), p


def sample_pack() -> bytes:
    """A synthetic RP6L pack holding the three shipped DL2 samples (shipped part shape)."""
    specs = []
    for k, n in enumerate(SAMPLE_NAMES):
        p = sample_parts(n)
        parts = [synth.part(t, p[t], flag_bits=synth.PHYS_BIT8) for t in PART_TYPES]
        specs.append(synth.ResourceSpec(name=b" " + n.encode(), type=0x10, flags=synth.LOGICAL_FLAGS_ONDEMAND_MESH,
                                        parts=parts))
    return synth.build_bytes(specs)


# ---- synthetic skinned DL2 mesh -------------------------------------------------------------------------------

class _Img:
    def __init__(self):
        self.buf = bytearray()
        self.records: list[Record] = []
        self.slots: list[Slot] = []

    def obj(self, class_id: int, count: int, data: bytes, align: int = 8) -> int:
        self.buf += b"\0" * (-len(self.buf) % align)
        off = len(self.buf)
        self.buf += data
        self.records.append(Record(off, (0xB0 << 24) | class_id, count))
        return off

    def string(self, s: bytes) -> int:
        return self.obj(0, 1, s + b"\0")

    def ptr(self, at: int, target: int | None) -> None:
        struct.pack_into("<Q", self.buf, at, 0 if target is None else target + 1)
        self.slots.append(Slot(at, 0))

    def finish(self) -> tuple[bytes, bytes]:
        self.buf += b"\0" * (-len(self.buf) % 8)
        size = len(self.buf)
        fx = Fixups(size, 1, sorted(self.records, key=lambda r: r.offset), sorted(self.slots, key=lambda s: s.offset))
        body = fx.to_bytes()
        fx.trailing = b"\0" * (-len(body) % 16)
        return bytes(self.buf) + b"\0" * (-size % 16), fx.to_bytes()


def _mat34(tx=0.0, ty=0.0, tz=0.0) -> bytes:
    return struct.pack("<12f", 1, 0, 0, tx, 0, 1, 0, ty, 0, 0, 1, tz)


def build_skinned_dl2(fmt: int = 6):
    """(image, fixups, vertex, index): 2 bones + 1 mesh entity, one geometry entry (format 6, 3 vertices, one
    submesh with a 2-entry palette), one material — laid out exactly as shipped DL2 meshes are."""
    im = _Img()
    root = im.obj(3, 1, bytearray(0x68))
    name = im.string(b"synth_dl2.msh")
    ents = im.obj(4, 3, bytearray(3 * 0xD0))
    names = [im.string(n) for n in (b"root_bone", b"child_bone", b"synth_dl2")]
    aux = im.obj(5, 1, bytearray(0x20))
    geo = im.obj(6, 1, bytearray(0x30))
    stream = im.obj(8, 1, bytearray(0x20) + struct.pack("<I", 3) + struct.pack("<H", 0) + b"\0\0")
    pal = im.obj(7, 1, bytearray(0x10) + struct.pack("<2H", 0, 1) + b"\0" * 4)
    mhdr = im.obj(10, 1, bytearray(0x10))
    ment = im.obj(11, 1, bytearray(0x20))
    mname = im.string(b"synth.mat")
    b = im.buf
    im.ptr(root + 0x00, name)
    im.ptr(root + 0x08, ents)
    im.ptr(root + 0x18, mhdr)
    struct.pack_into("<3I", b, root + 0x58, 3, 1, 0)
    for i, (parent, typ, gcount) in enumerate(((-1, 8, 0), (0, 8, 0), (-1, 1, 1))):
        e = ents + i * 0xD0
        b[e:e + 0x30] = _mat34(0.0, float(i), 0.0)
        b[e + 0x30:e + 0x60] = _mat34(0.0, -float(i), 0.0)
        b[e + 0x60:e + 0x78] = struct.pack("<6f", 0, 0.5, 0, 1, 0.5, 0.1)
        im.ptr(e + 0x78, names[i])
        struct.pack_into("<IHhBB", b, e + 0xC0, 0x4700, i, parent, typ, gcount)
        if gcount:
            im.ptr(e + 0x80, aux)
            im.ptr(e + 0x88, geo)
    im.ptr(geo + 0x08, stream)
    im.ptr(geo + 0x10, stream + 0x24)
    im.ptr(geo + 0x18, pal)
    struct.pack_into("<II", b, geo + 0x20, 1, 0x6E5A4632)
    im.ptr(stream + 0x00, stream + 0x20)
    struct.pack_into("<6I", b, stream + 0x08, 0, 3, 0, 1, fmt, 0)
    im.ptr(pal + 0x00, pal + 0x10)
    struct.pack_into("<Q", b, pal + 0x08, 2)
    im.ptr(mhdr + 0x00, ment)
    struct.pack_into("<HH", b, mhdr + 0x08, 1, 1)
    im.ptr(ment + 0x08, mname)
    struct.pack_into("<I", b, ment + 0x18, 4)
    image, fixups = im.finish()

    from nightrunner.mesh.vertex import VERTEX_DTYPES, align_to
    v = np.zeros(3, dtype=VERTEX_DTYPES[fmt])
    v["pos"] = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    v["weights"] = [[255, 0, 0, 0], [128, 127, 0, 0], [0, 255, 0, 0]]
    v["joints"] = [[0, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0]]
    v["qtan"] = [0, 0, 0, 32767]
    v["uv0"] = [[0, 0], [1, 0], [0, 1]]
    v["uv1"] = v["uv0"]
    v["raw_tail"] = 0xFFFFFFFF
    vb = v.tobytes()
    vb += b"\0" * (align_to(len(vb), 160) - len(vb))
    ib = np.array([0, 1, 2], dtype="<u2").tobytes()
    ib += b"\0" * (-len(ib) % 16)
    return image, fixups, vb, ib


# ---- tests ----------------------------------------------------------------------------------------------------

class LayoutDetectionTests(unittest.TestCase):
    def test_samples_are_dl2(self):
        for n in SAMPLE_NAMES:
            p = sample_parts(n)
            self.assertIs(detect_layout(Fixups.parse(p[0x11])), LAYOUT_DL2, n)

    def test_dltb_sample_pack_is_dltb(self):
        pack = SAMPLES / "meshes.rpack"
        if not pack.exists():
            self.skipTest("DLTB sample pack missing")
        with synth.Pack.open(pack) as pk:
            n = 0
            for r in pk.resources_of_type(0x10):
                fx = Fixups.parse(bytes(r.read_part_by_type(0x11)))
                self.assertIs(detect_layout(fx), LAYOUT_DLTB, r.name)
                self.assertEqual(decode_resource(r).layout, "dltb")
                n += 1
            self.assertGreater(n, 0)

    def test_hint(self):
        fx = Fixups.parse(sample_parts("alarm_lamp_a")[0x11])
        self.assertIs(detect_layout(fx, "dl2"), LAYOUT_DL2)
        with self.assertRaises(FormatError):
            detect_layout(fx, "dltb")
        with self.assertRaises(FormatError):
            detect_layout(fx, "dl3")

    def test_unrecognised(self):
        fx = Fixups(0x60, 1, [Record(0, (0xB0 << 24) | 3, 1)], [])
        with self.assertRaises(FormatError):
            detect_layout(fx)
        self.assertIs(detect_layout(fx, "dl2"), LAYOUT_DL2)       # the hint decides only when the data does not


class ShippedSampleTests(unittest.TestCase):
    def test_hotel_number(self):
        m, _ = decode_sample("prp_hotel_number_10_a")
        self.assertEqual(m.layout, "dl2")
        self.assertTrue(m.is_dl2)
        self.assertEqual(m.embedded_name, b" prp_hotel_number_10_a.msh")
        self.assertEqual(m.scr_name, b" prp_hotel_number_10_a.scr")
        self.assertEqual(len(m.root_raw), 0x68)
        self.assertEqual(len(m.entities), 1)
        en = m.entities[0]
        self.assertEqual((en.parent, en.type, en.geometry_count, en.flags), (-1, 1, 1, 0x24704))
        self.assertEqual(len(en.raw_ca), 6)
        self.assertEqual([x.name for x in m.materials], [b"prp_hotel_numbers.mat"])
        self.assertEqual(len(m.geometry_entries), 1)
        e = m.geometry_entries[0]
        self.assertEqual((e.format, e.vertex_base, e.vertex_count, e.index_base), (3, 0, 28, 0))
        self.assertEqual((e.offset, e.stream_offset, e.material_slots_offset, e.index_counts_offset),
                         (0x198, 0x1C8, 0x1EC, 0x1E8))
        self.assertEqual(len(e.raw_stream), 0x20)
        self.assertEqual(e.raw_34[:4], bytes.fromhex("32465a6e"))
        self.assertEqual([(s.material_slot, s.index_count, len(s.palette)) for s in e.submeshes], [(0, 42, 0)])
        self.assertEqual(e.owner_entity, 0)
        self.assertEqual(m.warnings, [])
        v = e.vertices
        self.assertEqual(v.count, 28)
        np.testing.assert_allclose(np.linalg.norm(v.normals, axis=1), 1.0, atol=1e-4)
        lo, hi = v.positions.min(0), v.positions.max(0)
        c, h = en.bounds_center, en.bounds_half
        self.assertTrue(np.all(lo >= c - h - 1e-3) and np.all(hi <= c + h + 1e-3))

    def test_alarm_lamp_lods_and_spare_materials(self):
        m, _ = decode_sample("alarm_lamp_a")
        self.assertEqual([(e.element, e.vertex_base, e.vertex_count, e.index_base) for e in m.geometry_entries],
                         [(0, 0, 74, 0), (1, 2400, 61, 540)])
        self.assertEqual((len(m.materials), m.material_capacity), (1, 6))
        self.assertEqual(m.materials[0].name, b"alarm_lamp_b.mat")
        self.assertEqual(m.materials[0].name_tag, 0)          # DL2 material names are plain (untagged) pointers
        self.assertEqual([en.name for en in m.entities], [b"alarm_lamp_a", b"sound_pos"])
        var = variants.decode(m.skin_raw, m)
        self.assertNotIn("error", var)
        self.assertEqual(len(var["variants"]), 6)
        self.assertEqual(len(variants.material_table(m)), 6)

    def test_reencode_byte_identical(self):
        for n in SAMPLE_NAMES:
            m, p = decode_sample(n)
            self.assertEqual(encode_vertex_buffer(m), p[0xF0], n)
            self.assertEqual(encode_index_buffer(m), p[0xF1], n)
            self.assertEqual(m.fixups.to_bytes(), p[0x11], n)
            self.assertEqual(variants.encode(variants.decode(p[0x12])), p[0x12], n)

    def test_sidecar_records_layout(self):
        m, _ = decode_sample("alarm_lamp_a")
        side = build_sidecar(m)
        self.assertEqual(side["layout"], "dl2")
        self.assertEqual(side["geometry_entries"][1]["stream_offset"], 0x308)
        self.assertEqual(len(bytes.fromhex(side["geometry_entries"][1]["raw_stream"])), 0x20)


class PackPathTests(unittest.TestCase):
    """The paths the GUI and the CLI use: Pack → decode_resource / roundtrip / gui.meshdata / Cast export."""

    @classmethod
    def setUpClass(cls):
        cls.data = sample_pack()

    def test_roundtrip_through_codec(self):
        pk = synth.open_bytes(self.data)
        for r in pk.resources_of_type(0x10):
            out = MeshCodec().roundtrip(r)
            self.assertEqual(sorted(out), [2, 3, 4])
            for k, i in enumerate(r.part_indices):
                if k in out:
                    self.assertEqual(bytes(pk.read_part(i)), out[k], (r.name, k))

    def test_gui_meshdata(self):
        from nightrunner.gui import meshdata
        pk = synth.open_bytes(self.data)
        lamp = pk.find(" alarm_lamp_a", type_id=0x10, fold=False)[0]
        geoms, info = meshdata.load_mesh_geometry(pk, lamp)
        self.assertIsNone(info["error"])
        self.assertEqual(info["errors"], [])
        self.assertEqual(info["layout"], "dl2")
        self.assertEqual(info["lod_count"], 2)
        self.assertEqual(info["formats"], [3])
        self.assertEqual(len(geoms), 2)
        self.assertEqual([g.lod for g in geoms], [0, 1])
        self.assertEqual(sum(len(g.indices) for g in geoms), 270 * 2)
        self.assertIsNone(info["variants_error"])
        g0, _ = meshdata.load_mesh_geometry(pk, lamp, lods=[0])
        self.assertEqual(len(g0), 1)
        self.assertEqual(meshdata.mesh_materials(pk, lamp), ["alarm_lamp_b.mat"])

    def test_cast_export(self):
        from nightrunner.cast.export import load_cast
        from nightrunner.gui import meshdata
        pk = synth.open_bytes(self.data)
        with synth.tmpdir("dl2cast_") as d:
            for r in pk.resources_of_type(0x10):
                out = Path(d) / f"{r.index}.cast"
                rep = meshdata.export_cast_files(pk, r.index, out)
                self.assertTrue(out.exists())
                self.assertEqual(rep["warnings"], [])
                self.assertIsNotNone(load_cast(out))
                self.assertTrue(out.with_name(f"{r.index}.mesh.json").exists())


class SyntheticSkinnedTests(unittest.TestCase):
    def test_decode_skinned(self):
        img, fx, vb, ib = build_skinned_dl2()
        m = decode_parts("synth_dl2", img, fx, vertex=vb, index=ib)
        self.assertEqual(m.layout, "dl2")
        self.assertEqual(m.warnings, [])
        self.assertEqual(m.embedded_name, b"synth_dl2.msh")
        self.assertEqual([en.parent for en in m.entities], [-1, 0, -1])
        self.assertTrue(m.skinned)
        e = m.geometry_entries[0]
        self.assertEqual((e.format, e.vertex_count, e.owner_entity), (6, 3, 2))
        self.assertEqual(e.submeshes[0].palette.tolist(), [0, 1])
        self.assertEqual(e.submeshes[0].indices.tolist(), [0, 1, 2])
        self.assertEqual(e.vertices.joints[1].tolist(), [0, 1, 0, 0])
        np.testing.assert_allclose(e.vertices.weights.sum(1), 1.0)
        np.testing.assert_allclose(m.entity_globals()[1][:3, 3], [0, 1, 0])
        self.assertEqual(encode_vertex_buffer(m), vb)
        self.assertEqual(encode_index_buffer(m), ib)
        self.assertEqual(m.fixups.to_bytes(), fx)

    def test_cast_has_bones_and_weights(self):
        from nightrunner.cast.export import build_cast
        img, fx, vb, ib = build_skinned_dl2()
        m = decode_parts("synth_dl2", img, fx, vertex=vb, index=ib)
        cast, rep = build_cast(m)
        self.assertEqual(rep.nodes.get("bone"), 3)
        self.assertEqual(rep.nodes.get("mesh"), 1)

    def test_bad_stream_pointer(self):
        img, fx, vb, ib = build_skinned_dl2()
        f = Fixups.parse(fx)
        geo = [r for r in f.records if r.class_id == 6][0].offset
        bad = bytearray(img)
        struct.pack_into("<Q", bad, geo + 0x08, 0)
        with self.assertRaises(FormatError):
            decode_parts("x", bytes(bad), fx, vertex=vb, index=ib)

    def test_graph_views(self):
        img, fx, _, _ = build_skinned_dl2()
        g = MeshGraph(Image(img, Fixups.parse(fx)))
        self.assertIs(g.layout, LAYOUT_DL2)
        self.assertEqual(g.entity(2).geometry_count, 1)
        self.assertEqual(len(g.entity(2).raw_ca), 6)
        v = g.geometry_arrays[0].entry(g.img, 0)
        self.assertEqual((v.format, v.submesh_count, v.stream_submesh_count), (6, 1, 1))
        self.assertEqual(v.index_counts(), (3,))
        self.assertEqual(v.material_slots(), (0,))
        self.assertEqual([op[1] for op in g.opaque_records()].count(8), 0)   # class 8 is decoded, not opaque


class ReadOnlyGuardTests(unittest.TestCase):
    def test_refuse_layout(self):
        m, _ = decode_sample("prp_hotel_number_10_a")
        with self.assertRaises(UnsupportedError) as cm:
            refuse_layout(m)
        self.assertIn("DL2", str(cm.exception))

    def test_import_refuses_before_reading_cast(self):
        p = sample_parts("prp_hotel_number_10_a")
        with synth.tmpdir("dl2imp_") as d:
            cast = Path(d) / "model.cast"
            side = Path(d) / "mesh.json"
            cast.write_bytes(b"not a cast")
            m, _ = decode_sample("prp_hotel_number_10_a")
            import json
            side.write_text(json.dumps(build_sidecar(m)))
            with self.assertRaises(UnsupportedError):
                rebuild_from_files(cast, side, p, "prp_hotel_number_10_a")

    def test_imagepatch_refuses(self):
        from nightrunner.errors import BuildError
        from nightrunner.mesh import imagepatch
        m, _ = decode_sample("prp_hotel_number_10_a")
        with self.assertRaises(BuildError):
            imagepatch.apply(m, None, [])


class VariantsFlagNibbleTests(unittest.TestCase):
    def test_remap_word_high_nibble_round_trips(self):
        # DL2 record: +0x14 = 0x10002C00 (remap at +0x2C, bit 28 set); DLTB records have a zero high byte
        d = bytes.fromhex(
            "01000000080000003000000000000010380000000000001020000000002c0010"
            "00000000030001000000000001000100020002000000000044656661756c7400"
            "000017ff000000ff0000000000000000")
        dec = variants.decode(d)
        self.assertNotIn("error", dec)
        self.assertEqual(dec["variants"][0]["raw"]["remap_rel"], 0x2C)
        self.assertEqual(dec["variants"][0]["raw"]["e_hi"], 1)
        self.assertEqual(variants.encode(dec), d)

    def test_name_word_nibble_round_trips(self):
        # 56/33,728 DL2 parts store a value in bits 20..23 of the name word (e.g. 0xE0099A)
        d = bytearray(sample_parts("alarm_lamp_a")[0x12])    # skips cleanly when the fixtures are absent
        w = struct.unpack_from("<I", d, 8)[0]
        struct.pack_into("<I", d, 8, w | 0x00E00000)
        dec = variants.decode(bytes(d))
        self.assertNotIn("error", dec)
        self.assertEqual(dec["variants"][0]["raw"]["n_nib"], 0xE)
        self.assertEqual(dec["variants"][0]["name"], "Anthena")
        self.assertEqual(variants.encode(dec), bytes(d))


if __name__ == "__main__":
    unittest.main()
