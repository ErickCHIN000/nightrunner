"""ClassReader fixups/image tests on the mesh sample pack (tools/make_samples_mesh.py) + corpus slice."""
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.classreader.fixups import Fixups, Record, Slot  # noqa: E402
from nightrunner.classreader.graph import MeshGraph, ROOT_SIZE, ENTITY_SIZE, GEOMETRY_SIZE  # noqa: E402
from nightrunner.classreader.image import Image, ImagePatch, embedded_mesh_name  # noqa: E402
from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.errors import FormatError  # noqa: E402
from nightrunner.mesh import identity  # noqa: E402
from tests.paths import ASSETS, SAMPLES, have_game  # noqa: E402

SAMPLE_PACK = SAMPLES / "meshes.rpack"


def _samples():
    if not SAMPLE_PACK.exists():
        raise unittest.SkipTest(f"{SAMPLE_PACK} missing (run tools/make_samples_mesh.py)")
    return Pack.open(SAMPLE_PACK)


class FixupsTests(unittest.TestCase):
    def test_parse_serialise_identity(self):
        with _samples() as pk:
            for r in pk.resources_of_type(0x10):
                raw = bytes(r.read_part_by_type(0x11))
                fx = Fixups.parse(raw)
                self.assertEqual(fx.to_bytes(), raw, r.name)
                self.assertFalse(fx.secondary_present)
                self.assertEqual(fx.records[0].class_id, 3)
                self.assertEqual(fx.records[0].offset, 0)
                offs = [x.offset for x in fx.records]
                self.assertEqual(offs, sorted(offs))
                self.assertTrue(all(s.kind in (0, 2) for s in fx.slots))
                self.assertTrue(all(s.offset % 8 == 0 for s in fx.slots))
                self.assertLessEqual(fx.primary_size, len(r.read_part_by_type(0x10)))

    def test_synthetic_secondary_stream(self):
        fx = Fixups(primary_size=64, object_count_raw=0x80000001, records=[Record(0, 0xB0000003, 1)],
                    slots=[Slot(0, 0), Slot(8, 2)], secondary=b"\x11" * 20, trailing=b"\0" * 4)
        blob = fx.to_bytes()
        back = Fixups.parse(blob)
        self.assertTrue(back.secondary_present)
        self.assertEqual(back.secondary, b"\x11" * 20)
        self.assertEqual(back.to_bytes(), blob)
        self.assertEqual([s.kind for s in back.slots], [0, 2])

    def test_bounds(self):
        with self.assertRaises(FormatError):
            Fixups.parse(b"\0" * 8)
        with self.assertRaises(FormatError):
            Fixups.parse(struct.pack("<3I", 16, 5, 1) + b"\0" * 8)   # 5 records do not fit


class ImageTests(unittest.TestCase):
    def test_embedded_name_and_pointers(self):
        with _samples() as pk:
            for r in pk.resources_of_type(0x10):
                img_b = bytes(r.read_part_by_type(0x10))
                fix_b = bytes(r.read_part_by_type(0x11))
                self.assertEqual(embedded_mesh_name(img_b, fix_b), r.name_raw + b".msh", r.name)
                img = Image.from_parts(img_b, fix_b)
                p = img.pointer(0)
                self.assertEqual(p.kind, 0)
                self.assertEqual(img.u64(0), p.target + 1)
                g = MeshGraph(img)
                self.assertEqual(g.root.entity_count, g.entity_count)
                self.assertEqual(g.root.entities_ptr.target, g.entity_offset)
                for i, ev in enumerate(g.entities()):
                    self.assertEqual(ev.own_index, i)
                    gp = ev.geometry_ptr
                    if ev.geometry_count:
                        self.assertIsNotNone(gp)
                        ga = g.geometry_array_at(gp.target)
                        self.assertIsNotNone(ga, f"{r.name} entity {i}")
                        self.assertEqual(ga.count, ev.geometry_count)
                for ga in g.geometry_arrays:
                    for k in range(ga.count):
                        e = ga.entry(img, k)
                        self.assertIn(e.format, (0, 3, 6, 8))
                        self.assertEqual(len(e.material_slots()), e.submesh_count)
                        self.assertEqual(len(e.index_counts()), e.submesh_count)
                        self.assertEqual(len(e.palette_descs()), e.submesh_count)
                for m in g.materials():
                    self.assertTrue(m.name.lower().endswith(b".mat"), m.name)

    def test_unslotted_nonzero_word_is_rejected(self):
        with _samples() as pk:
            r = pk.resource(1)
            img = Image.from_parts(bytes(r.read_part_by_type(0x10)), bytes(r.read_part_by_type(0x11)))
            # the root's entity-count word at 0x58 is not a slot and is non-zero
            with self.assertRaises(FormatError):
                img.pointer(0x58)

    def test_rename_roundtrip(self):
        with _samples() as pk:
            r = pk.resource(2)   # dlc_ft_safe_zone_cable_e
            img_b = bytes(r.read_part_by_type(0x10))
            fix_b = bytes(r.read_part_by_type(0x11))
            new_img, new_fix, info = identity.rename(img_b, fix_b, "my_custom_cable")
            self.assertTrue(info["changed"])
            self.assertEqual(embedded_mesh_name(new_img, new_fix), b"my_custom_cable.msh")
            self.assertEqual(len(new_img) % 16, 0)
            fx_new = Fixups.parse(new_fix)
            fx_old = Fixups.parse(fix_b)
            self.assertEqual(len(fx_new.slots), len(fx_old.slots))
            self.assertEqual(fx_new.records, fx_old.records)
            self.assertGreater(fx_new.primary_size, fx_old.primary_size)
            # every other byte of the old primary image is untouched
            self.assertEqual(new_img[8:fx_old.primary_size], img_b[8:fx_old.primary_size])
            # renaming to the same name is a no-op
            same_img, same_fix, info2 = identity.rename(new_img, new_fix, "my_custom_cable")
            self.assertFalse(info2["changed"])
            self.assertEqual(same_img, new_img)
            # and the model still decodes after the rename
            from nightrunner.mesh.decode import decode_parts
            m = decode_parts("my_custom_cable", new_img, new_fix, vertex=bytes(r.read_part_by_type(0xF0)),
                             index=bytes(r.read_part_by_type(0xF1)))
            self.assertEqual(m.embedded_name, b"my_custom_cable.msh")
            self.assertEqual(len(m.geometry_entries), 1)

    def test_patch_add_slot_and_record(self):
        with _samples() as pk:
            r = pk.resource(3)
            img = Image.from_parts(bytes(r.read_part_by_type(0x10)), bytes(r.read_part_by_type(0x11)))
            patch = ImagePatch(img)
            data_off = patch.append(b"\x01\x02\x03\x04", align=4)
            slot_off = patch.append(b"\0" * 8, align=8)
            patch.retarget(slot_off, data_off, kind=0)
            patch.add_record(data_off, 0, 1)
            new_img, new_fix = patch.finish()
            img2 = Image.from_parts(new_img, new_fix)
            self.assertEqual(img2.pointer(slot_off).target, data_off)
            self.assertEqual(img2.raw(data_off, 4), b"\x01\x02\x03\x04")
            self.assertEqual(len(img2.records), len(img.records) + 1)
            self.assertEqual(sorted(x.offset for x in img2.records), [x.offset for x in img2.records])


@unittest.skipUnless(have_game(), "game not installed")
class CorpusTests(unittest.TestCase):
    def test_embedded_names_first_500(self):
        with Pack.open(ASSETS / "common_meshes_pc.rpack") as pk:
            n = 0
            for r in pk.resources_of_type(0x10):
                if n >= 500:
                    break
                n += 1
                img_b = bytes(pk.read_part(r.part_by_type(0x10)))   # copies: memoryviews would pin the mmap
                fix_b = bytes(pk.read_part(r.part_by_type(0x11)))
                self.assertEqual(embedded_mesh_name(img_b, fix_b), r.name_raw + b".msh")
                fx = Fixups.parse(fix_b)
                self.assertEqual(fx.to_bytes(), fix_b)


if __name__ == "__main__":
    unittest.main()
