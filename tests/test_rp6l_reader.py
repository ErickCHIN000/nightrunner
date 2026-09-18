"""Container reader tests: header/table parsing, derived fields, names, part access, bounds errors.

Synthetic packs come from tests/synth.py (built with PackWriter); the six small shipped packs are used when
available (PC: tests/paths.ASSETS, cloud: the staged uploads). Provenance of every asserted rule: survey 01
(notes/survey/01-container-rp6l.md) and the census of 2026-09-15 (out/reports/census/SUMMARY.md).
"""
from __future__ import annotations

import struct
import unittest

from tests.synth import (
    Pack, ResourceSpec, PartSource, align_up,
    STOCK, MESH_SHAPE, payload, part, mesh, texture, single, prefab,
    build_bytes, open_bytes, patch, patch_u32, patch_u8, storage_off, physical_off, logical_off, name_offset_off,
    expected_table_end, small_packs, small_pack,
)
from nightrunner.container import catalogue
from nightrunner.container.rp6l import (
    Header, Storage, Physical, Logical, HEADER_SIZE, STORAGE_SIZE, PHYSICAL_SIZE, LOGICAL_SIZE, MAGIC, VERSION,
    PHYS_BIT8, PHYS_SPECIAL, PHYS_CHILD, PHYS_BIT14, PHYS_BIT15, STORAGE_FLAG_STREAM,
)
from nightrunner.errors import FormatError, UnsupportedError
from nightrunner.util.names import engine_fold


def _basic_pack() -> bytes:
    """2 textures + 1 area + 1 prefab, grouped layout (field08 = 0)."""
    return build_bytes([
        texture(b"Tex_A", 80, 1000, seed=1),
        texture(b"tex_b", 80, 500, seed=2),
        single(b"dlc_area_x", 0x5A, 700),
        prefab(),
    ])


class TestRecords(unittest.TestCase):
    """Raw record dataclasses and their derived fields (survey 01 §2-§5)."""

    def test_header_pack_unpack(self):
        h = Header(field08=0x1000, physical_count=5, storage_count=2, name_count=3, name_bytes=40, logical_count=3, flags=1)
        b = h.pack()
        self.assertEqual(len(b), HEADER_SIZE)
        self.assertEqual(struct.unpack_from("<I", b, 0)[0], MAGIC)
        self.assertEqual(b[:4], b"RP6L")
        self.assertEqual(struct.unpack_from("<I", b, 4)[0], VERSION)
        h2 = Header.unpack(b)
        self.assertEqual(h2, h)
        self.assertTrue(h2.ondemand)
        self.assertFalse(Header().ondemand)

    def test_storage_derived_fields(self):
        # alignment = 1 << ((align_raw >> 1) & 15): word0 bits 9..12 (ResourceCore +0x1C290/+0x236D0, engine +0xCEB530/+0xCEAA60)
        for raw in range(256):
            s = Storage(0x10, raw, 0xC9, 0x03, 0, 0, 0, 0)
            self.assertEqual(s.alignment, 1 << ((raw >> 1) & 15), raw)
            self.assertEqual(s.alignment, 1 << ((s.word0 >> 9) & 15), raw)
        self.assertEqual(Storage(0x10, 8, 0xC9, 3, 0, 0, 0, 0).alignment, 16)   # every shipped storage
        self.assertEqual(Storage(0x10, 12, 0xC9, 3, 0, 0, 0, 0).alignment, 64)
        self.assertEqual(Storage(0x10, 0, 0xC9, 3, 0, 0, 0, 0).alignment, 1)
        # method = flags & 3, version = (flags >> 4) | ((metadata & 15) << 4), codec = metadata >> 4
        cases = {  # (flags, metadata) -> (method, version, codec, stream)
            (0xC9, 0x03): (1, 60, 0, True),    # 0x10/0x11 on-demand mesh
            (0xC0, 0x03): (0, 60, 0, False),   # engine_pc mesh
            (0xD9, 0x00): (1, 13, 0, True),    # 0x12 skin
            (0x59, 0x00): (1, 5, 0, True),     # 0xF0
            (0x49, 0x00): (1, 4, 0, True),     # 0xF1
            (0x29, 0x00): (1, 2, 0, True),     # 0xF3 / 0x45
            (0xB1, 0x00): (1, 11, 0, False),   # textures
            (0x40, 0x00): (0, 4, 0, False),    # anims
            (0xC0, 0x08): (0, 140, 0, False),  # 0x47/0x48 animgraph (version 140 needs the metadata nibble)
            (0x20, 0x00): (0, 2, 0, False),    # 0x44
            (0x80, 0x00): (0, 8, 0, False),    # prefabs
            (0x21, 0x00): (1, 2, 0, False),    # area/envprobe/voxel
            (0x21, 0x50): (1, 2, 5, False),    # synthetic codec nibble
            (0x23, 0x00): (3, 2, 0, False),    # synthetic compressed method
        }
        for (f, m), (method, version, codec, stream) in cases.items():
            s = Storage(0x10, 8, f, m, 0, 0, 0, 0)
            self.assertEqual((s.method, s.version, s.codec, s.stream), (method, version, codec, stream), (f, m))
            self.assertEqual(s.flag_bit3, stream)
            self.assertEqual(s.flag_bit2, bool(f & 4))
            self.assertEqual(s.version, (s.word0 >> 20) & 0xFF)
            self.assertEqual(s.method, (s.word0 >> 16) & 3)
            self.assertEqual(s.codec, s.word0 >> 28)
        self.assertEqual(Storage(0x10, 8, 0xC9, 3, 0, 0, 0, 0).key, (0x10, 8, 0xC9, 3))

    def test_storage_40bit_sizes(self):
        s = Storage(0x21, 8, 0xB1, 0, 7, (3 << 32) | 0x11223344, (1 << 32) | 5, 65535)
        b = s.pack()
        self.assertEqual(len(b), STORAGE_SIZE)
        self.assertEqual(b[0x12], 3)        # size_hi
        self.assertEqual(b[0x13], 1)        # compressed_hi
        self.assertEqual(struct.unpack_from("<I", b, 8)[0], 0x11223344)
        self.assertEqual(Storage.unpack(b, 0), s)
        self.assertEqual(s.base_offset, 7 << 4)
        self.assertEqual(Storage.from_json(s.to_json()), s)

    def test_physical_derived_fields(self):
        p = Physical((0x1234 << 16) | PHYS_BIT8 | PHYS_BIT15 | (5 << 9) | 0x07, 100, 200, 0xDEAD)
        self.assertEqual(len(p.pack()), PHYSICAL_SIZE)
        self.assertEqual(p.storage_index, 7)
        self.assertEqual(p.owner, 0x1234)
        self.assertTrue(p.bit8)
        self.assertEqual(p.priority, 5)
        self.assertTrue(p.bit15)
        self.assertFalse(p.bit14)
        self.assertFalse(p.special)
        self.assertFalse(p.child)
        self.assertEqual(p.flag_bits, PHYS_BIT8 | PHYS_BIT15 | (5 << 9))
        self.assertEqual(Physical.unpack(p.pack(), 0), p)
        q = Physical(PHYS_SPECIAL | PHYS_CHILD | PHYS_BIT14, 0, 0, 0)
        self.assertTrue(q.special and q.child and q.bit14)
        self.assertEqual(q.to_json()["flag_bits"], "0x7000")

    def test_logical_derived_fields(self):
        l = Logical.make(5, 0x10, 0x81, 3, 40)
        self.assertEqual(len(l.pack()), LOGICAL_SIZE)
        self.assertEqual((l.part_count, l.type, l.flags, l.name_index, l.first_part), (5, 0x10, 0x81, 3, 40))
        self.assertEqual(Logical.unpack(l.pack(), 0), l)
        self.assertEqual(l.packed, 5 | (0x10 << 16) | (0x81 << 24))


class TestSyntheticPack(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.data = _basic_pack()
        cls.pk = open_bytes(cls.data)

    def test_header_and_table_bounds(self):
        pk = self.pk
        h = pk.header
        self.assertEqual((h.magic, h.version, h.field08, h.flags), (MAGIC, VERSION, 0, 1))
        self.assertEqual(h.logical_count, 4)
        self.assertEqual(h.name_count, 4)
        self.assertEqual(h.physical_count, 2 + 2 + 1 + 2)
        self.assertEqual(h.storage_count, 5)   # 0x20, 0x21, 0x5A, 0x61, 0x62
        names = [b"Tex_A", b"tex_b", b"dlc_area_x", b"Prefabs"]
        self.assertEqual(h.name_bytes, sum(len(n) + 1 for n in names))
        self.assertEqual(pk.table_end, expected_table_end(5, 7, 4, names))
        self.assertEqual(pk.storage_offset, HEADER_SIZE)
        self.assertEqual(pk.physical_offset, HEADER_SIZE + 5 * STORAGE_SIZE)
        self.assertEqual(pk.logical_offset, pk.physical_offset + 7 * PHYSICAL_SIZE)
        self.assertEqual(pk.name_offset_offset, pk.logical_offset + 4 * LOGICAL_SIZE)
        self.assertEqual(pk.name_blob_offset, pk.name_offset_offset + 4 * 4)
        self.assertEqual(pk.table_bytes(), self.data[: pk.table_end])
        self.assertEqual(pk.size % 16, 0)

    def test_storage_table(self):
        types = [s.type for s in self.pk.storages]
        self.assertEqual(types, [0x20, 0x21, 0x5A, 0x61, 0x62])   # stock order: no bit-3 groups, sorted by type
        for s in self.pk.storages:
            self.assertEqual((s.align_raw, s.flags, s.metadata), STOCK[s.type])
            self.assertEqual(s.version, catalogue.type_version(s.type))
            self.assertEqual(s.compressed, 0)
        self.assertEqual([s.count for s in self.pk.storages], [2, 2, 1, 1, 1])

    def test_part_identity_by_storage_type(self):
        pk = self.pk
        self.assertEqual([pk.part_type(i) for i in range(7)], [0x20, 0x21, 0x20, 0x21, 0x5A, 0x61, 0x62])
        for i, p in enumerate(pk.physicals):
            self.assertEqual(pk.part_offset(i), (pk.storages[p.storage_index].base_units + p.offset_units) << 4)
            self.assertEqual(pk.part_alignment(i), 16)
            self.assertTrue(pk.part_is_direct(i))
            self.assertEqual(pk.part_offset(i) % 16, 0)
            self.assertGreaterEqual(pk.part_offset(i), align_up(pk.table_end, 16))
            self.assertLessEqual(pk.part_offset(i) + p.size, pk.size)

    def test_owner_and_part_ranges(self):
        pk = self.pk
        self.assertEqual([r.logical.first_part for r in pk], [0, 2, 4, 5])
        self.assertEqual([r.logical.part_count for r in pk], [2, 2, 1, 2])
        for r in pk:
            self.assertEqual(list(r.part_indices), list(range(r.logical.first_part, r.logical.first_part + r.logical.part_count)))
            for i in r.part_indices:
                self.assertEqual(pk.owner_of(i), r.index)
                self.assertEqual(pk.physicals[i].owner, r.index)
        self.assertEqual(pk.resource(3).part_types, (0x61, 0x62))
        self.assertEqual(pk.resource(0).part_types, (0x20, 0x21))
        self.assertEqual(len(pk), 4)
        self.assertEqual([r.index for r in pk.resources_of_type(0x20)], [0, 1])
        self.assertEqual(pk.type_histogram(), {0x20: 2, 0x5A: 1, 0x61: 1})

    def test_read_part_bytes(self):
        pk = self.pk
        r = pk.resource(0)
        self.assertEqual(bytes(pk.read_part(0)), payload(80, 1))
        self.assertEqual(bytes(pk.read_part(1)), payload(1000, 2))
        self.assertEqual(bytes(r.read_part_by_type(0x21)), payload(1000, 2))
        self.assertEqual(r.part_by_type(0x21), 1)
        self.assertIsNone(r.part_by_type(0xF0))
        self.assertIsNone(r.read_part_by_type(0xF0))
        self.assertEqual(bytes(pk.resource(3).read_part_by_type(0x62)), payload(220, 4))
        self.assertIsInstance(pk.read_part(0), memoryview)

    def test_names(self):
        pk = self.pk
        self.assertEqual([r.name for r in pk], ["Tex_A", "tex_b", "dlc_area_x", "Prefabs"])
        self.assertEqual(pk.resource(0).name_raw, b"Tex_A")
        self.assertEqual([l.name_index for l in pk.logicals], [0, 1, 2, 3])
        self.assertEqual(pk.name_blob_order(), [0, 1, 2, 3])
        self.assertEqual(pk.name(1), "tex_b")
        self.assertTrue(pk.name_blob.endswith(b"\0"))
        self.assertEqual(pk.name_blob.count(b"\0"), 4)

    def test_find_engine_case_folding(self):
        pk = self.pk
        # FindLogicalResourceUsingName +0x19250: ASCII A..Z folded on both sides, linear scan, type filter
        self.assertEqual(pk.find("tex_a"), [0])
        self.assertEqual(pk.find("TEX_A"), [0])
        self.assertEqual(pk.find(b"TEX_B", type_id=0x20), [1])
        self.assertEqual(pk.find("tex_b", type_id=0x5A), [])
        self.assertEqual(pk.find("Tex_A", fold=False), [0])
        self.assertEqual(pk.find("tex_a", fold=False), [])
        self.assertEqual(pk.find("prefabs"), [3])
        self.assertEqual(pk.find("nope"), [])
        self.assertEqual(engine_fold(b"AbZ[\\]^_`{|}~\x80\xc4"), b"abz[\\]^_`{|}~\x80\xc4")   # only 'A'..'Z' move

    def test_to_json_views(self):
        j = self.pk.resource(0).to_json()
        self.assertEqual(j["type"], "0x20")
        self.assertEqual(j["type_name"], "Texture")
        self.assertEqual(j["name_hex"], b"Tex_A".hex())
        self.assertEqual([p["type"] for p in j["parts"]], ["0x20", "0x21"])
        self.assertEqual(j["parts"][0]["owner"], 0)
        t = self.pk.to_json_tables()
        self.assertEqual(t["header"]["field08"], "0x00000000")
        self.assertEqual(len(t["storages"]), 5)


class TestNamesEdgeCases(unittest.TestCase):

    def test_duplicates_and_leading_spaces(self):
        # 72 duplicate names and 8 leading-space names exist in the shipped corpus; both must survive byte-exact
        data = build_bytes([
            single(b"dup", 0x40, 16, seed=1), single(b"dup", 0x40, 32, seed=2),
            single(b" leading", 0x40, 8, seed=3), single(b"leading", 0x40, 8, seed=4), single(b"DUP", 0x42, 8, seed=5),
        ][:4] + [ResourceSpec(b"DUP", 0x42, 1, [part(0x42, payload(8)), part(0x43, payload(8))])])
        pk = open_bytes(data)
        self.assertEqual([r.name_raw for r in pk], [b"dup", b"dup", b" leading", b"leading", b"DUP"])
        self.assertEqual(pk.find("dup"), [0, 1, 4])          # lowest index first (engine takes the first)
        self.assertEqual(pk.find("dup", type_id=0x40), [0, 1])
        self.assertEqual(pk.find("dup", type_id=0x42), [4])
        self.assertEqual(pk.find(" LEADING"), [2])
        self.assertEqual(pk.find("leading"), [3])
        self.assertEqual(pk.name_blob, b"dup\0dup\0 leading\0leading\0DUP\0")

    def test_name_blob_order_permutation(self):
        res = [single(b"a", 0x55, 16), single(b"b", 0x55, 16), single(b"c", 0x55, 16)]
        data = build_bytes(res, name_blob_order=[2, 0, 1])
        pk = open_bytes(data)
        self.assertEqual(pk.name_blob, b"c\0a\0b\0")
        self.assertEqual(pk.name_offsets, [2, 4, 0])          # name_index == logical index, offsets permuted
        self.assertEqual([r.name for r in pk], ["a", "b", "c"])
        self.assertEqual(pk.name_blob_order(), [2, 0, 1])

    def test_name_blob_order_none_when_not_bijective(self):
        data = build_bytes([single(b"a", 0x55, 16), single(b"b", 0x55, 16)])
        pk = open_bytes(data)
        # make both logicals point at name 0
        data2 = patch_u32(data, logical_off(pk, 1) + 4, 0)
        pk2 = open_bytes(data2)
        self.assertEqual([r.name for r in pk2], ["a", "a"])
        self.assertIsNone(pk2.name_blob_order())

    def test_non_ascii_bytes_survive(self):
        raw = b"caf\xc3\xa9\xff"
        pk = open_bytes(build_bytes([single(raw, 0x55, 16)]))
        self.assertEqual(pk.resource(0).name_raw, raw)
        self.assertEqual(pk.resource(0).name.encode("utf-8", "surrogateescape"), raw)
        self.assertEqual(pk.find(raw), [0])


class TestRefusalsAndBounds(unittest.TestCase):

    def setUp(self):
        self.data = _basic_pack()
        self.pk = open_bytes(self.data)

    def test_read_part_refuses_child(self):
        pk = self.pk
        d = patch_u32(self.data, physical_off(pk, 4), pk.physicals[4].packed | PHYS_CHILD)
        pk2 = open_bytes(d)
        self.assertTrue(pk2.physicals[4].child)
        self.assertFalse(pk2.part_is_direct(4))
        with self.assertRaises(UnsupportedError):
            pk2.read_part(4)
        self.assertIsNotNone(pk2.read_part(3))   # other parts unaffected

    def test_read_part_refuses_compressed_methods(self):
        pk = self.pk
        for method in (2, 3):
            s = pk.storages[2]      # 0x5A area, flags 0x21
            d = patch_u8(self.data, storage_off(pk, 2) + 2, (s.flags & ~3) | method)
            pk2 = open_bytes(d)
            self.assertEqual(pk2.storages[2].method, method)
            self.assertFalse(pk2.part_is_direct(4))
            with self.assertRaises(UnsupportedError):
                pk2.read_part(4)

    def test_bad_magic_and_version(self):
        with self.assertRaises(FormatError):
            open_bytes(patch(self.data, 0, b"RP5L"))
        with self.assertRaises(FormatError):
            open_bytes(patch_u32(self.data, 4, 3))

    def test_truncated_files(self):
        pk = self.pk
        # shorter than the header
        for n in (0, 1, 35):
            with self.assertRaises((FormatError, struct.error)):
                open_bytes(self.data[:n])
        # tables cut: anywhere before table_end must be a FormatError ("tables extend past end of file")
        for cut in (HEADER_SIZE, HEADER_SIZE + 3, pk.physical_offset + 5, pk.logical_offset, pk.name_offset_offset + 1, pk.table_end - 1):
            with self.assertRaises(FormatError, msg=cut):
                open_bytes(self.data[:cut])
        # payload cut: tables parse, but read_part of the truncated span fails
        last = max(range(len(pk.physicals)), key=lambda i: pk.part_offset(i))
        cut = pk.part_offset(last) + pk.physicals[last].size - 1
        pk2 = open_bytes(self.data[:cut])
        with self.assertRaises(FormatError):
            pk2.read_part(last)
        first = min(range(len(pk.physicals)), key=lambda i: pk.part_offset(i))
        self.assertEqual(len(pk2.read_part(first)), pk.physicals[first].size)

    def test_truncated_file_on_disk(self):
        from tests.synth import tmpdir
        from pathlib import Path
        with tmpdir("trunc_") as d:
            p = Path(d) / "t.rpack"
            p.write_bytes(self.data[:20])
            with self.assertRaises(FormatError):
                Pack.open(p)
            p.write_bytes(self.data[: self.pk.table_end - 4])
            with self.assertRaises(FormatError):
                Pack.open(p)
            p.write_bytes(self.data)
            with Pack.open(p) as pk:
                self.assertEqual(pk.size, len(self.data))
                self.assertEqual(bytes(pk.read_part(0)), payload(80, 1))

    def test_storage_index_out_of_range(self):
        pk = self.pk
        d = patch_u32(self.data, physical_off(pk, 0), (pk.physicals[0].packed & ~0xFF) | 5)   # 5 storages → max index 4
        with self.assertRaises(FormatError):
            open_bytes(d)

    def test_storage_count_over_256(self):
        d = patch_u32(self.data, 0x10, 257)
        with self.assertRaises(FormatError):
            open_bytes(d)

    def test_logical_parts_out_of_range(self):
        pk = self.pk
        d = patch_u32(self.data, logical_off(pk, 3) + 8, 6)    # first_part 6 + 2 parts > 7
        with self.assertRaises(FormatError):
            open_bytes(d)

    def test_name_index_out_of_range(self):
        pk = self.pk
        d = patch_u32(self.data, logical_off(pk, 0) + 4, 4)
        with self.assertRaises(FormatError):
            open_bytes(d)

    def test_name_offset_outside_blob_and_unterminated(self):
        pk = self.pk
        d = patch_u32(self.data, name_offset_off(pk, 1), 10_000)
        pk2 = open_bytes(d)                      # tables parse lazily
        with self.assertRaises(FormatError):
            pk2.resource(1).name_raw
        self.assertEqual(pk2.resource(0).name, "Tex_A")
        # remove the final NUL of the blob
        d = patch_u8(self.data, pk.table_end - 1, ord("x"))
        pk3 = open_bytes(d)
        with self.assertRaises(FormatError):
            pk3.resource(3).name_raw

    def test_part_span_past_eof(self):
        pk = self.pk
        d = patch_u32(self.data, physical_off(pk, 1) + 8, 1 << 30)
        pk2 = open_bytes(d)
        with self.assertRaises(FormatError):
            pk2.read_part(1)


class TestShippedSmallPacks(unittest.TestCase):
    """The six small shipped packs (survey 01 §13 numbers, probe_layout 2026-09-15)."""

    EXPECT = {  # name: (field08, storages types, logical_count, physical_count, types histogram, table_end, size)
        "dlc_ft_prologue_envprobes_pc.rpack": (0, [0x55], 146, 146, {0x55: 146}, 9108, 1920832),
        "menu_level_ft_persistent_pc.rpack": (0, [0x61, 0x62], 1, 2, {0x61: 1}, 132, 559904),   # size re-measured after the 2026-09-17 game patch (was 904352)
        "reg_buffer_cst_pa_pc.rpack": (0, [0x61, 0x62], 1, 2, {0x61: 1}, 132, 976),
        "reg_pc.rpack": (0, [0x5A, 0x61, 0x62], 2, 3, {0x5A: 1, 0x61: 1}, 212, 6848),
        "dlc_frontier_cb_region_0_pc.rpack": (0, [0x5A], 4, 4, {0x5A: 4}, 299, 326800),
        "reg1_pc.rpack": (0, [0x5A, 0x61, 0x62], 6, 7, {0x5A: 5, 0x61: 1}, 595, 1516288),
    }

    def setUp(self):
        if not small_packs():
            self.skipTest("small shipped packs not available")

    def test_tables(self):
        for p in small_packs():
            f08, stypes, L, P, hist, tend, size = self.EXPECT[p.name]
            with Pack.open(p) as pk:
                h = pk.header
                self.assertEqual((h.field08, h.flags, h.version), (f08, 1, 4), p.name)
                self.assertEqual([s.type for s in pk.storages], stypes, p.name)
                self.assertEqual((h.logical_count, h.physical_count, h.name_count), (L, P, L), p.name)
                self.assertEqual(pk.type_histogram(), hist, p.name)
                self.assertEqual(pk.table_end, tend, p.name)
                self.assertEqual(pk.size, size, p.name)
                for s in pk.storages:
                    self.assertEqual(s.align_raw, 8, p.name)
                    self.assertEqual(s.alignment, 16)
                    self.assertEqual((s.align_raw, s.flags, s.metadata), STOCK[s.type], p.name)
                    self.assertEqual(s.version, catalogue.type_version(s.type))
                    self.assertEqual(s.compressed, 0)
                    self.assertIn(s.method, (0, 1))

    def test_relations(self):
        for p in small_packs():
            with Pack.open(p) as pk:
                self.assertEqual([l.name_index for l in pk.logicals], list(range(len(pk))), p.name)
                seen = set()
                for r in pk:
                    self.assertIn(r.part_types, catalogue.PART_SHAPES[r.type], p.name)
                    self.assertEqual(r.flags, 0x01, p.name)          # census: every non-mesh/non-ANM2 logical
                    for i in r.part_indices:
                        self.assertNotIn(i, seen)
                        seen.add(i)
                        ph = pk.physicals[i]
                        self.assertEqual(ph.owner, r.index, p.name)
                        self.assertEqual(ph.flag_bits, 0, p.name)    # bit 8 only on mesh-family/ANM2 parts
                        self.assertEqual(ph.fc, 0, p.name)
                        self.assertEqual(pk.part_offset(i) % 16, 0)
                        self.assertTrue(pk.part_is_direct(i))
                        self.assertEqual(len(pk.read_part(i)), ph.size)
                self.assertEqual(len(seen), len(pk.physicals))
                self.assertEqual(pk.name_blob_order() is not None, True, p.name)

    def test_find_on_shipped(self):
        p = small_pack("reg1_pc.rpack")
        if p is None:
            self.skipTest("reg1_pc.rpack missing")
        with Pack.open(p) as pk:
            self.assertEqual(pk.find("PREFABS"), [5])
            self.assertEqual(pk.find("Prefabs", type_id=0x61), [5])
            self.assertEqual(pk.find("Prefabs", type_id=0x5A), [])
            self.assertEqual(pk.find("dlc_ft_prologue_genpin_9x2ee007_area"), [4])
            self.assertEqual(pk.name_blob_order(), [3, 2, 1, 0, 4, 5])
            self.assertEqual(pk.resource(5).part_types, (0x61, 0x62))
            self.assertEqual(pk.part_offset(0), 608)
            self.assertEqual(pk.part_offset(6), 1515856)

    def test_name_blob_order_shipped(self):
        p = small_pack("dlc_frontier_cb_region_0_pc.rpack")
        if p is None:
            self.skipTest("missing")
        with Pack.open(p) as pk:
            self.assertEqual(pk.name_blob_order(), [3, 0, 1, 2])
            self.assertEqual(pk.name_offsets, [34, 61, 88, 0])


if __name__ == "__main__":
    unittest.main()
