"""Container writer tests: byte identity on the shipped small packs, table rendering, the stock storage-order
rule, hand-computed placement for the grouped and contiguous (bit-12) layouts, storage.size bookkeeping, tail
padding, u16 wraps, and every refusal (parts, storages, NUL names, preserve with changed sizes, permutation).

Placement expectations are computed by hand in the docstrings from the rules in notes/FORMATS/rp6l.md §9.
"""
from __future__ import annotations

import struct
import unittest
from pathlib import Path

from tests.synth import (
    Pack, PackWriter, ResourceSpec, PartSpec, PartSource, align_up,
    STOCK, METHOD0_MESH, MESH_SHAPE, payload, part, mesh, texture, single, prefab,
    write_pack, build_bytes, open_bytes, expected_table_end, small_packs, tmpdir,
)
from nightrunner.container.rp6l import (
    stock_storage_order, STORAGE_FLAG_STREAM, PHYS_BIT8, PHYS_OWNER_SHIFT, MAX_PARTS, MAX_STORAGES,
    HEADER_SIZE, STORAGE_SIZE, PHYSICAL_SIZE, LOGICAL_SIZE,
)
from nightrunner.container.validate import validate, ondemand_slices
from nightrunner.errors import BuildError


def _first_diff(a: bytes, b: bytes):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


class TestShippedIdentity(unittest.TestCase):
    """extract → build byte identity for the six small shipped packs, every applicable layout."""

    def setUp(self):
        if not small_packs():
            self.skipTest("small shipped packs not available")

    def _rebuild(self, src: Path, layout: str, *, from_scratch: bool = False, fill: bool = False) -> bytes:
        with Pack.open(src) as pk, tmpdir("ident_") as d:
            if from_scratch:
                w = PackWriter(pk.header.field08, pk.header.flags, layout, name_blob_order=pk.name_blob_order())
            else:
                w = PackWriter.from_pack(pk, layout)
            for r in pk:
                w.add(ResourceSpec.from_pack(pk, r.index))
            dst = Path(d) / "out.rpack"
            kw = {}
            if fill:
                kw = dict(fill=bytes(pk._data), final_size=pk.size)
            rep = w.write(dst, **kw)
            self.assertEqual(rep.layout, layout if layout != "auto" else "grouped")
            self.assertEqual(rep.size, dst.stat().st_size)
            self.assertEqual(rep.warnings, [])
            return dst.read_bytes()

    def test_byte_identity_all_layouts(self):
        for src in small_packs():
            orig = src.read_bytes()
            for layout in ("auto", "grouped", "preserve"):
                out = self._rebuild(src, layout)
                self.assertIsNone(_first_diff(orig, out), f"{src.name} layout={layout}")
            out = self._rebuild(src, "preserve", fill=True)
            self.assertEqual(out, orig, f"{src.name} preserve+fill")

    def test_from_scratch_identity(self):
        """No storage_order / template: only the name blob order is copied (it cannot be derived)."""
        for src in small_packs():
            orig = src.read_bytes()
            self.assertEqual(self._rebuild(src, "auto", from_scratch=True), orig, src.name)

    def test_render_table_identity(self):
        for src in small_packs():
            with Pack.open(src) as pk:
                w = PackWriter(pk.header.field08, pk.header.flags, "auto", name_blob_order=pk.name_blob_order())
                for r in pk:
                    w.add(ResourceSpec.from_pack(pk, r.index))
                tables, plan, header, warnings = w.render()
                self.assertEqual(tables, pk.table_bytes(), src.name)
                self.assertEqual(header, pk.header, src.name)
                self.assertEqual(w.part_offsets(plan), [pk.part_offset(i) for i in range(len(pk.physicals))], src.name)
                self.assertEqual([s for s in plan.storages], pk.storages, src.name)
                self.assertEqual(warnings, [])

    def test_name_blob_order_not_derived(self):
        """Without the recorded blob order the name table differs (reg1: [3,2,1,0,4,5]) but the pack is valid."""
        src = [p for p in small_packs() if p.name == "reg1_pc.rpack"]
        if not src:
            self.skipTest("reg1 missing")
        with Pack.open(src[0]) as pk:
            w = PackWriter(pk.header.field08, pk.header.flags, "auto")
            for r in pk:
                w.add(ResourceSpec.from_pack(pk, r.index))
            tables, plan, header, _ = w.render()
            self.assertNotEqual(tables, pk.table_bytes())
            self.assertEqual(tables[: pk.name_offset_offset], pk.table_bytes()[: pk.name_offset_offset])  # only the name table differs
            self.assertEqual(w.part_offsets(plan), [pk.part_offset(i) for i in range(len(pk.physicals))])
            nb = tables[pk.name_blob_offset:]
            self.assertEqual(nb.split(b"\0")[:-1], [r.name_raw for r in pk])


class TestStockStorageOrder(unittest.TestCase):

    def test_rule(self):
        # census 2026-09-15: bit-3 (stream) storages first, then the rest; each run sorted by type id
        mixed = [(0x20, 8, 0xB1, 0), (0x21, 8, 0xB1, 0), (0xF1, 8, 0x49, 0), (0x10, 8, 0xC9, 3), (0x12, 8, 0xD9, 0), (0x11, 8, 0xC9, 3), (0xF0, 8, 0x59, 0)]
        self.assertEqual([k[0] for k in stock_storage_order(mixed)], [0x10, 0x11, 0x12, 0xF0, 0xF1, 0x20, 0x21])
        anims_stream = [(0x4A, 8, 0x40, 0), (0x44, 8, 0x20, 0), (0x45, 8, 0x29, 0), (0x42, 8, 0x40, 0), (0x43, 8, 0x40, 0), (0x49, 8, 0x40, 0), (0x48, 8, 0xC0, 8), (0x47, 8, 0xC0, 8)]
        self.assertEqual([k[0] for k in stock_storage_order(anims_stream)], [0x45, 0x42, 0x43, 0x44, 0x47, 0x48, 0x49, 0x4A])
        engine = [(0xF1, 8, 0x40, 0), (0x62, 8, 0x80, 0), (0x10, 8, 0xC0, 3), (0x40, 8, 0x40, 0), (0x21, 8, 0xB1, 0), (0x11, 8, 0xC0, 3), (0xF0, 8, 0x50, 0), (0x20, 8, 0xB1, 0), (0x12, 8, 0xD0, 0), (0x61, 8, 0x80, 0)]
        self.assertEqual([k[0] for k in stock_storage_order(engine)], [0x10, 0x11, 0x12, 0x20, 0x21, 0x40, 0x61, 0x62, 0xF0, 0xF1])
        # same type, different attributes → ordered by (align_raw, flags, metadata)
        same = [(0x40, 8, 0x40, 1), (0x40, 8, 0x40, 0), (0x40, 6, 0x40, 0)]
        self.assertEqual(stock_storage_order(same), [(0x40, 6, 0x40, 0), (0x40, 8, 0x40, 0), (0x40, 8, 0x40, 1)])
        self.assertEqual(stock_storage_order([]), [])

    def test_writer_uses_stock_order(self):
        data = build_bytes([texture(b"t"), mesh(b"m"), single(b"a", 0x5A, 32)], field08=0x1000)
        pk = open_bytes(data)
        self.assertEqual([s.type for s in pk.storages], [0x10, 0x11, 0x12, 0xF0, 0xF1, 0x20, 0x21, 0x5A])

    def test_explicit_storage_order_with_extras(self):
        order = [(0x20, 8, 0xB1, 0), (0x21, 8, 0xB1, 0), (0x10, 8, 0xC9, 3), (0x12, 8, 0xD9, 0), (0x11, 8, 0xC9, 3), (0xF0, 8, 0x59, 0), (0xF1, 8, 0x49, 0)]  # assets_2_pc (DLE tool)
        w = PackWriter(0x1000, 1, "contiguous", storage_order=order)
        w.add(texture(b"t"))
        w.add(mesh(b"m"))
        w.add(single(b"a", 0x5A, 32))       # not in the explicit order → appended (stock order among extras)
        tables, plan, header, _ = w.render()
        self.assertEqual([s.type for s in plan.storages], [0x20, 0x21, 0x10, 0x12, 0x11, 0xF0, 0xF1, 0x5A])


class TestGroupedPlacement(unittest.TestCase):
    """field08 == 0: every storage group is one contiguous region, groups in storage-table order.

    Resources: texA(hdr 80, bmp 1000), texB(hdr 80, bmp 500), area(700), prefab(600, 220).
    storages (stock order): 0x20, 0x21, 0x5A, 0x61, 0x62 → 5; parts 7; names Tex_A/tex_b/dlc_area_x/Prefabs = 31 bytes.
    table_end = 36 + 5·20 + 7·16 + 4·12 + 4·4 + 31 = 343 → payload start 352.
      0x20 @ 352 (units 22): texA.hdr 352 (+0), texB.hdr 432 (+5)  size 160 → cursor 512
      0x21 @ 512 (units 32): texA.bmp 512 (+0), texB.bmp 1520 (+63) size 1008+512 = 1520 → cursor 2020 → 2032
      0x5A @ 2032 (127): area 2032, size 704 → 2736
      0x61 @ 2736 (171): prefab 2736, size 608 → 3344
      0x62 @ 3344 (209): fixups 3344, size 224 → 3564 → file padded to 3568
    """

    @classmethod
    def setUpClass(cls):
        cls.res = [texture(b"Tex_A", 80, 1000, seed=1), texture(b"tex_b", 80, 500, seed=2), single(b"dlc_area_x", 0x5A, 700), prefab()]
        cls.data = build_bytes(cls.res)
        cls.pk = open_bytes(cls.data)

    def test_offsets(self):
        pk = self.pk
        self.assertEqual(pk.table_end, 343)
        self.assertEqual([pk.part_offset(i) for i in range(7)], [352, 512, 432, 1520, 2032, 2736, 3344])
        self.assertEqual([s.base_units for s in pk.storages], [22, 32, 127, 171, 209])
        self.assertEqual([p.offset_units for p in pk.physicals], [0, 0, 5, 63, 0, 0, 0])
        self.assertEqual([s.size for s in pk.storages], [160, 1520, 704, 608, 224])
        self.assertEqual([s.count for s in pk.storages], [2, 2, 1, 1, 1])
        self.assertEqual(len(self.data), 3568)
        self.assertEqual(self.data[3564:], b"\0" * 4)
        # gaps are zero-filled
        self.assertEqual(self.data[1512:1520], b"\0" * 8)
        self.assertEqual(self.data[343:352], b"\0" * 9)

    def test_storage_size_is_sum_of_aligned_sizes(self):
        pk = self.pk
        for si, s in enumerate(pk.storages):
            members = [p for p in pk.physicals if p.storage_index == si]
            self.assertEqual(s.size, sum(align_up(p.size, 16) for p in members))
            self.assertEqual(s.count, len(members))

    def test_payload_bytes(self):
        pk = self.pk
        self.assertEqual(bytes(pk.read_part(3)), payload(500, 3))
        self.assertEqual(bytes(pk.read_part(6)), payload(220, 4))

    def test_validates(self):
        rep = validate(self.pk)
        self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
        self.assertEqual([f for f in rep.findings if f.level == "warning"], [])


class TestContiguousPlacement(unittest.TestCase):
    """field08 bit 12: non-stream groups first (grouped), then stream resources back-to-back in logical order.

    Resources: tex(hdr 80, bmp 1000), m0(200,40,100,1000,300), m1(50,20,30,500,100); logical order tex, m0, m1.
    storages: 10,11,12,F0,F1 (bit 3) then 20,21 → 7; parts 12; names tex/m0/m1 = 10 bytes.
    table_end = 36 + 140 + 192 + 36 + 12 + 10 = 426 → payload start 432.
      phase 1  0x20 @ 432 (27): tex.hdr 432;  0x21 @ 512 (32): tex.bmp 512 → cursor 1512
      phase 2  m0 start align 16 → 1520: image 1520, skin 1728, fixups 1776, vertex 1888, index 2896 → 3196
               m1 → 3200: image 3200, skin 3264, fixups 3296, vertex 3328, index 3840 → 3940 → file 3952
      mesh storages base_units 0, offset_units = absolute >> 4; sizes 0x10 208+64, 0x11 112+32, 0x12 48+32, 0xF0 1008+512, 0xF1 304+112.
    """

    @classmethod
    def setUpClass(cls):
        cls.res = [texture(b"tex", 80, 1000), mesh(b"m0", (200, 40, 100, 1000, 300), seed=1), mesh(b"m1", (50, 20, 30, 500, 100), seed=7)]
        cls.data = build_bytes(cls.res, field08=0x1000)
        cls.pk = open_bytes(cls.data)

    def test_offsets(self):
        pk = self.pk
        self.assertEqual(pk.table_end, 426)
        self.assertEqual([s.type for s in pk.storages], [0x10, 0x11, 0x12, 0xF0, 0xF1, 0x20, 0x21])
        offs = [pk.part_offset(i) for i in range(12)]
        self.assertEqual(offs, [432, 512, 1520, 1728, 1776, 1888, 2896, 3200, 3264, 3296, 3328, 3840])
        self.assertEqual([s.base_units for s in pk.storages], [0, 0, 0, 0, 0, 27, 32])
        self.assertEqual([p.offset_units for p in pk.physicals][2:7], [95, 108, 111, 118, 181])
        self.assertEqual([s.size for s in pk.storages], [272, 144, 80, 1520, 416, 80, 1008])
        self.assertEqual([s.count for s in pk.storages], [2, 2, 2, 2, 2, 1, 1])
        self.assertEqual(len(self.data), 3952)
        self.assertEqual([pk.part_type(i) for i in range(2, 7)], list(MESH_SHAPE))

    def test_engine_replay_matches(self):
        pk = self.pk
        for r in pk.resources_of_type(0x10):
            expected, total = ondemand_slices(pk, r.index)
            self.assertEqual(expected, [pk.part_offset(i) for i in r.part_indices])
        self.assertEqual(ondemand_slices(pk, 1)[1], 1376 + 300)
        rep = validate(pk, check_mesh_names=False)   # synthetic images carry no embedded .msh name
        self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
        self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=2, violating=0, incompatible=0))
        self.assertEqual(rep.stats["logical_flags"], {"0x01": 1, "0x81": 2})
        self.assertEqual(rep.stats["physical_flag_bits"], {"0x0000": 2, "0x0100": 10})

    def test_physical_words(self):
        pk = self.pk
        for r in pk:
            for i in r.part_indices:
                p = pk.physicals[i]
                self.assertEqual(p.packed >> PHYS_OWNER_SHIFT, r.index)
                self.assertEqual(p.flag_bits, PHYS_BIT8 if r.type == 0x10 else 0)
                self.assertEqual(p.fc, 0)

    def test_mixed_stream_resource_refused(self):
        r = ResourceSpec(b"bad", 0x40, 0x21, [part(0x44, payload(16)), part(0x45, payload(16), flag_bits=PHYS_BIT8)])
        with self.assertRaises(BuildError) as cm:
            build_bytes([r], field08=0x1000)
        self.assertIn("mixes stream and non-stream", str(cm.exception))
        # the same resource is fine in the grouped layout
        pk = open_bytes(build_bytes([r], field08=0))
        self.assertEqual(pk.resource(0).part_types, (0x44, 0x45))

    def test_auto_layout_selection(self):
        with tmpdir("auto_") as d:
            rep = write_pack([mesh(b"m")], Path(d) / "a.rpack", field08=0x1000)
            self.assertEqual(rep.layout, "contiguous")
            rep = write_pack([mesh(b"m")], Path(d) / "b.rpack", field08=0)
            self.assertEqual(rep.layout, "grouped")
            rep = write_pack([mesh(b"m")], Path(d) / "c.rpack", field08=0, layout="contiguous")
            self.assertEqual(rep.warnings, ["contiguous layout written into a pack without field08 bit 12"])
            rep = write_pack([mesh(b"m")], Path(d) / "d.rpack", field08=0x1000, layout="grouped")
            self.assertEqual(len(rep.warnings), 1)
            self.assertIn("grouped layout written into a pack WITH field08 bit 12", rep.warnings[0])


class TestAlignmentAbove16(unittest.TestCase):
    """No shipped storage has align_raw != 8; synthetic align_raw 12 → A = 64 (review item).

    Two meshes, all five storages at A=64 → 5 storages, 10 parts, names m0/m1 (6 bytes):
    table_end = 36 + 100 + 160 + 24 + 8 + 6 = 334 → payload 336 → m0 start align 64 → 384:
      image 384 (200) → 584 → 640 skin (40) → 680 → 704 fixups (100) → 804 → 832 vertex (1000) → 1832 → 1856 index (300) → 2156
      m1 → 2176: image 2176 (50) → 2226 → 2240 skin (20) → 2260 → 2304 fixups (30) → 2334 → 2368 vertex (500) → 2868 → 2880 index (100) → 2980 → file 2992
    """

    def test_a64_all_parts(self):
        res = [mesh(b"m0", (200, 40, 100, 1000, 300)), mesh(b"m1", (50, 20, 30, 500, 100), seed=7)]
        for r in res:
            for p in r.parts:
                p.align_raw = 12
        data = build_bytes(res, field08=0x1000)
        pk = open_bytes(data)
        self.assertEqual(pk.table_end, 334)
        self.assertTrue(all(s.alignment == 64 for s in pk.storages))
        self.assertEqual([pk.part_offset(i) for i in range(10)], [384, 640, 704, 832, 1856, 2176, 2240, 2304, 2368, 2880])
        self.assertEqual([s.size for s in pk.storages], [256 + 64, 128 + 64, 64 + 64, 1024 + 512, 320 + 128])
        self.assertEqual(len(data), 2992)
        for r in pk:
            expected, total = ondemand_slices(pk, r.index)
            self.assertEqual(expected, [pk.part_offset(i) for i in r.part_indices], r.name)
        rep = validate(pk, check_mesh_names=False)
        self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
        self.assertEqual([f.check for f in rep.findings if f.level != "info"], [])
        self.assertEqual(rep.stats["ondemand_meshes"]["ok"], 2)

    def test_mixed_alignment_within_resource(self):
        """image/skin/fixups at A=16, vertex/index at A=64: resource start aligned to 64, each part to its own A.
        One mesh: table_end = 36 + 100 + 80 + 12 + 4 + 3 = 235 → 240 → start align 64 → 256:
        image 256 (200) → 456 → 464 skin (40) → 504 → 512 fixups (100) → 612 → 640 vertex (1000) → 1640 → 1664 index."""
        r = mesh(b"m0", (200, 40, 100, 1000, 300))
        r.parts[3].align_raw = 12
        r.parts[4].align_raw = 12
        data = build_bytes([r], field08=0x1000)
        pk = open_bytes(data)
        self.assertEqual(pk.table_end, 235)
        self.assertEqual([pk.part_offset(i) for i in range(5)], [256, 464, 512, 640, 1664])
        expected, _ = ondemand_slices(pk, 0)
        self.assertEqual(expected, [256, 464, 512, 640, 1664])
        self.assertTrue(validate(pk, check_mesh_names=False).ok)

    def test_grouped_a64(self):
        """Grouped layout honours A=64 for group bases and part starts, storage.size = Σ align_up(size, 64)."""
        res = [single(b"a", 0x5A, 100, seed=1), single(b"b", 0x5A, 100, seed=2)]
        for r in res:
            r.parts[0].align_raw = 12
        pk = open_bytes(build_bytes(res))
        # table_end = 36 + 20 + 32 + 24 + 8 + 4 = 124 → 128 → align 64 → 128
        self.assertEqual(pk.table_end, 124)
        self.assertEqual([pk.part_offset(i) for i in range(2)], [128, 256])
        self.assertEqual(pk.storages[0].size, 256)
        self.assertTrue(validate(pk).ok)

    @unittest.expectedFailure
    def test_alignment_below_16_should_be_refused(self):
        """REVIEW FINDING (writer, low severity): for A < 16 the engine slices at align_up(cum, A) but file offsets
        are 16-byte units, so the writer places the part at align_up(cum, 16) — the two differ whenever the previous
        part's end is not 16-aligned and the contract is unsatisfiable. PackWriter should raise BuildError up front;
        today it writes the pack and only the post-build validation (ondemand check) catches it."""
        r = mesh(b"m", (200, 40, 100, 1000, 300))
        for p in r.parts:
            p.align_raw = 0      # A = 1
        with self.assertRaises(BuildError):
            build_bytes([r], field08=0x1000)

    def test_alignment_below_16_is_caught_by_validate(self):
        r = mesh(b"m", (200, 40, 100, 1000, 300))
        for p in r.parts:
            p.align_raw = 0
        pk = open_bytes(build_bytes([r], field08=0x1000))
        rep = validate(pk, check_mesh_names=False)
        self.assertFalse(rep.ok)
        self.assertEqual({f.check for f in rep.errors}, {"ondemand"})
        self.assertEqual(rep.stats["ondemand_meshes"]["violating"], 1)


class TestWrapsAndLimits(unittest.TestCase):

    def test_owner_wrap_warning(self):
        """65,537 logical resources: owner field (u16) wraps (dlc_frontier_envprobes does this: 124,398 logicals)."""
        res = [ResourceSpec(b"e%d" % i, 0x55, 1, [PartSpec(0x55, PartSource(b"\x01"), 8, 0x21, 0)]) for i in range(65537)]
        with tmpdir("wrap_") as d:
            rep = write_pack(res, Path(d) / "w.rpack")
            self.assertTrue(any("owner index wraps" in w for w in rep.warnings), rep.warnings)
            with Pack.open(Path(d) / "w.rpack") as pk:
                self.assertEqual(pk.physicals[65536].owner, 0)
                self.assertEqual(pk.physicals[65535].owner, 65535)
                self.assertEqual(pk.owner_of(65536), 0)
                v = validate(pk, max_findings=100)
                self.assertTrue(v.ok, [f.to_json() for f in v.errors][:3])
                infos = [f for f in v.findings if f.check == "owner"]
                self.assertEqual(len(infos), 1)
                self.assertEqual(infos[0].level, "info")
                self.assertEqual(infos[0].resource, 65536)

    def test_count_wrap_warning(self):
        """65,550 parts in one storage: count field (u16) wraps to 14 (dlc_frontier_envprobes: 123,777 → 58,241)."""
        res = [ResourceSpec(b"a%d" % i, 0x40, 1, [PartSpec(0x40, PartSource(b"\x01"), 8, 0x40, 0) for _ in range(15)]) for i in range(4370)]
        with tmpdir("cwrap_") as d:
            rep = write_pack(res, Path(d) / "c.rpack")
            self.assertTrue(any("count field wraps" in w for w in rep.warnings), rep.warnings)
            with Pack.open(Path(d) / "c.rpack") as pk:
                self.assertEqual(pk.storages[0].count, 65550 & 0xFFFF)
                self.assertEqual(pk.storages[0].size, 65550 * 16)
                v = validate(pk, max_findings=100)
                self.assertTrue(v.ok)
                self.assertTrue(any(f.check == "storage" and "wrapped" in f.message for f in v.findings))

    def test_part_count_limits(self):
        w = PackWriter()
        with self.assertRaises(BuildError):
            w.add(ResourceSpec(b"x", 0x40, 1, []))
        with self.assertRaises(BuildError):
            w.add(ResourceSpec(b"x", 0x40, 1, [part(0x40, payload(8)) for _ in range(MAX_PARTS + 1)]))
        w.add(ResourceSpec(b"x", 0x40, 1, [part(0x40, payload(8)) for _ in range(MAX_PARTS)]))
        tables, plan, header, _ = w.render()
        self.assertEqual(header.physical_count, 15)
        self.assertEqual(struct.unpack_from("<I", tables, HEADER_SIZE + STORAGE_SIZE + 15 * PHYSICAL_SIZE)[0] & 0xFFFF, 15)

    def test_unknown_part_type_refused(self):
        w = PackWriter()
        with self.assertRaises(BuildError):
            w.add(ResourceSpec(b"x", 0x40, 1, [PartSpec(0x99, PartSource(b"\x01"))]))

    def test_storage_group_limit(self):
        def res(n):
            out = []
            for m in range(n):
                f, meta = (0x40, m) if m < 256 else (0x41, 0)
                out.append(ResourceSpec(b"r%d" % m, 0x40, 1, [PartSpec(0x40, PartSource(b"\x01"), 8, f, meta)]))
            return out
        w = PackWriter()
        for r in res(MAX_STORAGES):
            w.add(r)
        tables, plan, header, _ = w.render()
        self.assertEqual(header.storage_count, 256)
        w = PackWriter()
        for r in res(MAX_STORAGES + 1):
            w.add(r)
        with self.assertRaises(BuildError) as cm:
            w.render()
        self.assertIn("257 storage groups", str(cm.exception))

    def test_nul_in_name_refused(self):
        w = PackWriter()
        w.add(single(b"bad\0name", 0x55, 16))
        with self.assertRaises(BuildError) as cm:
            w.render()
        self.assertIn("NUL", str(cm.exception))

    def test_empty_writer_refused(self):
        with self.assertRaises(BuildError):
            PackWriter().render()

    def test_flag_bits_range(self):
        w = PackWriter()
        w.add(ResourceSpec(b"x", 0x55, 1, [PartSpec(0x55, PartSource(b"\x01"), 8, 0x21, 0, flag_bits=0x10000)]))
        with self.assertRaises(BuildError):
            w.render()

    def test_name_blob_order_permutation_refused(self):
        res = [single(b"a", 0x55, 16), single(b"b", 0x55, 16)]
        for bad in ([0, 0], [0], [0, 1, 2], [1, 2]):
            with self.assertRaises(BuildError, msg=bad):
                build_bytes(res, name_blob_order=bad)
        pk = open_bytes(build_bytes(res, name_blob_order=[1, 0]))
        self.assertEqual(pk.name_blob, b"b\0a\0")
        self.assertEqual([r.name for r in pk], ["a", "b"])

    def test_name_index_kept_only_when_permutation(self):
        res = [single(b"a", 0x55, 16), single(b"b", 0x55, 16)]
        res[0].name_index, res[1].name_index = 1, 0
        pk = open_bytes(build_bytes(res))
        self.assertEqual([l.name_index for l in pk.logicals], [1, 0])
        self.assertEqual([r.name for r in pk], ["a", "b"])
        res[0].name_index, res[1].name_index = 1, 1        # not a permutation → sequential
        pk = open_bytes(build_bytes(res))
        self.assertEqual([l.name_index for l in pk.logicals], [0, 1])

    def test_unknown_layout(self):
        w = PackWriter(layout="bogus")
        w.add(single(b"a", 0x55, 16))
        with self.assertRaises(BuildError):
            w.render()


class TestPreserve(unittest.TestCase):
    """'preserve' re-uses the original storage indices/offset_units. Sources must be on disk: PartSpec.from_pack
    builds file-backed PartSources from pack.path (a Pack.from_bytes pack cannot feed a writer)."""

    def setUp(self):
        self._tmp = tmpdir("pres_")
        self.dir = Path(self._tmp.name)
        self.src = self.dir / "src.rpack"
        write_pack([texture(b"t", 80, 1000), single(b"a", 0x5A, 700), prefab()], self.src)
        self.data = self.src.read_bytes()
        self.pk = Pack.open(self.src)

    def tearDown(self):
        self.pk.close()
        self._tmp.cleanup()

    def test_preserve_roundtrip_and_fill(self):
        pk = self.pk
        w = PackWriter.from_pack(pk, "preserve")
        for r in pk:
            w.add(ResourceSpec.from_pack(pk, r.index))
        rep = w.write(self.dir / "p.rpack", fill=self.data, final_size=len(self.data))
        self.assertEqual(rep.layout, "preserve")
        self.assertEqual((self.dir / "p.rpack").read_bytes(), self.data)
        w = PackWriter.from_pack(pk, "preserve")
        for r in pk:
            w.add(ResourceSpec.from_pack(pk, r.index))
        w.write(self.dir / "q.rpack")
        self.assertEqual((self.dir / "q.rpack").read_bytes(), self.data)   # gaps/tail are zero anyway

    def test_preserve_needs_template(self):
        w = PackWriter(layout="preserve")
        w.add(single(b"a", 0x55, 16))
        with self.assertRaises(BuildError):
            w.render()
        # template but parts without original indices
        w = PackWriter(0, 1, "preserve", template_storages=self.pk.storages)
        w.add(single(b"a", 0x5A, 16))
        with self.assertRaises(BuildError):
            w.render()

    def test_preserve_changed_size_refused_by_overlap(self):
        pk = self.pk
        w = PackWriter.from_pack(pk, "preserve")
        w.add(ResourceSpec.from_pack(pk, 0, overrides={0: PartSource(payload(96))}))   # header 80 → 96 overlaps bitmap
        w.add(ResourceSpec.from_pack(pk, 1))
        w.add(ResourceSpec.from_pack(pk, 2))
        with self.assertRaises(BuildError) as cm:
            w.write(self.dir / "p.rpack")
        self.assertIn("overlaps", str(cm.exception))

    @unittest.expectedFailure
    def test_preserve_changed_size_of_last_part_should_be_refused(self):
        """REVIEW FINDING (writer, low severity): 'preserve' documents "requires every part to keep its original
        size" but only detects the change through payload overlap; enlarging the LAST part in file order (or
        shrinking any part) is silently accepted with stale storage.size values. build.py refuses at its level;
        PackWriter should too (PartSpec.from_pack could record the original size and _plan_preserve compare)."""
        pk = self.pk
        last = max(range(len(pk.physicals)), key=pk.part_offset)
        owner = pk.physicals[last].owner
        w = PackWriter.from_pack(pk, "preserve")
        for r in pk:
            ov = {last: PartSource(payload(pk.physicals[last].size + 100))} if r.index == owner else None
            w.add(ResourceSpec.from_pack(pk, r.index, overrides=ov))
        with self.assertRaises(BuildError):
            w.render()

    def test_preserve_tables_grown_refused(self):
        pk = self.pk
        w = PackWriter.from_pack(pk, "preserve")
        for r in pk:
            spec = ResourceSpec.from_pack(pk, r.index)
            spec.name = spec.name + b"_" * 64      # tables grow past the original payload start
            w.add(spec)
        with self.assertRaises(BuildError) as cm:
            w.render()
        self.assertIn("overlap the (larger) tables", str(cm.exception))


class TestPartSource(unittest.TestCase):

    def test_file_slice_source(self):
        with tmpdir("src_") as d:
            f = Path(d) / "blob.bin"
            f.write_bytes(payload(4096, 3))
            r = ResourceSpec(b"a", 0x55, 1, [PartSpec(0x55, PartSource(path=f, offset=100, size=1000), 8, 0x21, 0)])
            pk = open_bytes(build_bytes([r]))
            self.assertEqual(bytes(pk.read_part(0)), payload(4096, 3)[100:1100])
            with self.assertRaises(BuildError):
                PartSource(path=f)                    # size required
            with self.assertRaises(BuildError):
                PartSource()
            short = ResourceSpec(b"b", 0x55, 1, [PartSpec(0x55, PartSource(path=f, offset=4000, size=1000), 8, 0x21, 0)])
            with self.assertRaises(BuildError):
                build_bytes([short])                  # short read

    def test_from_pack_clones_storage_attributes(self):
        pk = open_bytes(build_bytes([mesh(b"m", ondemand=False)]))
        spec = ResourceSpec.from_pack(pk, 0)
        self.assertEqual(spec.name, b"m")
        self.assertEqual(spec.flags, 0x01)
        self.assertEqual([p.storage_key for p in spec.parts], [(t,) + METHOD0_MESH[t] for t in MESH_SHAPE])
        self.assertEqual([p.flag_bits for p in spec.parts], [0] * 5)
        self.assertEqual([p.storage_index for p in spec.parts], [0, 2, 1, 3, 4])
        self.assertEqual(spec.name_index, 0)


if __name__ == "__main__":
    unittest.main()
