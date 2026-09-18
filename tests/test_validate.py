"""validate.py: every check triggered by a deliberately broken synthetic pack, plus the on-demand replay arithmetic
re-derived independently from survey 01 §8 (engine +0xCEB530 / +0xCEAA60).
"""
from __future__ import annotations

import struct
import unittest

from tests.synth import (
    Pack, ResourceSpec, PartSpec, PartSource, align_up,
    STOCK, MESH_SHAPE, payload, part, mesh, texture, single, prefab,
    build_bytes, open_bytes, patch, patch_u32, patch_u8, storage_off, physical_off, logical_off, name_offset_off,
    small_packs,
)
from nightrunner.container.rp6l import (
    HEADER_SIZE, PHYS_BIT8, PHYS_SPECIAL, PHYS_CHILD, PHYS_BIT14, PHYS_BIT15, PHYS_OWNER_SHIFT, Storage, Physical, Logical,
)
from nightrunner.container.validate import validate, ondemand_slices, Report, Finding


def _checks(rep: Report, level: str | None = None) -> list[tuple[str, str]]:
    return [(f.level, f.check) for f in rep.findings if level is None or f.level == level]


def _has(rep: Report, level: str, check: str, needle: str | None = None) -> bool:
    return any(f.level == level and f.check == check and (needle is None or needle in f.message) for f in rep.findings)


def _grouped() -> bytes:
    return build_bytes([texture(b"Tex_A", 80, 1000, seed=1), texture(b"tex_b", 80, 500, seed=2), single(b"dlc_area_x", 0x5A, 700), prefab()])


def _meshes(field08=0x1000, layout="auto", **kw) -> bytes:
    return build_bytes([texture(b"tex", 80, 1000), mesh(b"m0", (200, 40, 100, 1000, 300), seed=1), mesh(b"m1", (50, 20, 30, 500, 100), seed=7)],
                       field08=field08, layout=layout, **kw)


def _engine_slices(base: int, parts: list[tuple[int, int]]) -> tuple[list[int], int]:
    """Independent transcription of OnDemandScheduleResourcePackCompatible (+0xCEB530) and
    OnDemandPreRegisterOnDemandReady_Mesh (+0xCEAA60), survey 01 §8:
        total = ((total - 1 + A) & -A) + size     (one read of `total` bytes from the first part's address)
        offset = (offset + A - 1) & -A; slice = buffer + offset; offset += size
    with A = 1 << ((storage_w >> 9) & 15) taken per part."""
    total = 0
    offset = 0
    slices = []
    for align_raw, size in parts:
        A = 1 << ((align_raw >> 1) & 15)          # (w >> 9) & 15 where w = type | align_raw << 8 | ...
        offset = (offset + A - 1) & -A
        slices.append(base + offset)
        offset += size
        total = ((total - 1 + A) & -A) + size
    return slices, total


class TestOnDemandArithmetic(unittest.TestCase):

    def test_formula_equivalence(self):
        # ((total - 1 + A) & -A) == align_up(total, A) for every total >= 0 and power-of-two A (incl. total == 0)
        for shift in range(0, 16):
            A = 1 << shift
            for total in list(range(0, 300)) + [4095, 4096, 4097, 65535, 1 << 20, (1 << 31) - 1]:
                self.assertEqual((total - 1 + A) & -A, align_up(total, A), (total, A))

    def test_replay_matches_independent_transcription(self):
        for align_raw, sizes in ((8, (200, 40, 100, 1000, 300)), (12, (200, 40, 100, 1000, 300)), (10, (1, 1, 1, 1, 1)), (14, (4096, 3, 129, 1000, 300))):
            r = mesh(b"m", sizes)
            for p in r.parts:
                p.align_raw = align_raw
            pk = open_bytes(build_bytes([r, texture(b"t")], field08=0x1000))
            base = pk.part_offset(0)
            exp_slices, exp_total = _engine_slices(base, [(align_raw, s) for s in sizes])
            got = ondemand_slices(pk, 0)
            self.assertIsNotNone(got)
            self.assertEqual(got, (exp_slices, exp_total), align_raw)
            self.assertEqual(got[0], [pk.part_offset(i) for i in pk.resource(0).part_indices], align_raw)
            self.assertEqual(pk.storages[0].alignment, 1 << ((align_raw >> 1) & 15))

    def test_replay_mixed_alignments(self):
        r = mesh(b"m", (200, 40, 100, 1000, 300))
        raws = [8, 12, 8, 14, 10]
        for p, a in zip(r.parts, raws):
            p.align_raw = a
        pk = open_bytes(build_bytes([r], field08=0x1000))
        base = pk.part_offset(0)
        self.assertEqual(base % 256, 0)      # resource start aligned to the largest part alignment (A=256 for raw 14)
        self.assertEqual(ondemand_slices(pk, 0), _engine_slices(base, list(zip(raws, (200, 40, 100, 1000, 300)))))

    def test_replay_refuses_method_and_special(self):
        pk = open_bytes(_meshes())
        data = pk.table_bytes() + b""
        # physical 0x1000 on any part → None
        i = pk.resource(1).logical.first_part + 2
        d = patch_u32(bytes(pk._data), physical_off(pk, i), pk.physicals[i].packed | PHYS_SPECIAL)
        self.assertIsNone(ondemand_slices(open_bytes(d), 1))
        self.assertIsNotNone(ondemand_slices(open_bytes(d), 2))
        # storage method 0 → None (0xC9 → 0xC8)
        d = patch_u8(bytes(pk._data), storage_off(pk, 0) + 2, 0xC8)
        self.assertIsNone(ondemand_slices(open_bytes(d), 1))
        # method 2/3 → None
        d = patch_u8(bytes(pk._data), storage_off(pk, 0) + 2, 0xCA)
        self.assertIsNone(ondemand_slices(open_bytes(d), 1))
        # non-mesh resources replay too (the function is type-agnostic; validate applies it to 0x10 only)
        self.assertEqual(ondemand_slices(pk, 0)[0], [pk.part_offset(0), pk.part_offset(1)])


class TestCleanPacks(unittest.TestCase):

    def test_synthetic_clean(self):
        rep = validate(open_bytes(_grouped()))
        self.assertTrue(rep.ok)
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.stats["types"], {0x20: 2, 0x5A: 1, 0x61: 1})
        self.assertFalse(rep.stats["ondemand"])
        self.assertNotIn("ondemand_meshes", rep.stats)
        j = rep.to_json()
        self.assertEqual(j["ok"], True)
        self.assertEqual(j["findings"], [])
        rep = validate(open_bytes(_meshes()), check_mesh_names=False)
        self.assertTrue(rep.ok)
        self.assertEqual(rep.findings, [])
        self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=2, violating=0, incompatible=0))

    def test_shipped_small_packs_clean(self):
        if not small_packs():
            self.skipTest("small shipped packs not available")
        for p in small_packs():
            with Pack.open(p) as pk:
                rep = validate(pk)
                self.assertTrue(rep.ok, p.name)
                self.assertEqual(rep.findings, [], p.name)
                self.assertEqual(rep.stats["logical_flags"], {"0x01": len(pk)}, p.name)
                self.assertEqual(rep.stats["physical_flag_bits"], {"0x0000": len(pk.physicals)}, p.name)


class TestStructureChecks(unittest.TestCase):

    def setUp(self):
        self.data = _grouped()
        self.pk = open_bytes(self.data)

    def test_name_offset_outside_blob(self):
        rep = validate(open_bytes(patch_u32(self.data, name_offset_off(self.pk, 2), 5000)))
        self.assertTrue(_has(rep, "error", "structure", "outside blob"))

    def test_name_not_terminated(self):
        rep = validate(open_bytes(patch_u8(self.data, self.pk.table_end - 1, ord("x"))))
        self.assertTrue(_has(rep, "error", "structure", "not NUL-terminated"))

    def test_header_flags_and_field08_info(self):
        rep = validate(open_bytes(patch_u32(self.data, 0x20, 0x41)))
        self.assertTrue(_has(rep, "info", "structure", "header.flags = 0x41"))
        rep = validate(open_bytes(patch_u32(self.data, 0x08, 0x1001)))
        self.assertTrue(_has(rep, "info", "structure", "bits other than 0x1000"))
        self.assertTrue(rep.stats["ondemand"])
        self.assertTrue(rep.ok)

    def test_name_count_mismatch_warning(self):
        """name_count > logical_count: hand-assembled (the writer always emits name_count == logical_count)."""
        payload_bytes = payload(100)
        names = b"one\0extra\0"
        table_end = HEADER_SIZE + 20 + 16 + 12 + 8 + len(names)
        start = align_up(table_end, 16)
        hdr = struct.pack("<9I", 0x4C365052, 4, 0, 1, 1, 2, len(names), 1, 1)
        sto = Storage(0x55, 8, 0x21, 0, start >> 4, 112, 0, 1).pack()
        phy = Physical(0, 0, 100, 0).pack()
        log = Logical.make(1, 0x55, 1, 0, 0).pack()
        data = hdr + sto + phy + log + struct.pack("<2I", 0, 4) + names
        data += b"\0" * (start - len(data)) + payload_bytes + b"\0" * 12
        pk = open_bytes(data)
        self.assertEqual(pk.header.name_count, 2)
        rep = validate(pk)
        self.assertTrue(rep.ok)
        self.assertTrue(_has(rep, "warning", "structure", "name_count 2 != logical_count 1"))


class TestStorageChecks(unittest.TestCase):

    def setUp(self):
        self.data = _grouped()
        self.pk = open_bytes(self.data)

    def test_unknown_type(self):
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2), 0x99)))
        self.assertTrue(_has(rep, "error", "storage", "unknown type 0x99"))

    def test_version_mismatch(self):
        # 0x5A area: flags 0x21 → version 2; 0x31 → version 3
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 2, 0x31)))
        self.assertTrue(_has(rep, "error", "storage", "version 3 != catalogue 2"))
        # metadata high nibble of the version (0x47 needs it: 140 = 0x8C → flags 0xC0, metadata 0x08)
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 3, 0x01)))
        self.assertTrue(_has(rep, "error", "storage", "version 18 != catalogue 2"))

    def test_compressed_method_warning(self):
        for m in (2, 3):
            rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 2, 0x20 | m)))
            self.assertTrue(_has(rep, "warning", "storage", f"compressed method {m}"), m)
            self.assertTrue(rep.ok)          # bounds/alignment are skipped for non-direct parts

    def test_unknown_bits_info(self):
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 3, 0x50)))       # codec nibble 5
        self.assertTrue(_has(rep, "info", "unknown", "codec nibble 5"))
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 2, 0x25)))       # flags bit 2
        self.assertTrue(_has(rep, "info", "unknown", "flags bit 2"))
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 1, 9)))          # align_raw bit 0
        self.assertTrue(_has(rep, "info", "unknown", "align_raw 0x09"))
        self.assertTrue(rep.ok)                                                                  # alignment still 16
        rep = validate(open_bytes(patch_u8(self.data, storage_off(self.pk, 2) + 1, 0x28)))       # align_raw bits 5..7
        self.assertTrue(_has(rep, "info", "unknown", "align_raw 0x28"))
        rep = validate(open_bytes(patch_u32(self.data, storage_off(self.pk, 2) + 0x0C, 77)))     # compressed size
        self.assertTrue(_has(rep, "info", "unknown", "compressed size 77"))
        self.assertTrue(rep.ok)

    def test_count_and_size_bookkeeping(self):
        rep = validate(open_bytes(patch(self.data, storage_off(self.pk, 0) + 0x10, struct.pack("<H", 7))))
        self.assertTrue(_has(rep, "warning", "storage", "count 7 != 2"))
        rep = validate(open_bytes(patch_u32(self.data, storage_off(self.pk, 1) + 8, 1)))
        self.assertTrue(_has(rep, "warning", "storage", "size 1 != Σ align_up(part.size) = 1520"))
        self.assertTrue(rep.ok)


class TestPartChecks(unittest.TestCase):

    def setUp(self):
        self.data = _grouped()
        self.pk = open_bytes(self.data)

    def test_owner_mismatch(self):
        p = self.pk.physicals[4]
        d = patch_u32(self.data, physical_off(self.pk, 4), (p.packed & 0xFFFF) | (7 << PHYS_OWNER_SHIFT))
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "owner", "part 4 owner 7 != 2"))
        self.assertEqual(len(rep.errors), 1)

    def test_part_count_zero(self):
        l = self.pk.logicals[2]
        d = patch_u32(self.data, logical_off(self.pk, 2), l.packed & ~0xFFFF)
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "parts", "0 parts"))
        self.assertTrue(_has(rep, "warning", "parts", "physical 4 is not referenced"))

    def test_part_count_over_15_and_double_reference(self):
        res = [ResourceSpec(b"a", 0x40, 1, [part(0x40, payload(8, k)) for k in range(8)]),
               ResourceSpec(b"b", 0x40, 1, [part(0x40, payload(8, k)) for k in range(8)])]
        data = build_bytes(res)
        pk = open_bytes(data)
        d = patch_u32(data, logical_off(pk, 0), (pk.logicals[0].packed & ~0xFFFF) | 16)
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "parts", "16 parts"))
        self.assertTrue(_has(rep, "error", "parts", "physical 8 referenced by logical 0 and 1"))
        self.assertTrue(_has(rep, "error", "owner"))

    def test_bounds_overlapping_tables(self):
        d = patch_u32(self.data, storage_off(self.pk, 0) + 4, 0)          # base_units of the 0x20 group → 0
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "bounds", "overlaps the tables"))

    def test_bounds_past_eof(self):
        d = patch_u32(self.data, physical_off(self.pk, 1) + 8, 1 << 24)
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "bounds", "exceeds file"))
        d = patch_u32(self.data, physical_off(self.pk, 1) + 4, 1 << 20)
        rep = validate(open_bytes(d))
        self.assertTrue(_has(rep, "error", "bounds", "exceeds file"))

    def test_bounds_skipped_for_child_parts(self):
        p = self.pk.physicals[1]
        d = patch_u32(self.data, physical_off(self.pk, 1) + 4, 1 << 20)
        d = patch_u32(d, physical_off(self.pk, 1), p.packed | PHYS_CHILD)
        rep = validate(open_bytes(d))
        self.assertTrue(rep.ok)

    def test_misaligned_offset(self):
        # payload starts at 352 (not a multiple of 64): declaring A=64 on storage 0 makes its parts misaligned
        d = patch_u8(self.data, storage_off(self.pk, 0) + 1, 12)
        pk = open_bytes(d)
        self.assertEqual(pk.storages[0].alignment, 64)
        rep = validate(pk)
        self.assertTrue(_has(rep, "error", "alignment", "offset 0x160 not aligned to 64"))
        self.assertTrue(_has(rep, "warning", "storage", "size 160 != Σ align_up(part.size) = 256"))

    def test_unknown_physical_bits_and_fc_info(self):
        p = self.pk.physicals[4]
        for bit, name in ((PHYS_BIT14, "physical bit 14"), (PHYS_BIT15, "physical bit 15"), (PHYS_SPECIAL, "physical bit 12 (0x1000)")):
            rep = validate(open_bytes(patch_u32(self.data, physical_off(self.pk, 4), p.packed | bit)))
            self.assertTrue(_has(rep, "info", "unknown", name), name)
            self.assertTrue(rep.ok, name)
        rep = validate(open_bytes(patch_u32(self.data, physical_off(self.pk, 4) + 12, 0xBEEF)))
        self.assertTrue(_has(rep, "info", "unknown", "fc = 0xBEEF"))
        self.assertTrue(rep.ok)
        self.assertEqual(rep.stats["physical_flag_bits"], {"0x0000": 7})

    def test_max_findings_cap(self):
        d = self.data
        for i in range(7):
            p = self.pk.physicals[i]
            d = patch_u32(d, physical_off(self.pk, i), (p.packed & 0xFFFF) | (9 << PHYS_OWNER_SHIFT))
        rep = validate(open_bytes(d), max_findings=3)
        # errors are never dropped by the cap (review 2026-09-15 F1): 7 owner errors survive, ok is False
        self.assertEqual(len(rep.errors), 7)
        self.assertFalse(rep.ok)
        # warnings/infos beyond the cap are counted, not stored
        d2 = self.data
        for i in range(7):
            d2 = patch_u32(d2, physical_off(self.pk, i) + 12, 0xBEEF)      # 7 'fc' infos
        rep2 = validate(open_bytes(d2), max_findings=3)
        self.assertTrue(rep2.ok)
        self.assertLessEqual(len(rep2.findings), 3)
        self.assertEqual(rep2.stats["findings_dropped"]["info"], 4)


class TestOnDemandChecks(unittest.TestCase):

    def test_grouped_layout_in_bit12_pack_violates(self):
        """The DLE crash (survey 01 §8) reproduced: storage-grouped payload under field08 = 0x1000."""
        data = _meshes(layout="grouped")
        pk = open_bytes(data)
        rep = validate(pk, check_mesh_names=False)
        self.assertFalse(rep.ok)
        self.assertEqual({f.check for f in rep.errors}, {"ondemand"})
        self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=0, violating=2, incompatible=0))
        msgs = [f.message for f in rep.errors]
        self.assertTrue(all("stored at" in m and "engine slices at" in m for m in msgs), msgs)
        # the same payload declared without bit 12 is not checked at all
        rep = validate(open_bytes(patch_u32(data, 8, 0)), check_mesh_names=False)
        self.assertTrue(rep.ok)

    def test_method0_mesh_in_bit12_pack_incompatible(self):
        data = build_bytes([mesh(b"legacy", ondemand=False), mesh(b"ok", seed=3)], field08=0x1000)
        pk = open_bytes(data)
        rep = validate(pk, check_mesh_names=False)
        self.assertFalse(rep.ok)
        self.assertTrue(_has(rep, "error", "ondemand", "'legacy': not on-demand compatible"))
        self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=1, violating=0, incompatible=1))

    def test_special_bit_makes_incompatible(self):
        data = _meshes()
        pk = open_bytes(data)
        i = pk.resource(2).logical.first_part + 3
        d = patch_u32(data, physical_off(pk, i), pk.physicals[i].packed | PHYS_SPECIAL)
        rep = validate(open_bytes(d), check_mesh_names=False)
        self.assertTrue(_has(rep, "error", "ondemand", "'m1': not on-demand compatible"))
        self.assertTrue(_has(rep, "info", "unknown", "physical bit 12"))
        self.assertEqual(rep.stats["ondemand_meshes"]["incompatible"], 1)

    def test_single_moved_part_reported_precisely(self):
        data = _meshes()
        pk = open_bytes(data)
        r = pk.resource(1)
        i = r.logical.first_part + 4                       # index buffer of m0
        d = patch_u32(data, physical_off(pk, i) + 4, pk.physicals[i].offset_units + 1)
        rep = validate(open_bytes(d), check_mesh_names=False)
        errs = [f for f in rep.errors if f.check == "ondemand"]
        self.assertEqual(len(errs), 1)
        self.assertEqual((errs[0].resource, errs[0].part), (1, i))
        self.assertIn(f"stored at 0x{pk.part_offset(i) + 16:X}, engine slices at 0x{pk.part_offset(i):X}", errs[0].message)

    def test_textures_not_replayed(self):
        """Only logical type 0x10 goes through the mesh slicing (engine +0xCEAF50); textures are grouped."""
        pk = open_bytes(_meshes())
        rep = validate(pk, check_mesh_names=False)
        self.assertEqual(rep.stats["ondemand_meshes"]["total"], 2)
        self.assertTrue(rep.ok)


class TestReportApi(unittest.TestCase):

    def test_report(self):
        rep = Report("x")
        self.assertTrue(rep.ok)
        rep.add("info", "unknown", "m")
        rep.add("error", "owner", "bad", 3, 4)
        self.assertFalse(rep.ok)
        self.assertEqual(rep.errors[0].to_json(), {"level": "error", "check": "owner", "message": "bad", "resource": 3, "part": 4})
        self.assertEqual(Finding("info", "unknown", "m").to_json(), {"level": "info", "check": "unknown", "message": "m"})


if __name__ == "__main__":
    unittest.main()
