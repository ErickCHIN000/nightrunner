"""Raw-bundle families: structural dumps, fixups parse/serialise identity, extract → build byte identity.

Sample-based tests use F:\\DLTB\\out\\samples\\types\\index.json (made by out/scratch/pull_type_samples.py) and skip
when absent; corpus tests use the game install and skip when it is not there."""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from tests.paths import ASSETS, SAMPLES, have_game

from nightrunner.container.rp6l import Pack
from nightrunner.types import classreader as cr
from nightrunner.types import anim, animscr, area, envprobe, prefab, voxelizer
from nightrunner.types.codecs import structural_dump

SAMPLE_DIRS = [SAMPLES / "types", Path("/mnt/user-data/uploads/DLTB/out/samples/types")]


def _samples():
    for d in SAMPLE_DIRS:
        if (d / "index.json").exists():
            idx = json.loads((d / "index.json").read_text(encoding="utf-8"))
            return d, idx
    return None, None


def _parts(d, entry):
    return [(int(p["type"], 16), (d / p["file"]).read_bytes()) for p in entry["parts"]]


class SampleDumps(unittest.TestCase):
    def setUp(self):
        self.dir, self.index = _samples()
        if self.index is None:
            self.skipTest("no type samples")

    def _entries(self, t):
        return [e for e in self.index if e["type"] == t]

    def test_every_sample_dumps(self):
        for e in self.index:
            d = structural_dump(int(e["type"], 16), _parts(self.dir, e))
            self.assertNotIn("dump_error", d, e["name"])
            self.assertNotEqual(d.get("form"), "unexpected", e["name"])

    def test_anim_relations(self):
        for e in self._entries("0x40"):
            d = anim.dump(_parts(self.dir, e))
            self.assertEqual(d["header"]["magic"], "ANM2")
            lay = d["layout"]
            self.assertTrue(lay["R2_w12_eq_w14"], e["name"])
            self.assertTrue(lay["R4_seg_sum_eq_w12"], e["name"])
            self.assertTrue(lay["R4_tail3_eq_w16_w14_w16"], e["name"])
            if d["form"] == "plain":
                self.assertTrue(d["R3_size_eq_header_plus_16xw08"], e["name"])
            else:
                self.assertTrue(d["R3_header_len_eq_part44"] and d["R3_part45_eq_16xw08"], e["name"])

    def test_static_equals_stream_concat(self):
        static = {e["name"]: e for e in self._entries("0x40") if e["pack"] == "player_anims_static_pc.rpack"}
        stream = {e["name"]: e for e in self._entries("0x40") if e["pack"] == "player_anims_stream_pc.rpack"}
        common = set(static) & set(stream)
        if not common:
            self.skipTest("no static/stream sample pair")
        for n in common:
            plain = _parts(self.dir, static[n])[0][1]
            hd, pd = [b for _, b in _parts(self.dir, stream[n])]
            self.assertEqual(plain, hd + pd, n)

    def test_animscr_legacy_contract(self):
        for e in self._entries("0x42"):
            p0, p1 = [b for _, b in _parts(self.dir, e)]
            sc = animscr.parse_script(p1)
            rc = animscr.parse_records(p0, sc["name_count"])
            self.assertEqual(sc["end"], sc["size"], e["name"])
            self.assertEqual(rc["end"], rc["size"], e["name"])
            self.assertEqual(animscr.serialise_script(sc), p1, e["name"])

    def test_prefab_graph(self):
        for e in self._entries("0x61"):
            g = prefab.parse(_parts(self.dir, e))
            self.assertEqual(sum(g.prefix), g.fx.object_count)
            # `stream_end` is the ABSOLUTE end position inside the blob that cr.parse(blob, offset=8) walked (the
            # 8-byte prefix included), so a stream that fills the part ends at len(blob), not len(blob) - 8.
            # Evidence (QA 2026-09-15, both samples): reg_cst_pc part 0x62 = 30332 bytes, engine_pc = 178068 bytes;
            # with the align-4/align-16 of the secondary image taken relative to the PART start the parse lands on
            # len(blob) exactly and 174/175 resp. 1693/1694 string-pool objects resolve; taken relative to the stream
            # start (offset 8) the secondary image would begin 8 bytes later and overrun the part on both samples.
            # The corpus census (out/reports/types/prefab.json) reports reserialise_identical 23/23 with the same parse.
            self.assertEqual(g.fx.stream_end, len(g.blob))
            self.assertEqual(g.fx.tail, b"")
            self.assertEqual(cr.serialise(g.fx, prefix=g.blob[:8]), g.blob)
            pool, bad = g.string_pool()
            self.assertGreater(len(pool), 0)
            self.assertTrue(all(d["type_name"] for d in g.descriptors()))

    def test_classreader_pairs(self):
        for e in self._entries("0x47") + self._entries("0x49"):
            parts = _parts(self.dir, e)
            for k in range(0, len(parts), 2):
                img, blob = parts[k][1], parts[k + 1][1]
                fx = cr.parse(blob)
                self.assertEqual(fx.stream_end, len(blob), e["name"])
                self.assertEqual(fx.data_size, len(img), e["name"])
                self.assertEqual(cr.serialise(fx), blob, e["name"])

    def test_area_chunks_tile(self):
        for e in self._entries("0x5A"):
            a = area.parse(_parts(self.dir, e)[0][1])
            self.assertEqual(a["form"], "chunked", e["name"])
            self.assertTrue(a["trailing_zero"] and a["trailing_bytes"] < 16, e["name"])
            self.assertEqual(a["chunks"][0]["tag"], "AREA")

    def test_envprobe_tags(self):
        for e in self._entries("0x55"):
            d = envprobe.dump(_parts(self.dir, e))
            tags = [s["tag"] for s in d["sections"]]
            self.assertEqual(tags[0], "ENVBIN_HEADER", e["name"])
            self.assertEqual(tags[-1], "ENVBIN_END", e["name"])

    def test_voxelizer_members(self):
        for e in self._entries("0x56"):
            m = voxelizer.members(_parts(self.dir, e)[0][1])
            self.assertGreater(m["member_count"], 0)
            self.assertEqual(m["members_compact_eq_len"], m["member_count"], e["name"])
            self.assertEqual(m["w00"], -18)


class FixupsSynthetic(unittest.TestCase):
    def test_roundtrip_with_secondary(self):
        import struct
        head = struct.pack("<3I", 32, 2, 0x80000001) + struct.pack("<3I", 0, 0xC0000003, 1) + struct.pack("<3I", 8, 0xB1000000, 0x40000001)
        head += struct.pack("<I", 2) + struct.pack("<2I", 0, 8) + bytes([0, 0x0E])
        pad1 = b"\0" * ((-len(head)) % 4)
        head += pad1 + struct.pack("<I", 24)
        head += b"\0" * ((-len(head)) % 16)
        blob = head + b"secondary-image-bytes!!!"
        fx = cr.parse(blob)
        self.assertTrue(fx.has_secondary)
        self.assertEqual(fx.secondary, b"secondary-image-bytes!!!")
        self.assertEqual(fx.stream_end, len(blob))
        self.assertEqual(cr.serialise(fx), blob)
        self.assertEqual([r.class_id for r in fx.records], [3, 0])
        self.assertTrue(fx.records[1].secondary)


class ExtractBuildIdentity(unittest.TestCase):
    """Extract → build → byte-identical on small packs of the raw families."""

    def _check(self, path: Path):
        tmp = Path(tempfile.mkdtemp(prefix="bp_types_"))
        try:
            from nightrunner.build import build
            from nightrunner.extract import extract
            with Pack.open(path) as pk:
                extract(pk, tmp / "x.rpx", progress=False)
            build(tmp / "x.rpx", tmp / "x.rpack")
            self.assertEqual((tmp / "x.rpack").read_bytes(), path.read_bytes(), path.name)
            sidecars = list((tmp / "x.rpx").rglob("*_*.json"))
            self.assertTrue(any(s.name != "pack.json" for s in sidecars))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_small_packs(self):
        cands = []
        if have_game():
            cands += [ASSETS / "dlc_frontier" / "reg_pc.rpack", ASSETS / "dlc_frontier" / "reg_buffer_cst_pa_pc.rpack",
                      ASSETS / "dlc_ft_prologue_envprobes_pc.rpack"]
        up = Path("/mnt/user-data/uploads/Dying Light The Beast/ph_ft/work/data_platform/pc/assets")
        cands += [up / "dlc_frontier" / "reg_pc.rpack", up / "dlc_ft_prologue_envprobes_pc.rpack"]
        cands = [c for c in cands if c.exists()]
        if not cands:
            self.skipTest("no small packs available")
        for c in cands:
            self._check(c)


class CorpusRoundtrip(unittest.TestCase):
    def test_roundtrip_counts_raw_families(self):
        if not have_game():
            self.skipTest("game not installed")
        from nightrunner.roundtrip import roundtrip_pack
        rep = roundtrip_pack(ASSETS / "player_anims_pc.rpack", limit=200, progress=False)
        self.assertEqual(rep["summary"]["checked"], 200)
        self.assertEqual(rep["summary"]["identical"], 200)


if __name__ == "__main__":
    unittest.main()
