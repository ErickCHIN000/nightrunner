"""CLI-level tests: `nr extract` tree layout and pack.json schema, `nr build` byte identity (auto/grouped/preserve),
`--field08` override, incomplete extraction (`--types`, `--limit`) building a valid smaller pack, edited raw parts.

Uses the small shipped packs (skips without them) and writes only under out/test.
"""
from __future__ import annotations

import contextlib
import io
import json
import unittest
from pathlib import Path

from tests.synth import Pack, small_pack, small_packs, tmpdir, payload
from nightrunner.cli import main
from nightrunner.container.validate import validate
from nightrunner.util.hashing import sha256_file
from nightrunner.extract import SCHEMA, PART_FILE_NAMES


def bp(*argv) -> tuple[int, str]:
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        rc = main([str(a) for a in argv])
    return rc, out.getvalue()


class TestExtract(unittest.TestCase):

    def setUp(self):
        self.src = small_pack("reg_pc.rpack")
        if self.src is None:
            self.skipTest("reg_pc.rpack not available")
        self._tmp = tmpdir("xb_")
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_tree_and_pack_json(self):
        rpx = self.dir / "reg.rpx"
        rc, out = bp("extract", self.src, rpx)
        self.assertEqual(rc, 0)
        self.assertIn("extracted 2 resources", out)
        spec = json.loads((rpx / "pack.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["schema"], SCHEMA)
        self.assertEqual(spec["complete"], True)
        self.assertEqual(spec["header"]["field08"], "0x00000000")
        self.assertEqual(spec["header"]["logical_count"], 2)
        self.assertEqual(spec["header"]["flags"], "0x00000001")
        self.assertEqual(spec["source"]["path"], str(self.src))
        self.assertEqual(spec["source"]["size"], 6848)
        self.assertEqual(spec["source"]["sha256"], sha256_file(self.src))
        self.assertEqual([s["type"] for s in spec["storages"]], ["0x5A", "0x61", "0x62"])
        self.assertEqual(spec["name_blob_order"], [0, 1])
        self.assertEqual(spec["warnings"], [])
        res = spec["resources"]
        self.assertEqual([r["index"] for r in res], [0, 1])
        self.assertEqual(res[0]["dir"], "area/000000_dlc_frontier_e_272cx56_area")
        self.assertEqual(res[1]["dir"], "prefab/000001_Prefabs")
        self.assertEqual(res[1]["name_hex"], b"Prefabs".hex())
        self.assertEqual(res[1]["type"], "0x61")
        self.assertEqual(res[1]["flags"], "0x01")
        self.assertEqual(res[1]["name_index"], 1)
        self.assertTrue(res[1]["editable"]["kind"].startswith("raw"), res[1]["editable"])   # raw / raw:prefab (types codec)
        parts = res[1]["parts"]
        self.assertEqual([p["type"] for p in parts], ["0x61", "0x62"])
        self.assertEqual([p["ordinal"] for p in parts], [0, 1])
        self.assertEqual([p["index"] for p in parts], [1, 2])
        self.assertEqual([p["storage_index"] for p in parts], [1, 2])
        self.assertEqual([p["raw"] for p in parts], ["prefab/000001_Prefabs/prefab.bin", "prefab/000001_Prefabs/prefab_fixups.bin"])
        self.assertEqual([p["offset"] for p in parts], [736, 4432])
        self.assertEqual([p["size"] for p in parts], [3696, 2416])
        self.assertEqual([p["flag_bits"] for p in parts], ["0x0000", "0x0000"])
        self.assertEqual([p["fc"] for p in parts], ["0x00000000", "0x00000000"])
        for r in res:
            for p in r["parts"]:
                f = rpx / p["raw"]
                self.assertTrue(f.exists(), f)
                self.assertEqual(f.stat().st_size, p["size"])
                self.assertEqual(sha256_file(f), p["sha256"])
                self.assertEqual(f.name, PART_FILE_NAMES[int(p["type"], 0)])
        self.assertIn("area.bin", {p.name for p in (rpx / "area" / "000000_dlc_frontier_e_272cx56_area").iterdir()})

    def test_types_and_limit(self):
        src = small_pack("reg1_pc.rpack")
        rpx = self.dir / "reg1_areas.rpx"
        rc, _ = bp("extract", src, rpx, "--types", "0x5A")
        self.assertEqual(rc, 0)
        spec = json.loads((rpx / "pack.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["complete"], False)
        self.assertEqual(len(spec["resources"]), 5)
        self.assertTrue(all(r["type"] == "0x5A" for r in spec["resources"]))
        self.assertFalse((rpx / "prefab").exists())
        rc, _ = bp("extract", src, self.dir / "lim.rpx", "--limit", "2")
        spec = json.loads((self.dir / "lim.rpx" / "pack.json").read_text(encoding="utf-8"))
        self.assertEqual([r["index"] for r in spec["resources"]], [0, 1])
        self.assertFalse(spec["complete"])

    def test_reextract_replaces(self):
        rpx = self.dir / "re.rpx"
        bp("extract", self.src, rpx)
        (rpx / "junk.txt").write_text("x")
        rc, _ = bp("extract", self.src, rpx)
        self.assertEqual(rc, 0)
        self.assertFalse((rpx / "junk.txt").exists())
        self.assertFalse((self.dir / "re.rpx.partial").exists())


class TestBuild(unittest.TestCase):

    def setUp(self):
        if not small_packs():
            self.skipTest("small shipped packs not available")
        self._tmp = tmpdir("bld_")
        self.dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _extract(self, name: str) -> tuple[Path, Path]:
        src = small_pack(name)
        rpx = self.dir / (name + ".rpx")
        rc, _ = bp("extract", src, rpx)
        self.assertEqual(rc, 0)
        return src, rpx

    def test_byte_identity_every_small_pack(self):
        for src in small_packs():
            rpx = self.dir / (src.name + ".rpx")
            self.assertEqual(bp("extract", src, rpx)[0], 0, src.name)
            orig = src.read_bytes()
            for layout in ("auto", "preserve", "grouped"):
                out = self.dir / f"{src.stem}.{layout}.rpack"
                rc, text = bp("build", rpx, out, "--layout", layout)
                self.assertEqual(rc, 0, (src.name, layout, text))
                self.assertEqual(out.read_bytes(), orig, (src.name, layout))
                self.assertIn("validation: OK", text)
                res = json.loads(text[: text.index("validation:")])
                self.assertEqual(res["layout"], "preserve" if layout == "auto" else layout)
                self.assertEqual(res["regenerated"], [])
                self.assertEqual(res["warnings"], [])
            self.assertEqual(bp("validate", out)[0], 0)

    def test_field08_override(self):
        src, rpx = self._extract("reg_pc.rpack")
        out = self.dir / "reg_0x1000.rpack"
        rc, text = bp("build", rpx, out, "--field08", "0x1000")
        self.assertEqual(rc, 0, text)
        res = json.loads(text[: text.index("validation:")])
        self.assertEqual(res["layout"], "contiguous")          # auto → contiguous because field08 changed
        with Pack.open(out) as pk:
            self.assertEqual(pk.header.field08, 0x1000)
            self.assertTrue(pk.header.ondemand)
            self.assertEqual(pk.header.flags, 1)
            self.assertEqual([r.name for r in pk], ["dlc_frontier_e_272cx56_area", "Prefabs"])
            rep = validate(pk)
            self.assertTrue(rep.ok)
            self.assertEqual(rep.stats["ondemand_meshes"], dict(total=0, ok=0, violating=0, incompatible=0))
            with Pack.open(src) as spk:
                for i in range(len(pk.physicals)):
                    self.assertEqual(bytes(pk.read_part(i)), bytes(spk.read_part(i)))
        # no bit 12 but explicit grouped → the same bytes as the original
        rc, text = bp("build", rpx, self.dir / "reg_g.rpack", "--layout", "grouped", "--field08", "0")
        self.assertEqual((self.dir / "reg_g.rpack").read_bytes(), src.read_bytes())

    def test_incomplete_extraction_builds_valid_smaller_pack(self):
        src = small_pack("reg1_pc.rpack")
        rpx = self.dir / "areas.rpx"
        self.assertEqual(bp("extract", src, rpx, "--types", "0x5A")[0], 0)
        out = self.dir / "areas.rpack"
        rc, text = bp("build", rpx, out)
        self.assertEqual(rc, 0, text)
        res = json.loads(text[: text.index("validation:")])
        self.assertEqual(res["layout"], "grouped")
        self.assertEqual((res["storages"], res["physicals"], res["logicals"]), (1, 5, 5))
        self.assertEqual(bp("validate", out)[0], 0)
        with Pack.open(out) as pk, Pack.open(src) as spk:
            self.assertEqual([s.type for s in pk.storages], [0x5A])
            self.assertEqual([r.name for r in pk], [r.name for r in spk][:5])
            self.assertEqual(pk.storages[0].count, 5)
            self.assertEqual(pk.storages[0].size, spk.storages[0].size)
            for i in range(5):
                self.assertEqual(bytes(pk.read_part(i)), bytes(spk.read_part(i)))
            self.assertEqual(pk.size, 16 * ((pk.table_end + 15) // 16) + spk.storages[0].size)
            rep = validate(pk)
            self.assertEqual(rep.findings, [])
        # preserve is refused for an incomplete spec
        rc, text = bp("build", rpx, self.dir / "x.rpack", "--layout", "preserve")
        self.assertEqual(rc, 2)

    def test_edited_raw_same_size_keeps_preserve(self):
        src, rpx = self._extract("reg_pc.rpack")
        raw = rpx / "area" / "000000_dlc_frontier_e_272cx56_area" / "area.bin"
        new = payload(raw.stat().st_size, 42)
        raw.write_bytes(new)
        out = self.dir / "edit.rpack"
        rc, text = bp("build", rpx, out)
        self.assertEqual(rc, 0, text)
        self.assertEqual(json.loads(text[: text.index("validation:")])["layout"], "preserve")
        orig = src.read_bytes()
        got = out.read_bytes()
        self.assertEqual(len(got), len(orig))
        with Pack.open(out) as pk:
            self.assertEqual(bytes(pk.read_part(0)), new)
            off = pk.part_offset(0)
            self.assertEqual(got[:off], orig[:off])
            self.assertEqual(got[off + len(new):], orig[off + len(new):])

    def test_edited_raw_changed_size_relayouts(self):
        src, rpx = self._extract("reg_pc.rpack")
        raw = rpx / "area" / "000000_dlc_frontier_e_272cx56_area" / "area.bin"
        new = payload(raw.stat().st_size + 100, 42)
        raw.write_bytes(new)
        out = self.dir / "grow.rpack"
        rc, text = bp("build", rpx, out)
        self.assertEqual(rc, 0, text)
        self.assertEqual(json.loads(text[: text.index("validation:")])["layout"], "grouped")
        with Pack.open(out) as pk:
            self.assertEqual(bytes(pk.read_part(0)), new)
            self.assertEqual(pk.physicals[0].size, 612)
            self.assertEqual(pk.storages[0].size, 624)
            self.assertTrue(validate(pk).ok)
        rc, text = bp("build", rpx, self.dir / "grow_p.rpack", "--layout", "preserve")
        self.assertEqual(rc, 2)                                   # refused: sizes changed
        self.assertFalse((self.dir / "grow_p.rpack").exists())

    def test_missing_raw_refused(self):
        src, rpx = self._extract("reg_pc.rpack")
        (rpx / "prefab" / "000001_Prefabs" / "prefab_fixups.bin").unlink()
        rc, text = bp("build", rpx, self.dir / "m.rpack")
        self.assertEqual(rc, 2)
        self.assertFalse((self.dir / "m.rpack").exists())

    def test_no_validate_flag_and_info_list(self):
        src, rpx = self._extract("reg_buffer_cst_pa_pc.rpack")
        out = self.dir / "nv.rpack"
        rc, text = bp("build", rpx, out, "--no-validate")
        self.assertEqual(rc, 0)
        self.assertNotIn("validation:", text)
        self.assertEqual(out.read_bytes(), src.read_bytes())
        rc, text = bp("info", out)
        self.assertEqual(rc, 0)
        self.assertIn("storages=2 physicals=2 logicals=1", text)
        rc, text = bp("list", out, "--parts")
        self.assertEqual(json.loads(text.strip())["name"], "Prefabs")


class TestNoRawTextures(unittest.TestCase):
    """`nr extract --no-raw` drops the raw copies of texture parts; `nr build` must regenerate them from the DDS /
    sidecar without `--force-codec` (QA 2026-09-15 regression: it failed with "part 0 has no raw file")."""

    def setUp(self):
        import numpy as np
        from tests.synth import ResourceSpec, part, write_pack
        from nightrunner.texture import imgc
        self._tmp = tmpdir("noraw_")
        self.dir = Path(self._tmp.name)
        rng = np.random.default_rng(7)
        hd = imgc.ImgcHeader(flags=0x44)
        hd.set_geometry(16, 8, 1, imgc.TYPE_2D, 5, 38)                                  # RGBA8, full chain
        self.levels = [bytes(rng.integers(0, 256, lv.size, dtype=np.uint8)) for lv in imgc.level_layout(hd)]
        bc = imgc.ImgcHeader(flags=0x64)
        bc.set_geometry(8, 8, 1, imgc.TYPE_2D, 4, 59)                                   # BC1 with the 8-byte tail levels
        bc_levels = [bytes(rng.integers(0, 256, lv.size, dtype=np.uint8)) for lv in imgc.level_layout(bc)]
        ho = imgc.ImgcHeader(flags=0x02, header_size=96, extension=b"default_prj.dds\0")   # header-only record
        self.pack = self.dir / "tex.rpack"
        write_pack([
            ResourceSpec(name=b"rgba.png", type=0x20, flags=1, parts=[part(0x20, imgc.pack_header(hd)), part(0x21, imgc.join_payload(hd, self.levels))]),
            ResourceSpec(name=b"office_lamp_b_prj.dds", type=0x20, flags=1, parts=[part(0x20, imgc.pack_header(ho))]),
            ResourceSpec(name=b"bc1.dds", type=0x20, flags=1, parts=[part(0x20, imgc.pack_header(bc)), part(0x21, imgc.join_payload(bc, bc_levels))]),
            payload_area(),
        ], self.pack)

    def tearDown(self):
        self._tmp.cleanup()

    def test_no_raw_extract_then_build_identity(self):
        rpx = self.dir / "tex.rpx"
        rc, text = bp("extract", self.pack, rpx, "--no-raw")
        self.assertEqual(rc, 0, text)
        spec = json.loads((rpx / "pack.json").read_text(encoding="utf-8"))
        self.assertEqual(spec["warnings"], [])
        tex = [r for r in spec["resources"] if r["type"] == "0x20"]
        self.assertEqual(len(tex), 3)
        for r in tex:
            for p in r["parts"]:
                self.assertIsNone(p["raw"])
                self.assertTrue(p["no_raw"])
                self.assertNotIn("unsupported", p)
        area = [r for r in spec["resources"] if r["type"] == "0x5A"][0]
        self.assertIsNotNone(area["parts"][0]["raw"])                                     # only textures drop raw
        self.assertEqual({p.name for p in (rpx / tex[0]["dir"]).iterdir()}, {"rgba.dds", "rgba.tex.json"})
        self.assertEqual({p.name for p in (rpx / tex[1]["dir"]).iterdir()}, {"office_lamp_b_prj.tex.json"})
        out = self.dir / "tex_out.rpack"
        rc, text = bp("build", rpx, out)
        self.assertEqual(rc, 0, text)
        res = json.loads(text[: text.index("validation:")])
        self.assertEqual(res["layout"], "preserve")                                       # sizes unchanged → byte identity
        self.assertEqual([(r["resource"], r["no_raw"], r["parts"]) for r in res["regenerated"]],
                         [(0, [0, 1], [0, 1]), (1, [0], [0]), (2, [0, 1], [0, 1])])
        self.assertEqual(out.read_bytes(), self.pack.read_bytes())
        # an edited DDS still goes through the codec; a missing DDS is refused, not silently zero-filled
        from nightrunner.texture import imgc, dds
        with Pack.open(self.pack) as spk:
            hd = imgc.parse_header(spk.read_part(0))
        new_levels = [bytes(reversed(self.levels[0]))] + self.levels[1:]
        (rpx / tex[0]["dir"] / "rgba.dds").write_bytes(dds.imgc_to_dds(hd, imgc.join_payload(hd, new_levels)))
        rc, text = bp("build", rpx, self.dir / "tex_edit.rpack")
        self.assertEqual(rc, 0, text)
        with Pack.open(self.dir / "tex_edit.rpack") as pk:
            self.assertEqual(bytes(pk.read_part(1)), imgc.join_payload(hd, new_levels))
        (rpx / tex[0]["dir"] / "rgba.dds").unlink()
        rc, text = bp("build", rpx, self.dir / "tex_missing.rpack")
        self.assertEqual(rc, 2)
        self.assertFalse((self.dir / "tex_missing.rpack").exists())


def payload_area():
    from tests.synth import single
    return single(b"some_area", 0x5A, 300)


if __name__ == "__main__":
    unittest.main()
