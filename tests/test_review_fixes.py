"""Regression tests for the review-report.md fixes of 2026-09-15 (F4, F5, F6, F7, F8, F11, F13).

F4/F5/F6/F11/F13 use the small synthetic / sample packs; F7 edits a sample Cast (skips without out/samples/meshes.rpx).
"""
from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import tests.synth as S
from nightrunner import codecs
from nightrunner.build import build
from nightrunner.container.rp6l import PackWriter
from nightrunner.errors import BuildError
from nightrunner.extract import extract
from nightrunner.mesh.rebuild import _joints
from tests.paths import SAMPLES


def _two_areas(path: Path, second=b"area_b"):
    S.write_pack([S.single(b"area_a", 0x5A, 64, seed=1), S.single(second, 0x5A, 64, seed=2)], path)


class TestBuildOutputSafety(unittest.TestCase):
    def setUp(self):
        self._tmp = S.tmpdir("rf_")
        self.d = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_f6_failed_build_keeps_previous_output(self):
        src = self.d / "src.rpack"
        # a bit-12 mesh pack built with the grouped layout violates the one-read contract → validation fails
        S.write_pack([S.mesh(b"m", sizes=(200, 40, 100, 1000, 300))], src, field08=0x1000, layout="contiguous")
        with S.Pack.open(src) as pk:
            extract(pk, self.d / "t.rpx", progress=False)
        out = self.d / "out.rpack"
        out.write_bytes(b"previous good output")
        with self.assertRaises(BuildError):
            build(self.d / "t.rpx", out, layout="grouped")
        self.assertEqual(out.read_bytes(), b"previous good output")
        self.assertTrue((self.d / "out.rpack.invalid").exists())
        self.assertFalse((self.d / "out.rpack.partial").exists())

    def test_f6_valid_build_replaces_output(self):
        src = self.d / "src.rpack"
        _two_areas(src)
        with S.Pack.open(src) as pk:
            extract(pk, self.d / "t.rpx", progress=False)
        out = self.d / "out.rpack"
        out.write_bytes(b"old")
        res = build(self.d / "t.rpx", out)          # auto → preserve; gaps read from the source path (F11)
        self.assertEqual(res["layout"], "preserve")
        self.assertEqual(res["output"], str(out))
        self.assertEqual(out.read_bytes(), src.read_bytes())
        self.assertFalse((self.d / "out.rpack.partial").exists())

    def test_f13_partial_removed_on_error(self):
        from nightrunner.container.rp6l import PartSource
        r = S.single(b"a", 0x5A, 64)
        r.parts[0].source = PartSource(path=self.d / "does_not_exist.bin", offset=0, size=64)
        w = PackWriter(0, 1, "auto")
        w.add(r)
        with self.assertRaises(OSError):
            w.write(self.d / "x.rpack")
        self.assertFalse((self.d / "x.rpack.partial").exists())
        self.assertFalse((self.d / "x.rpack").exists())


class TestCodecMissing(unittest.TestCase):
    def test_f4_changed_editable_without_codec_refuses(self):
        pack = SAMPLES / "textures.rpack"
        if not pack.exists():
            self.skipTest("textures.rpack sample missing")
        with S.tmpdir("rf4_") as d:
            d = Path(d)
            with S.Pack.open(pack) as pk:
                spec = extract(pk, d / "t.rpx", progress=False, limit=1)
            ent = spec["resources"][0]
            dds = next(f for f in (d / "t.rpx" / ent["dir"]).iterdir() if f.suffix == ".dds")
            data = bytearray(dds.read_bytes())
            data[-1] ^= 0xFF
            dds.write_bytes(bytes(data))
            saved = codecs._REGISTRY.pop(0x20)
            codecs._IMPORT_ERRORS[0x20] = "simulated: No module named 'numpy'"
            try:
                with self.assertRaisesRegex(BuildError, "codec for type 0x20 unavailable"):
                    build(d / "t.rpx", d / "o.rpack")
            finally:
                codecs._REGISTRY[0x20] = saved
                codecs._IMPORT_ERRORS.pop(0x20, None)
            self.assertFalse((d / "o.rpack").exists())


class TestNoRawFallback(unittest.TestCase):
    def test_f5_codec_failure_keeps_raw(self):
        with S.tmpdir("rf5_") as d:
            d = Path(d)
            src = d / "src.rpack"
            S.write_pack([S.texture(b"bad_tex", header_size=80, bitmap_size=64)], src)   # garbage IMGC header
            with S.Pack.open(src) as pk:
                spec = extract(pk, d / "t.rpx", no_raw=True, progress=False)
            ent = spec["resources"][0]
            self.assertIn("error", ent["editable"])
            self.assertTrue(all(p["raw"] for p in ent["parts"]))
            self.assertTrue(all("no_raw" not in p for p in ent["parts"]))
            build(d / "t.rpx", d / "o.rpack")
            self.assertEqual((d / "o.rpack").read_bytes(), src.read_bytes())


class TestJointsPalette(unittest.TestCase):
    def test_f8_inactive_lane_beyond_palette_zeroed(self):
        mesh = SimpleNamespace(name="m", positions=np.zeros((2, 3), np.float32),
                               weights=np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32),
                               bones=np.array([[3, 0, 0, 0], [3, 0, 0, 0]], np.int64))
        raw = np.array([[0, 1, 2, 0], [0, 0, 0, 0]], np.uint8)
        j = _joints(mesh, np.array([3], np.uint16), raw, np.array([True, True]))
        self.assertEqual(j.tolist(), [[0, 0, 0, 0], [0, 0, 0, 0]])

    def test_f8_inactive_lane_inside_palette_kept(self):
        mesh = SimpleNamespace(name="m", positions=np.zeros((1, 3), np.float32),
                               weights=np.array([[1, 0, 0, 0]], np.float32),
                               bones=np.array([[3, 0, 0, 0]], np.int64))
        raw = np.array([[0, 1, 2, 0]], np.uint8)
        j = _joints(mesh, np.array([3, 7, 9], np.uint16), raw, np.array([True]))
        self.assertEqual(j.tolist(), [[0, 1, 2, 0]])


class TestNonFinite(unittest.TestCase):
    def test_f7_nan_normal_refused(self):
        d = SAMPLES / "meshes.rpx" / "mesh" / "000003_dummybox_025m"
        if not (d / "model.cast").exists():
            self.skipTest("sample mesh missing")
        from nightrunner.cast import castlib
        from nightrunner.mesh.codec import rebuild_from_files
        from tests.test_mesh_build import _parts, _name
        for prop in ("vp", "vn", "wv"):
            c = castlib.Cast.load(str(d / "model.cast"))
            mesh = c.Roots()[0].ChildOfType(castlib.Model).Meshes()[0]
            if prop not in mesh.properties:
                continue
            vals = list(mesh.properties[prop].values)
            vals[1] = float("nan")
            mesh.properties[prop].values = vals
            with S.tmpdir("rf7_") as t:
                p = Path(t) / "e.cast"
                c.save(str(p))
                with self.assertRaisesRegex(BuildError, "non-finite"):
                    rebuild_from_files(p, d / "mesh.json", _parts(d), _name(d))


if __name__ == "__main__":
    unittest.main()
