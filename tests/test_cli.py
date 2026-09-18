"""Top-level CLI (nightrunner/cli.py): `nr list --sha256`, `nr build --ignore-bones`, help output on a cp1252 stream,
and the `-k` filter of tests/run.py. Sample-based tests use out/samples/meshes.rpx (skip without it)."""
from __future__ import annotations

import contextlib
import io
import json
import shutil
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.cli import build_parser, main  # noqa: E402
from nightrunner.util.hashing import sha256_file  # noqa: E402
from tests.paths import SAMPLES  # noqa: E402
from tests.synth import run_cli, tmpdir  # noqa: E402

SAMPLE_RPX = SAMPLES / "meshes.rpx"
SAMPLE_PACK = SAMPLES / "meshes.rpack"
PANTS = "mesh/000000_npc_b_man_pants_b_holster_bag_c"


def _need_samples():
    if not (SAMPLE_RPX / "pack.json").exists() or not SAMPLE_PACK.exists():
        raise unittest.SkipTest("out/samples/meshes.rpx missing (tools/make_samples_mesh.py)")


class ListSha256Tests(unittest.TestCase):
    def test_parts_records_carry_sha256_matching_pack_json(self):
        _need_samples()
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        rc, out = run_cli("list", SAMPLE_PACK, "--parts", "--sha256", "--limit", "3")
        self.assertEqual(rc, 0)
        recs = [json.loads(l) for l in out.splitlines() if l.strip()]
        self.assertEqual(len(recs), 3)
        for rec in recs:
            entry = spec["resources"][rec["index"]]
            self.assertEqual([p["sha256"] for p in rec["parts"]], [p["sha256"] for p in entry["parts"]])
            for p, prec in zip(rec["parts"], entry["parts"]):
                self.assertEqual(p["sha256"], sha256_file(SAMPLE_RPX / prec["raw"]))

    def test_compact_records_get_a_sha256_list(self):
        _need_samples()
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        rc, out = run_cli("list", SAMPLE_PACK, "--sha256", "--limit", "2")
        self.assertEqual(rc, 0)
        recs = [json.loads(l) for l in out.splitlines() if l.strip()]
        for rec in recs:
            self.assertEqual(rec["sha256"], [p["sha256"] for p in spec["resources"][rec["index"]]["parts"]])
            self.assertEqual(len(rec["sha256"]), len(rec["parts"]))

    def test_without_flag_no_sha256(self):
        _need_samples()
        rc, out = run_cli("list", SAMPLE_PACK, "--parts", "--limit", "1")
        self.assertEqual(rc, 0)
        rec = json.loads(out.strip())
        self.assertNotIn("sha256", rec)
        self.assertTrue(all("sha256" not in p for p in rec["parts"]))


class BuildIgnoreBonesTests(unittest.TestCase):
    """`nr build --ignore-bones` reaches MeshCodec.build as options['ignore_bone_changes'] (the switch of
    `nr mesh import --ignore-bones`): a Cast whose bone moved is refused without it and built with it."""

    def setUp(self):
        _need_samples()
        self._tmp = tmpdir("cli_")
        self.tmp = Path(self._tmp.name)
        spec = json.loads((SAMPLE_RPX / "pack.json").read_text(encoding="utf-8"))
        entry = next(r for r in spec["resources"] if r["dir"] == PANTS)
        self.rpx = self.tmp / "one.rpx"
        shutil.copytree(SAMPLE_RPX / PANTS, self.rpx / PANTS)
        one = dict(spec, resources=[dict(entry, index=0, name_index=0)], complete=False)
        one.pop("name_blob_order", None)
        (self.rpx / "pack.json").write_text(json.dumps(one), encoding="utf-8")
        # move one bone of the Cast beyond the import tolerance (test_mesh_build does the same through the library)
        from nightrunner.cast import castlib
        cast_path = self.rpx / PANTS / "model.cast"
        c = castlib.Cast.load(str(cast_path))
        b = c.Roots()[0].ChildOfType(castlib.Model).Skeleton().Bones()[5]
        lp = list(b.LocalPosition())
        lp[0] += 0.05
        b.SetLocalPosition(lp)
        c.save(str(cast_path))

    def tearDown(self):
        self._tmp.cleanup()

    def test_refused_without_flag_built_with_flag(self):
        out = self.tmp / "a.rpack"
        err = io.StringIO()
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            rc = main(["build", str(self.rpx), str(out)])
        self.assertEqual(rc, 2)
        self.assertIn("bone transforms differ", err.getvalue())
        self.assertFalse(out.exists())

        rc, text = run_cli("build", self.rpx, out, "--ignore-bones")
        self.assertEqual(rc, 0, text)
        self.assertTrue(out.exists())
        rep = json.loads(text[: text.rindex("}") + 1])
        self.assertEqual([r["name"] for r in rep["regenerated"]], [" npc_b_man_pants_b_holster_bag_c"])
        self.assertTrue(any("native skeleton kept" in w for w in rep["warnings"]), rep["warnings"])
        self.assertIn("validation: OK", text)
        # the native skeleton was kept: every part equals the raw one
        rc, listing = run_cli("list", out, "--parts", "--sha256")
        rec = json.loads(listing.strip())
        spec = json.loads((self.rpx / "pack.json").read_text(encoding="utf-8"))
        self.assertEqual([p["sha256"] for p in rec["parts"]], [p["sha256"] for p in spec["resources"][0]["parts"]])

    def test_parser_option(self):
        a = build_parser().parse_args(["build", "x", "y", "--ignore-bones"])
        self.assertTrue(a.ignore_bones)
        a = build_parser().parse_args(["build", "x", "y"])
        self.assertFalse(a.ignore_bones)


class HelpEncodingTests(unittest.TestCase):
    def test_help_prints_on_a_cp1252_stream(self):
        """`python nr.py --help > file` on Windows writes through cp1252; the help text must survive it."""
        buf = io.BytesIO()
        stream = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
        with contextlib.redirect_stdout(stream):
            with self.assertRaises(SystemExit) as cm:
                main(["--help"])
        self.assertEqual(cm.exception.code, 0)
        stream.flush()
        text = buf.getvalue().decode("cp1252")
        self.assertTrue(text.isascii() and "\\u" not in text, text)   # help strings are ASCII (no escapes needed)
        for word in ("info", "list", "validate", "census", "extract", "build", "roundtrip", "mesh", "texture",
                     "sdb", "model", "types", "select"):
            self.assertIn(word, text)
        for cmd in ("list", "build", "extract"):
            buf = io.BytesIO()
            stream = io.TextIOWrapper(buf, encoding="cp1252", errors="strict")
            with contextlib.redirect_stdout(stream):
                with self.assertRaises(SystemExit):
                    main([cmd, "--help"])
            stream.flush()
            self.assertIn("--", buf.getvalue().decode("cp1252"))


class RunnerFilterTests(unittest.TestCase):
    def test_k_filter_selects_by_substring(self):
        from tests import run as runner
        full = runner.build_suite([])
        sub = runner.build_suite(["HelpEncodingTests"])
        ids = [t.id() for t in runner._flatten(sub)]
        self.assertEqual(len(ids), 1)
        self.assertIn("test_cli.HelpEncodingTests.test_help_prints_on_a_cp1252_stream", ids[0])
        self.assertGreater(full.countTestCases(), sub.countTestCases())
        both = runner.build_suite(["helpencoding", "runnerfilter"])
        self.assertEqual(both.countTestCases(), 2)
        self.assertEqual(runner.build_suite(["no_such_test_zzz"]).countTestCases(), 0)


if __name__ == "__main__":
    unittest.main()
