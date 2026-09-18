"""`nr select`: composing a new spec from several extracted trees, then `nr build` + `nr validate` on the result.

Cloud-runnable: reg_pc.rpx (area + prefab) + reg1_pc.rpx (5 areas), synthetic mesh/texture trees.
PC only: the dlc_ft_prologue_pc mesh+texture composition under field08 = 0x1000 (5 meshes, 18 textures).
"""
from __future__ import annotations

import json
import os
import unittest
from pathlib import Path

from tests import paths
from tests.synth import (
    Pack, align_up, mesh, texture, single, write_pack, small_pack, small_packs, tmpdir, run_cli, payload, STOCK,
)
from nightrunner.container.validate import validate, ondemand_slices
from nightrunner.container.rp6l import STORAGE_FLAG_STREAM, stock_storage_order
from nightrunner.errors import BuildError
from nightrunner.select import compose, write_selection, split_tree_arg


def _spec(rpx: Path) -> dict:
    return json.loads((rpx / "pack.json").read_text(encoding="utf-8"))


class TestSelectShipped(unittest.TestCase):

    def setUp(self):
        if small_pack("reg_pc.rpack") is None or small_pack("reg1_pc.rpack") is None:
            self.skipTest("reg_pc / reg1_pc not available")
        self._tmp = tmpdir("sel_")
        self.dir = Path(self._tmp.name)
        self.reg = self.dir / "reg.rpx"
        self.reg1 = self.dir / "reg1.rpx"
        self.assertEqual(run_cli("extract", small_pack("reg_pc.rpack"), self.reg)[0], 0)
        self.assertEqual(run_cli("extract", small_pack("reg1_pc.rpack"), self.reg1)[0], 0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_compose_build_validate(self):
        out = self.dir / "new.rpx"
        rc, text = run_cli("select", self.reg, f"{self.reg1}:0-4", "--out", out)
        self.assertEqual(rc, 0, text)
        self.assertIn("selected 7 resources / 8 parts / 3 storages from 2 tree(s)", text)
        spec = _spec(out)
        self.assertEqual(spec["schema"], "nightrunner.rpx/1")
        self.assertFalse(spec["complete"])
        self.assertNotIn("name_blob_order", spec)
        self.assertEqual(spec["header"]["field08"], "0x00000000")
        self.assertEqual(spec["header"]["flags"], "0x00000001")
        self.assertEqual((spec["header"]["logical_count"], spec["header"]["physical_count"], spec["header"]["storage_count"]), (7, 8, 3))
        self.assertEqual([s["type"] for s in spec["storages"]], ["0x5A", "0x61", "0x62"])
        self.assertEqual([s["count"] for s in spec["storages"]], [6, 1, 1])
        self.assertIsInstance(spec["source"], list)
        self.assertEqual([s["resources"] for s in spec["source"]], [2, 5])
        res = spec["resources"]
        self.assertEqual([r["index"] for r in res], list(range(7)))
        self.assertEqual([r["name_index"] for r in res], list(range(7)))
        self.assertEqual([r["type"] for r in res], ["0x5A", "0x61"] + ["0x5A"] * 5)
        self.assertEqual(res[1]["dir"], "prefab/000001_Prefabs")
        self.assertEqual(res[2]["dir"], "area/000002_dlc_ft_prologue_genpin_9x2ee002_area")
        self.assertEqual(res[2]["source"], {"tree": str(self.reg1), "index": 0, "name": "dlc_ft_prologue_genpin_9x2ee002_area",
                                            "dir": "area/000000_dlc_ft_prologue_genpin_9x2ee002_area"})
        self.assertEqual([p["storage_index"] for p in res[1]["parts"]], [1, 2])
        self.assertEqual([p["raw"] for p in res[1]["parts"]], ["prefab/000001_Prefabs/prefab.bin", "prefab/000001_Prefabs/prefab_fixups.bin"])
        self.assertEqual([p["index"] for r in res for p in r["parts"]], list(range(8)))
        for r in res:
            self.assertNotIn("offset_units", r["parts"][0])
            self.assertNotIn("needs_identity_fix", r)
            for p in r["parts"]:
                self.assertTrue((out / p["raw"]).exists())
                self.assertEqual((out / p["raw"]).stat().st_size, p["size"])
        self.assertEqual(spec["warnings"], [])
        # build + validate
        pack = self.dir / "new.rpack"
        rc, text = run_cli("build", out, pack)
        self.assertEqual(rc, 0, text)
        self.assertIn("validation: OK", text)
        self.assertEqual(json.loads(text[: text.index("validation:")])["layout"], "grouped")
        self.assertEqual(run_cli("validate", pack)[0], 0)
        with Pack.open(pack) as pk, Pack.open(small_pack("reg_pc.rpack")) as a, Pack.open(small_pack("reg1_pc.rpack")) as b:
            self.assertEqual([r.name for r in pk], [r.name for r in a] + [r.name for r in b][:5])
            self.assertEqual([s.type for s in pk.storages], [0x5A, 0x61, 0x62])
            self.assertEqual(bytes(pk.read_part(0)), bytes(a.read_part(0)))
            self.assertEqual(bytes(pk.read_part(1)), bytes(a.read_part(1)))
            self.assertEqual(bytes(pk.read_part(2)), bytes(a.read_part(2)))
            for k in range(5):
                self.assertEqual(bytes(pk.read_part(3 + k)), bytes(b.read_part(k)))
            self.assertEqual(pk.storages[0].size, sum(align_up(pk.physicals[i].size, 16) for i in (0, 3, 4, 5, 6, 7)))
            rep = validate(pk)
            self.assertEqual(rep.findings, [])
            self.assertEqual(pk.find("prefabs"), [1])

    def test_rename_and_replace(self):
        out = self.dir / "rr.rpx"
        rc, text = run_cli("select", self.reg, f"{self.reg1}:dlc_ft_prologue_genpin_9x2ee002_area,4", "--out", out,
                           "--rename", "Prefabs=MyPrefabs", "--replace", f"dlc_frontier_e_272cx56_area={self.reg1}:4")
        self.assertEqual(rc, 0, text)
        spec = _spec(out)
        names = [r["name"] for r in spec["resources"]]
        self.assertEqual(names, ["dlc_frontier_e_272cx56_area", "MyPrefabs", "dlc_ft_prologue_genpin_9x2ee002_area", "dlc_ft_prologue_genpin_9x2ee007_area"])
        self.assertEqual(spec["resources"][1]["name_hex"], b"MyPrefabs".hex())
        self.assertEqual(spec["resources"][1]["dir"], "prefab/000001_MyPrefabs")
        self.assertTrue((out / "prefab/000001_MyPrefabs/prefab.bin").exists())
        r0 = spec["resources"][0]
        self.assertEqual(r0["source"]["index"], 4)            # bytes from reg1:4 ...
        self.assertEqual(r0["source"]["tree"], str(self.reg1))
        self.assertEqual(r0["parts"][0]["size"], 329696)      # ... (its size) ...
        self.assertEqual(r0["name"], "dlc_frontier_e_272cx56_area")   # ... under the kept name
        self.assertNotIn("needs_identity_fix", r0)           # not a mesh
        pack = self.dir / "rr.rpack"
        rc, text = run_cli("build", out, pack)
        self.assertEqual(rc, 0, text)
        with Pack.open(pack) as pk, Pack.open(small_pack("reg1_pc.rpack")) as b:
            self.assertEqual(pk.find("myprefabs"), [1])
            self.assertEqual(bytes(pk.read_part(0)), bytes(b.read_part(4)))
            self.assertEqual(bytes(pk.read_part(0)), bytes(pk.read_part(4)))   # same bytes twice under two names
            self.assertTrue(validate(pk).ok)

    def test_selectors(self):
        out = self.dir / "s.rpx"
        rc, text = run_cli("select", f"{self.reg1}:type=0x61,*", f"{self.reg}:1", "--out", out)
        self.assertEqual(rc, 0, text)
        spec = _spec(out)
        self.assertEqual([r["source"]["index"] for r in spec["resources"]], [5, 0, 1, 2, 3, 4, 1])
        self.assertEqual(len([w for w in spec["warnings"] if "duplicate" in w]), 1)    # Prefabs twice
        rc, _ = run_cli("select", f"{self.reg1}:PREFABS", "--out", self.dir / "f.rpx")   # engine case folding
        self.assertEqual(rc, 0)
        self.assertEqual([r["name"] for r in _spec(self.dir / "f.rpx")["resources"]], ["Prefabs"])
        rc, _ = run_cli("select", "--", f"{self.reg1}:2", "--out", self.dir / "dd.rpx")   # `--` form
        self.assertEqual(rc, 0)

    def test_errors(self):
        for argv in (
            [f"{self.reg}:nope"],                                            # unknown name
            [f"{self.reg}:9"],                                               # unknown index
            [f"{self.reg}:type=0x10"],                                       # no such type
            [str(self.dir / "missing.rpx")],                                 # tree missing
            [str(self.reg), "--rename", "nope=x"],                           # rename target missing
            [str(self.reg), "--rename", "Prefabs="],                         # empty new name
            [str(self.reg), "--replace", f"Prefabs={self.reg1}:0"],          # type mismatch (0x61 vs 0x5A)
            [str(self.reg), "--replace", f"Prefabs={self.reg1}:0-4"],        # not exactly one
            [str(self.reg), "--replace", "Prefabs=nope"],                    # bad syntax
            [str(self.reg), str(self.reg), "--rename", "Prefabs=x"],         # ambiguous
        ):
            rc, _ = run_cli("select", *argv, "--out", self.dir / "err.rpx")
            self.assertEqual(rc, 2, argv)
            self.assertFalse((self.dir / "err.rpx").exists(), argv)

    def test_link(self):
        probe = self.dir / "lnk_probe"
        (self.dir / "lnk_src").write_bytes(b"x")
        try:
            os.link(self.dir / "lnk_src", probe)
        except OSError:
            self.skipTest("hard links unsupported here")
        out = self.dir / "linked.rpx"
        rc, text = run_cli("select", self.reg, "--out", out, "--link")
        self.assertEqual(rc, 0, text)
        self.assertIn("linked", text)
        src = self.reg / "prefab/000001_Prefabs/prefab.bin"
        dst = out / "prefab/000001_Prefabs/prefab.bin"
        self.assertTrue(os.path.samefile(src, dst))
        rc, _ = run_cli("build", out, self.dir / "linked.rpack")
        self.assertEqual(rc, 0)

    def test_split_tree_arg(self):
        self.assertEqual(split_tree_arg(str(self.reg)), (self.reg, None))
        self.assertEqual(split_tree_arg(f"{self.reg}:1,Prefabs"), (self.reg, "1,Prefabs"))
        with self.assertRaises(BuildError):
            split_tree_arg(str(self.dir / "nope.rpx"))


class TestSelectSynthetic(unittest.TestCase):
    """Mesh/texture trees from synthetic packs (embedded .msh names absent → validate without the name check)."""

    def setUp(self):
        self._tmp = tmpdir("selm_")
        self.dir = Path(self._tmp.name)
        self.mesh_pack = self.dir / "m.rpack"
        write_pack([texture(b"tex_a", 80, 1000, seed=1), mesh(b"mesh_a", seed=2), mesh(b"mesh_b", (50, 20, 30, 500, 100), seed=3)],
                   self.mesh_pack, field08=0x1000)
        self.tex_pack = self.dir / "t.rpack"
        write_pack([texture(b"tex_b", 80, 700, seed=4), texture(b"tex_a", 80, 900, seed=5), single(b"area", 0x5A, 300)], self.tex_pack)
        self.mrpx = self.dir / "m.rpx"
        self.trpx = self.dir / "t.rpx"
        self.assertEqual(run_cli("extract", self.mesh_pack, self.mrpx)[0], 0)
        self.assertEqual(run_cli("extract", self.tex_pack, self.trpx)[0], 0)

    def tearDown(self):
        self._tmp.cleanup()

    def test_field08_disagreement_needs_override(self):
        rc, _ = run_cli("select", self.mrpx, self.trpx, "--out", self.dir / "x.rpx")
        self.assertEqual(rc, 2)
        rc, text = run_cli("select", f"{self.trpx}:tex_b", f"{self.mrpx}:mesh_b,mesh_a", "--out", self.dir / "x.rpx", "--field08", "0x1000")
        self.assertEqual(rc, 0, text)
        spec = _spec(self.dir / "x.rpx")
        self.assertEqual(spec["header"]["field08"], "0x00001000")
        self.assertEqual([s["type"] for s in spec["storages"]], ["0x10", "0x11", "0x12", "0xF0", "0xF1", "0x20", "0x21"])
        self.assertEqual([r["name"] for r in spec["resources"]], ["tex_b", "mesh_b", "mesh_a"])
        self.assertEqual([p["storage_index"] for p in spec["resources"][1]["parts"]], [0, 2, 1, 3, 4])
        self.assertEqual([p["flag_bits"] for p in spec["resources"][1]["parts"]], ["0x0100"] * 5)
        pack = self.dir / "x.rpack"
        rc, text = run_cli("build", self.dir / "x.rpx", pack, "--no-validate")
        self.assertEqual(rc, 0, text)
        self.assertEqual(json.loads(text)["layout"], "contiguous")
        with Pack.open(pack) as pk, Pack.open(self.mesh_pack) as src:
            rep = validate(pk, check_mesh_names=False)
            self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
            self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=2, violating=0, incompatible=0))
            # textures grouped first, meshes contiguous after (stock rule)
            tex_end = max(pk.part_offset(i) + pk.physicals[i].size for i in pk.resource(0).part_indices)
            mesh_start = min(pk.part_offset(i) for r in pk.resources_of_type(0x10) for i in r.part_indices)
            self.assertLess(tex_end, mesh_start + 1)
            self.assertEqual([s.base_units for s in pk.storages][:5], [0] * 5)
            self.assertTrue(all(s.base_units > 0 for s in pk.storages[5:]))
            # bytes: mesh_b's parts come from the source mesh_b (index 2 there)
            for k, i in enumerate(pk.resource(1).part_indices):
                self.assertEqual(bytes(pk.read_part(i)), bytes(src.read_part(list(src.resource(2).part_indices)[k])))

    def test_rename_mesh_flags_identity_fix(self):
        rc, text = run_cli("select", f"{self.mrpx}:mesh_a", "--out", self.dir / "r.rpx", "--rename", "mesh_a=hero")
        self.assertEqual(rc, 0, text)
        spec = _spec(self.dir / "r.rpx")
        r = spec["resources"][0]
        self.assertEqual((r["name"], r["name_hex"], r["dir"]), ("hero", b"hero".hex(), "mesh/000000_hero"))
        self.assertTrue(r["needs_identity_fix"])
        self.assertEqual(r["source"]["name"], "mesh_a")
        self.assertTrue((self.dir / "r.rpx" / "mesh/000000_hero/image.bin").exists())
        # --replace of a mesh by another mesh flags it too; texture replace does not
        rc, _ = run_cli("select", self.mrpx, "--out", self.dir / "r2.rpx", "--replace", f"mesh_a={self.mrpx}:mesh_b",
                        "--replace", f"tex_a={self.trpx}:tex_b")
        self.assertEqual(rc, 2)                       # the replacement tree is a source too: field08 0 vs 0x1000
        rc, _ = run_cli("select", self.mrpx, "--out", self.dir / "r2.rpx", "--replace", f"mesh_a={self.mrpx}:mesh_b",
                        "--replace", f"tex_a={self.trpx}:tex_b", "--field08", "0x1000")
        self.assertEqual(rc, 0)
        spec = _spec(self.dir / "r2.rpx")
        by = {r["name"]: r for r in spec["resources"]}
        self.assertTrue(by["mesh_a"].get("needs_identity_fix"))
        self.assertEqual(by["mesh_a"]["source"]["name"], "mesh_b")
        self.assertNotIn("needs_identity_fix", by["tex_a"])
        self.assertEqual(by["tex_a"]["source"]["name"], "tex_b")
        self.assertEqual(by["tex_a"]["parts"][1]["size"], 700)
        # replacing a mesh with itself under the same name is not flagged
        rc, _ = run_cli("select", self.mrpx, "--out", self.dir / "r3.rpx", "--replace", f"mesh_b={self.mrpx}:mesh_b")
        self.assertNotIn("needs_identity_fix", {r["name"]: r for r in _spec(self.dir / "r3.rpx")["resources"]}["mesh_b"])

    def test_api_compose(self):
        picks, info, warnings = compose([f"{self.mrpx}:type=0x10", str(self.trpx)], field08=0)
        self.assertEqual([p.name for p in picks], ["mesh_a", "mesh_b", "tex_b", "tex_a", "area"])
        self.assertEqual(info["field08"], 0)
        spec = write_selection(picks, info, self.dir / "api.rpx", warnings=warnings)
        self.assertEqual(spec["header"]["logical_count"], 5)
        with self.assertRaises(BuildError):
            compose([f"{self.mrpx}:mesh_a"], renames=["mesh_a=bad\0name"])
        with self.assertRaises(BuildError):
            compose([])


class TestSelectRenameBuild(unittest.TestCase):
    """`--rename` / `--replace` on real meshes (out/samples/meshes.rpack): `nr build` rewrites the embedded `.msh`
    name for entries flagged needs_identity_fix, so the validator's mesh_name check passes and every other part
    stays byte-identical (QA 2026-09-15 regression for build._apply_identity_fix)."""

    def setUp(self):
        cands = [paths.SAMPLES / "meshes.rpack", Path("/mnt/user-data/uploads/DLTB/out/samples/meshes.rpack")]
        self.src = next((p for p in cands if p.exists()), None)
        if self.src is None:
            self.skipTest("out/samples/meshes.rpack missing (tools/make_samples_mesh.py)")
        self._tmp = tmpdir("selrn_")
        self.dir = Path(self._tmp.name)
        self.rpx = self.dir / "meshes.rpx"
        rc, text = run_cli("extract", self.src, self.rpx, "--limit", "4")
        self.assertEqual(rc, 0, text)

    def tearDown(self):
        self._tmp.cleanup()

    def test_rename_rewrites_embedded_name(self):
        from nightrunner.classreader.image import embedded_mesh_name
        out = self.dir / "rn.rpx"
        rc, text = run_cli("select", f"{self.rpx}:2,3", "--out", out, "--rename", "dlc_ft_safe_zone_cable_e=my_custom_cable")
        self.assertEqual(rc, 0, text)
        spec = _spec(out)
        self.assertTrue(spec["resources"][0]["needs_identity_fix"])
        self.assertNotIn("needs_identity_fix", spec["resources"][1])
        pack = self.dir / "rn.rpack"
        rc, text = run_cli("build", out, pack)
        self.assertEqual(rc, 0, text)
        self.assertIn("validation: OK", text)
        res = json.loads(text[: text.index("validation:")])
        self.assertEqual(res["layout"], "contiguous")
        rg = [r for r in res["regenerated"] if r.get("identity")]
        self.assertEqual(len(rg), 1)
        self.assertEqual((rg[0]["resource"], rg[0]["parts"]), (0, [0, 2]))            # image ordinal 0, fixups ordinal 2
        self.assertEqual(rg[0]["identity"], {"before": "dlc_ft_safe_zone_cable_e.msh", "after": "my_custom_cable.msh", "changed": True})
        with Pack.open(pack) as pk, Pack.open(self.src) as src:
            rep = validate(pk)
            self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
            self.assertEqual(rep.stats["mesh_names"], dict(checked=2, mismatched=0))
            self.assertEqual(rep.stats["ondemand_meshes"], dict(total=2, ok=2, violating=0, incompatible=0))
            r0 = pk.resource(0)
            self.assertEqual(r0.name, "my_custom_cable")
            self.assertEqual(embedded_mesh_name(pk.read_part(r0.part_by_type(0x10)), pk.read_part(r0.part_by_type(0x11))),
                             b"my_custom_cable.msh")
            s2 = src.resource(2)
            for t in (0x12, 0xF0, 0xF1):                                                # untouched parts identical
                self.assertEqual(bytes(pk.read_part(r0.part_by_type(t))), bytes(src.read_part(s2.part_by_type(t))), hex(t))
            self.assertGreater(pk.physicals[r0.part_by_type(0x10)].size, src.physicals[s2.part_by_type(0x10)].size)
            # the unrenamed mesh is byte-identical in every part
            r1, s3 = pk.resource(1), src.resource(3)
            for k, i in enumerate(r1.part_indices):
                self.assertEqual(bytes(pk.read_part(i)), bytes(src.read_part(list(s3.part_indices)[k])), k)
        # a rebuild of the same spec is deterministic
        rc, _ = run_cli("build", out, self.dir / "rn2.rpack")
        self.assertEqual(rc, 0)
        self.assertEqual((self.dir / "rn2.rpack").read_bytes(), pack.read_bytes())

    def test_replace_keeps_name_and_fixes_identity(self):
        from nightrunner.classreader.image import embedded_mesh_name
        out = self.dir / "rp.rpx"
        rc, text = run_cli("select", f"{self.rpx}:2,3", "--out", out, "--replace", f"dummybox_025m={self.rpx}:2")
        self.assertEqual(rc, 0, text)
        spec = _spec(out)
        self.assertEqual([r["name"] for r in spec["resources"]], ["dlc_ft_safe_zone_cable_e", "dummybox_025m"])
        self.assertTrue(spec["resources"][1]["needs_identity_fix"])
        self.assertEqual(spec["resources"][1]["source"]["name"], "dlc_ft_safe_zone_cable_e")
        pack = self.dir / "rp.rpack"
        rc, text = run_cli("build", out, pack)
        self.assertEqual(rc, 0, text)
        with Pack.open(pack) as pk, Pack.open(self.src) as src:
            self.assertTrue(validate(pk).ok)
            r1 = pk.resource(1)
            self.assertEqual(embedded_mesh_name(pk.read_part(r1.part_by_type(0x10)), pk.read_part(r1.part_by_type(0x11))),
                             b"dummybox_025m.msh")
            s2 = src.resource(2)
            self.assertEqual(bytes(pk.read_part(r1.part_by_type(0xF0))), bytes(src.read_part(s2.part_by_type(0xF0))))
            self.assertEqual(pk.find("dummybox_025m"), [1])


class TestSelectPrologue(unittest.TestCase):
    """PC only: dlc_ft_prologue_pc.rpack (5 meshes + 18 textures, field08 0x1000) recomposed through nr select."""

    def setUp(self):
        if not paths.have_game():
            self.skipTest("game not installed")
        cands = [paths.ASSETS / "dlc_ft_prologue_pc.rpack"] + sorted(paths.ASSETS.rglob("dlc_ft_prologue_pc.rpack"))
        self.src = next((p for p in cands if p.exists()), None)
        if self.src is None:
            self.skipTest("dlc_ft_prologue_pc.rpack missing")
        self.rpx = paths.OUT / "prologue.rpx"
        if not (self.rpx / "pack.json").exists():
            rc, text = run_cli("extract", self.src, self.rpx)
            self.assertEqual(rc, 0, text)

    def test_mesh_texture_composition(self):
        out = paths.OUT / "prologue_sel.rpx"
        rc, text = run_cli("select", self.rpx, "--out", out, "--field08", "0x1000")
        self.assertEqual(rc, 0, text)
        spec = _spec(out)
        self.assertEqual(spec["header"]["logical_count"], 23)
        self.assertEqual([s["type"] for s in spec["storages"]], ["0x10", "0x11", "0x12", "0xF0", "0xF1", "0x20", "0x21"])
        pack = paths.OUT / "prologue_sel.rpack"
        rc, text = run_cli("build", out, pack)
        self.assertEqual(rc, 0, text)
        self.assertIn("validation: OK", text)
        self.assertEqual(json.loads(text[: text.index("validation:")])["layout"], "contiguous")
        self.assertEqual(run_cli("validate", pack)[0], 0)
        with Pack.open(pack) as pk, Pack.open(self.src) as src:
            rep = validate(pk)
            self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
            self.assertEqual(rep.stats["ondemand_meshes"], dict(total=5, ok=5, violating=0, incompatible=0))
            for r in pk.resources_of_type(0x10):
                expected, _ = ondemand_slices(pk, r.index)
                self.assertEqual(expected, [pk.part_offset(i) for i in r.part_indices], r.name)
            # stock layout rule: texture groups first as contiguous regions, meshes contiguous after
            keys = [s.key for s in pk.storages]
            self.assertEqual(keys, stock_storage_order(keys))
            self.assertEqual([s.base_units for s in pk.storages[:5]], [0] * 5)
            tex_parts = [i for r in pk.resources_of_type(0x20) for i in r.part_indices]
            mesh_parts = [i for r in pk.resources_of_type(0x10) for i in r.part_indices]
            tex_end = max(pk.part_offset(i) + pk.physicals[i].size for i in tex_parts)
            self.assertLessEqual(tex_end, min(pk.part_offset(i) for i in mesh_parts))
            for si in (5, 6):
                members = [i for i in tex_parts if pk.physicals[i].storage_index == si]
                offs = sorted(pk.part_offset(i) for i in members)
                self.assertEqual(offs[0], pk.storages[si].base_offset)
                self.assertEqual(pk.storages[si].size, sum(align_up(pk.physicals[i].size, 16) for i in members))
            # same resources, same order, same bytes as the shipped pack; only the name-blob order was dropped
            self.assertEqual([r.name_raw for r in pk], [r.name_raw for r in src])
            self.assertEqual([pk.part_offset(i) for i in range(len(pk.physicals))], [src.part_offset(i) for i in range(len(src.physicals))])
            self.assertEqual(pk.storages, src.storages)
            for i in range(len(pk.physicals)):
                self.assertEqual(pk.physicals[i], src.physicals[i])
                self.assertEqual(bytes(pk.read_part(i)), bytes(src.read_part(i)), i)
            self.assertEqual(pk.name_blob_order(), list(range(23)))
            self.assertNotEqual(src.name_blob_order(), list(range(23)))
        # a mesh-only selection with a renamed mesh is flagged, and the two-mesh subset stays on-demand valid
        out2 = paths.OUT / "prologue_two.rpx"
        rc, text = run_cli("select", f"{self.rpx}:type=0x20", f"{self.rpx}:0,2", "--out", out2, "--field08", "0x1000")
        self.assertEqual(rc, 0, text)
        rc, text = run_cli("build", out2, paths.OUT / "prologue_two.rpack")
        self.assertEqual(rc, 0, text)
        with Pack.open(paths.OUT / "prologue_two.rpack") as pk:
            rep = validate(pk)
            self.assertTrue(rep.ok, [f.to_json() for f in rep.errors])
            self.assertEqual(rep.stats["ondemand_meshes"]["ok"], 2)
            self.assertEqual(pk.type_histogram(), {0x10: 2, 0x20: 18})


if __name__ == "__main__":
    unittest.main()
