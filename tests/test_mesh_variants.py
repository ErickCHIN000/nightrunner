"""Mesh part 0x12 `_SKIN_` (material variants): decode, names, material maps, garbage safety, round trip.

Corpus: the 17-mesh sample pack plus, when present, dlc_ft_prologue_pc (5) and menu_level_ft_pc (3). The synthetic
dlc_stress pack is deliberately not used.
"""
import os
import re
import struct
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from nightrunner.container.rp6l import Pack  # noqa: E402
from nightrunner.mesh import variants as MV  # noqa: E402
from nightrunner.mesh.decode import decode_resource  # noqa: E402
from tests.paths import ASSETS, SAMPLES  # noqa: E402

SAMPLE_PACK = SAMPLES / "meshes.rpack"
SAMPLE_RPX = SAMPLES / "meshes.rpx" / "mesh"
_ASSET_DIRS = [ASSETS, Path(os.environ.get("BP_ASSETS", "/tmp/game/ph_ft/work/data_platform/pc/assets"))]
REAL_PACKS = ("dlc_ft_prologue_pc.rpack", "menu_level_ft_pc.rpack")

WN_NAMES = ["silencer", "no_silencer", "Default", "loot", "olive_plastic", "sand_plastic", "", "Default_hl", "",
            "olive_plastic_hl", "", "sand_plastic_hl"]
WN_PAIRS = {
    "silencer": [(3, "null.mat")],
    "no_silencer": [],
    "Default": [(0, "wn_pistol_b_frame.mat"), (1, "shadow_caster.mat"), (2, "wn_pistol_b_slide.mat"),
                (3, "wn_gunsilencer_a_tpp.mat")],
    "loot": [(0, "loot.mat"), (1, "loot.mat"), (2, "loot.mat")],
    "olive_plastic": [(0, "wn_pistol_b_frame_olive_plastic.mat"), (1, "shadow_caster.mat"),
                      (2, "wn_pistol_b_slide.mat")],
    "Default_hl": [(0, "wn_pistol_b_frame_hl.mat"), (1, "shadow_caster.mat"), (2, "wn_pistol_b_slide_hl.mat"),
                   (3, "wn_gunsilencer_a_tpp_hl.mat")],
}
IDENTITY_DEFAULT = {"npc_b_man_pants_b_holster_bag_c", "prp_hotel_number_10_a", "dlc_ft_safe_zone_cable_e",
                    "sh2_npc_aiden_beast", "sh2_npc_crane", "sh_npc_ft_crane_hair_a", "sh2_npc_ft_crane_beard_a",
                    "dlc_ft_freak_banshee_clothes_matriarch", "anim_hammer_a", "bdp_ce_a_ornament_str_d",
                    "alarm_siren_anm", "barrier", "dlc_ft_prologue_genpin_9x2ee002_lod1", "ui_bg_ph_ft",
                    "ui_bg_the_beast"}


def _packs() -> list[Path]:
    out = [SAMPLE_PACK] if SAMPLE_PACK.exists() else []
    for name in REAL_PACKS:
        for d in _ASSET_DIRS:
            if (d / name).exists():
                out.append(d / name)
                break
    return out


_CACHE: list | None = None


def _corpus() -> list:
    """[(name, part bytes, model)] of every mesh in the available real packs."""
    global _CACHE
    if _CACHE is None:
        _CACHE = []
        for p in _packs():
            pk = Pack.open(p)
            for r in pk.resources_of_type(0x10):
                raw = r.read_part_by_type(0x12)
                if raw is not None:
                    _CACHE.append((r.name.strip(), bytes(raw), decode_resource(r)))
    if not _CACHE:
        raise unittest.SkipTest("no real mesh pack available")
    return _CACHE


def _one(name: str):
    for n, raw, m in _corpus():
        if n == name:
            return raw, m
    raise unittest.SkipTest(f"sample {name!r} missing")


class CorpusTests(unittest.TestCase):
    def test_every_mesh_parses_to_its_end(self):
        for name, raw, m in _corpus():
            with self.subTest(mesh=name):
                r = MV.decode(raw, m)
                self.assertNotIn("error", r)
                self.assertTrue(r["complete"], r["notes"])
                self.assertEqual(r["size"], len(raw))
                # padding rule: zero bytes after the last object, up to the next 16-byte multiple
                self.assertEqual(len(raw), (r["consumed"] + 15) & ~15)
                self.assertFalse(any(raw[r["consumed"]:]))
                self.assertEqual(r["count"], struct.unpack_from("<I", raw)[0])
                self.assertEqual(len(r["variants"]), r["count"])

    def test_round_trip(self):
        for name, raw, m in _corpus():
            with self.subTest(mesh=name):
                self.assertEqual(MV.encode(MV.decode(raw, m)), raw)
                self.assertEqual(MV.encode(MV.decode(raw)), raw)

    def test_names_match_string_blob(self):
        for name, raw, _m in _corpus():
            with self.subTest(mesh=name):
                r = MV.decode(raw)
                names = MV.variant_names(raw)
                self.assertEqual(names, [v["name"] for v in r["variants"]])
                # every maximal NUL-terminated string in the part is a variant name (suffix sharing aside)
                blob = {m.group(1).decode() for m in re.finditer(rb"(?<![\x20-\x7e])([\x20-\x7e]{3,})\x00", raw)}
                self.assertEqual(blob, {n for n in names if n} - {n for n in names if n and any(
                    o != n and o.endswith(n) for o in names)})

    def test_default_present_and_covers_used_slots(self):
        for name, raw, m in _corpus():
            with self.subTest(mesh=name):
                r = MV.decode(raw, m)
                used = sorted({s.material_slot for g in m.geometry_entries for s in g.submeshes})
                if not used:
                    self.assertEqual(r["count"], 0)
                    continue
                default = [v for v in r["variants"] if v["name"] == "Default"]
                self.assertEqual(len(default), 1)
                slots = [e["slot"] for e in default[0]["material_map"]]
                self.assertTrue(set(used) <= set(slots))
                if name in IDENTITY_DEFAULT:
                    self.assertTrue(all(e["slot"] == e["material"] for e in default[0]["material_map"]))

    def test_pair_materials_inside_full_table(self):
        for name, raw, m in _corpus():
            with self.subTest(mesh=name):
                r = MV.decode(raw, m)
                table = r["material_table"]
                refd = set()
                for v in r["variants"]:
                    slots = [e["slot"] for e in v["material_map"]]
                    self.assertEqual(slots, sorted(set(slots)))
                    for e in v["material_map"]:
                        self.assertLess(e["material"], len(table))
                        self.assertLess(e["slot"], r["material_count"])
                        refd.add(e["material"])
                # every variant-only material (count..capacity) is used by some variant
                self.assertTrue(set(range(r["material_count"], len(table))) <= refd)


class WorkedExampleTests(unittest.TestCase):
    def test_wn_pistol(self):
        raw, m = _one("wn_pistol_b_b")
        r = MV.decode(raw, m)
        self.assertEqual(MV.variant_names(raw), WN_NAMES)
        by = {v["name"]: v for v in r["variants"]}
        for vname, pairs in WN_PAIRS.items():
            self.assertEqual([(e["slot"], e["material_name"]) for e in by[vname]["material_map"]], pairs, vname)
        self.assertEqual([v["index"] for v in r["variants"] if v["modifier"]], [0, 1, 6, 8, 10])
        self.assertEqual([x["record"] for x in by["Default"]["refs"]], [0, 1])
        self.assertEqual([x["record"] for x in by["sand_plastic_hl"]["refs"]], [10, 1])
        self.assertEqual([(e["slot"], e["material_name"]) for e in r["variants"][6]["material_map"]],
                         [(3, "null_hl.mat")])
        self.assertEqual(by["silencer"]["material_map"][0]["submeshes"], [[1, 0]])   # the silencer geometry
        self.assertEqual(len(r["unreferenced"]), 3)
        self.assertEqual(r["consumed"], len(raw))

    def test_ornament(self):
        raw, m = _one("bdp_ce_a_ornament_str_d")
        r = MV.decode(raw, m)
        self.assertEqual(r["count"], 116)
        names = MV.variant_names(raw)
        self.assertEqual(names[0], "Default")
        self.assertIn("bld_wall_rustic_planks_dark_a", names)
        self.assertEqual(len(set(names)), 116)
        self.assertTrue(all(not v["material_map"] for v in r["variants"][1:]))
        self.assertEqual([(x["key"], x["value"]) for x in r["variants"][0]["remap"]],
                         [(0, 13), (1, 13), (2, 13), (3, 13), (13, 13)])

    def test_default_only_and_variant_only_material(self):
        raw, m = _one("dummybox_025m")
        r = MV.decode(raw, m)
        self.assertEqual(MV.variant_names(raw), ["Default"])
        e = r["variants"][0]["material_map"][0]
        self.assertEqual((e["slot"], e["slot_name"], e["material"], e["material_name"], e["variant_only"]),
                         (0, "DUMMYBOX.MAT", 2, "dummy.mat", True))

    def test_barrier_differs_only_in_remap(self):
        raw, m = _one("barrier")
        r = MV.decode(raw, m)
        a, b = r["variants"]
        self.assertEqual((a["name"], b["name"]), ("Default", "no_climbing"))
        self.assertEqual(a["material_map"][0]["material"], b["material_map"][0]["material"])
        self.assertEqual([(x["key"], x["value"], x["hi"]) for x in a["remap"]], [(0x12, 0x12, 0x1000)])
        self.assertEqual([(x["key"], x["value"], x["hi"]) for x in b["remap"]], [(0x12, 0x12, 0x1110)])

    def test_skeleton_is_empty(self):
        raw, m = _one("man_basic_skeleton")
        r = MV.decode(raw, m)
        self.assertEqual((r["count"], r["consumed"], r["size"], r["complete"]), (0, 8, 16, True))

    def test_edit_material_value(self):
        raw, m = _one("anim_hammer_a")
        r = MV.decode(raw, m)
        r["variants"][1]["material_map"][0]["material"] = 0
        out = MV.encode(r)
        diff = [i for i in range(len(raw)) if raw[i] != out[i]]
        self.assertEqual(len(diff), 1)
        self.assertEqual(MV.decode(out, m)["variants"][1]["material_map"][0]["material_name"], "anim_hammer_a.mat")
        r["variants"][1]["material_map"].append({"slot": 1, "material": 1})
        with self.assertRaises(ValueError):
            MV.encode(r)


class GarbageTests(unittest.TestCase):
    def test_garbage_returns_error(self):
        import random
        rnd = random.Random(7)
        cases = [b"", b"\x01", b"\x01\x00\x00\x00", struct.pack("<2I", 0xFFFFFFFF, 8),
                 struct.pack("<2I", 1, 8) + b"\xff" * 32, struct.pack("<2I", 1, 3) + bytes(32),
                 struct.pack("<2I", 2, 8) + bytes(16)]
        cases += [bytes(rnd.getrandbits(8) for _ in range(rnd.randrange(0, 300))) for _ in range(300)]
        for c in cases:
            r = MV.decode(c)
            self.assertIsInstance(r, dict)
            if "error" not in r:
                self.assertIn("variants", r)
            self.assertIsInstance(MV.variant_names(c), list)
        for c in cases[:7]:
            self.assertIn("error", MV.decode(c), c.hex())
        self.assertIn("error", MV.decode(None))
        with self.assertRaises(ValueError):
            MV.encode({"error": "x"})

    def test_truncated_real_part(self):
        raw, _m = _one("wn_pistol_b_b")
        for cut in (8, 40, 300, 600, 700):
            self.assertIn("error", MV.decode(raw[:cut]))

    def test_sample_skin_bin_files(self):
        if not SAMPLE_RPX.exists():
            raise unittest.SkipTest("extracted sample tree missing")
        files = sorted(SAMPLE_RPX.glob("*/skin.bin"))
        self.assertTrue(files)
        for f in files:
            raw = f.read_bytes()
            r = MV.decode(raw)
            self.assertTrue(r.get("complete"), f)
            self.assertEqual(MV.encode(r), raw)


if __name__ == "__main__":
    unittest.main()
