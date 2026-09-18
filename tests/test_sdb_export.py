"""Whole-database JSON dumps (nightrunner/sdb/export.py).

The cleaning rules are tested against hand-built reader output, so they run without the game. The corpus class at
the end re-dumps the real database and checks the numbers this module's comments claim.
"""
from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner.sdb import export as E  # noqa: E402
from tests.paths import ASSETS, have_game  # noqa: E402
from tests.synth import tmpdir  # noqa: E402


def binding(param, texture, source="override", **extra):
    return {"binding": 0, "param_id": 1, "param": param, "texture": texture, "string_index": 7,
            "source": source, **extra}


def variant(selector, shader, passes, bindings):
    return {"selector": selector, "selector_hex": f"0x{selector:016X}", "shader": shader, "texture_array": 3,
            "render_pass_count": passes, "texture_bindings": bindings}


def param(pid, name, value, type_name="float", declared=True):
    return {"id": pid, "name": name, "offset": pid * 4, "flag_bit31": False, "raw": 12345, "declared": declared,
            "type": type_name, "value": value, "value_hex": "0000803f"}


def material(name="m.mat", index=1, routes=None):
    return {"index": index, "name": name, "database": "runtime_dx11.sdb", "non_rendering": False,
            "routes": routes if routes is not None else [route()]}


def route(preset="opaque", params=None, variants=None):
    return {"index": 0, "material": 1, "program": 2, "tokens_index": 3, "slots": 4, "values": 5,
            "tokens": f"{preset};cast_shadows_on;", "preset": preset, "preset_indices": [9],
            "parameters": params if params is not None else [param(1, "a", 1.0)],
            "variants": variants if variants is not None else [variant(1, 10, 1, [binding("dif_0_tex", "a.png")])]}


class CleanMaterialTests(unittest.TestCase):
    def test_no_raw_fields_survive(self):
        c = E.clean_material(material())
        blob = json.dumps(c)
        for banned in ("value_hex", "raw", "offset", "string_index", "texture_array", "tokens_index",
                       "preset_indices", "binding", "param_id", "selector_hex", "database"):
            self.assertNotIn(f'"{banned}"', blob, banned)

    def test_single_route_is_flattened(self):
        c = E.clean_material(material())
        self.assertNotIn("routes", c)                       # census: every material has exactly one route
        self.assertEqual(c["preset"], "opaque")
        self.assertEqual(c["variant_count"], 1)
        self.assertIn("parameters", c)

    def test_several_routes_keep_the_array(self):
        c = E.clean_material(material(routes=[route("opaque"), route("alpha")]))
        self.assertIn("routes", c)                          # never silently truncated
        self.assertEqual(len(c["routes"]), 2)
        self.assertEqual(c["presets"], ["opaque", "alpha"])

    def test_textures_are_collected_sorted_and_unique(self):
        r = route(variants=[variant(1, 10, 1, [binding("dif_0_tex", "b.png"), binding("nrm_0_tex", "a.png")]),
                            variant(2, 11, 1, [binding("dif_0_tex", "b.png")])])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual(c["textures"], ["a.png", "b.png"])

    def test_variants_with_the_same_bindings_are_grouped(self):
        same = [binding("dif_0_tex", "a.png")]
        r = route(variants=[variant(1, 10, 1, same), variant(2, 11, 1, list(same)),
                            variant(3, 12, 1, [binding("dif_0_tex", "b.png")])])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual(len(c["variants"]), 2)             # two distinct binding sets
        self.assertEqual(c["variant_count"], 3)             # but all three selectors are kept
        self.assertEqual([s["shader"] for s in c["variants"][0]["selectors"]], [10, 11])
        self.assertEqual(c["variants"][0]["selectors"][0]["selector"], "0x0000000000000001")

    def test_variants_differing_only_in_render_passes_stay_apart(self):
        same = [binding("dif_0_tex", "a.png")]
        r = route(variants=[variant(1, 10, 1, same), variant(2, 11, 0, list(same))])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual(len(c["variants"]), 2)

    def test_exact_duplicate_parameters_collapse(self):
        r = route(params=[param(1, "a", 1.0), param(1, "a", 1.0), param(2, "b", 2.0)])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual([(p["id"], p["value"]) for p in c["parameters"]], [(1, 1.0), (2, 2.0)])

    def test_conflicting_duplicate_parameters_are_both_kept(self):
        """3 materials in the shipped database give one id two different values; neither may be dropped."""
        r = route(params=[param(1, "det_0_a_rot_ang", 90.0), param(1, "det_0_a_rot_ang", 270.0)])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual([p["value"] for p in c["parameters"]], [90.0, 270.0])

    def test_unresolved_texture_keeps_its_runtime_index(self):
        r = route(variants=[variant(1, 10, 1, [binding("dif_0_tex", None, runtime_index=4)])])
        c = E.clean_material(material(routes=[r]))
        b = c["variants"][0]["textures"][0]
        self.assertIsNone(b["texture"])
        self.assertEqual(b["runtime_index"], 4)

    def test_shader_default_bindings_are_marked(self):
        r = route(variants=[variant(1, 10, 1, [binding("dif_0_tex", "a.png", source="shader_default")])])
        c = E.clean_material(material(routes=[r]))
        self.assertEqual(c["variants"][0]["textures"][0]["source"], "shader_default")


class FakeSdb:
    """Just enough of Sdb for the document builders."""
    path = Path("runtime_dx11.sdb")
    data = b"x" * 10

    def __init__(self, materials=None, presets=None):
        self._materials = materials or {"m.mat": material()}
        self._presets = presets or []

    def materials(self):
        return list(self._materials)

    def material(self, name):
        m = self._materials[name]
        if isinstance(m, Exception):
            raise m
        return m

    def preset(self, i):
        return self._presets[i]

    def table(self, key):
        return type("T", (), {"count": len(self._presets)})()

    def string_value(self, v):
        return {"string_index": v, "name": f"s{v}"}


class DocumentTests(unittest.TestCase):
    def test_materials_document_header_and_counts(self):
        d = E.materials_document(FakeSdb({"a.mat": material("a.mat"), "b.mat": material("b.mat", 2)}))
        self.assertEqual(d["schema"], E.SCHEMA_MATERIALS)
        self.assertEqual(d["counts"]["materials"], 2)
        self.assertEqual(d["counts"]["failed"], 0)
        self.assertEqual(d["database"]["name"], "runtime_dx11.sdb")
        self.assertIn("generated", d)

    def test_a_material_that_will_not_resolve_is_reported_not_fatal(self):
        d = E.materials_document(FakeSdb({"ok.mat": material("ok.mat"), "bad.mat": ValueError("broken")}))
        self.assertEqual(d["counts"]["materials"], 1)
        self.assertEqual(d["counts"]["failed"], 1)
        self.assertEqual(d["failed"][0]["name"], "bad.mat")
        self.assertIn("ValueError", d["failed"][0]["error"])

    def test_used_by_is_attached_and_absent_materials_get_empty_lists(self):
        used = {"models": {"a.mat": [{"model": "player.model"}]}, "meshes": {}, "complete": True,
                "models_scanned": 817, "meshes_scanned": 47}
        d = E.materials_document(FakeSdb({"a.mat": material("a.mat"), "b.mat": material("b.mat", 2)}), used_by=used)
        by_name = {m["name"]: m for m in d["materials"]}
        self.assertEqual(by_name["a.mat"]["used_by"]["models"], [{"model": "player.model"}])
        self.assertEqual(by_name["b.mat"]["used_by"], {"models": [], "meshes": []})
        self.assertTrue(d["used_by"]["complete"])

    def test_used_by_absent_when_not_supplied(self):
        d = E.materials_document(FakeSdb())
        self.assertNotIn("used_by", d)
        self.assertNotIn("used_by", d["materials"][0])

    def test_incomplete_scan_is_stated(self):
        d = E.materials_document(FakeSdb(), used_by={"models": {}, "meshes": {}, "complete": False})
        self.assertFalse(d["used_by"]["complete"])

    def test_cancel_raises_before_the_next_material(self):
        sdb = FakeSdb({f"{i}.mat": material(f"{i}.mat", i) for i in range(10)})
        with self.assertRaises(E.Cancelled):
            E.materials_document(sdb, cancel=lambda: True)

    def test_progress_reports_the_total(self):
        seen = []
        sdb = FakeSdb({f"{i}.mat": material(f"{i}.mat", i) for i in range(3)})
        E.materials_document(sdb, progress=lambda d, t: seen.append((d, t)))
        self.assertEqual(seen[-1], (3, 3))

    def test_presets_document_decodes_defaults(self):
        p = {"index": 0, "name": "opaque", "key": 5, "flags": 1, "masks": [0] * 7,
             "parameters": [{"id": 1, "name": "a", "expression": "", "annotation": "", "type": 2,
                             "type_name": "float", "default_hex": "0000803f"}],
             "groups_b": [{"key": 3, "values": [1, 2]}], "complete": True}
        d = E.presets_document(FakeSdb(presets=[p]))
        q = d["presets"][0]["parameters"][0]
        self.assertEqual(q["default"], 1.0)
        self.assertEqual(q["type"], "float")
        self.assertNotIn("default_hex", json.dumps(d))
        self.assertNotIn("masks", json.dumps(d))            # raw mask bytes are not decoded meaning
        self.assertEqual(d["counts"]["presets"], 1)

    def test_preset_default_that_does_not_fit_its_type_is_null(self):
        p = {"index": 0, "name": "x", "key": 0, "flags": 0, "masks": [],
             "parameters": [{"id": 1, "name": "a", "expression": "", "annotation": "", "type": 2,
                             "type_name": "float", "default_hex": "00"}], "groups_b": [], "complete": True}
        d = E.presets_document(FakeSdb(presets=[p]))
        self.assertIsNone(d["presets"][0]["parameters"][0]["default"])


class TexturesDocumentTests(unittest.TestCase):
    def doc(self, providers=None):
        r1 = route(variants=[variant(1, 10, 1, [binding("dif_0_tex", "a.png"), binding("nrm_0_tex", "b.png")])])
        r2 = route(variants=[variant(2, 11, 1, [binding("dif_0_tex", "a.png")])])
        mats = E.materials_document(FakeSdb({"one.mat": material("one.mat", 1, [r1]),
                                             "two.mat": material("two.mat", 2, [r2])}))
        return E.textures_document(FakeSdb(), mats, providers=providers)

    def test_reverse_index(self):
        d = self.doc()
        rows = {r["name"]: r for r in d["textures"]}
        self.assertEqual(rows["a.png"]["materials"], ["one.mat", "two.mat"])
        self.assertEqual(rows["a.png"]["material_count"], 2)
        self.assertEqual(rows["a.png"]["parameters"], ["dif_0_tex"])
        self.assertEqual(rows["b.png"]["materials"], ["one.mat"])
        self.assertEqual(d["counts"]["textures"], 2)
        self.assertEqual(d["counts"]["bindings"], 3)

    def test_sorted_by_name(self):
        self.assertEqual([r["name"] for r in self.doc()["textures"]], ["a.png", "b.png"])

    def test_providers_mark_what_the_game_does_not_ship(self):
        d = self.doc(providers={"a.png": ["common_textures_0_pc.rpack"]})
        rows = {r["name"]: r for r in d["textures"]}
        self.assertTrue(rows["a.png"]["in_game"])
        self.assertFalse(rows["b.png"]["in_game"])
        self.assertEqual(d["counts"]["missing_from_game"], 1)

    def test_reads_an_unflattened_material_too(self):
        """textures_document must not care whether the single route was flattened away."""
        mats = E.materials_document(FakeSdb({"m.mat": material("m.mat", 1, [route(), route("alpha")])}))
        self.assertIn("routes", mats["materials"][0])
        d = E.textures_document(FakeSdb(), mats)
        self.assertEqual([r["name"] for r in d["textures"]], ["a.png"])


class ExportAllTests(unittest.TestCase):
    def test_writes_three_files(self):
        sdb = FakeSdb({"a.mat": material("a.mat")})
        with tmpdir("sdb_dump") as d:
            res = E.export_all(sdb, Path(d))
            self.assertEqual(set(res), {"materials", "presets", "textures"})
            for kind, info in res.items():
                p = Path(info["path"])
                self.assertTrue(p.is_file(), kind)
                self.assertEqual(p.name, E.FILE_NAMES[kind])
                self.assertGreater(info["bytes"], 0)
                json.loads(p.read_text(encoding="utf-8"))          # every file is valid JSON

    def test_materials_are_compact_and_presets_readable(self):
        sdb = FakeSdb({"a.mat": material("a.mat")})
        with tmpdir("sdb_dump_fmt") as d:
            res = E.export_all(sdb, Path(d))
            mat = Path(res["materials"]["path"]).read_text(encoding="utf-8")
            pre = Path(res["presets"]["path"]).read_text(encoding="utf-8")
        self.assertNotIn('": "', mat)                              # compact separators, no padding
        self.assertIn('": "', pre)                                 # indented and readable


@unittest.skipUnless(have_game(), "game not installed")
class CorpusTests(unittest.TestCase):
    """The numbers the module's comments claim, re-measured against the shipped database."""

    @classmethod
    def setUpClass(cls):
        from nightrunner.sdb.reader import Sdb
        cls.sdb = Sdb.open(ASSETS / "runtime_dx11.sdb")

    @classmethod
    def tearDownClass(cls):
        cls.sdb.close()

    def test_every_material_has_exactly_one_route(self):
        """census 2026-09-17: 26,670/26,670 - the flattening in clean_material rests on this."""
        odd = [n for n in self.sdb.materials()
               if len(self.sdb.routes_of(self.sdb.material_index(n))) != 1]
        self.assertEqual(odd, [])

    def test_a_sample_dumps_without_raw_fields(self):
        names = self.sdb.materials()[:200]
        d = E.materials_document(self.sdb, names=names)
        self.assertEqual(d["counts"]["failed"], 0)
        self.assertEqual(len(d["materials"]), 200)
        blob = json.dumps(d)
        for banned in ("value_hex", "selector_hex", "texture_array", "preset_indices"):
            self.assertNotIn(f'"{banned}"', blob, banned)

    def test_textures_document_agrees_with_the_materials_it_came_from(self):
        d = E.materials_document(self.sdb, names=self.sdb.materials()[:300])
        t = E.textures_document(self.sdb, d)
        from_materials = {x for m in d["materials"] for x in m["textures"]}
        self.assertEqual({r["name"] for r in t["textures"]}, from_materials)


if __name__ == "__main__":
    unittest.main()
