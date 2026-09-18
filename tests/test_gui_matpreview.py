"""Material-aware 3D preview images (nightrunner/gui/matpreview.py): eye layers, hair cut-out, opacity blend.

Synthetic material dicts cover the maths; the real runtime_dx11.sdb (when present) checks that the shipped eye,
eye-shadow, wet-eye, hair, beard, forearm-hair and null materials pick the expected recipe.
"""
from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
try:
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def _mat(tokens, params=None, bindings=None, non_rendering=False):
    return {"non_rendering": non_rendering, "routes": [{
        "tokens": ";".join(tokens) + ";",
        "parameters": [{"name": k, "value": v} for k, v in (params or {}).items()],
        "variants": [{"texture_bindings": [{"param": k, "texture": v} for k, v in (bindings or {}).items()]}]}]}


def _solid(rgba, size=4):
    return np.tile(np.asarray(rgba, dtype=np.float32), (size, size, 1))


class _Fetch:
    def __init__(self, table):
        self.table, self.asked = table, []

    def __call__(self, name):
        self.asked.append(name)
        return None if self.table.get(name) is None else self.table[name].copy()

    def error(self, name):
        return "missing in test"


def _px(img, x=0, y=0):
    c = img.pixelColor(x, y)
    return np.array([c.redF(), c.greenF(), c.blueF(), c.alphaF()])


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_eye_layers(self):
        from nightrunner.gui import matpreview as mp
        eye = _mat(["opaque", "eyes_blicks_on", "od1_tex", "od2_tex", "off_tex", "dif_0_tex"],
                   bindings={"dif_0_tex": "veins", "od1_tex": "sclera", "od2_tex": "iris", "off_tex": "mask"})
        # sclera alpha 0 and mask red 0 -> pure iris colour
        f = _Fetch({"veins": _solid([1, 0, 0, 1]), "sclera": _solid([1, 1, 1, 0]), "iris": _solid([0, 0, 1, 1]),
                    "mask": _solid([0, 0, 0, 1])})
        sp = mp.compose(eye, f)
        self.assertEqual((sp.recipe, sp.alpha_mode), ("eye_layers", "opaque"))
        np.testing.assert_allclose(_px(sp.image), [0, 0, 1, 1], atol=0.01)
        # sclera alpha 1 -> sclera; mask red 1 -> veins win
        f.table["sclera"] = _solid([1, 1, 1, 1])
        np.testing.assert_allclose(_px(mp.compose(eye, f).image), [1, 1, 1, 1], atol=0.01)
        f.table["mask"] = _solid([1, 0, 0, 1])
        np.testing.assert_allclose(_px(mp.compose(eye, f).image), [1, 0, 0, 1], atol=0.01)

    def test_hair_cutout(self):
        from nightrunner.gui import matpreview as mp
        hair = _mat(["opaque", "dif_0_tex", "dit_0_tex"], {"dif_0_val": [0.5, 1.0, 1.0]},
                    {"dif_0_tex": "d", "dit_0_tex": "c"})
        sp = mp.compose(hair, _Fetch({"d": _solid([1, 1, 1, 1]), "c": _solid([0.2, 0.9, 0.9, 1])}))
        self.assertEqual((sp.recipe, sp.alpha_mode, sp.cutoff), ("dither_cutout", "mask", 0.25))
        np.testing.assert_allclose(_px(sp.image), [0.5, 1, 1, 0.2], atol=0.01)

    def test_opacity_blend_ranges(self):
        from nightrunner.gui import matpreview as mp
        shadow = _mat(["opaque", "dif_0_tex", "opc_0_tex", "eyes_blicks_on"],
                      {"dif_0_val": [0.27, 0.16, 0.16], "opc_3_ranges": [0.0, 153.0]},
                      {"dif_0_tex": "white", "opc_0_tex": "opc"})
        sp = mp.compose(shadow, _Fetch({"white": _solid([1, 1, 1, 1]), "opc": _solid([102 / 255, 0, 0, 1])}))
        self.assertEqual((sp.recipe, sp.alpha_mode), ("diffuse_opacity", "blend"))
        np.testing.assert_allclose(_px(sp.image), [0.27, 0.16, 0.16, 102 / 153], atol=0.01)

    def test_unsupported_opacity_falls_back_and_hidden(self):
        from nightrunner.gui import matpreview as mp
        watch = _mat(["opaque", "dif_0_tex", "opc_0_tex", "opc_uv_1_on"], {}, {"dif_0_tex": "d", "opc_0_tex": "o"})
        sp = mp.compose(watch, _Fetch({"d": _solid([0.1, 0.2, 0.3, 0.0]), "o": _solid([1, 0, 0, 1])}))
        self.assertEqual((sp.recipe, sp.alpha_mode), ("plain", "opaque"))
        self.assertTrue(any("UV1" in w for w in sp.warnings))
        self.assertAlmostEqual(_px(sp.image)[3], 1.0, places=2)
        self.assertTrue(mp.compose(_mat(["null"], non_rendering=True), _Fetch({})).hidden)
        self.assertFalse(mp.compose(None, _Fetch({})).hidden)

    def test_override_and_missing(self):
        from nightrunner.gui import matpreview as mp
        plain = _mat(["opaque", "dif_0_tex"], {}, {"dif_0_tex": "base"})
        f = _Fetch({"base": _solid([1, 0, 0, 1]), "bloody": _solid([0, 1, 0, 1])})
        sp = mp.compose(plain, f, {"dif_0_tex": "bloody"})
        np.testing.assert_allclose(_px(sp.image), [0, 1, 0, 1], atol=0.01)
        eye = _mat(["eyes_blicks_on", "od1_tex", "od2_tex", "off_tex"],
                   bindings={"dif_0_tex": "base", "od1_tex": "x", "od2_tex": "y", "off_tex": "z"})
        sp = mp.compose(eye, f)          # eye layers missing -> plain diffuse + warning
        self.assertEqual(sp.recipe, "plain")
        self.assertTrue(any("missing" in w for w in sp.warnings))


def _sdb_path() -> Path | None:
    from tests.paths import ASSETS
    for root in (os.environ.get("NIGHTRUNNER_GAME_ROOT"),):
        if root:
            p = Path(root) / "ph_ft/work/data_platform/pc/assets/runtime_dx11.sdb"
            if p.is_file():
                return p
    p = ASSETS / "runtime_dx11.sdb"
    return p if p.is_file() else None


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class RealSdbTests(unittest.TestCase):
    def test_shipped_materials_pick_recipes(self):
        path = _sdb_path()
        if path is None:
            self.skipTest("runtime_dx11.sdb not available")
        from nightrunner.gui import matpreview as mp
        from nightrunner.sdb.reader import Sdb

        class Any_:
            def __call__(self, name):
                return _solid([0.5, 0.5, 0.5, 0.5])

        want = {"sh2_eye.mat": ("eye_layers", "opaque"), "sh2_eye_kyle_crane.mat": ("eye_layers", "opaque"),
                "sh2_eye_shadow.mat": ("diffuse_opacity", "blend"), "sh2_wet_eye.mat": ("diffuse_opacity", "blend"),
                "npc_ft_crane_hair_a.mat": ("dither_cutout", "mask"),
                "sh_npc_ft_crane_beard_a.mat": ("dither_cutout", "mask"),
                "player_kc_forearm_hairs_a.mat": ("diffuse_opacity", "blend"),
                "sh2_npc_crane.mat": ("plain", "opaque")}
        with Sdb.open(path) as s:
            for name, (recipe, mode) in want.items():
                sp = mp.compose(s.material(name), Any_())
                self.assertEqual((sp.recipe, sp.alpha_mode), (recipe, mode), name)
            self.assertTrue(mp.compose(s.material("null.mat"), Any_()).hidden)
            sp = mp.compose(s.material("sh2_eye_shadow.mat"), _Fetch({"chr_white_dif.png": _solid([1, 1, 1, 1]),
                                                                       "eye_shadow_opc.png": _solid([1, 0, 0, 1])}))
            np.testing.assert_allclose(_px(sp.image)[:3], [0.2745, 0.1569, 0.1569], atol=0.01)


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class MeshViewAlphaTests(unittest.TestCase):
    def test_add_and_retexture_with_alpha(self):
        from PySide6.QtGui import QImage
        from nightrunner.gui.meshview import MeshView
        app = QApplication.instance() or QApplication([])
        v = MeshView()
        pos = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        uv = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
        img = QImage(4, 4, QImage.Format_RGBA8888)
        img.fill(0)
        v.add_mesh("a", pos, np.array([0, 1, 2]), uv=uv, texture=img, alpha_mode="mask", alpha_cutoff=0.25)
        v.set_texture("a", img, "blend")
        v.set_texture("a", None)
        self.assertEqual(v.keys(), ["a"])
        v.clear()
        app.processEvents()


if __name__ == "__main__":
    unittest.main()
