"""Nightrunner — Textures tab: list population, decode helper, garbage handling, DDS export, open_gid.

Headless (QT_QPA_PLATFORM=offscreen); skips without PySide6. Game data: $NIGHTRUNNER_GAME_ROOT, else /tmp/game (the
cloud fake install), else the real install (tests/paths.py); skipped when none exists.
"""
from __future__ import annotations

import os
import threading
import time
import unittest
from pathlib import Path

import tests.synth as S
from tests import paths

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


def game_root() -> Path | None:
    for cand in (os.environ.get("NIGHTRUNNER_GAME_ROOT"), "/tmp/game", str(paths.GAME)):
        if cand and (Path(cand) / "ph_ft" / "work" / "data_platform" / "pc" / "assets").is_dir():
            return Path(cand)
    return None


GAME_ROOT = game_root() if HAVE_QT else None


class _Sink:
    def emit(self, *a):
        pass


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
@unittest.skipUnless(GAME_ROOT is not None, "no game install (NIGHTRUNNER_GAME_ROOT, /tmp/game or the real install)")
class TestTexturesTab(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nightrunner.gui.context import AppContext
        from nightrunner.gui.game import GameInstall
        cls.app = QApplication.instance() or QApplication([])
        cls._tmp = S.tmpdir("guitex_")
        cls.d = Path(cls._tmp.name)
        settings = QSettings(str(cls.d / "settings.ini"), QSettings.IniFormat)
        cls.ctx = AppContext(settings, GameInstall(GAME_ROOT))
        assert cls.ctx.catalog.wait(600)
        cls.cat = cls.ctx.catalog

    @classmethod
    def tearDownClass(cls):
        cls.ctx.runner.pool.waitForDone(10000)
        cls.ctx.close()
        cls._tmp.cleanup()

    # ---- helpers ------------------------------------------------------------------------------------------------
    def pump(self, cond, timeout: float = 60.0) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.005)
        return False

    def make_tab(self):
        from nightrunner.gui.tabs.textures import Tab
        tab = Tab(self.ctx)
        self.addCleanup(tab.deleteLater)
        self.assertTrue(self.pump(lambda: self.idle(tab)), "search did not finish")
        return tab

    @staticmethod
    def idle(tab) -> bool:
        return (not tab._searching and not tab._search_deb._t.isActive() and not tab._pack_deb._t.isActive())

    def textures(self):
        import numpy as np
        return np.flatnonzero(self.cat._types[:len(self.cat)] == 0x20)

    def find(self, name: str):
        hits = self.cat.lookup(name, 0x20)
        return hits[0] if hits else None

    # ---- tests --------------------------------------------------------------------------------------------------
    def test_list_population(self):
        tab = self.make_tab()
        expected = self.textures()
        self.assertGreater(len(expected), 0)
        self.assertEqual(sorted(tab.model.gids.tolist()), sorted(expected.tolist()))
        total = sum(e.type_counts.get(0x20, 0) for e in self.cat.packs if e.pack is not None)
        self.assertEqual(tab._tex_total, total)
        # pack combo holds exactly the packs with textures
        combo_ids = {tab.search.pack_combo.itemData(i) for i in range(1, tab.search.pack_combo.count())}
        self.assertEqual(combo_ids, {e.id for e in self.cat.packs if e.type_counts.get(0x20)})
        # cells render (garbage headers become "?")
        m = tab.model
        for row in range(min(50, m.rowCount())):
            for col in range(m.columnCount()):
                self.assertIsInstance(m.data(m.index(row, col)), str)
        # search + pack filter
        g = int(expected[0])
        e = self.cat.entry(g)
        tab.search.pack_combo.setCurrentIndex(tab.search.pack_combo.findData(e.id))
        tab.search.edit.setText(self.cat.name(g))
        self.assertTrue(self.pump(lambda: self.idle(tab) and g in tab.model.gids.tolist()
                                  and all(self.cat.entry(int(x)).id == e.id for x in tab.model.gids)))
        # sort by name (descending on second click)
        tab.search.edit.clear()
        tab.search.pack_combo.setCurrentIndex(0)
        tab._on_header_clicked(0)
        tab._on_header_clicked(0)
        self.assertTrue(self.pump(lambda: self.idle(tab) and len(tab.model.gids) == len(expected)))
        names = [self.cat.name(int(x)).lower() for x in tab.model.gids[:200]]
        self.assertEqual(names, sorted(names, reverse=True))

    def test_decode_matches_header(self):
        from nightrunner.gui.texpreview import decode_texture, read_header
        checked = 0
        for g in self.textures()[:400].tolist():
            e, i = self.cat.split(g)
            try:
                h, _, bmp = read_header(e.pack, i)
            except Exception:  # noqa: BLE001 - garbage: covered by test_garbage
                continue
            if bmp is None or h.width * h.height > 1 << 22:
                continue
            img, info = decode_texture(e.pack, i)
            if info["format"]["decodable"]:
                self.assertIsNone(info["error"], self.cat.name(g))
                self.assertEqual((img.width(), img.height()), (h.width, h.height), self.cat.name(g))
                if h.mip_count > 1:
                    img2, info2 = decode_texture(e.pack, i, mip=h.mip_count - 1)
                    self.assertEqual((img2.width(), img2.height()),
                                     (max(1, h.width >> (h.mip_count - 1)), max(1, h.height >> (h.mip_count - 1))))
                checked += 1
            else:
                self.assertTrue(img.isNull())
                self.assertIn("no preview decoder", info["error"])
        self.assertGreater(checked, 0)

    def test_uncompressed_pixels_match_png_module(self):
        import numpy as np
        from nightrunner.gui.texpreview import decode_texture, qimage_to_rgba, read_header
        from nightrunner.texture.imgc import check_payload
        from nightrunner.texture.png import UNCOMPRESSED, decode_preview
        for g in self.textures().tolist():
            e, i = self.cat.split(g)
            try:
                h, _, bmp = read_header(e.pack, i)
            except Exception:  # noqa: BLE001
                continue
            if bmp is None or h.format not in UNCOMPRESSED or h.width * h.height > 1 << 23:
                continue
            bitmap = e.pack.read_part(bmp)
            lv = check_payload(h, len(bitmap))[0]
            ref = decode_preview(bitmap[lv.offset: lv.offset + lv.slice_size], lv.width, lv.height, h.format)
            img, info = decode_texture(e.pack, i)
            self.assertIsNone(info["error"])
            np.testing.assert_array_equal(qimage_to_rgba(img), ref)
            # thumbnail path: auto mip + downsample stays small
            th, tinfo = decode_texture(e.pack, i, mip=None, max_dim=64)
            self.assertLessEqual(max(th.width(), th.height()), max(128, 2 * 64 + 4))
            return
        self.skipTest("no uncompressed texture in the loaded packs")

    def test_garbage_is_error_not_exception(self):
        from nightrunner.gui.texpreview import decode_texture, header_summary
        from nightrunner.gui.tabs.textures import preview_job, thumb_job
        bad = None
        for g in self.textures().tolist():
            e, i = self.cat.split(g)
            if header_summary(e.pack, i)[0] == "?":
                bad = g
                break
        if bad is None:     # no garbage pack loaded: synthesise one
            p = self.d / "garbage.rpack"
            S.write_pack([S.texture(b"garbage_tex")], p)
            from nightrunner.container.rp6l import Pack
            with Pack.open(p) as pk:
                img, info = decode_texture(pk, 0)
                self.assertTrue(img.isNull())
                self.assertTrue(info["error"])
            return
        e, i = self.cat.split(bad)
        img, info = decode_texture(e.pack, i)
        self.assertTrue(img.isNull())
        self.assertIn("Error", info["error"])
        img, info = preview_job(self.cat, bad, 3, 2, 1)
        self.assertTrue(info["error"])
        self.assertIsInstance(thumb_job(self.cat, bad), str)

    def test_export_dds_matches_cli(self):
        from nightrunner.gui.tabs.textures import export_job, pack_dir
        from nightrunner.gui.texpreview import read_header
        from nightrunner.texture.codec import texture_stem
        target = None
        for g in self.textures().tolist():
            e, i = self.cat.split(g)
            try:
                h, _, bmp = read_header(e.pack, i)
            except Exception:  # noqa: BLE001
                continue
            if bmp is not None and e.pack.physicals[bmp].size < 4 << 20:
                target = g
                break
        self.assertIsNotNone(target)
        e, i = self.cat.split(target)
        out = self.d / "export"
        s = export_job(self.cat, [target], "dds", out, threading.Event(), _Sink())
        self.assertEqual((s["written"], s["errors"]), (1, []))
        got = out / pack_dir(e.label) / f"{texture_stem(self.cat.name(target))}.dds"
        self.assertTrue(got.is_file())
        cli = self.d / "cli.dds"
        rc, _ = S.run_cli("texture", "export", e.path, i, cli, "--force")
        self.assertEqual(rc, 0)
        self.assertEqual(got.read_bytes(), cli.read_bytes())
        # second export never overwrites
        s2 = export_job(self.cat, [target], "dds", out, threading.Event(), _Sink())
        self.assertEqual(s2["written"], 1)
        self.assertTrue(got.with_name(got.stem + ".1.dds").is_file())
        # raw + png modes
        s3 = export_job(self.cat, [target], "raw", out, threading.Event(), _Sink())
        self.assertEqual((s3["written"], s3["errors"]), (1, []))
        s4 = export_job(self.cat, [target], "png", out, threading.Event(), _Sink())
        self.assertEqual(s4["written"] + len(s4["errors"]), 1)
        # cancelled before start
        ev = threading.Event()
        ev.set()
        self.assertTrue(export_job(self.cat, [target], "dds", out, ev, _Sink())["cancelled"])

    def test_export_via_tab(self):
        tab = self.make_tab()
        g = int(tab.model.gids[0])
        tab.model.set_checked([g], True)
        tab._export("raw", str(self.d / "tab_export"))
        self.assertTrue(self.pump(lambda: tab.last_export is not None, 60))
        self.assertEqual(tab.last_export["total"], 1)
        self.assertFalse(tab.prog_box.isVisible())

    def test_open_gid(self):
        tab = self.make_tab()
        gids = self.textures()
        g = int(gids[len(gids) // 2])
        tab.search.edit.setText("zzz_no_such_texture_zzz")
        self.assertTrue(self.pump(lambda: self.idle(tab) and len(tab.model.gids) == 0))
        tab.open_gid(g)
        self.assertTrue(self.pump(lambda: tab._last_info is not None and tab._last_info.get("gid") == g))
        self.assertEqual(tab.search.text(), "")
        self.assertEqual(tab.model.gid_at(tab.table.currentIndex().row()), g)
        html = tab.info.toPlainText()
        self.assertIn(self.cat.name(g), html)
        self.assertIn("Used by materials", html)
        # already visible: select directly
        g2 = int(gids[0])
        tab.open_gid(g2)
        self.assertTrue(self.pump(lambda: tab._last_info is not None and tab._last_info.get("gid") == g2))
        # not a texture: ignored
        non_tex = next((x for x in range(len(self.cat)) if self.cat.type(x) != 0x20), None)
        if non_tex is not None:
            tab.open_gid(non_tex)
            self.assertEqual(tab._current, g2)

    def test_grid_thumbnails(self):
        tab = self.make_tab()
        tab.resize(900, 700)
        tab.show()
        tab.btn_grid.setChecked(True)
        self.assertTrue(self.pump(lambda: len(tab._thumbs) > 0, 30))
        tab.hide()


if __name__ == "__main__":
    unittest.main()
