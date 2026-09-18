"""SDB tab: the whole-database dump button and its callbacks.

Headless; skips without PySide6. Nothing is exported: `start_dump` is not called, and the finish / failure
callbacks are driven directly. Those callbacks are the part that only runs minutes into a real export, which is
exactly why two bugs in them (`svc.sdb()` on a property, a bare `TITLE`) survived until the feature was driven end
to end - so they are pinned here.
"""
from __future__ import annotations

import os
import sys
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication, QMessageBox, QWidget
    HAVE_QT = True
except ImportError:
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class DumpCallbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def stub(self, **attrs):
        w = QWidget()
        w.TITLE = "SDB"
        w.dump_btn = QWidget()
        w._dump_cancel = threading.Event()
        for k, v in attrs.items():
            setattr(w, k, v)
        return w

    def silence(self):
        """Message boxes must not block a test run."""
        info, warn = QMessageBox.information, QMessageBox.warning
        QMessageBox.information = staticmethod(lambda *a, **k: None)
        QMessageBox.warning = staticmethod(lambda *a, **k: None)
        return info, warn

    def test_done_clears_the_guard_and_reports_every_file(self):
        from nightrunner.gui.tabs.sdb import Tab
        said = []
        tab = self.stub(ctx=type("C", (), {"status": type("S", (), {"emit": staticmethod(said.append)})()})())
        res = {"materials": {"path": "x/sdb_materials.json", "count": 26670, "bytes": 112_460_363},
               "presets": {"path": "x/sdb_presets.json", "count": 791, "bytes": 1_330_116},
               "textures": {"path": "x/sdb_textures.json", "count": 59173, "bytes": 15_282_915}}
        info, warn = self.silence()
        try:
            Tab._on_dump_done(tab, res)
        finally:
            QMessageBox.information, QMessageBox.warning = info, warn
        self.assertIsNone(tab._dump_cancel)                 # a second export is possible again
        self.assertTrue(tab.dump_btn.isEnabled())
        self.assertEqual(len(said), 1)
        for name in ("sdb_materials.json", "sdb_presets.json", "sdb_textures.json"):
            self.assertIn(name, said[0])
        self.assertIn("26,670", said[0])

    def test_failure_clears_the_guard_and_warns(self):
        from nightrunner.gui.tabs.sdb import Tab
        tab = self.stub(ctx=type("C", (), {"status": type("S", (), {"emit": staticmethod(lambda s: None)})()})())
        info, warn = self.silence()
        try:
            Tab._on_dump_failed(tab, OSError("disk full"))
        finally:
            QMessageBox.information, QMessageBox.warning = info, warn
        self.assertIsNone(tab._dump_cancel)
        self.assertTrue(tab.dump_btn.isEnabled())

    def test_cancellation_is_not_reported_as_a_failure(self):
        from nightrunner.gui.tabs.sdb import Tab
        from nightrunner.sdb.export import Cancelled
        said, warned = [], []
        tab = self.stub(ctx=type("C", (), {"status": type("S", (), {"emit": staticmethod(said.append)})()})())
        info, warn = self.silence()
        QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
        try:
            Tab._on_dump_failed(tab, Cancelled("cancelled after 100 of 26,670"))
        finally:
            QMessageBox.information, QMessageBox.warning = info, warn
        self.assertEqual(warned, [])                        # cancelling is not an error
        self.assertIn("cancelled", said[0].casefold())

    def test_used_by_indices_shape(self):
        from nightrunner.gui.tabs.sdb import Tab
        scanner = type("S", (), {"index": {"a.mat": [1, 2]}, "scanned": {1},
                                 "complete": staticmethod(lambda: True)})()
        tab = self.stub(
            _model_refs={"refs": {"a.mat": [{"model": "p.model", "pak": "data0.pak", "slot": "TORSO",
                                             "mesh": "m", "kind": "resource", "selected": True,
                                             "rtti": [{"name": "drop me"}]}]}, "models": 817},
            scanner=scanner,
            ctx=type("C", (), {"catalog": type("K", (), {"name": staticmethod(lambda g: f"mesh{g}")})()})())
        u = Tab.used_by_indices(tab)
        self.assertEqual(u["meshes"]["a.mat"], ["mesh1", "mesh2"])
        self.assertEqual(u["models"]["a.mat"][0]["model"], "p.model")
        self.assertNotIn("rtti", u["models"]["a.mat"][0])    # the dump keeps the reference, not the whole record
        self.assertTrue(u["complete"])
        self.assertEqual(u["models_scanned"], 817)

    def test_used_by_is_marked_incomplete_before_the_scans_run(self):
        from nightrunner.gui.tabs.sdb import Tab
        scanner = type("S", (), {"index": {}, "scanned": set(), "complete": staticmethod(lambda: False)})()
        tab = self.stub(_model_refs=None, scanner=scanner,
                        ctx=type("C", (), {"catalog": type("K", (), {"name": staticmethod(lambda g: "")})()})())
        u = Tab.used_by_indices(tab)
        self.assertEqual((u["models"], u["meshes"]), ({}, {}))
        self.assertFalse(u["complete"])                      # never implies "nothing uses this"


if __name__ == "__main__":
    unittest.main()
