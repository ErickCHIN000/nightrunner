"""Update corner widget and dialog.

Headless (QT_QPA_PLATFORM=offscreen); skips without PySide6. No network: every Status is constructed by hand and
`refresh()` is never called, so these tests never reach GitHub and never need a reachable repository.
"""
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except ImportError:
    HAVE_QT = False

from nightrunner import __version__  # noqa: E402
from nightrunner import update as up  # noqa: E402

LOCAL = "b" * 40
REMOTE = "a" * 40


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class UpdateCornerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def corner(self):
        from nightrunner.gui.updater import UpdateCorner
        # runner=None would make refresh() check inline; these tests never call it.
        return UpdateCorner(SimpleNamespace(runner=None))

    def test_starts_with_the_version_and_no_network(self):
        c = self.corner()
        self.assertEqual(c.label.text(), f"v{__version__}")
        self.assertIsNone(c.status)

    def test_behind_shows_the_count_and_the_sha(self):
        c = self.corner()
        c._done(up.Status(state="behind", local=LOCAL, remote=REMOTE, behind=3, repo="o/r"))
        self.assertIn(f"v{__version__} (+3)", c.label.text())
        self.assertIn(LOCAL[:7], c.label.text())
        self.assertIn("3 new commit", c.label.toolTip())

    def test_up_to_date_has_no_count(self):
        c = self.corner()
        c._done(up.Status(state="up-to-date", local=LOCAL, remote=LOCAL, repo="o/r"))
        self.assertNotIn("(+", c.label.text())
        self.assertIn("Up to date", c.label.toolTip())

    def test_offline_keeps_the_sha_and_explains_itself(self):
        c = self.corner()
        c._done(up.Status(state="error", local=LOCAL, error="no connection", repo="o/r"))
        self.assertIn(LOCAL[:7], c.label.text())
        self.assertEqual(c.label.toolTip(), "no connection")

    def test_status_is_emitted(self):
        c = self.corner()
        seen = []
        c.checked.connect(seen.append)
        c._done(up.Status(state="up-to-date", local=LOCAL, remote=LOCAL, repo="o/r"))
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].state, "up-to-date")

    def test_a_new_check_drops_the_cached_changelog(self):
        c = self.corner()
        c._changelog = {"groups": {}}
        c._done(up.Status(state="behind", local=LOCAL, remote=REMOTE, behind=1, repo="o/r"))
        self.assertIsNone(c._changelog)


@unittest.skipUnless(HAVE_QT, "PySide6 not installed")
class UpdateDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def dialog(self, log, behind=3):
        from nightrunner.gui.updater import UpdateDialog
        st = up.Status(state="behind", local=LOCAL, remote=REMOTE, behind=behind, repo="o/r")
        return UpdateDialog(st, log)

    def texts(self, dlg):
        from PySide6.QtWidgets import QLabel
        return [w.text() for w in dlg.findChildren(QLabel)]

    def test_lists_the_groups_and_the_subjects(self):
        d = self.dialog({"groups": {"FIXED": ["fix: a crash"], "IMPROVED": ["feat: a thing"]},
                         "listed": 2, "total": 2, "more": 0})
        t = self.texts(d)
        self.assertIn("New update available", t)
        self.assertIn("FIXED", t)
        self.assertIn("IMPROVED", t)
        self.assertTrue(any("fix: a crash" in x for x in t))

    def test_more_line_only_when_there_is_more(self):
        d = self.dialog({"groups": {"FIXED": ["fix: one"]}, "listed": 1, "total": 2971, "more": 2970})
        self.assertTrue(any("2970 more change" in x for x in self.texts(d)))
        d2 = self.dialog({"groups": {"FIXED": ["fix: one"]}, "listed": 1, "total": 1, "more": 0})
        self.assertFalse(any("more change" in x for x in self.texts(d2)))

    def test_survives_an_empty_changelog(self):
        d = self.dialog({"groups": {}, "listed": 0, "total": 3, "more": 3, "error": "HTTPError"})
        self.assertTrue(any("Could not load" in x for x in self.texts(d)))

    def test_button_offers_github_when_the_tree_cannot_be_fast_forwarded(self):
        from nightrunner.gui import updater
        real = updater.up.can_apply
        updater.up.can_apply = lambda *a, **k: (False, "3 uncommitted change(s) - commit or stash them first")
        try:
            d = self.dialog({"groups": {}, "listed": 0, "total": 1, "more": 0})
        finally:
            updater.up.can_apply = real
        self.assertEqual(d.update_btn.text(), "Open on GitHub")
        self.assertTrue(any("uncommitted" in x for x in self.texts(d)))


if __name__ == "__main__":
    unittest.main()
