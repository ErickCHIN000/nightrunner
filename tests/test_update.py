"""Update check: local commit discovery, the GitHub comparison, changelog grouping and the apply guard.

No network. Every remote call goes through an injected opener that returns canned JSON, so these run anywhere and
say nothing about whether the real repository is reachable or public.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import unittest
import urllib.error
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nightrunner import update as up  # noqa: E402
from tests.synth import tmpdir  # noqa: E402


@contextlib.contextmanager
def tmproot(prefix: str):
    """tests.synth.tmpdir yields a str; everything here wants a Path."""
    with tmpdir(prefix) as d:
        yield Path(d)

HEAD = "a" * 40
LOCAL = "b" * 40


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def opener_for(routes: dict, seen: list | None = None):
    """An opener returning canned payloads keyed by a substring of the URL."""
    def _open(req, timeout=None):
        url = req.full_url
        if seen is not None:
            seen.append((url, dict(req.header_items())))
        for key, payload in routes.items():
            if key in url:
                if isinstance(payload, Exception):
                    raise payload
                return FakeResponse(json.dumps(payload).encode())
        raise urllib.error.HTTPError(url, 404, "not found", None, None)
    return _open


def git_tree(root: Path, head_content: str, refs: dict | None = None, packed: str | None = None) -> Path:
    g = root / ".git"
    (g / "refs" / "heads").mkdir(parents=True)
    (g / "HEAD").write_text(head_content, encoding="utf-8")
    for name, sha in (refs or {}).items():
        (g / "refs" / "heads" / name).write_text(sha + "\n", encoding="utf-8")
    if packed is not None:
        (g / "packed-refs").write_text(packed, encoding="utf-8")
    return root


class LocalCommitTests(unittest.TestCase):
    def test_loose_ref(self):
        with tmproot("upd_loose") as d:
            git_tree(d, "ref: refs/heads/main\n", {"main": LOCAL})
            self.assertEqual(up.local_commit(d), LOCAL)
            self.assertEqual(up.local_branch(d), "main")

    def test_packed_ref(self):
        with tmproot("upd_packed") as d:
            git_tree(d, "ref: refs/heads/main\n", packed=f"# pack-refs with: peeled\n{LOCAL} refs/heads/main\n")
            self.assertEqual(up.local_commit(d), LOCAL)

    def test_detached_head(self):
        with tmproot("upd_detached") as d:
            git_tree(d, LOCAL + "\n")
            self.assertEqual(up.local_commit(d), LOCAL)
            self.assertIsNone(up.local_branch(d))

    def test_gitdir_file_indirection(self):
        with tmproot("upd_worktree") as d:
            real = d / "real_git"
            (real / "refs" / "heads").mkdir(parents=True)
            (real / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
            (real / "refs" / "heads" / "main").write_text(LOCAL, encoding="utf-8")
            wt = d / "wt"
            wt.mkdir()
            (wt / ".git").write_text(f"gitdir: {real}\n", encoding="utf-8")
            self.assertEqual(up.local_commit(wt), LOCAL)

    def test_build_info_fallback_and_nothing_at_all(self):
        with tmproot("upd_archive") as d:
            self.assertIsNone(up.local_commit(d))
            (d / "build_info.json").write_text(json.dumps({"commit": LOCAL}), encoding="utf-8")
            self.assertEqual(up.local_commit(d), LOCAL)

    def test_corrupt_build_info_is_not_fatal(self):
        with tmproot("upd_corrupt") as d:
            (d / "build_info.json").write_text("{not json", encoding="utf-8")
            self.assertIsNone(up.local_commit(d))


class CheckTests(unittest.TestCase):
    def setUp(self):
        self.d = tmproot("upd_check")
        self.root = self.d.__enter__()
        git_tree(self.root, "ref: refs/heads/main\n", {"main": LOCAL})

    def tearDown(self):
        self.d.__exit__(None, None, None)

    def check(self, routes, seen=None):
        return up.check(repo="o/r", branch="main", root=self.root, opener=opener_for(routes, seen))

    def test_up_to_date_makes_one_request(self):
        seen = []
        st = self.check({"/commits/main": {"sha": LOCAL}}, seen)
        self.assertEqual(st.state, "up-to-date")
        self.assertEqual(len(seen), 1)                       # no compare call when the shas already match
        self.assertEqual(st.label, f"{LOCAL[:7]} · latest")

    def test_behind(self):
        st = self.check({"/commits/main": {"sha": HEAD},
                         "/compare/": {"status": "ahead", "ahead_by": 3, "behind_by": 0, "total_commits": 3}})
        self.assertEqual((st.state, st.behind, st.ahead), ("behind", 3, 0))
        self.assertEqual(st.compare_url, f"https://github.com/o/r/compare/{LOCAL}...{HEAD}")
        self.assertIn("3 behind", st.label)

    def test_ahead_and_diverged_are_not_reported_as_behind(self):
        st = self.check({"/commits/main": {"sha": HEAD},
                         "/compare/": {"status": "behind", "ahead_by": 0, "behind_by": 2, "total_commits": 2}})
        self.assertEqual((st.state, st.ahead, st.behind), ("ahead", 2, 0))
        st = self.check({"/commits/main": {"sha": HEAD},
                         "/compare/": {"status": "diverged", "ahead_by": 1, "behind_by": 4, "total_commits": 5}})
        self.assertEqual((st.state, st.behind, st.ahead), ("diverged", 1, 4))

    def test_network_failure_is_a_status_not_an_exception(self):
        st = self.check({"/commits/main": urllib.error.URLError("down")})
        self.assertEqual(st.state, "error")
        self.assertEqual(st.error, "no connection")
        self.assertEqual(st.local, LOCAL)                    # the local side still works offline

    def test_private_repo_without_token_reads_as_404(self):
        st = self.check({"/commits/main": urllib.error.HTTPError("u", 404, "nf", None, None)})
        self.assertEqual(st.state, "error")
        self.assertIn("not found", st.error)

    def test_unpushed_local_commit_is_not_reported_as_offline(self):
        # remote_head succeeds, then compare 404s because our own commit is not on the remote yet. Seen live
        # against the real repository on 2026-09-17, where it used to surface as "offline".
        st = self.check({"/commits/main": {"sha": HEAD},
                         "/compare/": urllib.error.HTTPError("u", 404, "nf", None, None)})
        self.assertEqual(st.state, "local-only")
        self.assertIn("unpushed", st.label)
        self.assertEqual(st.remote, HEAD)

    def test_compare_failure_other_than_404_is_still_an_error(self):
        st = self.check({"/commits/main": {"sha": HEAD},
                         "/compare/": urllib.error.HTTPError("u", 500, "boom", None, None)})
        self.assertEqual(st.state, "error")
        self.assertEqual(st.error, "HTTP 500")

    def test_unknown_local_commit(self):
        with tmproot("upd_nolocal") as d:
            st = up.check(repo="o/r", branch="main", root=d, opener=opener_for({"/commits/main": {"sha": HEAD}}))
        self.assertEqual(st.state, "unknown")
        self.assertEqual(st.short, "unknown")

    def test_token_is_sent_only_when_given_and_never_stored(self):
        seen = []
        self.check({"/commits/main": {"sha": LOCAL}}, seen)
        self.assertNotIn("Authorization", seen[0][1])
        seen.clear()
        st = up.check(repo="o/r", branch="main", root=self.root, token="secret-token",
                      opener=opener_for({"/commits/main": {"sha": LOCAL}}, seen))
        self.assertEqual(seen[0][1].get("Authorization"), "Bearer secret-token")
        self.assertNotIn("secret-token", repr(st))

    def test_repo_comes_from_the_environment_when_not_given(self):
        old = os.environ.get("NIGHTRUNNER_UPDATE_REPO")
        os.environ["NIGHTRUNNER_UPDATE_REPO"] = "fork/nightrunner"
        try:
            st = up.check(root=self.root, opener=opener_for({"/commits/": {"sha": LOCAL}}))
        finally:
            os.environ.pop("NIGHTRUNNER_UPDATE_REPO", None)
            if old is not None:
                os.environ["NIGHTRUNNER_UPDATE_REPO"] = old
        self.assertEqual(st.repo, "fork/nightrunner")


class CategoriseTests(unittest.TestCase):
    def test_groups(self):
        self.assertEqual(up.categorise("fix: stop the crash"), "FIXED")
        self.assertEqual(up.categorise("Fixed a leak"), "FIXED")
        self.assertEqual(up.categorise("fix(gui): tab order"), "FIXED")
        self.assertEqual(up.categorise("feat: add the updater"), "IMPROVED")
        self.assertEqual(up.categorise("refactor!: split the codec"), "IMPROVED")
        self.assertEqual(up.categorise("Merge pull request #12"), "OTHER")
        self.assertEqual(up.categorise(""), "OTHER")

    def test_a_colon_far_into_the_subject_is_not_a_prefix(self):
        self.assertEqual(up.categorise("rewrote the thing that does: everything"), "OTHER")


class ChangelogTests(unittest.TestCase):
    def payload(self, subjects, total=None):
        return {"status": "ahead", "ahead_by": len(subjects), "behind_by": 0,
                "total_commits": total if total is not None else len(subjects),
                "commits": [{"commit": {"message": s}} for s in subjects]}

    def test_grouped_newest_first(self):
        log = up.changelog("o/r", LOCAL, HEAD, opener=opener_for({"/compare/": self.payload(
            ["feat: add a thing", "fix: a bug", "chore: tidy"])}))
        self.assertEqual(list(log["groups"]), ["FIXED", "IMPROVED", "OTHER"])
        self.assertEqual(log["groups"]["FIXED"], ["fix: a bug"])
        self.assertEqual(log["listed"], 3)
        self.assertEqual(log["more"], 0)

    def test_only_the_subject_line_is_used(self):
        log = up.changelog("o/r", LOCAL, HEAD, opener=opener_for({"/compare/": self.payload(
            ["fix: one thing\n\nA long body that must not appear in the dialog."])}))
        self.assertEqual(log["groups"]["FIXED"], ["fix: one thing"])

    def test_limit_and_more_count(self):
        log = up.changelog("o/r", LOCAL, HEAD, limit=2, opener=opener_for({"/compare/": self.payload(
            [f"fix: bug {i}" for i in range(5)], total=2970)}))
        self.assertEqual(log["listed"], 2)
        self.assertEqual(log["total"], 2970)
        self.assertEqual(log["more"], 2968)

    def test_duplicate_subjects_collapse(self):
        log = up.changelog("o/r", LOCAL, HEAD, opener=opener_for({"/compare/": self.payload(
            ["fix: same", "fix: same", "fix: other"])}))
        self.assertEqual(log["groups"]["FIXED"], ["fix: other", "fix: same"])


class ApplyGuardTests(unittest.TestCase):
    """`apply_update` must refuse before it can touch anything it should not."""

    def test_refuses_without_a_git_dir(self):
        with tmproot("upd_nogit") as d:
            ok, why = up.can_apply(d)
            self.assertFalse(ok)
            self.assertIn("not a git checkout", why)
            with self.assertRaises(up.UpdateError):
                up.apply_update(d)

    def test_refuses_a_detached_head(self):
        with tmproot("upd_det") as d:
            git_tree(d, LOCAL + "\n")
            real = [up._git]
            up._git = lambda root, *a, **k: type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()
            try:
                ok, why = up.can_apply(d)
            finally:
                up._git = real[0]
            self.assertFalse(ok)
            self.assertIn("detached", why)

    def test_refuses_a_dirty_tree(self):
        with tmproot("upd_dirty") as d:
            git_tree(d, "ref: refs/heads/main\n", {"main": LOCAL})
            real = up._git

            def fake(root, *a, **k):
                out = " M nightrunner/cli.py\n?? scratch.py\n" if a[:1] == ("status",) else ""
                return type("R", (), {"returncode": 0, "stdout": out, "stderr": ""})()

            up._git = fake
            try:
                ok, why = up.can_apply(d)
            finally:
                up._git = real
            self.assertFalse(ok)
            self.assertIn("2 uncommitted", why)


if __name__ == "__main__":
    unittest.main()
