"""Run every test module: python tests/run.py [-k substring ...] [-q]

-k keeps only the tests whose id (module.Class.method) contains the substring (case-insensitive); several -k are
OR-ed. Modules that fail to import are always reported. Exit code 0 when the run was successful.
"""
import argparse
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")    # GUI tests never open real windows (they can hang)


def _flatten(suite):
    for t in suite:
        if isinstance(t, unittest.TestSuite):
            yield from _flatten(t)
        else:
            yield t


def build_suite(patterns) -> unittest.TestSuite:
    discovered = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py", top_level_dir=str(ROOT))
    if not patterns:
        return discovered
    pats = [p.lower() for p in patterns]
    keep = [t for t in _flatten(discovered)
            if any(p in t.id().lower() for p in pats) or type(t).__name__ == "_FailedTest"]
    return unittest.TestSuite(keep)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tests/run.py", description=__doc__.splitlines()[0])
    ap.add_argument("-k", action="append", default=[], metavar="SUBSTRING", help="only tests whose id contains this")
    ap.add_argument("-q", "--quiet", action="store_true", help="verbosity 1 instead of 2")
    a = ap.parse_args(argv)
    suite = build_suite(a.k)
    if a.k and suite.countTestCases() == 0:
        print(f"no test id matches {a.k}", file=sys.stderr)
        return 1
    result = unittest.TextTestRunner(verbosity=1 if a.quiet else 2).run(suite)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
