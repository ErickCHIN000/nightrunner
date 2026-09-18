#!/usr/bin/env python3
"""Shim: run nightrunner's CLI from the repo root without installing anything."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nightrunner.cli import main  # noqa: E402

sys.exit(main())
