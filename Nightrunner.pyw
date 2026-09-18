"""Nightrunner launcher (double-click).

Runs inside the repo's .venv: when started by any other interpreter it re-launches itself with
.venv\\Scripts\\pythonw.exe (create the venv once with setup.bat).
"""
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
VENV_PYW = VENV / ("Scripts/pythonw.exe" if os.name == "nt" else "bin/python")

if VENV_PYW.exists() and Path(sys.prefix).resolve() != VENV.resolve():
    subprocess.Popen([str(VENV_PYW), str(Path(__file__).resolve()), *sys.argv[1:]], cwd=str(ROOT))
    sys.exit(0)

sys.path.insert(0, str(ROOT))
from nightrunner.gui.app import main  # noqa: E402

sys.exit(main())
