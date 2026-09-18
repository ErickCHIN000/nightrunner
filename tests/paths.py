"""Shared locations for tests. Corpus tests skip when the game is not installed."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GAME = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Dying Light The Beast")
ASSETS = GAME / "ph_ft" / "work" / "data_platform" / "pc" / "assets"
PAKS = GAME / "ph_ft" / "source"
SAMPLES = ROOT / "out" / "samples"          # small extracted samples used by unit tests (created by tools/make_samples.py)
OUT = ROOT / "out" / "test"


def have_game() -> bool:
    return ASSETS.exists()
