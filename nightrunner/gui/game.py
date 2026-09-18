"""Compatibility shim: install lookup now lives in `nightrunner.games` (game profiles, Qt-free)."""
from __future__ import annotations

from ..games import (DL2, DLTB, PROFILES, GameInstall, GameProfile, detect_profile, find_game,  # noqa: F401
                     find_installs, normalize_root, profile, steam_roots)

GAME_DIR_NAME = DLTB.steam_dir
_steam_roots = steam_roots
