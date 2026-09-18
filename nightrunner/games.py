"""Game profiles and install lookup (Qt-free).

A *profile* holds what differs between Chrome Engine games: the Steam folder, the data folder under the root
(`ph_ft` for Dying Light: The Beast, `ph` for Dying Light 2), the executable, the stock archives the Build tab
must never take, and the paths of the player-appearance scripts. A `GameInstall` is a root folder plus its profile.

Layouts (checked on the real installs, 2026-09-16):

* DLTB  `<root>/ph_ft/{source,work}`; `<root>/ph` holds only `fs.ini`; exe
  `ph_ft/work/bin/x64/DyingLightGame_TheBeast_x64_rwdi.exe`; `source/data0.pak`, `data1.pak`, `data_lang/`.
* DL2   `<root>/ph/{source,work,dlc_opera}`; `<root>/DevTools`; exe `ph/work/bin/x64/DyingLightGame_x64_rwdi.exe`;
  `source/data0.pak`, `data1.pak`, `data_devtools0.pak`, `data_lang/`; `dlc_opera/data0.pak`, `data1.pak`,
  `dlc_opera/data/*.rpack`.

Both games ship `scripts/playerappearances.scr`, `scripts/player/player_outfit_slots.scr` and
`models/player/player_{tpp,fpp}_skeleton.model` in `source/data0.pak`.

Lookup (`find_game`): explicit root, then `$NIGHTRUNNER_GAME_ROOT` (legacy `$BEASTPACK_GAME_ROOT`), then every
Steam library (`libraryfolders.vdf`) under `steamapps/common/<steam_dir>`. Nothing here writes to an install.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

ASSETS_SUB = ("work", "data_platform", "pc", "assets")
EXE_SUB = ("work", "bin", "x64")


@dataclass(frozen=True)
class GameProfile:
    id: str
    name: str
    steam_dir: str
    data_dirs: tuple[str, ...]                 # candidates under the root, preferred first
    exe_names: tuple[str, ...] = ()
    markers: tuple[str, ...] = ()              # root-relative paths that only this game has
    sdb_names: dict = field(default_factory=lambda: {"dx11": "runtime_dx11.sdb", "dx12": "runtime_dx12.sdb"})
    stock_paks: tuple[str, ...] = ("data0.pak", "data1.pak")   # never offered as the mod PAK name
    base_pak: str = "data0.pak"                                 # holds the scripts / player models
    rpack_first: int = 2                                        # first assets_N_pc.rpack for mods
    appearances_script: str | None = None      # player-appearance flow (None = hidden)
    outfit_script: str | None = None           # "No gear" (None = disabled)
    player_models: frozenset = frozenset()

    @property
    def short(self) -> str:
        return self.id.upper()


_PLAYER = frozenset({"player_tpp_skeleton.model", "player_fpp_skeleton.model"})

DLTB = GameProfile(
    id="dltb", name="Dying Light: The Beast", steam_dir="Dying Light The Beast", data_dirs=("ph_ft",),
    exe_names=("DyingLightGame_TheBeast_x64_rwdi.exe",),
    markers=("ph_ft/work/data_platform/pc/assets/dlc_frontier", "ph_ft/work/data_platform/pc/assets/menu_level_ft"),
    appearances_script="scripts/playerappearances.scr",
    outfit_script="scripts/player/player_outfit_slots.scr",
    player_models=_PLAYER)

DL2 = GameProfile(
    id="dl2", name="Dying Light 2", steam_dir="Dying Light 2", data_dirs=("ph", "ph_ft"),
    exe_names=("DyingLightGame_x64_rwdi.exe",),
    markers=("DevTools", "ph/dlc_opera", "ph_ft/dlc_opera", "ph/source/data_devtools0.pak",
             "ph_ft/source/data_devtools0.pak"),
    appearances_script="scripts/playerappearances.scr",
    outfit_script="scripts/player/player_outfit_slots.scr",
    player_models=_PLAYER)

PROFILES: dict[str, GameProfile] = {p.id: p for p in (DLTB, DL2)}
DEFAULT_ID = "dltb"


def profile(pid: str | GameProfile | None) -> GameProfile:
    """Profile by id (case-insensitive); None -> the default (DLTB). Unknown ids raise KeyError."""
    if isinstance(pid, GameProfile):
        return pid
    if not pid:
        return PROFILES[DEFAULT_ID]
    return PROFILES[str(pid).lower()]


def _assets_of(root: Path, data: str) -> Path:
    return root.joinpath(data, *ASSETS_SUB)


def normalize_root(path: str | Path) -> Path:
    """The picked folder, or its parent when the user picked the data folder itself (`.../ph`)."""
    p = Path(path)
    names = {d.lower() for pr in PROFILES.values() for d in pr.data_dirs}
    if p.name.lower() in names and p.joinpath(*ASSETS_SUB).is_dir():
        return p.parent
    return p


def detect_profile(root: str | Path) -> GameProfile | None:
    """Best profile for a root folder, or None when no data folder with assets exists.

    Scores: exe under `<data>/work/bin/x64` (+8), marker path (+4), Steam folder name (+3), a data folder with
    assets that is the profile's preferred one (+2) or merely a candidate (+1). Ties go to registry order (DLTB)."""
    root = normalize_root(root)
    best, best_score = None, 0
    for pr in PROFILES.values():
        datas = [d for d in pr.data_dirs if _assets_of(root, d).is_dir()]
        if not datas:
            continue
        score = 2 if datas[0] == pr.data_dirs[0] else 1
        if any(root.joinpath(d, *EXE_SUB, exe).is_file() for d in datas for exe in pr.exe_names):
            score += 8
        if any(root.joinpath(m).exists() for m in pr.markers):
            score += 4
        if root.name.lower() == pr.steam_dir.lower():
            score += 3
        if score > best_score:
            best, best_score = pr, score
    return best


class GameInstall:
    """An install root plus its profile. Without a profile the root is detected (default DLTB)."""

    def __init__(self, root: str | Path, profile_: GameProfile | str | None = None):
        self.root = normalize_root(root)
        if profile_ is None:
            self.profile = detect_profile(self.root) or profile(None)
        else:
            self.profile = profile(profile_)

    def __repr__(self) -> str:
        return f"GameInstall({str(self.root)!r}, {self.profile.id!r})"

    def __eq__(self, other) -> bool:
        return isinstance(other, GameInstall) and (str(self.root).lower(), self.profile.id) == \
            (str(other.root).lower(), other.profile.id)

    def __hash__(self) -> int:
        return hash((str(self.root).lower(), self.profile.id))

    @property
    def id(self) -> str:
        return self.profile.id

    @property
    def name(self) -> str:
        return self.profile.name

    @property
    def data_name(self) -> str:
        """The data folder in use: the first candidate with assets, else the first that exists, else the first."""
        cands = self.profile.data_dirs
        for d in cands:
            if _assets_of(self.root, d).is_dir():
                return d
        for d in cands:
            if (self.root / d).is_dir() and (self.root / d / "source").is_dir():
                return d
        return cands[0]

    @property
    def data(self) -> Path:
        return self.root / self.data_name

    @property
    def assets(self) -> Path:
        return self.data.joinpath(*ASSETS_SUB)

    @property
    def source(self) -> Path:
        return self.data / "source"

    def rpacks(self) -> list[Path]:
        """Every .rpack under the assets folder, recursively (level packs live in sub-folders), sorted by
        relative path. `custom_rpacks` (third-party loader folder) is included; it is just more packs."""
        if not self.assets.is_dir():
            return []
        out = [p for p in self.assets.rglob("*.rpack") if p.is_file()]
        return sorted(out, key=lambda p: str(p.relative_to(self.assets)).lower())

    def sdb(self, api: str = "dx11") -> Path:
        return self.assets / self.profile.sdb_names.get(api, f"runtime_{api}.sdb")

    def paks(self) -> list[Path]:
        """dataN.pak archives in <data>/source, numeric order (data0 first)."""
        if not self.source.is_dir():
            return []

        def key(p: Path):
            m = re.match(r"data(\d+)\.pak$", p.name, re.I)
            return (int(m.group(1)) if m else 1 << 30, p.name.lower())
        return sorted((p for p in self.source.glob("*.pak") if re.match(r"data\d+\.pak$", p.name, re.I)), key=key)

    def exe(self) -> Path | None:
        for exe in self.profile.exe_names:
            p = self.data.joinpath(*EXE_SUB, exe)
            if p.is_file():
                return p
        return None

    def valid(self) -> bool:
        return self.assets.is_dir()


def steam_roots() -> list[Path]:
    """Steam library folders (registry SteamPath / default Program Files locations + libraryfolders.vdf)."""
    roots: list[Path] = []
    if os.name == "nt":
        try:
            import winreg
            for hive, key in ((winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam"),
                              (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Valve\Steam")):
                try:
                    with winreg.OpenKey(hive, key) as k:
                        for name in ("SteamPath", "InstallPath"):
                            try:
                                roots.append(Path(winreg.QueryValueEx(k, name)[0]))
                            except OSError:
                                pass
                except OSError:
                    pass
        except ImportError:
            pass
    roots += [Path(r"C:\Program Files (x86)\Steam"), Path(r"C:\Program Files\Steam")]
    libs: list[Path] = []
    for r in roots:
        vdf = r / "steamapps" / "libraryfolders.vdf"
        libs.append(r)
        try:
            for m in re.finditer(r'"path"\s+"([^"]+)"', vdf.read_text(encoding="utf-8", errors="replace")):
                libs.append(Path(m.group(1).replace("\\\\", "\\")))
        except OSError:
            pass
    seen, out = set(), []
    for p in libs:
        k = str(p).lower()
        if k not in seen:
            seen.add(k)
            out.append(p)
    return out


def _env_root() -> str | None:
    return os.environ.get("NIGHTRUNNER_GAME_ROOT") or os.environ.get("BEASTPACK_GAME_ROOT") or None


def _libraries() -> list[Path]:
    return steam_roots()


def find_game(explicit: str | Path | None = None, game: str | GameProfile | None = None) -> GameInstall | None:
    """First valid install. *game* (profile id) restricts the search to that game; an explicit / env root whose
    detected profile differs is then skipped. Without *game* the root's own profile is used."""
    want = profile(game) if game else None
    for cand in (explicit, _env_root()):
        if not cand:
            continue
        gi = GameInstall(cand, want) if want else GameInstall(cand)
        if not gi.valid():
            continue
        if want is not None and (detect_profile(gi.root) or want).id != want.id:
            continue
        return gi
    for pr in ([want] if want else PROFILES.values()):
        for lib in _libraries():
            gi = GameInstall(lib / "steamapps" / "common" / pr.steam_dir, pr)
            if gi.valid():
                return gi
    return None


def find_installs(extra: dict[str, str | Path] | None = None) -> dict[str, GameInstall]:
    """{profile id: install} for every game found. *extra* ({id: root}, e.g. remembered roots) wins over the env
    root, which wins over the Steam libraries."""
    out: dict[str, GameInstall] = {}
    for pid, root in (extra or {}).items():
        if pid in PROFILES and root and pid not in out:
            gi = GameInstall(root, pid)
            if gi.valid():
                out[pid] = gi
    env = _env_root()
    if env:
        gi = GameInstall(env)
        if gi.valid() and gi.id not in out:
            out[gi.id] = gi
    libs = None
    for pr in PROFILES.values():
        if pr.id in out:
            continue
        if libs is None:
            libs = _libraries()
        for lib in libs:
            gi = GameInstall(lib / "steamapps" / "common" / pr.steam_dir, pr)
            if gi.valid():
                out[pr.id] = gi
                break
    return out
