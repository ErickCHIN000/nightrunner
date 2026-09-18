"""AppContext: the objects every tab shares, plus cross-tab navigation.

Tabs get the context in their constructor and must not reach into each other directly; to jump, they emit one
of the `open*` signals and the main window switches tabs and forwards the request to the tab's `open_*` slot.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QSettings, Signal

from .catalog import Catalog
from ..games import PROFILES, GameInstall, detect_profile, find_installs
from .services import PakService, SdbService
from .tasks import TaskRunner


class AppContext(QObject):
    openTexture = Signal(int)        # gid
    openMesh = Signal(int)           # gid
    openMaterial = Signal(str)       # SDB material name
    openModel = Signal(str)          # .model member name (or basename)
    openRaw = Signal(int)            # gid
    openModelDoc = Signal(str, object, object)   # name, in-memory .model doc, extra pack paths (list[Path])
    status = Signal(str)             # transient status-bar message
    sdbChanged = Signal(str)         # "dx11" | "dx12" after set_sdb_api()

    def __init__(self, settings: QSettings | None = None, game: GameInstall | None = None, parent=None,
                 autoload: bool = True):
        super().__init__(parent)
        self.settings = settings or _settings()
        self.game = game if game is not None else resolve_game(self.settings)
        self.catalog = Catalog(self)
        self.runner = TaskRunner(self)
        api = self.settings.value("sdb/api", "dx11")
        self.sdb = SdbService(self.game.sdb(api) if self.game else None, self)
        self.paks = PakService(self.game.paks() if self.game else [], self)
        if autoload:
            self.load_game()

    def sdb_api(self) -> str:
        return str(self.settings.value("sdb/api", "dx11"))

    def set_sdb_api(self, api: str) -> None:
        """Switch runtime_dx11.sdb / runtime_dx12.sdb for every tab (menu and SDB tab dropdown both call this)."""
        api = "dx12" if str(api).lower() == "dx12" else "dx11"
        if api == self.sdb_api() and self.sdb.path is not None:
            return
        self.settings.setValue("sdb/api", api)
        if self.game:
            self.sdb.set_path(self.game.sdb(api))
        self.sdbChanged.emit(api)

    def game_name(self) -> str:
        return self.game.name if self.game is not None else ""

    def load_game(self) -> None:
        if self.game is not None:
            self.catalog.load(self.game.rpacks(), assets_root=self.game.assets)

    def open_user_packs(self, paths: list[Path]) -> None:
        self.catalog.load([Path(p).resolve() for p in paths], user=True)

    def export_dir(self) -> str:
        return self.settings.value("paths/export_dir", "") or ""

    def set_export_dir(self, d: str) -> None:
        self.settings.setValue("paths/export_dir", d)

    def close(self) -> None:
        self.catalog.close()
        self.sdb.close()
        self.paks.close()


# ---- game choice (QSettings: game/current = id, game/root/<id> = root folder) ------------------------------------

def remembered_roots(settings) -> dict[str, str]:
    """{id: root} saved per game; the pre-profile `paths/game_root` counts for whichever game it holds."""
    out = {}
    for pid in PROFILES:
        r = str(settings.value(f"game/root/{pid}", "") or "")
        if r:
            out[pid] = r
    legacy = str(settings.value("paths/game_root", "") or "")
    if legacy:
        pr = detect_profile(legacy)
        if pr is not None and pr.id not in out:
            out[pr.id] = legacy
    return out


def detected_installs(settings) -> dict[str, GameInstall]:
    """{id: install} for every game found (remembered roots first, then env / Steam), registry order."""
    found = find_installs(remembered_roots(settings))
    return {pid: found[pid] for pid in PROFILES if pid in found}


def resolve_game(settings, installs: dict[str, GameInstall] | None = None) -> GameInstall | None:
    """The install to open: the last used game, else DLTB, else any detected one."""
    installs = detected_installs(settings) if installs is None else installs
    cur = str(settings.value("game/current", "") or "")
    for pid in (cur, "dltb", *installs):
        if pid in installs:
            return installs[pid]
    return None


def remember_game(settings, game: GameInstall) -> None:
    settings.setValue("game/current", game.id)
    settings.setValue(f"game/root/{game.id}", str(game.root))


def _settings():
    """The app's QSettings; the first run after the rename copies the old beastpack settings over."""
    from PySide6.QtCore import QSettings
    s = QSettings("nightrunner", "nightrunner")
    if not s.allKeys():
        old = QSettings("beastpack", "explorer")
        for k in old.allKeys():
            s.setValue(k, old.value(k))
        s.sync()
    return s
