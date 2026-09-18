"""Shared, lazily-opened data sources: the SDB material database and the PAK archives (.model documents).

Both open on first use (cheap: mmap / ZIP central directory). Expensive derived data (the texture → materials
reverse index over all 26k materials, ~5 s) is built on a background thread and cached on disk keyed by the SDB's
size + mtime, so later launches load it in milliseconds.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from ..pak.model_json import PakIndex
from ..sdb.reader import Sdb

CACHE_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "nightrunner" / "cache"


class SdbService(QObject):
    """One SDB (dx11 by default). `sdb` opens it on first access; `material_info(name)` is the resolver dict;
    `materials_using(texture)` needs the reverse index (`reverseReady` fires once it is available)."""
    reverseReady = Signal()
    failed = Signal(str)

    def __init__(self, path: Path | None, parent=None):
        super().__init__(parent)
        self.path = path
        self._sdb: Sdb | None = None
        self._lock = threading.RLock()
        self._reverse: dict[str, list[str]] | None = None
        self._reverse_started = False
        self._generation = 0
        self.error: str | None = None

    def available(self) -> bool:
        return self.path is not None and self.path.is_file()

    @property
    def sdb(self) -> Sdb | None:
        with self._lock:
            if self._sdb is None and self.available() and self.error is None:
                try:
                    self._sdb = Sdb.open(self.path)
                except Exception as exc:  # noqa: BLE001
                    self.error = f"{type(exc).__name__}: {exc}"
                    self.failed.emit(self.error)
            return self._sdb

    def set_path(self, path: Path | None) -> None:
        with self._lock:
            self.close()
            self.path, self.error = path, None
            self._reverse, self._reverse_started = None, False
            self._generation += 1

    def close(self) -> None:
        with self._lock:
            if self._sdb is not None:
                try:
                    self._sdb.close()
                except BufferError:
                    pass
                self._sdb = None

    # ---- lookups ----------------------------------------------------------------------------------------------
    # The reader builds its lookup indices lazily and is not thread-safe while doing so: every call that can
    # trigger that goes through the service lock.
    def material_names(self) -> list[str]:
        with self._lock:
            s = self.sdb
            return s.materials() if s else []

    def material_info(self, name_or_index) -> dict | None:
        """`Sdb.material()` dict (routes, parameters, variants, texture bindings) or None if unknown."""
        with self._lock:
            s = self.sdb
            if s is None:
                return None
            try:
                return s.material(name_or_index)
            except Exception:  # noqa: BLE001 - reader raises its own error types for unknown names
                return None

    def textures_for(self, material: str) -> list[str]:
        with self._lock:
            s = self.sdb
            if s is None:
                return []
            try:
                return sorted(s.textures_for_material(material))
            except Exception:  # noqa: BLE001
                return []

    def locked(self):
        """Context manager for direct `self.sdb` use from worker threads."""
        return self._lock

    # ---- reverse index ------------------------------------------------------------------------------------------
    def materials_using(self, texture: str) -> list[str] | None:
        """Materials whose resolved bindings name *texture* (case-insensitive); None while the index builds."""
        idx = self._reverse
        if idx is None:
            self.ensure_reverse()
            return None
        return idx.get(texture.lower(), [])

    def reverse_ready(self) -> bool:
        return self._reverse is not None

    def texture_index(self) -> dict[str, list[str]] | None:
        """Lower-case texture name -> material names, or None while building."""
        return self._reverse

    def ensure_reverse(self) -> None:
        with self._lock:
            if self._reverse is not None or self._reverse_started or not self.available():
                return
            self._reverse_started = True
        threading.Thread(target=self._build_reverse, args=(self._generation,), daemon=True,
                         name="sdb-reverse").start()

    def _cache_file(self) -> Path:
        st = self.path.stat()
        return CACHE_DIR / f"sdb_reverse_{self.path.stem}_{st.st_size}_{int(st.st_mtime)}.json"

    def _build_reverse(self, generation: int) -> None:
        try:
            cf = self._cache_file()
            if cf.is_file():
                rev = json.loads(cf.read_text(encoding="utf-8"))
            else:
                with self._lock:
                    s = self.sdb
                    names = s.materials() if s else []
                rev = {}
                for i, m in enumerate(names):
                    with self._lock:
                        if generation != self._generation or self._sdb is None:
                            return
                        try:
                            texs = self._sdb.textures_for_material(i)
                        except Exception:  # noqa: BLE001
                            continue
                    for t in texs:
                        rev.setdefault(t.lower(), []).append(m)
                try:
                    CACHE_DIR.mkdir(parents=True, exist_ok=True)
                    tmp = cf.with_suffix(".partial")
                    tmp.write_text(json.dumps(rev), encoding="utf-8")
                    os.replace(tmp, cf)
                except OSError:
                    pass
            with self._lock:
                if generation != self._generation:
                    return              # the SDB was switched meanwhile: drop the stale index
                self._reverse = rev
            self.reverseReady.emit()
        except Exception as exc:  # noqa: BLE001
            self.error = f"reverse index: {exc}"
            self.failed.emit(self.error)


class PakService(QObject):
    """The dataN.pak archives. Later archives override earlier ones member-by-member for the listing (a
    documented RT convention for root-level data3.pak members, see notes/FORMATS/pak-and-model.md §4 —
    precedence is not native-verified, so every provider is kept and shown)."""

    def __init__(self, paths: list[Path], parent=None):
        super().__init__(parent)
        self.paths = list(paths)
        self._idx: dict[Path, PakIndex] = {}
        self._lock = threading.RLock()
        self._models: list[dict] | None = None

    def index(self, path: Path) -> PakIndex | None:
        with self._lock:
            if path not in self._idx:
                try:
                    self._idx[path] = PakIndex.open(path)
                except Exception:  # noqa: BLE001
                    return None
            return self._idx[path]

    def models(self) -> list[dict]:
        """[{name, member, pak, basename, overridden_by: [pak names]}] for every .model member, data0 first."""
        with self._lock:
            if self._models is not None:
                return self._models
            out: list[dict] = []
            by_base: dict[str, list[dict]] = {}
            for p in self.paths:
                ix = self.index(p)
                if ix is None:
                    continue
                for m in ix.list_models():
                    base = m.name.replace("\\", "/").rsplit("/", 1)[-1].lower()
                    rec = {"name": m.name, "member": m, "pak": p, "basename": base, "overridden_by": []}
                    for prev in by_base.get(base, []):
                        prev["overridden_by"].append(p.name)
                    by_base.setdefault(base, []).append(rec)
                    out.append(rec)
            self._models = out
            return out

    def find_model(self, name: str) -> dict | None:
        """Model record by member path (either slash) or basename (with or without .model); the last provider
        in pak order wins, matching the listing's override marker."""
        key = name.replace("\\", "/").lower()
        base = key.rsplit("/", 1)[-1]
        if not base.endswith(".model"):
            base += ".model"
        models = self.models()
        exact = [m for m in models if m["name"].replace("\\", "/").lower() == key]
        if exact:
            return exact[-1]
        hits = [m for m in models if m["basename"] == base]
        return hits[-1] if hits else None

    def load_model(self, rec: dict) -> dict:
        ix = self.index(rec["pak"])
        return ix.load_model(rec["member"])

    def close(self) -> None:
        with self._lock:
            for ix in self._idx.values():
                try:
                    ix.close()
                except Exception:  # noqa: BLE001
                    pass
            self._idx.clear()
