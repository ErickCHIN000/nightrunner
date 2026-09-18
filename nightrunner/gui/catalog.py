"""Global, lazily-built index of every logical resource in every loaded pack.

A resource is addressed everywhere in the GUI by one integer, its *global id* (gid):
``gid = pack.base + logical_index``. Packs are opened (mmap + table parse only) and indexed on a background thread
when the catalog starts; tabs receive `packAdded` / `progress` / `ready` signals and can search at any time —
a search only sees the packs indexed so far, and re-running it later sees more.

Thread-safety: indexing appends under `_lock`; readers take a snapshot length first. `search()` is pure Python /
numpy and is meant to be run through `TaskRunner` so typing never blocks the GUI.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, Signal

from ..container import catalogue
from ..container.rp6l import Pack, Resource
from ..util.names import engine_fold


@dataclass(eq=False)
class PackEntry:
    id: int
    path: Path
    label: str                     # path relative to the assets folder (or the file name)
    base: int = 0                  # gid of logical 0
    count: int = 0
    pack: Pack | None = None
    error: str | None = None
    user: bool = False             # opened by hand (not from the install)
    type_counts: dict = field(default_factory=dict)


class Catalog(QObject):
    progress = Signal(int, int)      # packs indexed, packs total
    packAdded = Signal(int)          # pack id (indexed, or failed with .error)
    ready = Signal()                 # the initial batch is done

    def __init__(self, parent=None):
        super().__init__(parent)
        self._lock = threading.RLock()
        self.packs: list[PackEntry] = []
        self._names: list[str] = []
        self._lower: list[str] = []
        self._types = np.zeros(0, dtype=np.uint8)
        self._pack_of = np.zeros(0, dtype=np.int32)
        self._total = 0
        self._by_name: dict[int, dict[bytes, list[int]]] = {}
        self._pending = 0
        self._thread: threading.Thread | None = None
        self.is_ready = False
        self.assets_root: Path | None = None
        self.closed = False            # set by close(): a background index run stops at the next pack

    # ---- loading ----------------------------------------------------------------------------------------------
    def load(self, paths: list[Path], assets_root: Path | None = None, user: bool = False) -> None:
        """Queue packs for background indexing (duplicates ignored)."""
        if assets_root is not None:
            self.assets_root = assets_root
        with self._lock:
            known = {str(p.path).lower() for p in self.packs}
            new = []
            for p in paths:
                if str(p).lower() in known:
                    continue
                known.add(str(p).lower())
                label = p.name
                if self.assets_root is not None:
                    try:
                        label = str(p.relative_to(self.assets_root)).replace("\\", "/")
                    except ValueError:
                        pass
                e = PackEntry(len(self.packs), p, label, user=user)
                self.packs.append(e)
                new.append(e)
            self._pending += len(new)
        if new:
            t = threading.Thread(target=self._index_all, args=(new,), daemon=True, name="catalog-index")
            t.start()
            self._thread = t

    def wait(self, timeout: float = 60.0) -> bool:
        """Block until every queued pack is indexed (tests / scripts only)."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if self._pending == 0:
                    return True
            time.sleep(0.01)
        return False

    def _index_all(self, entries: list[PackEntry]) -> None:
        for e in entries:
            if self.closed:
                with self._lock:
                    self._pending = 0
                return
            self._index(e)
            with self._lock:
                self._pending -= 1
                done = sum(1 for p in self.packs if p.pack is not None or p.error)
                total = len(self.packs)
            self.packAdded.emit(e.id)
            self.progress.emit(done, total)
        with self._lock:
            finished = self._pending == 0
        if finished:
            self.is_ready = True
            self.ready.emit()

    def _index(self, e: PackEntry) -> None:
        try:
            pk = Pack.open(e.path)
        except Exception as exc:  # noqa: BLE001
            e.error = f"{type(exc).__name__}: {exc}"
            return
        blob = pk.name_blob
        offs = pk.name_offsets
        names = []
        types = np.fromiter((lg.type for lg in pk.logicals), dtype=np.uint8, count=len(pk.logicals))
        for lg in pk.logicals:
            o = offs[lg.name_index]
            end = blob.find(b"\0", o)
            names.append(blob[o:end if end >= 0 else len(blob)].decode("utf-8", "surrogateescape"))
        lower = [n.lower() for n in names]
        tc: dict[int, int] = {}
        for t in types.tolist():
            tc[t] = tc.get(t, 0) + 1
        with self._lock:
            if self.closed:              # closed while this pack was being read
                try:
                    pk.close()
                except BufferError:
                    pass
                return
            e.pack = pk
            e.base = self._total
            e.count = len(names)
            e.type_counts = dict(sorted(tc.items()))
            self._names += names
            self._lower += lower
            self._types = np.concatenate([self._types, types])
            self._pack_of = np.concatenate([self._pack_of, np.full(len(names), e.id, dtype=np.int32)])
            self._total += len(names)
            self._by_name.clear()

    def close(self) -> None:
        self.closed = True
        with self._lock:
            for e in self.packs:
                if e.pack is not None:
                    try:
                        e.pack.close()
                    except BufferError:
                        pass

    def indexed(self) -> tuple[int, list[int]]:
        """Consistent snapshot: (resource count, ids of packs that finished indexing, failed ones included)."""
        with self._lock:
            return self._total, [p.id for p in self.packs if p.pack is not None or p.error]

    # ---- addressing -------------------------------------------------------------------------------------------
    def __len__(self) -> int:
        return self._total

    def entry(self, gid: int) -> PackEntry:
        return self.packs[int(self._pack_of[gid])]

    def split(self, gid: int) -> tuple[PackEntry, int]:
        e = self.entry(gid)
        return e, gid - e.base

    def pack(self, gid: int) -> Pack:
        return self.entry(gid).pack

    def resource(self, gid: int) -> Resource:
        e, i = self.split(gid)
        return e.pack.resource(i)

    def name(self, gid: int) -> str:
        return self._names[gid]

    def type(self, gid: int) -> int:
        return int(self._types[gid])

    def gid(self, pack_id: int, index: int) -> int:
        return self.packs[pack_id].base + index

    def part_size(self, gid: int) -> int:
        e, i = self.split(gid)
        pk = e.pack
        lg = pk.logicals[i]
        return sum(pk.physicals[j].size for j in range(lg.first_part, lg.first_part + lg.part_count))

    def type_counts(self) -> dict[int, int]:
        with self._lock:
            n = self._total
            t = self._types[:n]
        vals, counts = np.unique(t, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    # ---- search -----------------------------------------------------------------------------------------------
    def search(self, text: str = "", types=None, packs=None) -> np.ndarray:
        """gids matching every whitespace-separated word of *text* (case-insensitive substring), restricted to
        *types* (iterable of type ids) and *packs* (iterable of pack ids). Order: pack load order, then logical
        order. Safe to call from a worker thread."""
        with self._lock:
            n = self._total
            tarr = self._types[:n]
            parr = self._pack_of[:n]
            lower = self._lower
        mask = np.ones(n, dtype=bool)
        if types is not None:
            mask &= np.isin(tarr, np.fromiter(types, dtype=np.uint8))
        if packs is not None:
            mask &= np.isin(parr, np.fromiter(packs, dtype=np.int32))
        cand = np.flatnonzero(mask)
        words = [w for w in text.lower().split() if w]
        if not words:
            return cand
        w0 = words[0]
        rest = words[1:]
        if len(cand) == n:
            hit = [i for i, s in enumerate(lower[:n]) if w0 in s and all(w in s for w in rest)]
        else:
            hit = [i for i in cand.tolist() if w0 in lower[i] and all(w in lower[i] for w in rest)]
        return np.asarray(hit, dtype=np.int64)

    def lookup(self, name: str | bytes, type_id: int) -> list[int]:
        """gids of resources of *type_id* whose name equals *name* under engine folding (ASCII case-insensitive).
        Meshes also match with/without a `.msh` suffix; textures fall back to the basename after '/' or '\\'."""
        key = name.encode("utf-8", "surrogateescape") if isinstance(name, str) else name
        key = engine_fold(key)
        table = self._name_table(type_id)
        hits = table.get(key)
        if not hits and type_id == 0x10 and key.endswith(b".msh"):
            hits = table.get(key[:-4])
        if not hits and type_id == 0x20:
            base = key.replace(b"\\", b"/").rsplit(b"/", 1)[-1]
            hits = table.get(base)
        return list(hits or [])

    def _name_table(self, type_id: int) -> dict[bytes, list[int]]:
        with self._lock:
            t = self._by_name.get(type_id)
            if t is not None:
                return t
            n = self._total
            t = {}
            for gid in np.flatnonzero(self._types[:n] == type_id).tolist():
                k = engine_fold(self._names[gid].encode("utf-8", "surrogateescape"))
                t.setdefault(k, []).append(gid)
            self._by_name[type_id] = t
            return t

    # ---- preferred packs (previews of built mods) ------------------------------------------------------------
    def pack_ids(self, paths) -> list[int]:
        """Ids of loaded packs by path (same order as *paths*; unknown paths skipped)."""
        with self._lock:
            by = {str(e.path).lower(): e.id for e in self.packs}
        out = []
        for p in paths:
            i = by.get(str(Path(p)).lower())
            if i is None:
                i = by.get(str(Path(p).resolve()).lower())
            if i is not None:
                out.append(i)
        return out

    def wait_packs(self, pack_ids, timeout: float = 30.0) -> bool:
        """Block until the given packs are indexed (or failed). Worker threads only."""
        end = time.time() + timeout
        while time.time() < end:
            with self._lock:
                if all(self.packs[i].pack is not None or self.packs[i].error for i in pack_ids):
                    return True
            time.sleep(0.01)
        return False

    def prefer(self, pack_ids) -> "CatalogView":
        """A view whose lookup() lists gids of *pack_ids* first (in that order); everything else is this catalog."""
        return CatalogView(self, pack_ids)

    @staticmethod
    def type_label(t: int) -> str:
        return f"0x{t:02X} {catalogue.type_name(t)}"


class CatalogView:
    """Catalog proxy for previews: same gids, lookup() ranks the preferred packs first."""

    def __init__(self, catalog: Catalog, pack_ids):
        self._cat = catalog
        self._rank = {int(i): n for n, i in enumerate(pack_ids)}

    def lookup(self, name, type_id: int) -> list[int]:
        hits = self._cat.lookup(name, type_id)
        if not self._rank or len(hits) < 2:
            return hits
        far = len(self._rank)
        return sorted(hits, key=lambda g: self._rank.get(int(self._cat._pack_of[g]), far))

    def __getattr__(self, name):
        return getattr(self._cat, name)
