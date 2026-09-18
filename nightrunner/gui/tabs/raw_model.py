"""Lazy, checkable tree over every pack of the catalog (Raw tab).

Tree shape::

    <pack label>                      pack file (path relative to assets)
      Header / Storage table / Name table          byte ranges of the tables (hidden while a filter is active)
      <Type>  (N)                     one group per logical type, resources in logical order (fetched in chunks)
        <name>                        logical record
          part k  <PartType>          physical record -> part bytes

Only packs, table nodes and groups are Python objects. Resource and part rows are encoded in the index's
`internalId` (kind | group node | row | part ordinal), so a fully-scrolled 60k group costs no per-row objects.

Check state is stored per pack as a bool array over logical indices (resource fully checked), a small dict of
partially-checked resources (logical -> set of part ordinals) and a set of checked table nodes. Group and pack
states are derived from those on display (cached per check-state version), so checking a 39k-texture group or a
whole pack is one numpy assignment and never materialises rows.

The pure helpers (`build_info`, `group_rows`, `parse_query`, `compute_view`, `iter_export`) run on worker threads.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PySide6.QtCore import QAbstractItemModel, QModelIndex, Qt, Signal
from PySide6.QtGui import QColor

from ...container import catalogue
from ...container.rp6l import HEADER_SIZE, STORAGE_SIZE, Pack
from ...extract import part_filename
from ...util.names import resource_dirname, safe_filename
from ..widgets import human_size, mono_font

COLUMNS = ("Name", "Type", "Index", "Offset", "Size")
COL_NAME, COL_TYPE, COL_INDEX, COL_OFFSET, COL_SIZE = range(5)
CHUNK = 2000                                  # minimum resource rows added per fetchMore()
# Every insert makes QTreeView re-lay-out the whole expanded tree (a few Python calls per shown row), so a chunk is
# as large as what is already loaded: a 60k group is complete after 6 fetches instead of 30 (linear total cost).

TABLES = ("header", "storages", "names")
TABLE_FILES = {"header": "header.bin", "storages": "storage_table.bin", "names": "name_table.bin"}

# internalId layout (always < 2**62, so it fits quintptr on every platform)
_KIND_SHIFT, _NODE_SHIFT, _ROW_SHIFT = 60, 36, 4
K_NODE, K_RES, K_PART = 0, 1, 2
_NODE_MASK, _ROW_MASK, _PART_MASK = (1 << 24) - 1, (1 << 32) - 1, 0xF

_F_BASE = Qt.ItemFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
_F_CHECK = Qt.ItemFlags(_F_BASE | Qt.ItemIsUserCheckable)
_F_PART = Qt.ItemFlags(_F_BASE | Qt.ItemNeverHasChildren)
_F_PART_CHECK = Qt.ItemFlags(_F_CHECK | Qt.ItemNeverHasChildren)
_RES_ID, _PART_ID = K_RES << _KIND_SHIFT, K_PART << _KIND_SHIFT

GREY = QColor(140, 140, 140)
RED = QColor(230, 90, 90)


# ---- per-pack data (built on a worker) -------------------------------------------------------------------------------

@dataclass(eq=False)
class PackInfo:
    """Numpy views of one pack's tables: everything the tree and the check counter need per logical index."""
    pid: int
    pack: Pack
    types: np.ndarray            # uint8, per logical
    first: np.ndarray            # int64, first physical index
    nparts: np.ndarray           # int64
    sizes: np.ndarray            # int64, sum of the part sizes
    tables: dict[str, tuple[int, int]]

    @property
    def count(self) -> int:
        return len(self.types)

    def table_label(self, which: str) -> str:
        h = self.pack.header
        return {"header": "Header", "storages": f"Storage table ({h.storage_count})",
                "names": f"Name table ({h.name_count:,})"}[which]


def build_info(pid: int, pk: Pack) -> PackInfo:
    n = len(pk.logicals)
    types = np.fromiter((lg.type for lg in pk.logicals), dtype=np.uint8, count=n)
    first = np.fromiter((lg.first_part for lg in pk.logicals), dtype=np.int64, count=n)
    nparts = np.fromiter((lg.part_count for lg in pk.logicals), dtype=np.int64, count=n)
    psize = np.fromiter((p.size for p in pk.physicals), dtype=np.int64, count=len(pk.physicals))
    cs = np.concatenate([np.zeros(1, np.int64), np.cumsum(psize)])
    sizes = cs[first + nparts] - cs[first] if n else np.zeros(0, np.int64)
    h = pk.header
    tables = {"header": (0, HEADER_SIZE),
              "storages": (pk.storage_offset, h.storage_count * STORAGE_SIZE),
              "names": (pk.name_offset_offset, pk.table_end - pk.name_offset_offset)}
    return PackInfo(pid, pk, types, first, nparts, sizes, tables)


def group_rows(info: PackInfo, li: np.ndarray | None = None) -> list[tuple[int, np.ndarray]]:
    """[(type, ascending logical indices)] for the logical indices *li* (all when None), types ascending."""
    if li is None:
        li = np.arange(info.count, dtype=np.int64)
    if not len(li):
        return []
    t = info.types[li]
    order = np.argsort(t, kind="stable")
    st, sl = t[order], li[order]
    uniq, starts = np.unique(st, return_index=True)
    bounds = list(starts.tolist()) + [len(st)]
    return [(int(u), sl[bounds[k]:bounds[k + 1]]) for k, u in enumerate(uniq.tolist())]


def parse_query(text: str) -> tuple[list[str], list[int], bool]:
    """(words, logical indices from `#123` tokens, malformed-#-token seen)."""
    words, idx, bad = [], [], False
    for w in text.split():
        if w.startswith("#"):
            if w[1:].isdigit():
                idx.append(int(w[1:]))
            else:
                bad = True
        else:
            words.append(w)
    return words, idx, bad


@dataclass(eq=False)
class PackView:
    entry: object                          # catalog.PackEntry
    info: PackInfo | None
    groups: list[tuple[int, np.ndarray]]
    error: str | None = None


def indexed_entries(cat) -> tuple[int, list]:
    """(resource count, packs fully indexed or failed) — a consistent snapshot of the catalog."""
    lock = getattr(cat, "_lock", None)
    if lock is None:
        n = len(cat)
        return n, [p for p in list(cat.packs) if p.error or (p.pack is not None and p.base + p.count <= n)]
    with lock:
        n = len(cat)
        return n, [p for p in cat.packs if p.error or p.pack is not None]


def compute_view(cat, text: str, type_id: int | None, pids=None, infos: dict | None = None) -> list[PackView]:
    """Tree content for the query: every indexed pack (or *pids*), only packs with matches when filtering.
    Runs on a worker thread; *infos* are reused and the ones built here are added to it (pass a private copy)."""
    infos = {} if infos is None else infos
    n, entries = indexed_entries(cat)
    if pids is not None:
        want = set(pids)
        entries = [e for e in entries if e.id in want]
    words, idx, bad = parse_query(text)
    filtering = bool(words or idx or bad or type_id is not None)
    gids = None
    if filtering and not bad:
        live = [e.id for e in entries if e.pack is not None]
        gids = cat.search(" ".join(words), types=None if type_id is None else [type_id], packs=live) if live else \
            np.zeros(0, np.int64)
        gids = np.asarray(gids, dtype=np.int64)
        gids = gids[gids < n]
    out = []
    for e in entries:
        if e.pack is None:
            if not filtering:
                out.append(PackView(e, None, [], e.error or "not open"))
            continue
        info = infos.get(e.id)
        if info is None:
            try:
                info = infos[e.id] = build_info(e.id, e.pack)
            except Exception as exc:  # noqa: BLE001 - a broken table must not kill the tree
                out.append(PackView(e, None, [], f"{type(exc).__name__}: {exc}"))
                continue
        if not filtering:
            out.append(PackView(e, info, group_rows(info)))
            continue
        if gids is None:
            continue
        lo, hi = np.searchsorted(gids, [e.base, e.base + e.count])
        li = gids[lo:hi] - e.base
        for k in idx:
            li = li[li == k]
        if len(li):
            out.append(PackView(e, info, group_rows(info, li)))
    return out


# ---- check state -------------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class PackChecks:
    mask: np.ndarray                                    # bool per logical: every part checked
    partial: dict[int, set[int]] = field(default_factory=dict)   # logical -> checked part ordinals (not all)
    tables: set[str] = field(default_factory=set)

    def any(self) -> bool:
        return bool(self.tables or self.partial or self.mask.any())

    def copy(self) -> "PackChecks":
        return PackChecks(self.mask.copy(), {k: set(v) for k, v in self.partial.items()}, set(self.tables))

    def set_rows(self, rows: np.ndarray, on: bool) -> None:
        self.mask[rows] = on
        if self.partial:
            for k in np.intersect1d(np.fromiter(self.partial, np.int64), rows).tolist():
                del self.partial[k]


def checks_totals(info: PackInfo, ch: PackChecks) -> tuple[int, int]:
    """(parts, bytes) selected by *ch*."""
    parts = int(info.nparts[ch.mask].sum())
    size = int(info.sizes[ch.mask].sum())
    phys = info.pack.physicals
    for li, ks in ch.partial.items():
        parts += len(ks)
        f = int(info.first[li])
        size += sum(phys[f + k].size for k in ks)
    for t in ch.tables:
        parts += 1
        size += info.tables[t][1]
    return parts, size


@dataclass
class ExportFile:
    pack: Pack
    offset: int
    size: int
    rel: str                  # path relative to the output folder ('/' separated)
    direct: bool = True       # False: child-pack / compressed part, not extractable raw


def pack_dirname(label: str) -> str:
    """Output folder of a pack: its label without `.rpack`, each path segment made filesystem-safe."""
    lab = label[:-6] if label.lower().endswith(".rpack") else label
    return "/".join(safe_filename(s) for s in lab.replace("\\", "/").split("/") if s not in ("", ".", ".."))


def iter_export(info: PackInfo, label: str, ch: PackChecks, names=None) -> Iterator[ExportFile]:
    """Files for the checked items of one pack, in logical order: tables first, then
    `<pack>/<family>/<index>_<name>/<part file>` (part file names from `extract.part_filename`).
    *names(li)* returns a resource name (defaults to the pack's name table)."""
    pk = info.pack
    root = pack_dirname(label)
    for t in TABLES:
        if t in ch.tables:
            off, size = info.tables[t]
            yield ExportFile(pk, off, size, f"{root}/_tables/{TABLE_FILES[t]}")
    full = np.flatnonzero(ch.mask)
    lis = full if not ch.partial else np.union1d(full, np.fromiter(ch.partial, np.int64))
    for li in lis.tolist():
        lg = pk.logicals[li]
        want = None if ch.mask[li] else ch.partial.get(li, set())
        name = names(li) if names else pk.name(lg.name_index)
        rdir = f"{root}/{catalogue.family_dir(lg.type)}/{resource_dirname(li, name)}"
        seen: dict[str, int] = {}
        for k in range(lg.part_count):
            j = lg.first_part + k
            fname = part_filename(k, pk.part_type(j), seen)
            if want is not None and k not in want:
                continue
            yield ExportFile(pk, pk.part_offset(j), pk.physicals[j].size, f"{rdir}/{fname}", pk.part_is_direct(j))


# ---- the model ---------------------------------------------------------------------------------------------------------

@dataclass(eq=False)
class _Node:
    kind: str                      # pack | table | group
    pid: int
    idx: int = 0                   # position in RawTreeModel._nodes
    row: int = 0
    parent: int = -1               # node idx (-1: top level)
    which: str = ""                # table name
    type: int = 0                  # group type
    rows: np.ndarray | None = None  # group: logical indices
    fetched: int = 0
    children: list[int] = field(default_factory=list)
    error: str | None = None
    label: str = ""


class RawTreeModel(QAbstractItemModel):
    checksChanged = Signal()

    def __init__(self, catalog, parent=None):
        super().__init__(parent)
        self.cat = catalog
        self.infos: dict[int, PackInfo] = {}
        self.checks: dict[int, PackChecks] = {}
        self.filtering = False
        self._nodes: list[_Node] = []
        self._top: list[int] = []            # pack node idxs, ordered by pack id
        self._by_pid: dict[int, int] = {}
        self._groups: dict[tuple[int, int], int] = {}
        self._ver = 0
        self._state_cache: dict[int, Qt.CheckState] = {}
        self._mono = mono_font()

    # ---- building ---------------------------------------------------------------------------------------------
    def _new_node(self, **kw) -> _Node:
        n = _Node(idx=len(self._nodes), **kw)
        self._nodes.append(n)
        return n

    def _make_pack(self, pv: PackView) -> _Node:
        e = pv.entry
        pn = self._new_node(kind="pack", pid=e.id, error=pv.error, label=e.label)
        if pv.info is None:
            return pn
        self.infos.setdefault(e.id, pv.info)
        if e.id not in self.checks:
            self.checks[e.id] = PackChecks(np.zeros(pv.info.count, dtype=bool))
        kids = []
        if not self.filtering:
            for t in TABLES:
                kids.append(self._new_node(kind="table", pid=e.id, which=t, parent=pn.idx,
                                           label=pv.info.table_label(t)).idx)
        for t, rows in pv.groups:
            g = self._new_node(kind="group", pid=e.id, type=t, rows=rows, parent=pn.idx,
                               label=f"{catalogue.type_name(t)}  ({len(rows):,})")
            self._groups[(e.id, t)] = g.idx
            kids.append(g.idx)
        for r, k in enumerate(kids):
            self._nodes[k].row = r
        pn.children = kids
        return pn

    def reset_view(self, views: list[PackView], filtering: bool) -> None:
        self.beginResetModel()
        self.filtering = filtering
        self._nodes, self._top, self._by_pid, self._groups = [], [], {}, {}
        for pv in sorted(views, key=lambda v: v.entry.id):
            pn = self._make_pack(pv)
            pn.row = len(self._top)
            self._top.append(pn.idx)
            self._by_pid[pn.pid] = pn.idx
        self._touch()
        self.endResetModel()

    def add_pack_view(self, pv: PackView) -> None:
        """Insert (or ignore, if present) one pack in pack-id order without resetting the view."""
        if pv.entry.id in self._by_pid:
            return
        if self.filtering and not pv.groups:
            return
        pids = [self._nodes[i].pid for i in self._top]
        pos = int(np.searchsorted(np.asarray(pids, dtype=np.int64), pv.entry.id))
        self.beginInsertRows(QModelIndex(), pos, pos)
        pn = self._make_pack(pv)
        self._top.insert(pos, pn.idx)
        for r, i in enumerate(self._top):
            self._nodes[i].row = r
        self._by_pid[pn.pid] = pn.idx
        self._touch()
        self.endInsertRows()

    def has_pack(self, pid: int) -> bool:
        return pid in self._by_pid

    # ---- addressing -------------------------------------------------------------------------------------------
    @staticmethod
    def _rid(kind: int, node: int, row: int = 0, part: int = 0) -> int:
        return (kind << _KIND_SHIFT) | (node << _NODE_SHIFT) | (row << _ROW_SHIFT) | part

    @staticmethod
    def decode(index: QModelIndex) -> tuple[int, int, int, int]:
        """(kind, node idx, row in group, part ordinal) of a valid index."""
        v = int(index.internalId())
        return v >> _KIND_SHIFT, (v >> _NODE_SHIFT) & _NODE_MASK, (v >> _ROW_SHIFT) & _ROW_MASK, v & _PART_MASK

    def node_of(self, index: QModelIndex) -> _Node | None:
        """The pack / table / group object behind *index* (None for resource and part rows)."""
        if not index.isValid():
            return None
        k, n, _, _ = self.decode(index)
        return self._nodes[n] if k == K_NODE else None

    def item(self, index: QModelIndex) -> dict:
        """Plain description of a row: kind, pid, and li / part / which / type / group as applicable."""
        k, n, r, p = self.decode(index)
        node = self._nodes[n]
        if k == K_NODE:
            d = {"kind": node.kind, "pid": node.pid, "node": node}
            if node.kind == "table":
                d["which"] = node.which
            if node.kind == "group":
                d["type"] = node.type
            return d
        li = int(node.rows[r])
        d = {"kind": "resource" if k == K_RES else "part", "pid": node.pid, "li": li, "group": node, "row": r}
        if k == K_PART:
            d["part"] = p
        return d

    def gid_of(self, index: QModelIndex) -> int | None:
        if not index.isValid():
            return None
        it = self.item(index)
        if "li" not in it:
            return None
        return self.cat.packs[it["pid"]].base + it["li"]

    def node_index(self, node: _Node, column: int = 0) -> QModelIndex:
        return self.createIndex(node.row, column, self._rid(K_NODE, node.idx))

    def pack_index(self, pid: int) -> QModelIndex:
        i = self._by_pid.get(pid)
        return QModelIndex() if i is None else self.node_index(self._nodes[i])

    def group_node(self, pid: int, type_id: int) -> _Node | None:
        i = self._groups.get((pid, type_id))
        return None if i is None else self._nodes[i]

    def resource_row(self, pid: int, li: int) -> tuple[_Node, int] | None:
        """(group node, row) where logical *li* of pack *pid* sits in the current view, or None."""
        info = self.infos.get(pid)
        if info is None or not 0 <= li < info.count:
            return None
        g = self.group_node(pid, int(info.types[li]))
        if g is None:
            return None
        r = int(np.searchsorted(g.rows, li))
        if r >= len(g.rows) or int(g.rows[r]) != li:
            return None
        return g, r

    def resource_index(self, g: _Node, row: int, column: int = 0) -> QModelIndex:
        """Index of a resource row, fetching the chunk that contains it first."""
        self.ensure_fetched(g, row + 1)
        return self.createIndex(row, column, self._rid(K_RES, g.idx, row))

    def ensure_fetched(self, g: _Node, upto: int) -> None:
        upto = min(len(g.rows), upto)
        if upto > g.fetched:
            upto = min(len(g.rows), max(upto, g.fetched + max(CHUNK, g.fetched)))
            self.beginInsertRows(self.node_index(g), g.fetched, upto - 1)
            g.fetched = upto
            self.endInsertRows()

    def visible_pack_ids(self) -> list[int]:
        return [self._nodes[i].pid for i in self._top]

    def group_nodes(self, pid: int | None = None) -> list[_Node]:
        return [n for n in self._nodes if n.kind == "group" and (pid is None or n.pid == pid)]

    # ---- QAbstractItemModel -----------------------------------------------------------------------------------
    def index(self, row, column, parent=QModelIndex()):
        if row < 0 or column < 0 or column >= len(COLUMNS):
            return QModelIndex()
        if not parent.isValid():
            if row < len(self._top):
                return self.createIndex(row, column, self._rid(K_NODE, self._top[row]))
            return QModelIndex()
        v = parent.internalId()
        if v < _RES_ID:                                  # structural parent
            node = self._nodes[v >> _NODE_SHIFT]
            if node.rows is not None:
                if row < node.fetched:
                    return self.createIndex(row, column, _RES_ID | v | (row << _ROW_SHIFT))
                return QModelIndex()
            if node.kind == "pack" and row < len(node.children):
                return self.createIndex(row, column, node.children[row] << _NODE_SHIFT)
            return QModelIndex()
        k, n, r, _ = self.decode(parent)
        node = self._nodes[n]
        if k == K_RES:
            info = self.infos[node.pid]
            if row < int(info.nparts[int(node.rows[r])]):
                return self.createIndex(row, column, self._rid(K_PART, n, r, row))
        return QModelIndex()

    def parent(self, index=QModelIndex()):
        if not index.isValid():
            return QModelIndex()
        k, n, r, _ = self.decode(index)
        node = self._nodes[n]
        if k == K_NODE:
            if node.parent < 0:
                return QModelIndex()
            return self.node_index(self._nodes[node.parent])
        if k == K_RES:
            return self.node_index(node)
        return self.createIndex(r, 0, self._rid(K_RES, n, r))

    def rowCount(self, parent=QModelIndex()):
        if not parent.isValid():
            return len(self._top)
        if parent.column() != 0:
            return 0
        k, n, r, _ = self.decode(parent)
        node = self._nodes[n]
        if k == K_NODE:
            if node.kind == "pack":
                return len(node.children)
            if node.kind == "group":
                return node.fetched
            return 0
        if k == K_RES:
            return int(self.infos[node.pid].nparts[int(node.rows[r])])
        return 0

    def hasChildren(self, parent=QModelIndex()):
        if not parent.isValid():
            return bool(self._top)
        v = parent.internalId()
        if v >= _PART_ID or parent.column() != 0:
            return False
        if v >= _RES_ID:
            return True                                  # every shipped resource has parts (rowCount is exact)
        k, n, r, _ = self.decode(parent)
        node = self._nodes[n]
        if k == K_NODE:
            if node.kind == "pack":
                return bool(node.children)
            return node.kind == "group" and len(node.rows) > 0
        if k == K_RES:
            return int(self.infos[node.pid].nparts[int(node.rows[r])]) > 0
        return False

    def canFetchMore(self, parent):
        node = self.node_of(parent)
        return node is not None and node.kind == "group" and node.fetched < len(node.rows)

    def fetchMore(self, parent):
        node = self.node_of(parent)
        if node is not None and node.kind == "group":
            self.ensure_fetched(node, node.fetched + CHUNK)

    def columnCount(self, parent=QModelIndex()):
        return len(COLUMNS)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return COLUMNS[section]
        return None

    def flags(self, index):
        v = index.internalId()
        if v >= _PART_ID:
            return _F_PART_CHECK if index.column() == 0 else _F_PART
        if v < _RES_ID:
            if not index.isValid():
                return Qt.NoItemFlags
            if index.column() != 0 or self._nodes[v >> _NODE_SHIFT].error is not None:
                return _F_BASE
            return _F_CHECK
        return _F_CHECK if index.column() == 0 else _F_BASE

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        col = index.column()
        k, n, r, p = self.decode(index)
        node = self._nodes[n]
        if role == Qt.DisplayRole:
            try:
                return self._display(k, node, r, p, col)
            except Exception as exc:  # noqa: BLE001 - garbage tables show as text, never as a crash
                return f"<{type(exc).__name__}>"
        if role == Qt.CheckStateRole and col == 0:
            if k == K_NODE and node.error:
                return None
            return self.check_state(index)
        if role == Qt.FontRole and col != COL_NAME:
            return self._mono
        if role == Qt.TextAlignmentRole and col in (COL_INDEX, COL_OFFSET, COL_SIZE):
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.ForegroundRole:
            if k == K_NODE and node.error:
                return RED
            if k == K_PART:
                info = self.infos[node.pid]
                j = int(info.first[int(node.rows[r])]) + p
                if not info.pack.part_is_direct(j):
                    return GREY
            return None
        if role == Qt.ToolTipRole and col == COL_NAME:
            if k == K_NODE and node.kind == "pack":
                e = self.cat.packs[node.pid]
                return f"{e.path}\n{node.error}" if node.error else str(e.path)
            if k == K_RES:
                info = self.infos[node.pid]
                li = int(node.rows[r])
                return repr(info.pack.name_bytes(info.pack.logicals[li].name_index))
            if k == K_PART:
                info = self.infos[node.pid]
                j = int(info.first[int(node.rows[r])]) + p
                if not info.pack.part_is_direct(j):
                    return "child-pack (.rpacz) or compressed part: exported bytes are not the payload"
        return None

    def _display(self, k: int, node: _Node, r: int, p: int, col: int):
        if k == K_NODE:
            return self._display_node(node, col)
        info = self.infos[node.pid]
        pk = info.pack
        li = int(node.rows[r])
        if k == K_RES:
            if col == COL_NAME:
                return visible_spaces(self.cat.name(self.cat.packs[node.pid].base + li))
            if col == COL_TYPE:
                lg = pk.logicals[li]
                return f"0x{lg.type:02X} f{lg.flags:02X}"
            if col == COL_INDEX:
                return str(li)
            if col == COL_OFFSET:
                return f"0x{pk.part_offset(int(info.first[li])):X}" if info.nparts[li] else ""
            if col == COL_SIZE:
                return human_size(int(info.sizes[li]))
            return None
        j = int(info.first[li]) + p
        if col == COL_NAME:
            return f"part {p}  {catalogue.type_name(pk.part_type(j))}"
        if col == COL_TYPE:
            return f"0x{pk.part_type(j):02X}"
        if col == COL_INDEX:
            return str(j)
        if col == COL_OFFSET:
            return f"0x{pk.part_offset(j):X}"
        if col == COL_SIZE:
            return human_size(pk.physicals[j].size)
        return None

    def _display_node(self, node: _Node, col: int):
        info = self.infos.get(node.pid)
        if node.kind == "pack":
            if col == COL_NAME:
                return node.label + (f"   [{node.error}]" if node.error else "")
            if info is None:
                return ""
            if col == COL_TYPE:
                return f"f08 0x{info.pack.header.field08:X}"
            if col == COL_INDEX:
                return f"{info.count:,}"
            if col == COL_OFFSET:
                return "0x0"
            if col == COL_SIZE:
                return human_size(info.pack.size)
            return None
        if col == COL_NAME:
            return node.label
        if node.kind == "group":
            return f"0x{node.type:02X}" if col == COL_TYPE else ""
        off, size = info.tables[node.which]
        if col == COL_OFFSET:
            return f"0x{off:X}"
        if col == COL_SIZE:
            return human_size(size)
        return ""

    # ---- check state ------------------------------------------------------------------------------------------
    def _touch(self) -> None:
        self._ver += 1
        self._state_cache.clear()

    def _group_state(self, g: _Node) -> Qt.CheckState:
        s = self._state_cache.get(g.idx)
        if s is None:
            ch = self.checks[g.pid]
            on = int(np.count_nonzero(ch.mask[g.rows]))
            part = bool(ch.partial) and bool(len(np.intersect1d(np.fromiter(ch.partial, np.int64), g.rows)))
            if on == len(g.rows) and on:
                s = Qt.Checked
            elif on or part:
                s = Qt.PartiallyChecked
            else:
                s = Qt.Unchecked
            self._state_cache[g.idx] = s
        return s

    def _pack_state(self, pn: _Node) -> Qt.CheckState:
        s = self._state_cache.get(pn.idx)
        if s is None:
            states = {self._node_state(self._nodes[c]) for c in pn.children}
            if not states:
                s = Qt.Unchecked
            elif len(states) == 1:
                s = states.pop()
            else:
                s = Qt.PartiallyChecked
            self._state_cache[pn.idx] = s
        return s

    def _node_state(self, node: _Node) -> Qt.CheckState:
        if node.kind == "pack":
            return self._pack_state(node)
        if node.kind == "group":
            return self._group_state(node)
        return Qt.Checked if node.which in self.checks[node.pid].tables else Qt.Unchecked

    def check_state(self, index: QModelIndex) -> Qt.CheckState:
        k, n, r, p = self.decode(index)
        node = self._nodes[n]
        if k == K_NODE:
            if node.error:
                return Qt.Unchecked
            return self._node_state(node)
        ch = self.checks[node.pid]
        li = int(node.rows[r])
        if ch.mask[li]:
            return Qt.Checked
        ks = ch.partial.get(li)
        if k == K_RES:
            return Qt.PartiallyChecked if ks else Qt.Unchecked
        return Qt.Checked if ks and p in ks else Qt.Unchecked

    def setData(self, index, value, role=Qt.EditRole):
        if role != Qt.CheckStateRole or not index.isValid() or index.column() != 0:
            return False
        on = Qt.CheckState(value) == Qt.Checked
        self.set_checked(index, on)
        return True

    def set_checked(self, index: QModelIndex, on: bool) -> None:
        """Check / uncheck a row and everything under it (as currently shown)."""
        k, n, r, p = self.decode(index)
        node = self._nodes[n]
        if k == K_NODE:
            if node.error:
                return
            targets = [node] if node.kind != "pack" else [self._nodes[c] for c in node.children]
            ch = self.checks[node.pid]
            for t in targets:
                if t.kind == "table":
                    (ch.tables.add if on else ch.tables.discard)(t.which)
                else:
                    ch.set_rows(t.rows, on)
        else:
            ch = self.checks[node.pid]
            li = int(node.rows[r])
            if k == K_RES:
                ch.mask[li] = on
                ch.partial.pop(li, None)
            else:
                total = int(self.infos[node.pid].nparts[li])
                ks = set(range(total)) if ch.mask[li] else set(ch.partial.get(li, ()))
                (ks.add if on else ks.discard)(p)
                ch.mask[li] = len(ks) == total
                if ch.mask[li] or not ks:
                    ch.partial.pop(li, None)
                else:
                    ch.partial[li] = ks
        self._changed(index)

    def set_rows_checked(self, pid: int, rows, on: bool) -> None:
        ch = self.checks.get(pid)
        if ch is not None:
            ch.set_rows(np.asarray(rows, dtype=np.int64), on)
            self._changed(None)

    def clear_checks(self) -> None:
        for ch in self.checks.values():
            ch.mask[:] = False
            ch.partial.clear()
            ch.tables.clear()
        self._changed(None)

    def _changed(self, index: QModelIndex | None) -> None:
        self._touch()
        roles = [Qt.CheckStateRole]
        if index is not None:
            i = index.siblingAtColumn(0)
            while i.isValid():
                self.dataChanged.emit(i, i, roles)
                i = i.parent()
            rows = self.rowCount(index.siblingAtColumn(0))
            if rows:
                p = index.siblingAtColumn(0)
                self.dataChanged.emit(self.index(0, 0, p), self.index(rows - 1, 0, p), roles)
        elif self._top:
            self.dataChanged.emit(self.index(0, 0), self.index(len(self._top) - 1, 0), roles)
        self.checksChanged.emit()

    def checked_totals(self) -> tuple[int, int]:
        parts = size = 0
        for pid, ch in self.checks.items():
            if ch.any():
                a, b = checks_totals(self.infos[pid], ch)
                parts += a
                size += b
        return parts, size

    def checked_spec(self) -> list[tuple[PackInfo, str, PackChecks]]:
        """Snapshot of every pack with something checked (safe to hand to a worker)."""
        return [(self.infos[pid], self.cat.packs[pid].label, ch.copy())
                for pid, ch in sorted(self.checks.items()) if ch.any()]

    def selection_spec(self, indexes) -> list[tuple[PackInfo, str, PackChecks]]:
        """Like `checked_spec`, for a set of selected rows (each row selects itself and what it shows)."""
        spec: dict[int, PackChecks] = {}

        def get(pid):
            if pid not in spec:
                spec[pid] = PackChecks(np.zeros(self.infos[pid].count, dtype=bool))
            return spec[pid]

        for index in indexes:
            if not index.isValid() or index.column() != 0:
                continue
            k, n, r, p = self.decode(index)
            node = self._nodes[n]
            if k == K_NODE:
                if node.error or node.pid not in self.infos:
                    continue
                ch = get(node.pid)
                for t in ([node] if node.kind != "pack" else [self._nodes[c] for c in node.children]):
                    if t.kind == "table":
                        ch.tables.add(t.which)
                    else:
                        ch.set_rows(t.rows, True)
                continue
            ch = get(node.pid)
            li = int(node.rows[r])
            if k == K_RES:
                ch.mask[li] = True
                ch.partial.pop(li, None)
            elif not ch.mask[li]:
                ks = ch.partial.setdefault(li, set())
                ks.add(p)
                if len(ks) == int(self.infos[node.pid].nparts[li]):
                    ch.mask[li] = True
                    del ch.partial[li]
        return [(self.infos[pid], self.cat.packs[pid].label, ch) for pid, ch in sorted(spec.items())]


def all_spec_for(cat, entries, infos: dict) -> list[tuple[PackInfo, str, PackChecks]]:
    """Every pack of *entries* in full (tables included); infos missing from *infos* are built and added."""
    out = []
    for e in entries:
        info = infos.get(e.id)
        if info is None:
            try:
                info = infos[e.id] = build_info(e.id, e.pack)
            except Exception:  # noqa: BLE001 - unreadable tables: nothing to export
                continue
        out.append((info, e.label, PackChecks(np.ones(info.count, dtype=bool), {}, set(TABLES))))
    return out


def visible_spaces(name: str) -> str:
    """Leading / trailing spaces are part of shipped names (" npc_b_man_pants_…"); show them as `·`."""
    core = name.strip(" ")
    if core == name:
        return name
    lead = len(name) - len(name.lstrip(" "))
    trail = len(name) - len(name.rstrip(" "))
    return "·" * lead + core + "·" * trail


def spec_totals(spec) -> tuple[int, int]:
    parts = size = 0
    for info, _, ch in spec:
        a, b = checks_totals(info, ch)
        parts += a
        size += b
    return parts, size


def export_path(out: Path, rel: str) -> Path:
    return Path(out).joinpath(*rel.split("/"))
