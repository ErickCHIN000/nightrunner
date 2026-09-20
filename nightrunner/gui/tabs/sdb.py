"""SDB tab: browse the compiled material database (runtime_dx11/dx12.sdb).

Left: a mode switch (Materials | Presets | Textures) over virtual lists with a debounced, pooled search.
Right: the detail of the selection — a material (overview, typed parameters vs preset defaults, per-variant
texture bindings with catalog status, models / meshes that use it, raw records), a preset (declarations and the
materials that use it), a texture (materials binding it, catalog providers) — or the database stats page.

Other tabs link here through `ctx.openMaterial.emit(name)` → `open_material(name)`. The SDB itself is opened in a
worker (first access to `ctx.sdb.sdb`); nothing here parses on the GUI thread.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
import shiboken6
from PySide6.QtCore import QAbstractTableModel, QItemSelectionModel, QModelIndex, Qt, QTimer, QUrl
from PySide6.QtGui import QBrush, QColor, QFont
from PySide6.QtWidgets import (QComboBox, QAbstractItemView, QButtonGroup, QCheckBox, QFileDialog, QHBoxLayout, QHeaderView, QSizePolicy,
                               QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QStackedWidget, QTableView, QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...sdb import export as sdbexport
from ...sdb.reader import Sdb
from .. import sdbscan as scan
from ..widgets import human_size, mono_font

MODES = ("Materials", "Presets", "Textures")
P_MSG, P_MAT, P_PRESET, P_TEX, P_STATS = range(5)
MISSING = QColor("#e06c75")
OVERRIDE = QColor("#e5c07b")
DIM = QColor("#888888")


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class RowsModel(QAbstractTableModel):
    """Read-only virtual table over an int array of row ids; cells come from per-column getters (visible rows only)."""

    def __init__(self, headers: list[str], getters: list[Callable[[int], object]], mono: tuple[int, ...] = (),
                 right: tuple[int, ...] = (), parent=None):
        super().__init__(parent)
        self.headers, self.getters = headers, getters
        self.rows = np.zeros(0, dtype=np.int64)
        self._mono_cols, self._right = set(mono), set(right)
        self._mono = mono_font()

    def set_rows(self, rows) -> None:
        self.beginResetModel()
        self.rows = np.asarray(rows, dtype=np.int64)
        self.endResetModel()

    def row_id(self, row: int) -> int:
        return int(self.rows[row])

    def row_of(self, rid: int) -> int:
        hits = np.flatnonzero(self.rows == rid)
        return int(hits[0]) if len(hits) else -1

    def refresh(self) -> None:
        if len(self.rows):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.rows) - 1, len(self.headers) - 1))

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent=QModelIndex()):
        return len(self.headers)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.headers[section]
        return None

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        c = index.column()
        if role == Qt.DisplayRole:
            try:
                v = self.getters[c](int(self.rows[index.row()]))
            except Exception as exc:  # noqa: BLE001
                v = f"<{type(exc).__name__}>"
            return "" if v is None else str(v)
        if role == Qt.FontRole and c in self._mono_cols:
            return self._mono
        if role == Qt.TextAlignmentRole and c in self._right:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.UserRole:
            return int(self.rows[index.row()])
        return None


def _table_view(model: RowsModel, stretch: int) -> QTableView:
    v = QTableView()
    v.setModel(model)
    v.setSelectionBehavior(QAbstractItemView.SelectRows)
    v.setSelectionMode(QAbstractItemView.SingleSelection)
    v.setEditTriggers(QAbstractItemView.NoEditTriggers)
    v.verticalHeader().hide()
    v.verticalHeader().setSectionResizeMode(QHeaderView.Fixed)
    v.verticalHeader().setDefaultSectionSize(20)
    v.setWordWrap(False)
    h = v.horizontalHeader()
    h.setSectionResizeMode(QHeaderView.Interactive)
    h.setSectionResizeMode(stretch, QHeaderView.Stretch)
    for c in range(model.columnCount()):
        if c != stretch:
            v.setColumnWidth(c, 70)
    return v


def _item_table(headers: list[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.verticalHeader().hide()
    t.verticalHeader().setDefaultSectionSize(20)
    t.setWordWrap(False)
    t.horizontalHeader().setStretchLastSection(True)
    return t


def _cell(v, mono: bool = False, color: QColor | None = None, data=None) -> QTableWidgetItem:
    it = QTableWidgetItem("" if v is None else str(v))
    if mono:
        it.setFont(mono_font())
    if color is not None:
        it.setForeground(QBrush(color))
    if data is not None:
        it.setData(Qt.UserRole, data)
    return it


class Tab(QWidget):
    TITLE = "SDB"

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.svc = ctx.sdb
        self._epoch = 0
        self._sdb = None
        self._lists: scan.SdbLists | None = None
        self._tex_names: list[str] = []
        self._tex_lower: list[str] = []
        self._reverse: dict[str, list[str]] | None = None
        self._search_text = {m: "" for m in MODES}
        self._pending_material: str | None = None
        self._detail: dict | None = None
        self._preset_cur: dict | None = None
        self._tex_cur: str | None = None
        self._model_refs: dict | None = None
        self._dump_cancel: threading.Event | None = None
        self._model_refs_loading = False
        self._mesh_scan_requested = False
        self.timings: dict[str, float] = {}

        self.scanner = scan.MeshMaterialScanner(ctx.catalog, self)
        self.scanner.progress.connect(self._on_scan_progress)
        self.scanner.finished.connect(self._on_scan_finished)
        self.validator = scan.ValidateJob(self)
        self.validator.progress.connect(self._on_validate_progress)
        self.validator.done.connect(self._on_validate_done)
        self.validator.failed.connect(self._on_validate_failed)

        self._build_ui()
        self.svc.reverseReady.connect(self._on_reverse_ready)
        self.svc.failed.connect(self._on_svc_failed)
        ctx.catalog.ready.connect(self._on_catalog_ready)
        self._search_timer = QTimer(self, singleShot=True, interval=200)
        self._search_timer.timeout.connect(self._run_search)
        self._start_load()

    # ================================================================================================================
    # UI
    # ================================================================================================================
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        head = QHBoxLayout()
        self.header = QLabel()
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.header.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)   # long path must not widen the window
        self.stats_btn = QPushButton("Stats")
        self.stats_btn.clicked.connect(self.show_stats)
        self.api_combo = QComboBox()
        self.api_combo.addItem("DX11  (runtime_dx11.sdb)", "dx11")
        self.api_combo.addItem("DX12  (runtime_dx12.sdb)", "dx12")
        self.api_combo.setToolTip("Which compiled material database to browse (also File > SDB; every tab follows)")
        self._sync_api_combo()
        self.api_combo.currentIndexChanged.connect(self._on_api_combo)
        head.addWidget(self.header, 1)
        head.addWidget(QLabel("SDB"))
        head.addWidget(self.api_combo)
        head.addWidget(self.stats_btn)
        self.dump_btn = QPushButton("Export all…")
        self.dump_btn.setToolTip("Dump the whole database to JSON: materials, presets, textures")
        self.dump_btn.clicked.connect(self._dump_clicked)
        head.addWidget(self.dump_btn)
        root.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        # ---- left --------------------------------------------------------------------------------------------------
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        mrow = QHBoxLayout()
        self.mode_group = QButtonGroup(self)
        self.mode_buttons: dict[str, QToolButton] = {}
        for i, m in enumerate(MODES):
            b = QToolButton()
            b.setText(m)
            b.setCheckable(True)
            b.setAutoRaise(False)
            self.mode_group.addButton(b, i)
            self.mode_buttons[m] = b
            mrow.addWidget(b)
        mrow.addStretch(1)
        self.mode_buttons["Materials"].setChecked(True)
        self.mode_group.idClicked.connect(lambda i: self.set_mode(MODES[i]))
        ll.addLayout(mrow)
        srow = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("search…  (words must all match; #123 = index)")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._on_search_text)
        self.count = QLabel()
        srow.addWidget(self.search, 1)
        srow.addWidget(self.count)
        ll.addLayout(srow)

        L = lambda: self._lists  # noqa: E731
        self.mat_model = RowsModel(
            ["#", "material", "preset"],
            [str, lambda i: L().names[i], lambda i: (L().material_preset[i] if L().material_preset else "…")],
            right=(0,))
        self.preset_model = RowsModel(
            ["#", "preset", "params", "materials"],
            [str, lambda i: L().presets[i]["name"] or "(empty name)", lambda i: L().presets[i]["params"],
             lambda i: L().presets[i]["materials"]], right=(0, 2, 3))
        self.tex_model = RowsModel(
            ["texture", "materials"],
            [lambda i: self._tex_names[i], lambda i: len((self._reverse or {}).get(self._tex_names[i], ()))],
            right=(1,))
        self.list_stack = QStackedWidget()
        self.views: dict[str, QTableView] = {}
        for m, model, stretch in (("Materials", self.mat_model, 1), ("Presets", self.preset_model, 1),
                                  ("Textures", self.tex_model, 0)):
            v = _table_view(model, stretch)
            if m == "Textures":
                v.setColumnWidth(1, 80)
            v.selectionModel().currentRowChanged.connect(lambda cur, _prev, m=m: self._on_current(m, cur))
            self.views[m] = v
            self.list_stack.addWidget(v)
        self.views["Materials"].setColumnWidth(0, 55)
        self.views["Materials"].setColumnWidth(2, 110)
        self.views["Presets"].setColumnWidth(0, 45)
        ll.addWidget(self.list_stack, 1)
        self.list_msg = QLabel()
        self.list_msg.setWordWrap(True)
        ll.addWidget(self.list_msg)
        split.addWidget(left)

        # ---- right -------------------------------------------------------------------------------------------------
        self.pages = QStackedWidget()
        self.msg = QLabel()
        self.msg.setAlignment(Qt.AlignCenter)
        self.msg.setWordWrap(True)
        self.pages.addWidget(self.msg)
        self.pages.addWidget(self._build_material_page())
        self.pages.addWidget(self._build_preset_page())
        self.pages.addWidget(self._build_texture_page())
        self.pages.addWidget(self._build_stats_page())
        split.addWidget(self.pages)
        split.setSizes([430, 1100])
        split.setStretchFactor(1, 1)

    def _build_material_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        top = QHBoxLayout()
        self.mat_title = QLabel()
        f = self.mat_title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 2)
        self.mat_title.setFont(f)
        self.mat_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.export_btn = QPushButton("Export JSON…")
        self.export_btn.clicked.connect(self._export_clicked)
        top.addWidget(self.mat_title, 1)
        top.addWidget(self.export_btn)
        lay.addLayout(top)
        self.mat_tabs = QTabWidget()
        lay.addWidget(self.mat_tabs, 1)

        # overview
        self.overview = QTextBrowser()
        self.overview.setOpenLinks(False)
        self.overview.anchorClicked.connect(self._on_anchor)
        self.mat_tabs.addTab(self.overview, "Overview")

        # parameters
        pw = QWidget()
        pl = QVBoxLayout(pw)
        self.show_preset_only = QCheckBox("show preset declarations the material does not set")
        self.show_preset_only.toggled.connect(lambda _: self._fill_params())
        pl.addWidget(self.show_preset_only)
        self.params = _item_table(["id", "name", "type", "value", "offset", "bit31", "preset default", "status",
                                   "annotation"])
        for c, wdt in enumerate((45, 190, 55, 220, 50, 45, 160, 110)):
            self.params.setColumnWidth(c, wdt)
        pl.addWidget(self.params, 1)
        self.mat_tabs.addTab(pw, "Parameters")

        # textures / variants
        tw = QWidget()
        tl = QVBoxLayout(tw)
        brow = QHBoxLayout()
        self.var_summary = QLabel()
        exp = QPushButton("Expand all")
        col = QPushButton("Collapse all")
        brow.addWidget(self.var_summary, 1)
        brow.addWidget(exp)
        brow.addWidget(col)
        tl.addLayout(brow)
        ts = QSplitter(Qt.Vertical)
        self.variants = QTreeWidget()
        self.variants.setHeaderLabels(["variant / binding", "param", "texture", "source", "catalog"])
        self.variants.setUniformRowHeights(True)
        for c, wdt in enumerate((330, 150, 260, 100)):
            self.variants.setColumnWidth(c, wdt)
        self.variants.itemDoubleClicked.connect(self._on_texture_item)
        exp.clicked.connect(self.variants.expandAll)
        col.clicked.connect(self.variants.collapseAll)
        self.used_textures = QTreeWidget()
        self.used_textures.setRootIsDecorated(False)
        self.used_textures.setHeaderLabels(["texture (de-duplicated)", "params", "variants", "catalog"])
        for c, wdt in enumerate((300, 220, 70)):
            self.used_textures.setColumnWidth(c, wdt)
        self.used_textures.itemDoubleClicked.connect(self._on_texture_item)
        ts.addWidget(self.variants)
        ts.addWidget(self.used_textures)
        ts.setSizes([500, 220])
        tl.addWidget(ts, 1)
        tl.addWidget(QLabel("Double-click a texture to open it in the Textures tab."))
        self.mat_tabs.addTab(tw, "Textures / Variants")

        # used by
        uw = QWidget()
        ul = QVBoxLayout(uw)
        us = QSplitter(Qt.Vertical)
        mw = QWidget()
        ml = QVBoxLayout(mw)
        ml.setContentsMargins(0, 0, 0, 0)
        self.models_label = QLabel("Models: …")
        ml.addWidget(self.models_label)
        self.models_tree = QTreeWidget()
        self.models_tree.setHeaderLabels(["model / slot", "mesh", "kind", "number", "selected", "rttiValues", "pak"])
        for c, wdt in enumerate((320, 230, 75, 55, 60, 300)):
            self.models_tree.setColumnWidth(c, wdt)
        self.models_tree.itemDoubleClicked.connect(self._on_model_item)
        ml.addWidget(self.models_tree, 1)
        us.addWidget(mw)
        shw = QWidget()
        sl = QVBoxLayout(shw)
        sl.setContentsMargins(0, 0, 0, 0)
        srow = QHBoxLayout()
        self.meshes_label = QLabel("Meshes: not scanned")
        self.scan_btn = QPushButton("Scan meshes")
        self.scan_btn.clicked.connect(self.start_mesh_scan)
        self.scan_prog = QProgressBar()
        self.scan_prog.setMaximumWidth(220)
        self.scan_prog.setFormat("%v / %m meshes")
        self.scan_prog.hide()
        self.scan_cancel = QPushButton("Cancel")
        self.scan_cancel.clicked.connect(self.scanner.cancel)
        self.scan_cancel.hide()
        srow.addWidget(self.meshes_label, 1)
        srow.addWidget(self.scan_prog)
        srow.addWidget(self.scan_cancel)
        srow.addWidget(self.scan_btn)
        sl.addLayout(srow)
        self.meshes_table = _item_table(["mesh (embedded submesh material)", "pack", "gid"])
        self.meshes_table.setColumnWidth(0, 380)
        self.meshes_table.setColumnWidth(1, 300)
        self.meshes_table.cellDoubleClicked.connect(self._on_mesh_cell)
        sl.addWidget(self.meshes_table, 1)
        us.addWidget(shw)
        ul.addWidget(us, 1)
        ul.addWidget(QLabel("Double-click a model or mesh to open it."))
        self.usedby_index = self.mat_tabs.addTab(uw, "Used by")

        # raw
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setFont(mono_font())
        self.raw.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.mat_tabs.addTab(self.raw, "Raw")
        self.mat_tabs.currentChanged.connect(self._on_mat_tab)
        return w

    def _build_preset_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.preset_title = QLabel()
        self.preset_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.preset_title.setWordWrap(True)
        lay.addWidget(self.preset_title)
        sp = QSplitter(Qt.Vertical)
        self.preset_params = _item_table(["id", "name", "type", "expression", "default", "annotation"])
        for c, wdt in enumerate((45, 200, 60, 220, 160)):
            self.preset_params.setColumnWidth(c, wdt)
        sp.addWidget(self.preset_params)
        bw = QWidget()
        bl = QVBoxLayout(bw)
        bl.setContentsMargins(0, 0, 0, 0)
        self.preset_users_label = QLabel()
        bl.addWidget(self.preset_users_label)
        L = lambda: self._lists  # noqa: E731
        self.preset_users_model = RowsModel(["#", "material using this preset"], [str, lambda i: L().names[i]],
                                            right=(0,))
        self.preset_users = _table_view(self.preset_users_model, 1)
        self.preset_users.setColumnWidth(0, 60)
        self.preset_users.doubleClicked.connect(
            lambda ix: self._open_material_index(self.preset_users_model.row_id(ix.row())))
        bl.addWidget(self.preset_users, 1)
        sp.addWidget(bw)
        sp.setSizes([450, 300])
        lay.addWidget(sp, 1)
        return w

    def _build_texture_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.tex_title = QLabel()
        self.tex_title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        lay.addWidget(self.tex_title)
        sp = QSplitter(Qt.Vertical)
        self.tex_users = _item_table(["material binding this texture", "preset"])
        self.tex_users.setColumnWidth(0, 420)
        self.tex_users.cellDoubleClicked.connect(
            lambda r, _c: self.open_material(self.tex_users.item(r, 0).text()))
        self.tex_providers = _item_table(["catalog provider (pack)", "gid", "logical name"])
        self.tex_providers.setColumnWidth(0, 380)
        self.tex_providers.cellDoubleClicked.connect(
            lambda r, _c: self.ctx.openTexture.emit(int(self.tex_providers.item(r, 1).text())))
        sp.addWidget(self.tex_users)
        sp.addWidget(self.tex_providers)
        sp.setSizes([450, 200])
        lay.addWidget(sp, 1)
        lay.addWidget(QLabel("Double-click a material to show it, a provider to open the texture."))
        return w

    def _build_stats_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        self.stats_text = QPlainTextEdit()
        self.stats_text.setReadOnly(True)
        self.stats_text.setFont(mono_font())
        self.stats_text.setMaximumHeight(170)
        lay.addWidget(self.stats_text)
        self.stats_tables = _item_table(["key", "shape", "width", "count", "bytes", "meaning"])
        for c, wdt in enumerate((55, 100, 55, 90, 100)):
            self.stats_tables.setColumnWidth(c, wdt)
        lay.addWidget(self.stats_tables, 1)
        vrow = QHBoxLayout()
        self.validate_btn = QPushButton("Validate (resolve every material)")
        self.validate_btn.clicked.connect(self.start_validate)
        self.validate_prog = QProgressBar()
        self.validate_prog.setFormat("%v / %m materials")
        self.validate_prog.hide()
        self.validate_cancel = QPushButton("Cancel")
        self.validate_cancel.clicked.connect(self.validator.cancel)
        self.validate_cancel.hide()
        vrow.addWidget(self.validate_btn)
        vrow.addWidget(self.validate_prog, 1)
        vrow.addWidget(self.validate_cancel)
        lay.addLayout(vrow)
        self.validate_out = QPlainTextEdit()
        self.validate_out.setReadOnly(True)
        self.validate_out.setFont(mono_font())
        self.validate_out.setMaximumHeight(200)
        lay.addWidget(self.validate_out)
        return w

    # ================================================================================================================
    # task helpers
    # ================================================================================================================
    def _drop_dead(self, key) -> None:
        """Work around TaskRunner calling pool.tryTake() on an auto-deleted job whose result is still queued
        (RuntimeError 'Internal C++ object already deleted')."""
        runner = self.ctx.runner
        old = runner._jobs.get(key)
        if old is not None and not shiboken6.isValid(old):
            runner._jobs.pop(key, None)

    def _submit(self, key, fn, *args, **kwargs) -> int:
        self._drop_dead(key)
        return self.ctx.runner.submit(key, fn, *args, **kwargs)

    def _cancel_job(self, key) -> None:
        self._drop_dead(key)
        self.ctx.runner.cancel(key)

    # ================================================================================================================
    # loading
    # ================================================================================================================
    def _db_label(self) -> str:
        p = self.svc.path
        return p.name if p is not None else "runtime SDB"

    def _show_message(self, text: str) -> None:
        self.msg.setText(text)
        self.pages.setCurrentIndex(P_MSG)

    def _start_load(self) -> None:
        self._epoch += 1
        ep = self._epoch
        self._sdb = None
        self._lists = None
        self._reverse = None
        self._tex_names, self._tex_lower = [], []
        self._detail = self._preset_cur = self._tex_cur = None
        for m in (self.mat_model, self.preset_model, self.tex_model, self.preset_users_model):
            m.set_rows([])
        self._cancel_job(("sdb", "search"))
        self.count.clear()
        self.list_msg.clear()
        if self.validator.running():
            self.validator.cancel()
        if not self.svc.available():
            where = str(self.svc.path) if self.svc.path is not None else "(no game — Game > Browse…)"
            self.header.setText(f"<b>No SDB loaded.</b> {_esc(self._db_label())} not found: {_esc(where)}")
            self._show_message("The material database is not available.\n\n"
                               "Pick a game (Game menu) or the other SDB (File > SDB).")
            self.list_msg.setText("no SDB")
            return
        self.header.setText(f"Opening <b>{_esc(self._db_label())}</b> …  <span style='color:#888'>"
                            f"{_esc(self.svc.path)}</span>")
        self._show_message(f"Opening {self._db_label()} …")
        t0 = time.perf_counter()
        svc = self.svc

        def job():
            s = svc.sdb
            if s is None:
                raise RuntimeError(svc.error or "the SDB could not be opened")
            t1 = time.perf_counter()
            lists = scan.load_names(s)
            return s, lists, t1 - t0, time.perf_counter() - t1

        self._submit(("sdb", "open"), job, on_done=lambda r: self._on_opened(ep, r),
                               on_error=lambda e: self._on_open_failed(ep, e))

    def _on_open_failed(self, ep: int, err: str) -> None:
        if ep != self._epoch:
            return
        first = err.splitlines()[0] if err else "unknown error"
        self.header.setText(f"<b>{_esc(self._db_label())}</b>: <span style='color:#e06c75'>cannot open — "
                            f"{_esc(first)}</span>")
        self._show_message(f"{self._db_label()} could not be opened:\n\n{first}")
        self.list_msg.setText("SDB failed to open")

    def _on_svc_failed(self, err: str) -> None:
        self.list_msg.setText(f"SDB service: {err}")

    def _on_opened(self, ep: int, res) -> None:
        if ep != self._epoch:
            return
        s, lists, t_open, t_names = res
        self.timings.update(open=t_open, names=t_names)
        self._sdb, self._lists = s, lists
        st = s.stats()
        self.header.setText(
            f"<b>{_esc(self._db_label())}</b>  {len(lists.names):,} materials · {st['presets']:,} presets · "
            f"{st['routes']:,} routes · {st['size'] / (1 << 20):,.0f} MB  <span style='color:#888'>"
            f"{_esc(s.path)}  (opened in {t_open:.2f} s)</span>")
        self._search_now()
        t0 = time.perf_counter()
        self._submit(("sdb", "presets"), scan.load_presets, s, lists,
                               on_done=lambda _r: self._on_presets(ep, time.perf_counter() - t0),
                               on_error=lambda e: self.list_msg.setText(f"presets: {e.splitlines()[0]}"))
        self.svc.ensure_reverse()
        if self.svc.reverse_ready():
            self._on_reverse_ready()
        if self._pending_material:
            name, self._pending_material = self._pending_material, None
            self.open_material(name)
        elif self.pages.currentIndex() == P_MSG:
            self._show_message("Select a material, preset or texture on the left.")

    def _on_presets(self, ep: int, dt: float) -> None:
        if ep != self._epoch:
            return
        self.timings["presets"] = dt
        self.mat_model.refresh()
        if self._mode() == "Presets":
            self._search_now()

    def _on_reverse_ready(self) -> None:
        ep = self._epoch
        svc = self.svc
        t0 = time.perf_counter()

        def job():
            fn = getattr(svc, "texture_index", None)
            rev = fn() if callable(fn) else getattr(svc, "_reverse", None)
            if rev is None:
                return None
            names = sorted(rev)
            return rev, names, [n.casefold() for n in names]

        def done(r):
            if ep != self._epoch or r is None:
                return
            self._reverse, self._tex_names, self._tex_lower = r
            self.timings["reverse_list"] = time.perf_counter() - t0
            if self._mode() == "Textures":
                self._search_now()

        self._submit(("sdb", "reverse"), job, on_done=done)

    def _sync_api_combo(self) -> None:
        api_fn = getattr(self.ctx, "sdb_api", None)
        api = api_fn() if callable(api_fn) else "dx11"
        i = self.api_combo.findData(api)
        if i >= 0 and i != self.api_combo.currentIndex():
            self.api_combo.blockSignals(True)
            self.api_combo.setCurrentIndex(i)
            self.api_combo.blockSignals(False)

    def _on_api_combo(self, _i: int) -> None:
        setter = getattr(self.ctx, "set_sdb_api", None)
        if callable(setter):
            setter(self.api_combo.currentData())       # the main window calls sdb_changed() on every tab

    def sdb_changed(self) -> None:
        self._sync_api_combo()
        """The main window switched dx11/dx12: drop everything SDB-derived and reopen in a worker."""
        name = self._detail["material"]["name"] if self._detail else None
        self._start_load()
        if name:
            self._pending_material = name

    def shutdown(self) -> None:
        self.scanner.cancel()
        self.validator.cancel()

    # ================================================================================================================
    # list / search
    # ================================================================================================================
    def _mode(self) -> str:
        return MODES[max(0, self.mode_group.checkedId())]

    def set_mode(self, mode: str) -> None:
        self.mode_buttons[mode].setChecked(True)
        self.list_stack.setCurrentWidget(self.views[mode])
        self.search.blockSignals(True)
        self.search.setText(self._search_text[mode])
        self.search.blockSignals(False)
        self._search_now()
        cur = self.views[mode].currentIndex()
        if cur.isValid():
            self._on_current(mode, cur)

    def _on_search_text(self, text: str) -> None:
        self._search_text[self._mode()] = text
        self._search_timer.start()

    def _search_now(self) -> None:
        self._search_timer.stop()
        self._run_search()

    def _source(self, mode: str):
        L = self._lists
        if mode == "Materials":
            return (L.lower if L else None), self.mat_model
        if mode == "Presets":
            return (L.preset_lower if L and L.presets else None), self.preset_model
        return (self._tex_lower if self._reverse is not None else None), self.tex_model

    def _run_search(self) -> None:
        mode = self._mode()
        lower, model = self._source(mode)
        if lower is None:
            model.set_rows([])
            self.count.clear()
            if self._lists is None:
                self.list_msg.setText("" if not self.svc.available() else "opening SDB…")
            elif mode == "Presets":
                self.list_msg.setText("reading presets…")
            elif mode == "Textures":
                self.list_msg.setText("indexing texture → material bindings… (first run ≈5–10 s, then cached)")
            return
        self.list_msg.clear()
        ep, text = self._epoch, self._search_text[mode]
        t0 = time.perf_counter()

        def done(rows):
            if ep != self._epoch:
                return
            view = self.views[mode]
            keep = view.currentIndex()
            keep_id = model.row_id(keep.row()) if keep.isValid() else None
            model.set_rows(rows)
            self.timings[f"search_{mode}"] = time.perf_counter() - t0
            if mode == self._mode():
                self.count.setText(f"{len(rows):,} / {len(lower):,}")
            if keep_id is not None:
                r = model.row_of(keep_id)
                if r >= 0:
                    self._select_row(view, r, notify=False)

        self._submit(("sdb", "search", mode), scan.filter_rows, lower, text,
                               index_prefix=(mode != "Textures"), on_done=done)

    def _select_row(self, view: QTableView, row: int, notify: bool = True) -> None:
        sm = view.selectionModel()
        ix = view.model().index(row, 0)
        if not notify:
            sm.blockSignals(True)
        sm.setCurrentIndex(ix, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
        if not notify:
            sm.blockSignals(False)
        view.scrollTo(ix, QAbstractItemView.PositionAtCenter)

    def _on_current(self, mode: str, cur: QModelIndex) -> None:
        if not cur.isValid() or self._sdb is None:
            return
        rid = self.views[mode].model().row_id(cur.row())
        if mode == "Materials":
            self._load_material(rid)
        elif mode == "Presets":
            self._load_preset(rid)
        else:
            self._load_texture(self._tex_names[rid])

    # ================================================================================================================
    # navigation
    # ================================================================================================================
    def open_material(self, name: str) -> bool:
        """Switch to Materials, clear the search, select and show *name*. Returns False for an unknown name."""
        name = (name or "").strip()
        if self._lists is None:
            self._pending_material = name
            if not self.svc.available():
                self._show_message(f"Cannot show {name}: no SDB is loaded.")
            else:
                self._show_message(f"Opening {self._db_label()} … (then {name})")
            return False
        key = name.casefold()
        hits = self._lists.by_name.get(key)
        if not hits and ("/" in key or "\\" in key):
            hits = self._lists.by_name.get(key.replace("\\", "/").rsplit("/", 1)[-1])
        if not hits:
            msg = f"Material '{name}' is not in {self._db_label()}."
            self._show_message(msg)
            self.ctx.status.emit(msg)
            return False
        self._open_material_index(hits[0])
        return True

    def _open_material_index(self, index: int) -> None:
        self._search_text["Materials"] = ""
        self._cancel_job(("sdb", "search", "Materials"))
        self.mode_buttons["Materials"].setChecked(True)
        self.list_stack.setCurrentWidget(self.views["Materials"])
        self.search.blockSignals(True)
        self.search.setText("")
        self.search.blockSignals(False)
        self._search_timer.stop()
        n = len(self._lists.names)
        if len(self.mat_model.rows) != n:
            self.mat_model.set_rows(np.arange(n, dtype=np.int64))
        self.count.setText(f"{n:,} / {n:,}")
        self.list_msg.clear()
        view = self.views["Materials"]
        self._select_row(view, index, notify=False)
        self._load_material(index)

    def _on_anchor(self, url: QUrl) -> None:
        kind, _, arg = url.toString().partition(":")
        if kind == "preset" and arg.isdigit():
            self.open_preset(int(arg))
        elif kind == "material":
            self.open_material(QUrl.fromPercentEncoding(arg.encode()))

    def open_preset(self, index: int) -> None:
        if self._lists is None:
            return
        self._search_text["Presets"] = ""
        self.set_mode("Presets")
        self._load_preset(index)
        r = self.preset_model.row_of(index)
        if r >= 0:
            self._select_row(self.views["Presets"], r, notify=False)

    # ================================================================================================================
    # material detail
    # ================================================================================================================
    def _load_material(self, index: int) -> None:
        ep, s, cat = self._epoch, self._sdb, self.ctx.catalog
        name = self._lists.names[index]
        self.pages.setCurrentIndex(P_MAT)
        self.mat_title.setText(f"{name or '(empty name)'}   #{index}   …")
        t0 = time.perf_counter()

        def done(d):
            if ep != self._epoch:
                return
            self.timings["material"] = time.perf_counter() - t0
            self._show_material(d)

        def fail(err):
            if ep == self._epoch:
                self._show_message(f"Material #{index} {name} could not be resolved:\n\n{err.splitlines()[0]}")

        self._submit(("sdb", "detail"), scan.material_detail, s, cat, index, on_done=done, on_error=fail)

    def _show_material(self, d: dict) -> None:
        self._detail = d
        m = d["material"]
        self.pages.setCurrentIndex(P_MAT)
        routes = m["routes"]
        nvar = sum(len(r["variants"]) for r in routes)
        self.mat_title.setText(f"{m['name'] or '(empty name)'}   #{m['index']}")
        self._fill_overview()
        self._fill_params()
        self._fill_textures()
        self.raw.setPlainText(self._raw_text())
        self._fill_used_by()
        self.var_summary.setText(f"{nvar} variant(s) · "
                                 f"{sum(len(v['texture_bindings']) for r in routes for v in r['variants'])} bindings · "
                                 f"{len(d['providers'])} distinct textures")

    def _fill_overview(self) -> None:
        d = self._detail
        m = d["material"]
        h = [f"<h2>{_esc(m['name'] or '(empty name)')}</h2><table cellspacing=4>"]

        def row(k, v):
            h.append(f"<tr><td style='color:#888;padding-right:12px'>{k}</td><td>{v}</td></tr>")

        row("database", _esc(m["database"]))
        row("material index (0xB2)", m["index"])
        row("routes (0xCA)", ", ".join(str(r["index"]) for r in m["routes"]) or "<b>none</b>")
        nvar = sum(len(r["variants"]) for r in m["routes"])
        row("variants", nvar)
        row("non-rendering", "<b>yes</b> — every variant has 0 render passes" if m["non_rendering"] else "no")
        missing = [t for t, g in d["providers"].items() if not g]
        row("textures", f"{len(d['providers'])} distinct, "
            + (f"<span style='color:#e06c75'>{len(missing)} not found in the catalog</span>" if missing else
               "all found in the catalog")
            + ("" if d["catalog_ready"] else " <i>(catalog still indexing)</i>"))
        h.append("</table>")
        for r in m["routes"]:
            h.append(f"<h3>Route {r['index']}</h3><table cellspacing=4>")
            if r["preset_indices"]:
                links = ", ".join(f"<a href='preset:{i}'>{_esc(r['preset'] or '(empty)')} #{i}</a>"
                                  for i in r["preset_indices"])
            else:
                links = f"<span style='color:#e06c75'>{_esc(r['preset'])} — no preset record</span>"
            row("preset", links + (" <b>(ambiguous)</b>" if len(r["preset_indices"]) > 1 else ""))
            row("program / selectors (0xC2)", r["program"])
            row("tokens (0xB6)", r["tokens_index"])
            row("slots (0xCE) / values (0xD2)", f"{r['slots']} / {r['values']}")
            row("parameters set", len(r["parameters"]))
            h.append("</table>")
            feats = [t for t in r["tokens"].split(";")[1:] if t]
            h.append(f"<p><b>Features</b> ({len(feats)})</p><ul>")
            h.extend(f"<li><code>{_esc(t)}</code></li>" for t in feats)
            h.append("</ul>")
        self.overview.setHtml("".join(h))

    def _fill_params(self) -> None:
        d = self._detail
        if d is None:
            return
        t = self.params
        t.setUpdatesEnabled(False)
        t.setRowCount(0)
        decls = d["decls"]
        rows = []
        seen = set()
        for r in d["material"]["routes"]:
            for p in r["parameters"]:
                seen.add(p["id"])
                dc = decls.get(p["id"])
                if not p["declared"] or dc is None:
                    status, color, dflt = "undeclared", MISSING, ""
                else:
                    if dc["default"] is None and p.get("value_hex") is not None:
                        same, status = False, "set (default untyped)"
                    else:
                        same = p.get("value_hex") == dc["default_hex"]
                        status = "= preset default" if same else "overridden"
                    color = None if same else OVERRIDE
                    dflt = scan.format_value(dc["default"]) if dc["default"] is not None else f"hex {dc['default_hex']}"
                val = scan.format_value(p.get("value")) if "value" in p else f"raw 0x{p['raw']:08X}"
                if p.get("string") and p["string"].get("name") is None:
                    val = f"<runtime name {p['string'].get('runtime_index')}>"
                rows.append(([p["id"], p["name"], p.get("type", "?"), val, p["offset"], "1" if p["flag_bit31"] else "",
                              dflt, status, dc["annotation"] if dc else ""], color, False))
        if self.show_preset_only.isChecked():
            for pid, dc in sorted(decls.items()):
                if pid in seen:
                    continue
                dflt = scan.format_value(dc["default"]) if dc["default"] is not None else f"hex {dc['default_hex']}"
                rows.append(([pid, dc["name"], dc["type_name"], "", "", "", dflt, "preset only", dc["annotation"]],
                             DIM, True))
        t.setRowCount(len(rows))
        for i, (vals, color, dim) in enumerate(rows):
            for c, v in enumerate(vals):
                it = _cell(v, mono=c in (3, 6), color=color if c == 7 or dim else None)
                if c in (0, 4):
                    it.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                if c == 3 and color is OVERRIDE:
                    f = QFont(it.font())
                    f.setBold(True)
                    it.setFont(f)
                t.setItem(i, c, it)
        t.setUpdatesEnabled(True)
        self.mat_tabs.setTabText(1, f"Parameters ({sum(1 for r in rows if not r[2])})")

    def _catalog_text(self, gids: list[int]) -> tuple[str, QColor | None]:
        if not gids:
            ready = self._detail["catalog_ready"] if self._detail else True
            return ("missing" if ready else "missing (catalog indexing…)"), MISSING
        cat = self.ctx.catalog
        labels = []
        for g in gids:
            try:
                lb = cat.entry(g).label
            except Exception:  # noqa: BLE001
                lb = f"gid {g}"
            if lb not in labels:
                labels.append(lb)
        return "found in " + ", ".join(labels), None

    def _fill_textures(self) -> None:
        d = self._detail
        prov = d["providers"]
        self.variants.clear()
        per_tex: dict[str, dict] = {}
        multi = len(d["material"]["routes"]) > 1
        k = 0
        for r in d["material"]["routes"]:
            for v in r["variants"]:
                b = v["texture_bindings"]
                label = (f"route {r['index']} · " if multi else "") + \
                    f"variant {k}  {v['selector_hex']}  shader {v['shader']}  passes {v['render_pass_count']}"
                top = QTreeWidgetItem([label, "", f"{len(b)} textures", f"A2 {v['texture_array']}", ""])
                if v["render_pass_count"] == 0:
                    top.setForeground(0, QBrush(DIM))
                top.setFont(2, mono_font())
                for x in b:
                    tex = x["texture"]
                    gids = prov.get(tex, []) if tex else []
                    ctext, ccol = self._catalog_text(gids) if tex else ("", None)
                    param = f"{x['param']} ({x['param_id']})" if x["param"] else "—"
                    ttxt = tex if tex else f"<runtime name {x.get('runtime_index')}>"
                    ch = QTreeWidgetItem([f"slot {x['binding']}", param, ttxt, x["source"], ctext])
                    ch.setData(0, Qt.UserRole, gids)
                    if ccol is not None:
                        ch.setForeground(4, QBrush(ccol))
                    if x["source"] == "shader_default":
                        ch.setForeground(3, QBrush(DIM))
                    top.addChild(ch)
                    if tex:
                        e = per_tex.setdefault(tex, {"params": [], "variants": set()})
                        if x["param"] and x["param"] not in e["params"]:
                            e["params"].append(x["param"])
                        e["variants"].add(k)
                self.variants.addTopLevelItem(top)
                top.setExpanded(k < 2)
                k += 1
        self.used_textures.clear()
        for tex in sorted(per_tex):
            e = per_tex[tex]
            gids = prov.get(tex, [])
            ctext, ccol = self._catalog_text(gids)
            it = QTreeWidgetItem([tex, ", ".join(e["params"]) or "(shader default)", str(len(e["variants"])), ctext])
            it.setData(0, Qt.UserRole, gids)
            if ccol is not None:
                it.setForeground(3, QBrush(ccol))
            self.used_textures.addTopLevelItem(it)
        self.mat_tabs.setTabText(2, f"Textures / Variants ({len(per_tex)})")

    def _raw_text(self) -> str:
        d = self._detail
        out = [f"# {d['material']['name']}  — every record the material references ({d['material']['database']})", ""]
        for r in d["raw"]:
            n = len(r["hex"]) // 2 if not r["hex"].startswith("<") else 0
            out.append(f"{r['table']}[{r['index']}]  {n} bytes")
            out.append(scan.hexdump(r["hex"]))
            out.append("")
        return "\n".join(out)

    def _on_texture_item(self, item: QTreeWidgetItem, _col: int) -> None:
        gids = item.data(0, Qt.UserRole)
        if gids:
            self.ctx.openTexture.emit(int(gids[0]))
        elif item.childCount() == 0 and item.text(2 if item.treeWidget() is self.variants else 0):
            self.ctx.status.emit("That texture is not in any loaded pack.")

    def material_json(self) -> dict | None:
        return self._detail["material"] if self._detail else None

    def export_json(self, path: Path) -> None:
        """Write the current material dict (as `nr sdb material` prints it) to *path* in a worker."""
        m = self.material_json()
        if m is None:
            return
        path = Path(path)

        def job():
            tmp = path.with_name(path.name + ".partial")
            tmp.write_text(json.dumps(m, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
            tmp.replace(path)
            return path

        self._submit(("sdb", "export"), job, on_done=lambda p: self.ctx.status.emit(f"wrote {p}"),
                               on_error=lambda e: QMessageBox.warning(self, "Export JSON", e.splitlines()[0]))

    def _export_clicked(self) -> None:
        m = self.material_json()
        if m is None:
            return
        stem = (m["name"] or f"material_{m['index']}").replace("/", "_").replace("\\", "_")
        start = str(Path(self.ctx.export_dir() or Path.home()) / f"{stem}.json")
        fn, _ = QFileDialog.getSaveFileName(self, "Export material JSON", start, "JSON (*.json)")
        if fn:
            self.ctx.set_export_dir(str(Path(fn).parent))
            self.export_json(Path(fn))

    # ---- whole-database dump ------------------------------------------------------------------------------------
    def used_by_indices(self) -> dict:
        """What the tab already knows about material usage, in the shape `sdb.export` wants.

        Only what has actually been scanned: the model refs when they are loaded, the mesh index as far as the
        background scan got. `complete` says whether both were finished, so the document never implies that an
        empty `used_by` means "nothing uses this" when the scan simply had not run.
        """
        models: dict[str, list] = {}
        for key, rows in (self._model_refs or {}).get("refs", {}).items():
            models[key] = [{"model": r.get("model"), "pak": r.get("pak"), "slot": r.get("slot"),
                            "mesh": r.get("mesh"), "kind": r.get("kind"),
                            "selected": r.get("selected")} for r in rows]
        meshes: dict[str, list] = {}
        cat = self.ctx.catalog
        for key, gids in (self.scanner.index or {}).items():
            meshes[key] = sorted({cat.name(g) for g in gids})
        return {"models": models, "meshes": meshes,
                "models_scanned": (self._model_refs or {}).get("models"),
                "meshes_scanned": len(self.scanner.scanned) if self.scanner.index else 0,
                "complete": bool(self._model_refs) and self.scanner.complete()}

    def _dump_clicked(self) -> None:
        sdb = self.svc.sdb                       # a property, not a call
        if sdb is None:
            QMessageBox.information(self, self.TITLE, "No database open.")
            return
        if self._dump_cancel is not None:
            return
        if not self.scanner.complete() or not self._model_refs:
            box = QMessageBox(QMessageBox.Question, "Export all",
                              "The 'used by' scan has not finished.\n\n"
                              "Export now and the materials file records which models and meshes are known so far, "
                              "marked incomplete. Run the mesh scan first for the full picture.", parent=self)
            go = box.addButton("Export anyway", QMessageBox.AcceptRole)
            box.addButton("Scan first", QMessageBox.RejectRole)
            box.addButton(QMessageBox.Cancel)
            box.exec()
            clicked = box.clickedButton()
            if clicked is None or box.buttonRole(clicked) == QMessageBox.RejectRole:
                if clicked is not None and box.buttonRole(clicked) == QMessageBox.RejectRole:
                    self.ensure_model_refs()
                    self.start_mesh_scan()
                return
            if clicked is not go:
                return
        d = QFileDialog.getExistingDirectory(self, "Export the database to", self.ctx.export_dir())
        if not d:
            return
        self.ctx.set_export_dir(d)
        self.start_dump(Path(d))

    def start_dump(self, out_dir: Path) -> bool:
        """Run the three-file dump in a worker. False when one is already running."""
        if self._dump_cancel is not None:
            return False
        self._dump_cancel = threading.Event()
        self.dump_btn.setEnabled(False)
        used_by = self.used_by_indices()
        path = self.svc.path
        cancel = self._dump_cancel

        def job():
            with Sdb.open(path) as own:                      # the worker reads its own handle, never the tab's
                return sdbexport.export_all(own, out_dir, used_by=used_by, cancel=cancel.is_set)

        self._submit(("sdb", "dump"), job, on_done=self._on_dump_done, on_error=self._on_dump_failed)
        self.ctx.status.emit(f"Exporting the database to {out_dir}…")
        return True

    def _on_dump_done(self, res: dict) -> None:
        self._dump_cancel = None
        self.dump_btn.setEnabled(True)
        lines = [f"{k}: {v['count']:,} in {Path(v['path']).name} ({human_size(v['bytes'])})"
                 for k, v in res.items()]
        self.ctx.status.emit("Database exported: " + "; ".join(lines))
        QMessageBox.information(self, self.TITLE, "Exported:\n\n" + "\n".join(lines))

    def _on_dump_failed(self, err) -> None:
        self._dump_cancel = None
        self.dump_btn.setEnabled(True)
        if isinstance(err, sdbexport.Cancelled):
            self.ctx.status.emit("Database export cancelled.")
            return
        QMessageBox.warning(self, self.TITLE, f"Export failed:\n{err}")

    # ---- used by ------------------------------------------------------------------------------------------------
    def _on_mat_tab(self, i: int) -> None:
        if i == self.usedby_index and not self._mesh_scan_requested:
            self.start_mesh_scan()

    def _fill_used_by(self) -> None:
        self._fill_models()
        self._fill_meshes()

    def ensure_model_refs(self) -> None:
        if self._model_refs is not None or self._model_refs_loading:
            return
        self._model_refs_loading = True
        self.models_label.setText("Models: scanning .model documents…")

        def done(r):
            self._model_refs_loading = False
            self._model_refs = r
            self.timings["model_refs"] = r["seconds"]
            self._fill_models()

        def fail(e):
            self._model_refs_loading = False
            self.models_label.setText(f"Models: scan failed — {e.splitlines()[0]}")

        self._submit(("sdb", "model-refs"), scan.build_model_refs, self.ctx.paks, on_done=done, on_error=fail)

    def model_refs_for(self, name: str) -> list[dict] | None:
        if self._model_refs is None:
            return None
        return self._model_refs["refs"].get(name.casefold(), [])

    def _fill_models(self) -> None:
        tree = self.models_tree
        tree.clear()
        if self._detail is None:
            return
        name = self._detail["material"]["name"]
        refs = self.model_refs_for(name)
        if refs is None:
            self.ensure_model_refs()
            if self._model_refs is None:
                self.models_label.setText("Models: scanning .model documents…" if self.ctx.paks.paths else
                                          "Models: no dataN.pak found")
                return
            refs = self.model_refs_for(name)
        by_model: dict[str, list[dict]] = {}
        for r in refs:
            by_model.setdefault(r["model"], []).append(r)
        for model in sorted(by_model, key=str.casefold):
            rows = by_model[model]
            first = rows[0]
            pak = first["pak"] + (f" (overridden by {', '.join(first['overridden_by'])})" if first["overridden_by"]
                                  else "")
            top = QTreeWidgetItem([model, f"{len(rows)} reference(s)", "", "", "", "", pak])
            top.setData(0, Qt.UserRole, model)
            for r in rows:
                sel = "" if r["selected"] is None else ("yes" if r["selected"] else "no")
                if r["kind"] == "resource" and r["alternatives"] and r["alternatives"] > 1:
                    sel += f" (1 of {r['alternatives']})"
                kind = "base" if r["kind"] == "resource" else "embedded"
                if r["kind"] == "resource" and r["embedded"]:
                    kind += f" ← {r['embedded']}"
                ch = QTreeWidgetItem([str(r["slot"]), str(r["mesh"]), kind, str(r["number"]), sel,
                                      scan.rtti_text(r["rtti"]), ""])
                ch.setData(0, Qt.UserRole, model)
                ch.setToolTip(5, scan.rtti_text(r["rtti"]).replace("; ", "\n"))
                top.addChild(ch)
            tree.addTopLevelItem(top)
            if len(by_model) <= 20:
                top.setExpanded(True)
        errs = self._model_refs["errors"]
        self.models_label.setText(
            f"Models: {len(by_model)} of {self._model_refs['models']} .model documents reference this material "
            f"({len(refs)} rows; 'base' = materialsResources entry, 'embedded' = materialsData name)"
            + (f" — {len(errs)} documents unreadable" if errs else ""))

    def _on_model_item(self, item: QTreeWidgetItem, _col: int) -> None:
        name = item.data(0, Qt.UserRole)
        if name:
            self.ctx.openModel.emit(name)

    def start_mesh_scan(self) -> None:
        self._mesh_scan_requested = True
        if self.scanner.start():
            self.scan_prog.setRange(0, 0)
            self.scan_prog.show()
            self.scan_cancel.show()
            self.scan_btn.setEnabled(False)
            self.meshes_label.setText("Meshes: scanning catalog meshes for embedded material names…")

    def _on_scan_progress(self, done: int, total: int) -> None:
        self.scan_prog.setRange(0, max(total, 1))
        self.scan_prog.setValue(done)

    def _on_scan_finished(self, ok: bool) -> None:
        self.scan_prog.hide()
        self.scan_cancel.hide()
        self.scan_btn.setEnabled(True)
        self.timings["mesh_scan"] = self.scanner.seconds
        if not ok:
            self.ctx.status.emit("Mesh scan cancelled")
        self._fill_meshes()

    def _on_catalog_ready(self) -> None:
        if self._mesh_scan_requested and self.scanner.pending_packs() and not self.scanner.running():
            self.start_mesh_scan()
        if self._detail is not None and not self._detail["catalog_ready"]:
            idx = self._detail["material"]["index"]
            if self._sdb is not None:
                self._load_material(idx)

    def _fill_meshes(self) -> None:
        t = self.meshes_table
        t.setRowCount(0)
        sc = self.scanner
        pending = sc.pending_packs()
        if self._detail is None:
            return
        if not sc.scanned and not sc.running():
            self.meshes_label.setText("Meshes: not scanned yet — open this page or press Scan meshes "
                                      "(one pass over every mesh, cached for the session)")
            return
        gids = sc.lookup(self._detail["material"]["name"])
        cat = self.ctx.catalog
        t.setRowCount(len(gids))
        for i, g in enumerate(gids):
            try:
                nm, lb = cat.name(g), cat.entry(g).label
            except Exception:  # noqa: BLE001
                nm, lb = f"gid {g}", ""
            t.setItem(i, 0, _cell(nm, data=g))
            t.setItem(i, 1, _cell(lb))
            t.setItem(i, 2, _cell(g))
        state = "scanning…" if sc.running() else (
            f"{len(pending)} pack(s) not scanned yet — press Scan meshes" if pending else "scan complete")
        self.meshes_label.setText(
            f"Meshes: {len(gids)} embed this material · {sc.meshes:,} meshes scanned in {sc.seconds:.1f} s"
            f" ({sc.failed:,} without readable materials) · {state}")

    def _on_mesh_cell(self, row: int, _col: int) -> None:
        it = self.meshes_table.item(row, 0)
        g = it.data(Qt.UserRole) if it else None
        if g is not None:
            self.ctx.openMesh.emit(int(g))

    # ================================================================================================================
    # preset / texture / stats pages
    # ================================================================================================================
    def _load_preset(self, index: int) -> None:
        ep, s = self._epoch, self._sdb
        if s is None:
            return
        self.pages.setCurrentIndex(P_PRESET)

        def done(p):
            if ep == self._epoch:
                self._show_preset(p)

        self._submit(("sdb", "detail"), scan.preset_detail, s, index, on_done=done,
                               on_error=lambda e: self._show_message(f"Preset #{index}: {e.splitlines()[0]}"))

    def _show_preset(self, p: dict) -> None:
        self._preset_cur = p
        self.pages.setCurrentIndex(P_PRESET)
        users = self._lists.preset_materials.get(p["name"]) if self._lists.presets else None
        self.preset_title.setText(
            f"<h2>{_esc(p['name'] or '(empty name)')}</h2>preset #{p['index']} · {len(p['parameters'])} declarations"
            f" · key 0x{p['key']:016X} · flags 0x{p['flags']:08X} · masks {p['masks']} · group B {len(p['groups_b'])}"
            f" · record {'complete' if p['complete'] else '<b>not fully consumed</b>'}")
        t = self.preset_params
        t.setUpdatesEnabled(False)
        t.setRowCount(len(p["parameters"]))
        for i, d in enumerate(p["parameters"]):
            dflt = scan.format_value(d["default"]) if d["default"] is not None else f"hex {d['default_hex']}"
            tn = d["type_name"] + ("" if not d["type_flags"] & 0x8000 else " ·8000")
            for c, v in enumerate((d["id"], d["name"], tn, d["expression"], dflt, d["annotation"])):
                t.setItem(i, c, _cell(v, mono=c in (3, 4)))
        t.setUpdatesEnabled(True)
        if users is None:
            self.preset_users_label.setText("Materials using this preset: computing…")
            self.preset_users_model.set_rows([])
        else:
            self.preset_users_label.setText(f"{len(users):,} material(s) use this preset (route token prefix)")
            self.preset_users_model.set_rows(users)

    def materials_using_preset(self, name: str) -> list[int] | None:
        L = self._lists
        return None if L is None or not L.presets else list(L.preset_materials.get(name, []))

    def _load_texture(self, tex: str) -> None:
        self._tex_cur = tex
        self.pages.setCurrentIndex(P_TEX)
        users = (self._reverse or {}).get(tex, [])
        self.tex_title.setText(f"<h2>{_esc(tex)}</h2>{len(users):,} material(s) bind this texture")
        t = self.tex_users
        t.setRowCount(len(users))
        L = self._lists
        pmap = {}
        if L is not None and L.material_preset:
            for u in users:
                hits = L.by_name.get(u.casefold())
                pmap[u] = L.material_preset[hits[0]] if hits else ""
        for i, u in enumerate(users):
            t.setItem(i, 0, _cell(u))
            t.setItem(i, 1, _cell(pmap.get(u, "")))
        self.tex_providers.setRowCount(0)
        cat = self.ctx.catalog

        def done(r):
            if self._tex_cur != tex:
                return
            gids = r.get(tex, [])
            tp = self.tex_providers
            tp.setRowCount(max(1, len(gids)))
            if not gids:
                tp.setItem(0, 0, _cell("not found in any loaded pack" +
                                       ("" if cat.is_ready else " (catalog still indexing)"), color=MISSING))
                tp.setItem(0, 1, _cell(""))
                return
            for i, g in enumerate(gids):
                tp.setItem(i, 0, _cell(cat.entry(g).label))
                tp.setItem(i, 1, _cell(g))
                tp.setItem(i, 2, _cell(cat.name(g)))

        self._submit(("sdb", "tex-providers"), scan.texture_providers, cat, [tex], on_done=done)

    def show_stats(self) -> None:
        s = self._sdb
        self.pages.setCurrentIndex(P_STATS)
        if s is None:
            self.stats_text.setPlainText("No SDB loaded.")
            self.stats_tables.setRowCount(0)
            return

        def done(st):
            lines = [f"{st['path']}",
                     f"size {st['size']:,} B   block A {st['block_a']:,}   block B {st['block_b']:,}",
                     f"materials {st['materials']:,}   routes {st['routes']:,}   presets {st['presets']:,}",
                     f"compact int tags {st['compact_tags']}",
                     f"inner header opaque {st['inner_opaque_hex']}"]
            for k, v in st["programs"].items():
                lines.append(f"program kind {k}: {v['count']:,} blobs, {v['bytes']:,} B")
            if self._reverse is not None:
                lines.append(f"distinct texture names bound (reverse index): {len(self._reverse):,}")
            self.stats_text.setPlainText("\n".join(lines))
            t = self.stats_tables
            t.setRowCount(len(st["tables"]))
            for i, tb in enumerate(st["tables"]):
                for c, v in enumerate((tb["key"], tb["shape"], tb["width"], f"{tb['count']:,}", f"{tb['bytes']:,}",
                                       tb["meaning"])):
                    it = _cell(v, mono=c == 0)
                    if c in (2, 3, 4):
                        it.setTextAlignment(int(Qt.AlignRight | Qt.AlignVCenter))
                    t.setItem(i, c, it)

        self._submit(("sdb", "stats"), s.stats, on_done=done)

    def start_validate(self) -> None:
        s = self._sdb
        if s is None or self.validator.running():
            return
        self._validate_total = len(self._lists.names) if self._lists else 0
        self.validate_prog.setRange(0, max(1, self._validate_total))
        self.validate_prog.setValue(0)
        self.validate_prog.show()
        self.validate_cancel.show()
        self.validate_btn.setEnabled(False)
        self.validate_out.setPlainText("resolving every material…")
        self._validate_epoch = self._epoch
        self.validator.start(s)

    def _on_validate_progress(self, msg: str) -> None:
        try:
            self.validate_prog.setValue(int(msg.split("/", 1)[0]))
        except ValueError:
            pass

    def _validate_finish(self) -> None:
        self.validate_prog.hide()
        self.validate_cancel.hide()
        self.validate_btn.setEnabled(True)

    def _on_validate_done(self, res) -> None:
        self._validate_finish()
        if res is None:
            self.validate_out.setPlainText("validation cancelled")
            return
        if getattr(self, "_validate_epoch", None) != self._epoch:
            self.validate_out.setPlainText("(result discarded: the SDB was switched)")
            return
        self.last_validation = res
        tot = res["totals"]
        lines = [f"validated in {res.get('seconds', 0):.1f} s"]
        lines += [f"  {k:<28} {v:>12,}" for k, v in tot.items()]
        for e in res["errors"]:
            lines.append(f"  error: material {e['material']}: {e['error']}")
        self.validate_out.setPlainText("\n".join(lines))

    def _on_validate_failed(self, err: str) -> None:
        self._validate_finish()
        self.validate_out.setPlainText(f"validation failed: {err}")
