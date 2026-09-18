"""Textures tab: every type-0x20 resource of every loaded pack in one searchable, sortable, checkable list (or a
thumbnail grid), a zoomable preview, the full IMGC/level/part info, other providers of the same name, the SDB
materials that bind the texture, and a streaming export (DDS lossless / PNG preview / raw parts).

Threading: searches + sorting, preview decodes, thumbnails, channel filtering and exports all run through
`ctx.runner`; list cells read only the 80-byte header part of visible rows (cached).
"""
from __future__ import annotations

import html
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
from PySide6.QtCore import QEvent, QItemSelectionModel, QModelIndex, QObject, QRect, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QListView, QMenu, QMessageBox, QProgressBar, QPushButton, QSplitter, QStackedWidget,
                               QStyle, QStyledItemDelegate, QStyleOptionButton, QTableView, QTextBrowser,
                               QToolButton, QVBoxLayout, QWidget)

from ...texture.codec import texture_stem
from ...texture.dds import iter_dds
from ...util.names import safe_filename
from ..catalog import Catalog
from ..tasks import Debouncer
from ..texpreview import (TEXTURE_TYPE, TexturePreview, checker_brush, decode_texture, header_summary,
                          read_header)
from ..widgets import Column, RefListModel, SearchBar, human_size, mono_font, unique_path, write_span

TITLE = "Textures"
THUMB = 112                       # thumbnail box (px)
THUMB_CACHE = 3000                # LRU entries
PREVIEW_MAX = 4096                # preview surfaces larger than 2×this on a side are stride-downsampled
COL_NAME, COL_FORMAT, COL_SIZE, COL_MIPS, COL_PACK, COL_BYTES = range(6)


# ----------------------------------------------------------------------------------------------------------------
# worker functions (no Qt widgets touched)
# ----------------------------------------------------------------------------------------------------------------

def _sort_key(catalog: Catalog, col: int):
    """Key function gid -> sortable value for column *col* (runs in a worker)."""
    if col == COL_NAME:
        return lambda g: catalog.name(g).lower()
    if col == COL_PACK:
        return lambda g: (catalog.entry(g).label.lower(), catalog.name(g).lower())
    if col == COL_BYTES:
        return catalog.part_size

    def hdr(g):
        e, i = catalog.split(g)
        try:
            h, _, _ = read_header(e.pack, i)
        except Exception:  # noqa: BLE001 - garbage sorts last
            return (1, "", 0)
        if col == COL_FORMAT:
            return (0, h.format_name, 0)
        if col == COL_SIZE:
            return (0, "", h.width * h.height * max(1, h.depth))
        return (0, "", h.mip_count)
    return hdr


def search_textures(catalog: Catalog, text: str, packs, sort: tuple[int, int] | None) -> tuple[np.ndarray, float]:
    """(gids, seconds): texture gids matching *text* in *packs*, optionally sorted by (column, Qt.SortOrder)."""
    t0 = time.perf_counter()
    gids = catalog.search(text, types=[TEXTURE_TYPE], packs=packs)
    if sort is not None and len(gids) > 1:
        col, order = sort
        key = _sort_key(catalog, col)
        lst = sorted(gids.tolist(), key=key, reverse=(order == int(Qt.DescendingOrder.value)))
        gids = np.asarray(lst, dtype=np.int64)
    return gids, time.perf_counter() - t0


def preview_job(catalog: Catalog, gid: int, mip: int, face: int, slice_: int):
    """Decode for the preview + the same-name providers (lookup builds its table on first use: keep it here)."""
    e, i = catalog.split(gid)
    img, info = decode_texture(e.pack, i, mip, face, slice_, max_dim=PREVIEW_MAX)
    name = catalog.name(gid)
    info["providers"] = catalog.lookup(name, TEXTURE_TYPE)
    h = info.get("header")
    if h is not None and h.header_only and h.reference:
        info["reference_providers"] = catalog.lookup(h.reference, TEXTURE_TYPE)
    return img, info


def thumb_job(catalog: Catalog, gid: int) -> QImage | str:
    e, i = catalog.split(gid)
    img, info = decode_texture(e.pack, i, mip=None, max_dim=THUMB)
    if img.isNull():
        return info.get("error") or "?"
    if img.width() > THUMB or img.height() > THUMB:
        img = img.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)
    elif max(img.width(), img.height()) < THUMB // 2:          # tiny textures: blow up with visible texels
        k = max(1, THUMB // max(img.width(), img.height()))
        img = img.scaled(img.width() * k, img.height() * k, Qt.KeepAspectRatio, Qt.FastTransformation)
    return img


def pack_dir(label: str) -> Path:
    """`dlc_samples/samples_textures_pc.rpack` -> Path('dlc_samples/samples_textures_pc') (components sanitised)."""
    parts = [safe_filename(p) for p in label.replace("\\", "/").split("/") if p not in ("", ".", "..")]
    if not parts:
        parts = ["pack"]
    last = parts[-1]
    if last.lower().endswith(".rpack"):
        last = last[:-6] or "pack"
    parts[-1] = last
    return Path(*parts)


class ExportCancelled(Exception):
    pass


def _write_dds(pack, index: int, dest: Path, cancel: threading.Event) -> int:
    """Same bytes as `nr texture export … out.dds` (texture.cli), streamed to *dest*."""
    h, _, bmp_i = read_header(pack, index)
    if bmp_i is None:
        raise ValueError(f"header-only record (reference {h.reference!r}); no texels")
    tmp = dest.with_name(dest.name + ".partial")
    n = 0
    try:
        with open(tmp, "wb") as fh:
            for chunk in iter_dds(h, pack.read_part(bmp_i)):
                if cancel.is_set():
                    raise ExportCancelled()
                fh.write(chunk)
                n += len(chunk)
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return n


def export_job(catalog: Catalog, gids: list[int], mode: str, out: Path, cancel: threading.Event, progress) -> dict:
    """Export *gids* under *out* (`<out>/<pack label stem>/<texture stem>.<ext>`). Never overwrites (a `.N` suffix
    is added). mode: 'dds' | 'png' | 'raw'. Returns a summary; per-texture errors are collected, not raised."""
    summary = {"written": 0, "bytes": 0, "errors": [], "cancelled": False, "out": str(out), "mode": mode,
               "total": len(gids)}
    last = 0.0
    for k, gid in enumerate(gids):
        if cancel.is_set():
            summary["cancelled"] = True
            break
        name = catalog.name(gid)
        now = time.perf_counter()
        if now - last > 0.05:
            progress.emit(k, len(gids), name)
            last = now
        e, i = catalog.split(gid)
        d = out / pack_dir(e.label)
        stem = texture_stem(name)
        try:
            d.mkdir(parents=True, exist_ok=True)
            if mode == "dds":
                summary["bytes"] += _write_dds(e.pack, i, unique_path(d / f"{stem}.dds"), cancel)
                summary["written"] += 1
            elif mode == "png":
                img, info = decode_texture(e.pack, i, 0, 0, 0)
                if img.isNull():
                    raise ValueError(info.get("error") or "decode failed")
                dest = unique_path(d / f"{stem}.png")
                tmp = dest.with_name(dest.name + ".partial")
                if not img.save(str(tmp), "PNG"):
                    tmp.unlink(missing_ok=True)
                    raise OSError(f"could not write {dest}")
                tmp.replace(dest)
                summary["bytes"] += dest.stat().st_size
                summary["written"] += 1
            elif mode == "raw":
                res = e.pack.resource(i)
                for k2, pi in enumerate(res.part_indices):
                    if not e.pack.part_is_direct(pi):
                        raise ValueError(f"part {pi} is compressed / in a child pack; raw copy not supported")
                    t = e.pack.part_type(pi)
                    dest = unique_path(d / f"{stem}.part{k2}_0x{t:02X}.bin")
                    summary["bytes"] += write_span(e.pack, e.pack.part_offset(pi), e.pack.physicals[pi].size, dest)
                summary["written"] += 1
            else:
                raise ValueError(f"unknown export mode {mode!r}")
        except ExportCancelled:
            summary["cancelled"] = True
            break
        except Exception as exc:  # noqa: BLE001 - reported in the summary
            summary["errors"].append(f"{e.label} / {name}: {type(exc).__name__}: {exc}")
    progress.emit(len(gids), len(gids), "")
    return summary


class _Progress(QObject):
    """Signal carrier created on the GUI thread; emitted from the export worker (queued delivery)."""
    progress = Signal(int, int, str)

    def emit(self, done: int, total: int, name: str) -> None:
        self.progress.emit(done, total, name)


# ----------------------------------------------------------------------------------------------------------------
# thumbnail grid
# ----------------------------------------------------------------------------------------------------------------

class _ThumbDelegate(QStyledItemDelegate):
    """Paints a cached thumbnail (or a placeholder and a request) + check box + elided name."""

    def __init__(self, tab: "Tab"):
        super().__init__(tab)
        self.tab = tab

    def sizeHint(self, option, index) -> QSize:
        return QSize(THUMB + 16, THUMB + 34)

    @staticmethod
    def _check_rect(cell: QRect) -> QRect:
        return QRect(cell.left() + 4, cell.top() + 4, 16, 16)

    def paint(self, painter, option, index) -> None:
        gid = index.data(Qt.UserRole)
        if gid is None:
            return
        r = option.rect
        pal = option.palette
        painter.save()
        if option.state & QStyle.State_Selected:
            painter.fillRect(r, pal.highlight())
        box = QRect(r.left() + 8, r.top() + 4, THUMB, THUMB)
        th = self.tab.thumb(gid, index.row())
        if isinstance(th, QPixmap):
            x = box.left() + (THUMB - th.width()) // 2
            y = box.top() + (THUMB - th.height()) // 2
            painter.fillRect(QRect(x, y, th.width(), th.height()), self.tab.checker)
            painter.drawPixmap(x, y, th)
            painter.setPen(QColor(110, 110, 110))
            painter.drawRect(QRect(x - 1, y - 1, th.width() + 1, th.height() + 1))
        else:
            painter.fillRect(box, QColor(60, 60, 60))
            painter.setPen(QColor(200, 120, 120) if th == "error" else QColor(150, 150, 150))
            painter.drawText(box, Qt.AlignCenter, "?" if th == "error" else "…")
        painter.setPen(pal.highlightedText().color() if option.state & QStyle.State_Selected else pal.text().color())
        text_r = QRect(r.left() + 2, box.bottom() + 2, r.width() - 4, r.bottom() - box.bottom() - 2)
        name = option.fontMetrics.elidedText(str(index.data(Qt.DisplayRole)), Qt.ElideMiddle, text_r.width())
        painter.drawText(text_r, Qt.AlignHCenter | Qt.AlignTop, name)
        cb = QStyleOptionButton()
        cb.rect = self._check_rect(r)
        cb.state = QStyle.State_Enabled | (QStyle.State_On if index.data(Qt.CheckStateRole) == Qt.Checked
                                           else QStyle.State_Off)
        QApplication.style().drawControl(QStyle.CE_CheckBox, cb, painter)
        painter.restore()

    def editorEvent(self, event, model, option, index) -> bool:
        if event.type() == QEvent.MouseButtonRelease and self._check_rect(option.rect).contains(event.position().toPoint()):
            on = index.data(Qt.CheckStateRole) == Qt.Checked
            model.setData(index, Qt.Unchecked if on else Qt.Checked, Qt.CheckStateRole)
            return True
        return False


# ----------------------------------------------------------------------------------------------------------------
# tab
# ----------------------------------------------------------------------------------------------------------------

class Tab(QWidget):
    TITLE = TITLE

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.cat: Catalog = ctx.catalog
        self._hdr_cache: dict[int, tuple[str, str, str]] = {}
        self._packs_listed: set[int] = set()
        self._tex_total = 0
        self._sort: tuple[int, int] | None = None
        self._current: int | None = None
        self._pending_select: int | None = None
        self._last_info: dict | None = None
        self._mat_links: list[str] = []
        self._searching = False
        self._search_again = False
        self.last_search_s = 0.0
        self._thumbs: OrderedDict[int, QPixmap | str] = OrderedDict()
        self._thumb_wanted: dict[int, int] = {}
        self._thumb_inflight: set[int] = set()
        self._export_cancel: threading.Event | None = None
        self.last_export: dict | None = None
        self._build()
        self._search_deb = Debouncer(self._run_search, 200, self)
        self._pack_deb = Debouncer(self._run_search, 400, self)
        self._thumb_timer = QTimer(self, singleShot=True, interval=15)
        self._thumb_timer.timeout.connect(self._pump_thumbs)
        self.search.changed.connect(self._search_deb.trigger)
        self.cat.packAdded.connect(self._on_pack_added)
        self.cat.ready.connect(self._pack_deb.trigger)
        self.ctx.sdb.reverseReady.connect(self._refresh_info)
        for e in list(self.cat.packs):
            if e.pack is not None:
                self._on_pack_added(e.id)
        self._run_search()

    # ---- construction -------------------------------------------------------------------------------------------
    def _build(self) -> None:
        cols = [
            Column("Name", self.cat.name),
            Column("Format", lambda g: self._hdr(g)[0]),
            Column("Size", lambda g: self._hdr(g)[1], mono=True),
            Column("Mips", lambda g: self._hdr(g)[2], align_right=True),
            Column("Pack", lambda g: self.cat.entry(g).label),
            Column("Bytes", lambda g: human_size(self.cat.part_size(g)), align_right=True),
        ]
        self.model = RefListModel(cols, self)
        self.model.checkedChanged.connect(self._update_count)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(4, 4, 4, 4)
        self.search = SearchBar("search textures")
        lv.addWidget(self.search)

        bar = QHBoxLayout()
        self.btn_all = QPushButton("Check all visible")
        self.btn_none = QPushButton("Check none")
        self.btn_export = QToolButton(text="Export Checked ▾")
        self.btn_export.setPopupMode(QToolButton.InstantPopup)
        m = QMenu(self)
        self.act_dds = m.addAction("DDS (lossless)…", lambda: self._export("dds"))
        self.act_png = m.addAction("PNG (mip 0, 8-bit preview)…", lambda: self._export("png"))
        self.act_raw = m.addAction("Raw parts (.bin)…", lambda: self._export("raw"))
        self.btn_export.setMenu(m)
        self.btn_grid = QToolButton(text="Grid", checkable=True)
        self.btn_grid.setToolTip("Thumbnail grid (only visible items are decoded)")
        self.btn_all.clicked.connect(lambda: self.model.check_all_visible(True))
        self.btn_none.clicked.connect(self.model.clear_checks)
        self.btn_grid.toggled.connect(self._set_grid)
        for w in (self.btn_all, self.btn_none, self.btn_export):
            bar.addWidget(w)
        bar.addStretch(1)
        self.busy = QLabel("")
        bar.addWidget(self.busy)
        bar.addWidget(self.btn_grid)
        lv.addLayout(bar)

        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setWordWrap(False)
        self.table.setAlternatingRowColors(True)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(QHeaderView.Interactive)
        hh.setStretchLastSection(True)
        for c, w in ((COL_NAME, 300), (COL_FORMAT, 90), (COL_SIZE, 90), (COL_MIPS, 40), (COL_PACK, 170),
                     (COL_BYTES, 80)):
            hh.resizeSection(c, w)
        hh.setSectionsClickable(True)
        hh.setSortIndicatorShown(False)
        hh.sectionClicked.connect(self._on_header_clicked)
        self.table.selectionModel().currentRowChanged.connect(self._on_current)

        self.grid = QListView()
        self.grid.setModel(self.model)
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setUniformItemSizes(True)
        self.grid.setLayoutMode(QListView.Batched)
        self.grid.setBatchSize(2000)
        self.grid.setGridSize(QSize(THUMB + 16, THUMB + 34))
        self.grid.setSpacing(0)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self._delegate = _ThumbDelegate(self)
        self.grid.setItemDelegate(self._delegate)
        self.grid.selectionModel().currentChanged.connect(lambda cur, _prev: self._on_current(cur, None))
        self.checker = checker_brush(6)

        self.stack = QStackedWidget()
        self.stack.addWidget(self.table)
        self.stack.addWidget(self.grid)
        lv.addWidget(self.stack, 1)

        prow = QHBoxLayout()
        self.prog = QProgressBar()
        self.prog_label = QLabel("")
        self.prog_label.setMinimumWidth(10)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._cancel_export)
        prow.addWidget(self.prog, 1)
        prow.addWidget(self.btn_cancel)
        self.prog_box = QWidget()
        pv = QVBoxLayout(self.prog_box)
        pv.setContentsMargins(0, 0, 0, 0)
        pv.addLayout(prow)
        pv.addWidget(self.prog_label)
        self.prog_box.hide()
        lv.addWidget(self.prog_box)
        self.result_label = QLabel("")
        self.result_label.setWordWrap(True)
        self.result_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.result_label.hide()
        lv.addWidget(self.result_label)
        self._progress = _Progress(self)
        self._progress.progress.connect(self._on_export_progress)

        right = QSplitter(Qt.Vertical)
        self.preview = TexturePreview(self.ctx.runner)
        self.preview.levelRequested.connect(self._request_preview)
        right.addWidget(self.preview)
        info_box = QWidget()
        iv = QVBoxLayout(info_box)
        iv.setContentsMargins(0, 0, 0, 0)
        ib = QHBoxLayout()
        self.btn_raw = QPushButton("Show in Raw")
        self.btn_raw.clicked.connect(lambda: self._current is not None and self.ctx.openRaw.emit(self._current))
        self.btn_raw.setEnabled(False)
        ib.addWidget(self.btn_raw)
        ib.addStretch(1)
        iv.addLayout(ib)
        self.info = QTextBrowser()
        self.info.setOpenLinks(False)
        self.info.setFont(mono_font())
        self.info.anchorClicked.connect(self._on_link)
        iv.addWidget(self.info, 1)
        right.addWidget(info_box)
        right.setSizes([600, 350])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setSizes([760, 740])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

    # ---- list data ----------------------------------------------------------------------------------------------
    def _hdr(self, gid: int) -> tuple[str, str, str]:
        v = self._hdr_cache.get(gid)
        if v is None:
            e, i = self.cat.split(gid)
            v = header_summary(e.pack, i)
            if len(self._hdr_cache) > 200_000:
                self._hdr_cache.clear()
            self._hdr_cache[gid] = v
        return v

    def _on_pack_added(self, pack_id: int) -> None:
        e = self.cat.packs[pack_id]
        if pack_id in self._packs_listed or e.pack is None:
            if e.error:
                self._pack_deb.trigger()
            return
        self._packs_listed.add(pack_id)
        n = e.type_counts.get(TEXTURE_TYPE, 0)
        if n:
            self._tex_total += n
            self.search.add_pack(pack_id, f"{e.label} ({n:,})")
        self._pack_deb.trigger()

    # ---- search -------------------------------------------------------------------------------------------------
    def _run_search(self) -> None:
        if self._searching:
            self._search_again = True
            return
        self._searching = True
        self.busy.setText("searching…")
        self.ctx.runner.submit(("textures", "search"), search_textures, self.cat, self.search.text(),
                               self.search.packs(), self._sort, on_done=self._on_search_done,
                               on_error=self._on_search_error)

    def _on_search_error(self, err: str) -> None:
        self._searching = False
        self.busy.setText("search failed")
        self.ctx.status.emit(f"Textures: search failed: {err.splitlines()[0]}")

    def _on_search_done(self, res) -> None:
        gids, secs = res
        self._searching = False
        self.last_search_s = secs
        keep = self._pending_select if self._pending_select is not None else self._current
        self.model.set_gids(gids)
        self._update_count()
        self.busy.setText("")
        self._thumb_wanted.clear()
        if keep is not None:
            row = self.model.row_of(keep)
            if row >= 0:
                self._select_row(row, force=self._pending_select is not None)
                self._pending_select = None
        if self._search_again:
            self._search_again = False
            self._run_search()

    def _update_count(self, *_):
        self.search.set_count(len(self.model.gids), self._tex_total, len(self.model.checked))

    def _on_header_clicked(self, col: int) -> None:
        hh = self.table.horizontalHeader()
        if self._sort is not None and self._sort[0] == col:
            if self._sort[1] == int(Qt.AscendingOrder.value):
                order = Qt.DescendingOrder
            else:                              # third click: back to pack order
                self._sort = None
                hh.setSortIndicatorShown(False)
                self._run_search()
                return
        else:
            order = Qt.AscendingOrder
        self._sort = (col, int(order.value))
        hh.setSortIndicatorShown(True)
        hh.setSortIndicator(col, order)
        self._run_search()

    # ---- selection / preview ------------------------------------------------------------------------------------
    def _select_row(self, row: int, force: bool = False) -> None:
        idx = self.model.index(row, 0)
        view = self.grid if self.stack.currentWidget() is self.grid else self.table
        gid = self.model.gid_at(row)
        if force:
            self._current = None
        for v in (self.table, self.grid):
            sm = v.selectionModel()
            flags = QItemSelectionModel.ClearAndSelect
            if v is self.table:
                flags |= QItemSelectionModel.Rows
            sm.blockSignals(True)
            sm.setCurrentIndex(idx, flags)
            sm.blockSignals(False)
            if v is view:
                v.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        if self._current != gid:
            self._show_gid(gid)

    def _on_current(self, cur: QModelIndex, _prev) -> None:
        if not cur.isValid():
            return
        gid = self.model.gid_at(cur.row())
        other = self.grid if self.stack.currentWidget() is self.table else self.table
        sm = other.selectionModel()
        flags = QItemSelectionModel.ClearAndSelect
        if other is self.table:
            flags |= QItemSelectionModel.Rows
        sm.blockSignals(True)
        sm.setCurrentIndex(self.model.index(cur.row(), 0), flags)
        sm.blockSignals(False)
        if gid != self._current:
            self._show_gid(gid)

    def _show_gid(self, gid: int) -> None:
        self._current = gid
        self.btn_raw.setEnabled(True)
        self.preview.set_levels(1, 1, 1)
        self.preview.status.setText("decoding…")
        self._request_preview(0, 0, 0)

    def _request_preview(self, mip: int, face: int, slice_: int) -> None:
        if self._current is None:
            return
        gid = self._current
        self._preview_t0 = time.perf_counter()
        self.ctx.runner.submit("tex-preview", preview_job, self.cat, gid, mip, face, slice_,
                               on_done=lambda r, gid=gid: self._on_preview(gid, r),
                               on_error=lambda err, gid=gid: self._on_preview_error(gid, err))

    def _on_preview(self, gid: int, res) -> None:
        if gid != self._current:
            return
        img, info = res
        info["gid"] = gid
        self.last_preview_ms = (time.perf_counter() - self._preview_t0) * 1000.0
        h = info.get("header")
        if h is not None and not h.header_only and info.get("levels"):
            depth = max((lv.depth for lv in info["levels"] if lv.mip == info.get("mip", 0)), default=1)
            self.preview.set_levels(h.mip_count, h.faces, depth, info.get("mip", 0), info.get("face", 0),
                                    info.get("slice", 0))
        self.preview.show_image(img, info)
        self._last_info = info
        self._render_info(info)

    def _on_preview_error(self, gid: int, err: str) -> None:
        if gid != self._current:
            return
        self.preview.show_message(err.splitlines()[0])
        self._last_info = {"gid": gid, "error": err.splitlines()[0], "parts": [], "levels": []}
        self._render_info(self._last_info)

    def _refresh_info(self) -> None:
        if self._last_info is not None and self._last_info.get("gid") == self._current:
            self._render_info(self._last_info)

    def sdb_changed(self) -> None:
        self._refresh_info()

    # ---- info panel ---------------------------------------------------------------------------------------------
    def _render_info(self, info: dict) -> None:
        gid = info["gid"]
        e, i = self.cat.split(gid)
        name = self.cat.name(gid)
        esc = html.escape
        out = [f"<h3 style='margin:2px'>{esc(name)}</h3>",
               f"<p>pack <b>{esc(e.label)}</b> &nbsp; logical #{i} &nbsp; gid {gid} &nbsp; "
               f"<a href='raw:{gid}'>Show in Raw</a></p>"]
        if info.get("error"):
            out.append(f"<p style='color:#e06060'><b>{esc(info['error'])}</b></p>")

        def table(rows, head=None):
            s = ["<table cellspacing=0 cellpadding=2 border=0>"]
            if head:
                s.append("<tr>" + "".join(f"<th align=left>{esc(h)}</th>" for h in head) + "</tr>")
            for r in rows:
                s.append("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
            s.append("</table>")
            return "".join(s)

        h = info.get("header")
        j = info.get("imgc")
        if h is not None and j is not None:
            f = info.get("format") or {}
            st = j["stats"]
            fmt = f"{h.format} {esc(h.format_name)}"
            if f.get("dxgi") is not None:
                fmt += f" &nbsp; DXGI {f['dxgi']} {esc(f.get('dxgi_name') or '')}"
            fmt += f" &nbsp; {esc(str(f.get('kind')))} tier {esc(str(f.get('tier')))}"
            if f.get("srgb"):
                fmt += " sRGB"
            if not f.get("decodable"):
                fmt += " <span style='color:#e0a040'>(no preview decoder)</span>"
            rows = [
                ("magic", esc(h.magic.decode("latin-1"))), ("version", j["version"]),
                ("header_size", j["header_size"]), ("flags", j["flags"] + (" (0x02 header-only)" if h.header_only
                                                                         else "")),
                ("width × height × depth", f"{h.width} × {h.height} × {h.depth}"),
                ("reserved", j["reserved"]), ("format", fmt),
                ("type", f"{h.tex_type} {esc(h.type_name)}"), ("mip_count", h.mip_count),
                ("packed", f"0x{h.packed:02X}"), ("mip_split", j["mip_split"]),
                ("minimum", esc(_vec(st["minimum"]))), ("maximum", esc(_vec(st["maximum"]))),
                ("mean", esc(_vec(st["mean"]))),
                ("extension", esc(j["extension_hex"]) or "—"), ("tail", esc(j["tail_hex"]) or "—"),
            ]
            if h.header_only:
                ref = h.reference or ""
                links = ", ".join(f"<a href='tex:{g}'>{esc(self.cat.entry(g).label)}</a>"
                                  for g in info.get("reference_providers", [])) or "not loaded"
                rows.append(("reference", f"{esc(ref)} → {links}"))
            out.append("<h4>IMGC header</h4>" + table(rows))
        if info.get("levels"):
            out.append(f"<h4>Bitmap: {info['payload_size']:,} bytes, {len(info['levels'])} levels, level padding "
                       f"{info['level_padding']}{'' if info['level_padding'] == 16 else ' (third-party tight)'}</h4>")
            out.append(table(((lv.mip, lv.face, f"{lv.width}×{lv.height}×{lv.depth}", f"{lv.offset:,}",
                               f"{lv.size:,}", lv.padded_size - lv.size) for lv in info["levels"]),
                             ("mip", "face", "size", "offset", "bytes", "pad")))
        if info.get("parts"):
            out.append("<h4>Parts</h4>")
            out.append(table(((p["index"], f"0x{p['type']:02X}", f"0x{p['offset']:X}", f"{p['size']:,}",
                               p["method"], "yes" if p["direct"] else "no" + (" (child pack)" if p["child"] else ""))
                              for p in info["parts"]), ("phys", "type", "offset", "size", "method", "direct")))
        provs = [g for g in info.get("providers", []) if g != gid]
        out.append(f"<h4>Other providers of this name ({len(provs)})</h4>")
        out.append("<br>".join(f"<a href='tex:{g}'>{esc(self.cat.entry(g).label)}</a> #{g - self.cat.entry(g).base}"
                               f" &nbsp; {esc(self.cat.name(g))}" for g in provs) or "none")
        out.append("<h4>Used by materials (SDB)</h4>")
        sdb = self.ctx.sdb
        if not sdb.available():
            out.append("no SDB found")
        elif sdb.error:
            out.append(f"<span style='color:#e06060'>{esc(sdb.error)}</span>")
        else:
            mats = sdb.materials_using(name)
            if mats is None:
                out.append("indexing… (the list appears when the SDB reverse index is ready)")
            elif not mats:
                out.append("none")
            else:
                self._mat_links = list(mats)
                out.append(f"{len(mats):,} material(s):<br>" + "<br>".join(
                    f"<a href='mat:{k}'>{esc(m)}</a>" for k, m in enumerate(mats)))
        self.info.setHtml("".join(out))

    def _on_link(self, url) -> None:
        s = url.toString()
        kind, _, val = s.partition(":")
        if kind == "raw":
            self.ctx.openRaw.emit(int(val))
        elif kind == "tex":
            self.open_gid(int(val))
        elif kind == "mat":
            k = int(val)
            if 0 <= k < len(self._mat_links):
                self.ctx.openMaterial.emit(self._mat_links[k])

    # ---- navigation ---------------------------------------------------------------------------------------------
    def open_gid(self, gid: int) -> None:
        """Make *gid* visible (clearing filters when needed), select it, scroll to it and preview it."""
        gid = int(gid)
        if not 0 <= gid < len(self.cat) or self.cat.type(gid) != TEXTURE_TYPE:
            self.ctx.status.emit(f"Textures: gid {gid} is not a texture")
            return
        row = self.model.row_of(gid)
        if row >= 0:
            self._select_row(row, force=True)
            return
        self._pending_select = gid
        for w in (self.search.edit, self.search.pack_combo):
            w.blockSignals(True)
        self.search.edit.clear()
        self.search.pack_combo.setCurrentIndex(0)
        for w in (self.search.edit, self.search.pack_combo):
            w.blockSignals(False)
        self._run_search()

    # ---- grid ---------------------------------------------------------------------------------------------------
    def _set_grid(self, on: bool) -> None:
        self.stack.setCurrentWidget(self.grid if on else self.table)
        view = self.grid if on else self.table
        if self._current is not None:
            row = self.model.row_of(self._current)
            if row >= 0:
                view.scrollTo(self.model.index(row, 0), QAbstractItemView.PositionAtCenter)

    def thumb(self, gid: int, row: int) -> QPixmap | str:
        """Cached thumbnail, or 'loading' / 'error' (queues a decode for visible items)."""
        v = self._thumbs.get(gid)
        if v is not None:
            self._thumbs.move_to_end(gid)
            return v if isinstance(v, QPixmap) else "error"
        if gid not in self._thumb_inflight:
            self._thumb_wanted[gid] = row
            if not self._thumb_timer.isActive():
                self._thumb_timer.start()
        return "loading"

    def _pump_thumbs(self) -> None:
        limit = max(2, self.ctx.runner.pool.maxThreadCount())
        vp = self.grid.viewport().rect()
        n = len(self.model.gids)
        for gid, row in list(self._thumb_wanted.items()):
            if len(self._thumb_inflight) >= limit:
                break
            del self._thumb_wanted[gid]
            if row >= n or self.model.gid_at(row) != gid:
                continue
            if not self.grid.visualRect(self.model.index(row, 0)).intersects(vp):
                continue
            self._thumb_inflight.add(gid)
            self.ctx.runner.submit(("thumb", gid), thumb_job, self.cat, gid,
                                   on_done=lambda r, g=gid: self._on_thumb(g, r),
                                   on_error=lambda _e, g=gid: self._on_thumb(g, "error"))

    def _on_thumb(self, gid: int, res) -> None:
        self._thumb_inflight.discard(gid)
        self._thumbs[gid] = QPixmap.fromImage(res) if isinstance(res, QImage) else "error"
        while len(self._thumbs) > THUMB_CACHE:
            self._thumbs.popitem(last=False)
        self.grid.viewport().update()
        if self._thumb_wanted:
            self._thumb_timer.start()

    # ---- export -------------------------------------------------------------------------------------------------
    def _export(self, mode: str, out_dir: str | None = None) -> None:
        gids = sorted(self.model.checked)
        if not gids:
            QMessageBox.information(self, TITLE, "Check at least one texture first.")
            return
        if self._export_cancel is not None:
            return
        if out_dir is None:
            out_dir = QFileDialog.getExistingDirectory(self, f"Export {len(gids):,} texture(s) as {mode.upper()}",
                                                       self.ctx.export_dir())
            if not out_dir:
                return
            self.ctx.set_export_dir(out_dir)
        self._export_cancel = threading.Event()
        self.btn_export.setEnabled(False)
        self.result_label.hide()
        self.prog.setRange(0, len(gids))
        self.prog.setValue(0)
        self.prog_label.setText("")
        self.prog_box.show()
        self._export_t0 = time.perf_counter()
        self.ctx.runner.submit(("textures", "export"), export_job, self.cat, gids, mode, Path(out_dir),
                               self._export_cancel, self._progress,
                               on_done=self._on_export_done, on_error=self._on_export_error)

    def _on_export_progress(self, done: int, total: int, name: str) -> None:
        self.prog.setValue(done)
        self.prog_label.setText(f"{done:,} / {total:,}  {name}")

    def _cancel_export(self) -> None:
        if self._export_cancel is not None:
            self._export_cancel.set()
            self.prog_label.setText("cancelling…")

    def _export_finished(self) -> None:
        self._export_cancel = None
        self.btn_export.setEnabled(True)
        self.prog_box.hide()

    def _on_export_done(self, s: dict) -> None:
        self._export_finished()
        self.last_export = s
        secs = time.perf_counter() - self._export_t0
        msg = (f"{'Cancelled' if s['cancelled'] else 'Done'}: {s['written']:,} / {s['total']:,} texture(s) → "
               f"{s['out']} ({human_size(s['bytes'])}, {secs:.1f} s)")
        if s["errors"]:
            msg += f"; {len(s['errors'])} failed:\n" + "\n".join(s["errors"][:20])
            if len(s["errors"]) > 20:
                msg += f"\n… {len(s['errors']) - 20} more"
        self.result_label.setText(msg)
        self.result_label.show()
        self.ctx.status.emit(msg.splitlines()[0])

    def _on_export_error(self, err: str) -> None:
        self._export_finished()
        self.last_export = {"error": err}
        self.result_label.setText("Export failed: " + err.splitlines()[0])
        self.result_label.show()

    def shutdown(self) -> None:
        if self._export_cancel is not None:
            self._export_cancel.set()


def _vec(v) -> str:
    return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"


__all__ = ["Tab", "TITLE", "search_textures", "export_job", "preview_job", "thumb_job", "pack_dir"]
