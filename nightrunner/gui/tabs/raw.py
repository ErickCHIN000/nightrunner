"""Raw tab: every pack of the install as one lazy tree; check or select anything and export its raw bytes.

No viewers here — the tab is an extractor. Packs appear as the catalog indexes them; groups fetch their rows in
chunks while the user scrolls (never capped); the search runs on the worker pool and shows a filtered tree with
every match. Exports stream on a background thread with an in-tab progress bar and a cancel button.

Export layout: `<out>/<pack label without .rpack>/<family>/<index>_<name>/<part file>` and
`<out>/<pack>/_tables/{header,storage_table,name_table}.bin`. Existing files are never overwritten: they are
skipped and counted (so an interrupted export can simply be re-run). Child-pack / compressed parts are skipped and
counted too (their bytes are not the payload).
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from PySide6.QtCore import QObject, QPoint, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QLineEdit, QMenu, QMessageBox, QProgressBar, QPushButton, QTreeView, QVBoxLayout,
                               QWidget)

from ...container import catalogue
from ..context import AppContext
from ..tasks import Debouncer
from ..widgets import human_size, write_span
from .raw_model import (COL_INDEX, COL_NAME, COL_OFFSET, COL_SIZE, COL_TYPE, K_NODE, TABLE_FILES, TABLES,
                        ExportFile, PackChecks, RawTreeModel, all_spec_for, compute_view, export_path,
                        indexed_entries, iter_export, parse_query, spec_totals)

TITLE = "Raw"
FETCH_MARGIN = 600          # fetch the next chunk when the loaded end of a group is this many rows away


# ---- export worker ------------------------------------------------------------------------------------------------

class _ExportBridge(QObject):
    progress = Signal(object)           # dict
    finished = Signal(object)           # stats dict


def run_export(spec, out: Path, cancel: threading.Event, report=None, files: list[ExportFile] | None = None,
               overwrite: bool = False) -> dict:
    """Write *spec* ([(PackInfo, label, PackChecks)]) or explicit *files* under *out*. Thread-safe, no Qt."""
    out = Path(out)
    total = sum(f.size for f in files) if files is not None else spec_totals(spec)[1]
    st = {"out": str(out), "written": 0, "bytes": 0, "existing": 0, "unsupported": 0, "errors": 0,
          "messages": [], "cancelled": False, "total": total, "done": 0}
    last = 0.0

    def items():
        if files is not None:
            yield from files
            return
        for info, label, ch in spec:
            yield from iter_export(info, label, ch)

    for f in items():
        if cancel.is_set():
            st["cancelled"] = True
            break
        dest = export_path(out, f.rel)
        try:
            if not f.direct:
                st["unsupported"] += 1
            elif dest.exists() and not overwrite:
                st["existing"] += 1
            else:
                st["bytes"] += write_span(f.pack, f.offset, f.size, dest)
                st["written"] += 1
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the export
            st["errors"] += 1
            if len(st["messages"]) < 20:
                st["messages"].append(f"{f.rel}: {type(exc).__name__}: {exc}")
        st["done"] += f.size
        now = time.monotonic()
        if report is not None and now - last > 0.1:
            last = now
            report(dict(st, current=f.rel))
    return st


def _all_spec_job(cat, infos: dict):
    """Export All: every indexed pack in full (infos built where missing) + totals."""
    _, entries = indexed_entries(cat)
    spec = all_spec_for(cat, [e for e in entries if e.pack is not None], infos)
    parts, size = spec_totals(spec)
    return spec, parts, size, infos


def _filter_job(cat, text: str, type_id, infos: dict):
    _, entries = indexed_entries(cat)
    ids = [e.id for e in entries]
    views = compute_view(cat, text, type_id, ids, infos)
    return views, set(ids), infos


# ---- the tab --------------------------------------------------------------------------------------------------------

class Tab(QWidget):
    TITLE = TITLE

    def __init__(self, ctx: AppContext, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.cat = ctx.catalog
        self.model = RawTreeModel(self.cat, self)
        self._gen = 0                    # filter generation; pack jobs of an older generation are dropped
        self._filter_busy = False
        self._reveal_gid: int | None = None
        self._known_types: set[int] = set()
        self._fetch_scheduled = False
        self._export_thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self._last_export: dict | None = None
        self._jobno = 0

        # toolbar
        self.search = QLineEdit()
        self.search.setPlaceholderText("search names…   (words must all match;  #123 = logical index)")
        self.search.setClearButtonEnabled(True)
        self.type_combo = QComboBox()
        self.type_combo.addItem("all types", None)
        self.type_combo.setMinimumContentsLength(16)
        self.type_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.btn_checked = QPushButton("Export Checked…")
        self.btn_selected = QPushButton("Export Selected…")
        self.btn_all = QPushButton("Export All…")
        self.btn_none = QPushButton("Check none")
        self.counter = QLabel()
        self.counter.setMinimumWidth(170)
        top = QHBoxLayout()
        top.addWidget(self.search, 1)
        top.addWidget(self.type_combo)
        top.addWidget(self.btn_none)
        top.addWidget(self.counter)
        top.addWidget(self.btn_checked)
        top.addWidget(self.btn_selected)
        top.addWidget(self.btn_all)

        # tree
        self.view = QTreeView()
        self.view.setModel(self.model)
        self.view.setUniformRowHeights(True)
        self.view.setAlternatingRowColors(True)
        self.view.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.view.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.view.setContextMenuPolicy(Qt.CustomContextMenu)
        self.view.setAnimated(False)
        hdr = self.view.header()
        hdr.setStretchLastSection(False)
        hdr.setSectionResizeMode(COL_NAME, QHeaderView.Stretch)
        for c, w in ((COL_TYPE, 90), (COL_INDEX, 80), (COL_OFFSET, 110), (COL_SIZE, 95)):
            hdr.setSectionResizeMode(c, QHeaderView.Interactive)
            hdr.resizeSection(c, w)

        # status + export bar
        self.info = QLabel()
        self.prog = QProgressBar()
        self.prog.setRange(0, 1000)
        self.prog.setTextVisible(True)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_open = QPushButton("Open Folder")
        self.export_label = QLabel()
        self.export_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        bottom = QHBoxLayout()
        bottom.addWidget(self.info)
        bottom.addStretch(1)
        bottom.addWidget(self.export_label)
        bottom.addWidget(self.prog)
        bottom.addWidget(self.btn_cancel)
        bottom.addWidget(self.btn_open)
        self.prog.setMaximumWidth(260)
        self.prog.hide()
        self.btn_cancel.hide()
        self.btn_open.hide()

        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.addLayout(top)
        lay.addWidget(self.view, 1)
        lay.addLayout(bottom)

        self._bridge = _ExportBridge(self)
        self._bridge.progress.connect(self._on_export_progress)
        self._bridge.finished.connect(self._on_export_finished)
        self._debounce = Debouncer(self._run_filter, 220, self)
        self.search.textChanged.connect(self._debounce.trigger)
        self.search.returnPressed.connect(self._debounce.flush)
        self.type_combo.currentIndexChanged.connect(self._run_filter)
        self.btn_none.clicked.connect(self.model.clear_checks)
        self.btn_checked.clicked.connect(self.export_checked_dialog)
        self.btn_selected.clicked.connect(self.export_selected_dialog)
        self.btn_all.clicked.connect(self.export_all_dialog)
        self.btn_cancel.clicked.connect(self.cancel_export)
        self.btn_open.clicked.connect(self._open_last_folder)
        self.model.checksChanged.connect(self._on_checks)
        self.view.customContextMenuRequested.connect(self._menu)
        self.view.expanded.connect(self._schedule_fetch)
        self.view.verticalScrollBar().valueChanged.connect(self._schedule_fetch)
        self.model.rowsInserted.connect(self._schedule_fetch)

        self.cat.packAdded.connect(self._on_pack_added)
        for e in list(self.cat.packs):
            if e.pack is not None or e.error:
                self._on_pack_added(e.id)
        self._on_checks()
        self._update_info()

    def _submit(self, fn, *args, on_done, on_error=None) -> None:
        """Run *fn* on the pool under a fresh key. Staleness is handled here with `_gen`, not by key reuse:
        re-submitting a key whose finished job was auto-deleted makes `TaskRunner.submit` raise (tryTake on a
        deleted QRunnable)."""
        self._jobno += 1
        self.ctx.runner.submit(("raw", self._jobno), fn, *args, on_done=on_done,
                               on_error=on_error or self._job_failed)

    # ---- query --------------------------------------------------------------------------------------------------
    def query(self) -> tuple[str, int | None]:
        return self.search.text().strip(), self.type_combo.currentData()

    def filtering(self) -> bool:
        text, t = self.query()
        return bool(text) or t is not None

    def _run_filter(self, *_) -> None:
        text, t = self.query()
        self._gen += 1
        gen = self._gen
        self._filter_busy = True
        self.info.setText("searching…")
        self._submit(_filter_job, self.cat, text, t, dict(self.model.infos),
                               on_done=lambda res: self._filter_done(gen, text, t, res),
                               on_error=self._job_failed)

    def _filter_done(self, gen: int, text: str, t, res) -> None:
        if gen != self._gen:
            return
        views, covered, infos = res
        self._filter_busy = False
        for k, v in infos.items():
            self.model.infos.setdefault(k, v)
        filtering = bool(text) or t is not None
        self.model.reset_view(views, filtering)
        if filtering and len(views) <= 5:
            for pid in self.model.visible_pack_ids():
                pi = self.model.pack_index(pid)
                self.view.expand(pi)
                groups = self.model.group_nodes(pid)
                if len(groups) <= 3:
                    for g in groups:
                        self.view.expand(self.model.node_index(g))
        for e in list(self.cat.packs):
            if (e.pack is not None or e.error) and e.id not in covered:
                self._submit_pack(e.id)
        self._update_info()
        self._try_reveal()

    def _job_failed(self, err: str) -> None:
        self._filter_busy = False
        self.info.setText("error: " + err.splitlines()[0])
        self.ctx.status.emit("Raw: " + err.splitlines()[0])

    def _on_pack_added(self, pid: int) -> None:
        e = self.cat.packs[pid]
        new = set(e.type_counts) - self._known_types
        if new:
            self._known_types |= new
            self._rebuild_type_combo()
        if not self._filter_busy:
            self._submit_pack(pid)

    def _submit_pack(self, pid: int) -> None:
        text, t = self.query()
        gen = self._gen
        info = self.model.infos.get(pid)
        infos = {pid: info} if info is not None else {}
        self._submit(compute_view, self.cat, text, t, [pid], infos,
                               on_done=lambda views: self._pack_done(gen, views, infos),
                               on_error=self._job_failed)

    def _pack_done(self, gen: int, views, infos: dict) -> None:
        for k, v in infos.items():
            self.model.infos.setdefault(k, v)
        if gen != self._gen:
            return
        for pv in views:
            self.model.add_pack_view(pv)
        self._update_info()
        self._try_reveal()

    def _rebuild_type_combo(self) -> None:
        cur = self.type_combo.currentData()
        self.type_combo.blockSignals(True)
        self.type_combo.clear()
        self.type_combo.addItem("all types", None)
        for t in sorted(self._known_types):
            self.type_combo.addItem(f"{catalogue.type_name(t)}  0x{t:02X}", t)
        i = self.type_combo.findData(cur)
        self.type_combo.setCurrentIndex(max(0, i))
        self.type_combo.blockSignals(False)

    def _update_info(self) -> None:
        m = self.model
        pids = m.visible_pack_ids()
        if m.filtering:
            n = sum(len(g.rows) for g in m.group_nodes())
            s = f"{n:,} matches in {len(pids)} packs"
        else:
            n = sum(self.cat.packs[p].count for p in pids)
            s = f"{len(pids)} packs · {n:,} resources"
        if self._filter_busy:
            s += "   (searching…)"
        elif len(pids) < sum(1 for e in self.cat.packs) and not m.filtering:
            s += "   (loading…)"
        self.info.setText(s)

    # ---- lazy fetching ----------------------------------------------------------------------------------------------
    def _schedule_fetch(self, *_) -> None:
        if not self._fetch_scheduled:
            self._fetch_scheduled = True
            QTimer.singleShot(0, self._fetch_visible)

    def _fetch_visible(self) -> None:
        """QTreeView only fetches for the very last row; groups in the middle of the tree are fetched here
        whenever their loaded end comes near the viewport."""
        self._fetch_scheduled = False
        vp = self.view.viewport()
        first = self.view.indexAt(QPoint(4, 2))
        if not first.isValid():
            return
        rh = max(8, self.view.visualRect(first).height())
        x = self.view.header().sectionViewportPosition(COL_SIZE) + 4
        done = set()
        for y in range(2, vp.height() + rh, rh):
            idx = self.view.indexAt(QPoint(x, y))
            if not idx.isValid():
                continue
            kind, _, row, _ = self.model.decode(idx)
            if kind == K_NODE:
                continue
            g = self.model.item(idx)["group"]
            if g.idx in done:
                continue
            done.add(g.idx)
            if g.fetched < len(g.rows) and g.fetched - row < FETCH_MARGIN:
                self.model.fetchMore(self.model.node_index(g))

    # ---- navigation -------------------------------------------------------------------------------------------------
    def open_gid(self, gid: int) -> None:
        """Reveal, select and scroll to a resource (clears the filter when it hides the resource)."""
        try:
            e, li = self.cat.split(gid)
        except (IndexError, ValueError):
            return
        self._reveal_gid = gid
        if self._try_reveal():
            return
        if self.filtering():
            self.search.blockSignals(True)
            self.type_combo.blockSignals(True)
            self.search.clear()
            self.type_combo.setCurrentIndex(0)
            self.search.blockSignals(False)
            self.type_combo.blockSignals(False)
            self._run_filter()
        # otherwise the pack is still loading: the pack job reveals it

    def _try_reveal(self) -> bool:
        gid = self._reveal_gid
        if gid is None:
            return False
        e, li = self.cat.split(gid)
        hit = self.model.resource_row(e.id, li)
        if hit is None:
            return False
        self._reveal_gid = None
        g, row = hit
        idx = self.model.resource_index(g, row)
        self.view.expand(self.model.pack_index(e.id))
        self.view.expand(self.model.node_index(g))
        self.view.setCurrentIndex(idx)
        self.view.scrollTo(idx, QAbstractItemView.PositionAtCenter)
        self._schedule_fetch()
        return True

    # ---- checks -----------------------------------------------------------------------------------------------------
    def _on_checks(self) -> None:
        parts, size = self.model.checked_totals()
        self.counter.setText(f"✓ {parts:,} parts · {human_size(size)}" if parts else "nothing checked")
        self.btn_checked.setEnabled(parts > 0)
        self.view.viewport().update()

    # ---- context menu -----------------------------------------------------------------------------------------------
    def _menu(self, pos: QPoint) -> None:
        idx = self.view.indexAt(pos)
        if not idx.isValid():
            return
        idx = idx.siblingAtColumn(0)
        m = self.model
        it = m.item(idx)
        menu = QMenu(self)
        clip = QGuiApplication.clipboard()
        name = m.data(idx.siblingAtColumn(COL_NAME))
        if it["kind"] in ("resource", "part"):
            pk = m.infos[it["pid"]].pack
            raw = self.cat.name(m.gid_of(idx))
            menu.addAction("Copy Name", lambda: clip.setText(raw))
        else:
            menu.addAction("Copy Name", lambda: clip.setText(str(name)))
        off = m.data(idx.siblingAtColumn(COL_OFFSET))
        if off:
            menu.addAction("Copy Offset", lambda: clip.setText(str(off)))
        gid = m.gid_of(idx)
        if gid is not None:
            menu.addAction("Copy gid", lambda: clip.setText(str(gid)))
            t = pk.logicals[it["li"]].type
            if t == 0x20:
                menu.addAction("Show in Textures", lambda: self.ctx.openTexture.emit(gid))
            elif t == 0x10:
                menu.addAction("Show in Meshes", lambda: self.ctx.openMesh.emit(gid))
        menu.addSeparator()
        state = m.check_state(idx)
        on = state != Qt.Checked
        menu.addAction("Check" if on else "Uncheck", lambda: m.set_checked(idx, on))
        menu.addAction("Export This…", lambda: self.export_rows_dialog([idx]))
        menu.addAction("Export Selected…", self.export_selected_dialog)
        menu.addSeparator()
        if m.hasChildren(idx):
            if self.view.isExpanded(idx):
                menu.addAction("Collapse", lambda: self.view.collapse(idx))
            else:
                menu.addAction("Expand", lambda: self.view.expand(idx))
            menu.addAction("Expand Children", lambda: self._expand_children(idx))
        menu.addAction("Collapse All", self.view.collapseAll)
        menu.exec(self.view.viewport().mapToGlobal(pos))

    def _expand_children(self, idx) -> None:
        """Expand a node and its structural children (packs → groups); never resources en masse."""
        self.view.expand(idx)
        if self.model.node_of(idx) is not None and self.model.node_of(idx).kind == "pack":
            for r in range(self.model.rowCount(idx)):
                self.view.expand(self.model.index(r, 0, idx))

    # ---- export -----------------------------------------------------------------------------------------------------
    def _ask_dir(self, title: str) -> Path | None:
        d = QFileDialog.getExistingDirectory(self, title, self.ctx.export_dir())
        if not d:
            return None
        self.ctx.set_export_dir(d)
        return Path(d)

    def exporting(self) -> bool:
        return self._export_thread is not None and self._export_thread.is_alive()

    def _busy_msg(self) -> bool:
        if self.exporting():
            QMessageBox.information(self, TITLE, "An export is already running.")
            return True
        return False

    def export_checked_dialog(self) -> None:
        if self._busy_msg():
            return
        spec = self.model.checked_spec()
        if not spec:
            return
        out = self._ask_dir("Export checked items to…")
        if out is not None:
            self.start_export(spec, out)

    def export_selected_dialog(self) -> None:
        self.export_rows_dialog([i for i in self.view.selectionModel().selectedRows(0)])

    def export_rows_dialog(self, rows) -> None:
        """Export exactly *rows*. The context menu passes the row under the cursor, so a right-click exports what
        was clicked rather than whatever happened to be selected."""
        if self._busy_msg():
            return
        if not rows:
            return
        single = self._single_file(rows)
        if single is not None:
            f, suggested = single
            start = str(Path(self.ctx.export_dir() or ".") / suggested)
            dest, _ = QFileDialog.getSaveFileName(self, "Save part as", start)
            if dest:
                dest = Path(dest)
                self.ctx.set_export_dir(str(dest.parent))
                f = ExportFile(f.pack, f.offset, f.size, dest.name, f.direct)
                self.start_export(None, dest.parent, files=[f], overwrite=True)   # the dialog asked already
            return
        spec = self.model.selection_spec(rows)
        if not spec:
            return
        out = self._ask_dir("Export selected items to…")
        if out is not None:
            self.start_export(spec, out)

    def _single_file(self, rows) -> tuple[ExportFile, str] | None:
        if len(rows) != 1:
            return None
        it = self.model.item(rows[0])
        if it["kind"] == "part":
            info = self.model.infos[it["pid"]]
            spec = [(info, self.cat.packs[it["pid"]].label, self.model.selection_spec(rows)[0][2])]
            f = next(iter_export(*spec[0]))
            return f, f"{Path(f.rel).parent.name}_{Path(f.rel).name}"
        if it["kind"] == "table":
            info = self.model.infos[it["pid"]]
            off, size = info.tables[it["which"]]
            stem = Path(self.cat.packs[it["pid"]].label).stem
            return ExportFile(info.pack, off, size, TABLE_FILES[it["which"]]), f"{stem}_{TABLE_FILES[it['which']]}"
        return None

    def export_all_dialog(self) -> None:
        if self._busy_msg():
            return
        self.info.setText("sizing the install…")
        self._submit(_all_spec_job, self.cat, dict(self.model.infos),
                               on_done=self._confirm_all, on_error=self._job_failed)

    def _confirm_all(self, res) -> None:
        spec, parts, size, infos = res
        for k, v in infos.items():
            self.model.infos.setdefault(k, v)
        self._update_info()
        if not spec:
            return
        pending = sum(1 for e in self.cat.packs if e.pack is None and not e.error)
        msg = (f"Export every part of {len(spec)} packs?\n\n{parts:,} files, {human_size(size)} total"
               f"{f'  ({pending} packs still indexing are not included)' if pending else ''}.\n\n"
               "Existing files are skipped.")
        if QMessageBox.question(self, TITLE, msg) != QMessageBox.Yes:
            return
        out = self._ask_dir("Export the whole install to…")
        if out is not None:
            self.start_export(spec, out)

    def start_export(self, spec, out: Path, files: list[ExportFile] | None = None, overwrite: bool = False) -> None:
        """Run an export on a background thread (progress + cancel in the tab)."""
        if self.exporting():
            return
        self._cancel = threading.Event()
        self.prog.setValue(0)
        self.prog.setFormat("exporting… %p%")
        self.prog.show()
        self.btn_cancel.show()
        self.btn_cancel.setEnabled(True)
        self.btn_open.hide()
        self.export_label.setText("")
        for b in (self.btn_checked, self.btn_selected, self.btn_all):
            b.setEnabled(False)
        cancel = self._cancel
        bridge = self._bridge

        def work():
            try:
                st = run_export(spec, out, cancel, bridge.progress.emit, files=files, overwrite=overwrite)
            except Exception as exc:  # noqa: BLE001
                st = {"out": str(out), "fatal": f"{type(exc).__name__}: {exc}"}
            bridge.finished.emit(st)

        self._export_thread = threading.Thread(target=work, name="raw-export", daemon=True)
        self._export_thread.start()

    def cancel_export(self) -> None:
        self._cancel.set()
        self.btn_cancel.setEnabled(False)

    def wait_export(self, timeout: float = 60.0) -> bool:
        """Tests / shutdown: join the export thread (callbacks still need the event loop)."""
        t = self._export_thread
        if t is not None:
            t.join(timeout)
            return not t.is_alive()
        return True

    def _on_export_progress(self, st: dict) -> None:
        total = st.get("total") or 1
        self.prog.setValue(int(1000 * st["done"] / total))
        self.export_label.setText(f"{st['written']:,} files · {human_size(st['bytes'])}")
        self.export_label.setToolTip(st.get("current", ""))

    def _on_export_finished(self, st: dict) -> None:
        self._last_export = st
        self.prog.hide()
        self.btn_cancel.hide()
        self.btn_selected.setEnabled(True)
        self.btn_all.setEnabled(True)
        self._on_checks()
        if "fatal" in st:
            self.export_label.setText("export failed: " + st["fatal"])
            return
        s = f"{'cancelled — ' if st['cancelled'] else ''}{st['written']:,} files · {human_size(st['bytes'])}"
        extra = []
        if st["existing"]:
            extra.append(f"{st['existing']:,} existed (skipped)")
        if st["unsupported"]:
            extra.append(f"{st['unsupported']:,} child/compressed (skipped)")
        if st["errors"]:
            extra.append(f"{st['errors']:,} errors")
        if extra:
            s += " · " + ", ".join(extra)
        self.export_label.setText(s)
        self.export_label.setToolTip("\n".join([st["out"]] + st["messages"]))
        self.btn_open.show()
        self.ctx.status.emit(f"Raw export to {st['out']}: {s}")

    def _open_last_folder(self) -> None:
        if self._last_export:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self._last_export["out"]))

    def shutdown(self) -> None:
        self._cancel.set()
        self.wait_export(3.0)


# keep a reference to names used only by type-checkers / re-export for tests
__all__ = ["Tab", "TITLE", "run_export", "PackChecks", "TABLES", "parse_query"]
