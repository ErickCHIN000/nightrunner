"""Build tab: a project (nightrunner/project.py) edited in the GUI and built into one mod rpack + optional dataN.pak.

Layout
  toolbar   project menu · name · rpack / PAK names · output folder · Add · Validate · Build · Open folder
  pages     Resources (this module: drop list + inspector) · Models (tabs/build_models.py)
  bottom    Log · Problems · Settings · History

Both pages share one `BuildState` (gui/buildstate.py). Validation and builds run in `ctx.runner`; the worker
only sees a deep copy of the project, and progress lines reach the GUI through a Qt signal.
"""
from __future__ import annotations

import copy
import json
import re
import threading
import time
import traceback
from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, QObject, Qt, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMenu, QMessageBox, QPlainTextEdit,
                               QProgressBar, QPushButton, QScrollArea, QSplitter, QTableWidget, QTableWidgetItem, QTabWidget,
                               QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...project import (PAK_RANGE, Cancelled, Problem, ProjectError, add_files, build_project, scene_objects,
                        scene_targets, validate)
from ...util.jsonio import load_json
from ..buildstate import BuildState
from ..tasks import Debouncer
from ..widgets import WheelGuard, human_size, mono_font
from ...util.schema import startswith as schema_startswith

TITLE = "Build"

GROUPS = ("Textures", "Meshes", "Raw")
GROUP_OF = {"texture": "Textures", "mesh": "Meshes", "scene": "Meshes", "raw": "Raw"}
GREEN, AMBER, RED, GREY = QColor(90, 180, 110), QColor(220, 170, 60), QColor(230, 90, 90), QColor(140, 140, 140)
NEW = QColor(110, 170, 255)
PEND = "…"
_SUFFIX = re.compile(r"\.\d{3}$")
ROLE_ID = Qt.UserRole + 1
C_SRC, C_TARGET, C_PACK, C_OUT, C_STATUS, C_SIZE = range(6)
RECENT_MAX = 8
HISTORY_MAX = 50
THUMB = 150


def _cell(text, color=None, tip=None):
    it = QTableWidgetItem(text)
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if color is not None:
        it.setForeground(QBrush(color))
    if tip:
        it.setToolTip(tip)
    return it


def _as_list(v) -> list:
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    return [v]


def item_status(problems: list) -> tuple[str, QColor, str]:
    """(text, color, tooltip) for one item's problems."""
    errs = [p for p in problems if p.level == "error"]
    warns = [p for p in problems if p.level != "error"]
    tip = "\n".join(p.message for p in problems)
    if errs:
        return f"✗ {errs[0].message.split(':')[0]}", RED, tip
    if warns:
        return f"⚠ {len(warns)}", AMBER, tip
    return "✓", GREEN, ""


def default_new_name(item) -> str:
    p = Path(item.source)
    if item.kind == "texture":
        return p.name
    return p.stem if p.stem.lower() != "model" else f"{item.target_name}_new"


def scene_meshes(item) -> list[str]:
    """Logical mesh names of a scene item (from its export report)."""
    try:
        parts = load_json(item.options["report"]).get("parts", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for pr in parts:
        m = str(pr.get("mesh") or "")
        m = m[:-4] if m.lower().endswith(".msh") else m
        if m and m not in out:
            out.append(m)
    return out


def change_text(item) -> str:
    src = Path(item.source).name
    if item.kind == "scene":
        lines = [f"split {src}"]
        only = item.options.get("meshes") or []
        if only:
            lines.append("only " + ", ".join(only))
        for a, b in (item.options.get("renames") or {}).items():
            lines.append(f"new {b} from {a}")
        for a, b in (item.options.get("assign") or {}).items():
            lines.append(f"{a} → {b}" if b else f"skip {a}")
        for h in item.options.get("hide") or []:
            lines.append(f"hide {h}")
        if item.options.get("skip_rest"):
            lines.append("skip rest")
        for o in item.options.get("double_sided") or []:
            lines.append(f"2-sided {o}")
        if not item.options.get("report"):
            lines.append("no report")
        return "\n".join(lines)
    if item.new_name:
        lines = [f"new {item.new_name} from {item.target_name}"]
    else:
        lines = [f"replace {item.target_name} · {item.target_pack or 'any'}"]
    if item.kind == "texture":
        fmt = item.options.get("format", "auto")
        if fmt != "auto":
            lines.append(f"format {fmt}")
        if item.options.get("mips") == "none":
            lines.append("no mips")
    if item.kind == "raw":
        lines.append(f"part {item.options.get('part', '?')}")
    return "\n".join(lines)


def target_gid(ctx, item) -> int | None:
    cat = getattr(ctx, "catalog", None)
    if cat is None or not item.target_name:
        return None
    for gid in cat.lookup(item.target_name, item.type):
        e, _ = cat.split(gid)
        if e.user or (item.target_pack and e.label != item.target_pack):
            continue
        return gid
    return None


# ---- worker functions (no widgets) -------------------------------------------------------------------------------

def _thumb_source(path: str) -> QImage:
    p = Path(path)
    if p.suffix.lower() == ".dds":
        from ...texture.dds import dds_to_levels, read_dds
        from ..texpreview import _decode_surface, rgba_to_qimage
        buf = p.read_bytes()
        d = read_dds(buf, allow_unobserved=True)
        lv = dds_to_levels(d, buf)[0]
        step = max(1, max(d.width, d.height) // (2 * THUMB))
        img = rgba_to_qimage(_decode_surface(bytes(lv), d.width, d.height, d.il_format, step))
    else:
        img = QImage(str(p))
    if img.isNull():
        raise ValueError("unreadable")
    return img.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _thumb_target(pack, index: int) -> QImage:
    from ..texpreview import decode_texture
    img, info = decode_texture(pack, index, mip=None, max_dim=256)
    if img.isNull():
        raise ValueError(info.get("error") or "no preview")
    return img.scaled(THUMB, THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation)


def _build_job(project, env, say, cancel, out_dir) -> dict:
    try:
        return {"ok": True, "report": build_project(project, env, progress=say, cancel=cancel, out_dir=out_dir)}
    except Cancelled:
        return {"ok": False, "cancelled": True, "error": "cancelled"}
    except ProjectError as exc:
        return {"ok": False, "error": str(exc)}


class _Lines(QObject):
    line = Signal(str)


# ---- resource list ---------------------------------------------------------------------------------------------

class DropTree(QTreeWidget):
    """The resource list: accepts files/folders, rows grouped by kind, one row per project item."""

    COLS = ("Source file", "Target", "Pack", "Output name", "Status", "Size")
    filesDropped = Signal(list)          # [Path]
    deletePressed = Signal()
    toggled = Signal(str, bool)          # item id, enabled

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DropOnly)
        self.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.setAlternatingRowColors(True)
        self.setUniformRowHeights(True)
        self.setHeaderLabels(self.COLS)
        self.setContextMenuPolicy(Qt.CustomContextMenu)
        h = self.header()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        for c, w in ((1, 180), (2, 130), (3, 170), (4, 130), (5, 70)):
            h.setSectionResizeMode(c, QHeaderView.Interactive)
            self.setColumnWidth(c, w)
        self.groups: dict[str, QTreeWidgetItem] = {}
        for g in GROUPS:
            gi = QTreeWidgetItem([g])
            f = gi.font(0)
            f.setBold(True)
            gi.setFont(0, f)
            gi.setFlags(Qt.ItemIsEnabled)
            self.addTopLevelItem(gi)
            gi.setExpanded(True)
            self.groups[g] = gi
        self.rows: dict[str, QTreeWidgetItem] = {}
        self._filling = False
        self.hint = QLabel("Drop files here", self.viewport())
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.hint.setStyleSheet("color: palette(placeholder-text); border: 2px dashed palette(mid); "
                                "border-radius: 8px; font-size: 13px;")
        self.itemChanged.connect(self._on_changed)
        self._refresh()

    # ---- rows ----------------------------------------------------------------------------------------------
    def set_items(self, items, statuses: dict, sizes: dict) -> None:
        """Rebuild from project items (updated in place when the ids are unchanged)."""
        ids = [it.id for it in items]
        same = ids == list(self.rows)
        sel = self.selected_ids()
        cur = self.current_id()
        self._filling = True
        blocked = self.blockSignals(not same)       # a rebuild must not report selection churn
        try:
            if not same:
                for gi in self.groups.values():
                    gi.takeChildren()
                self.rows.clear()
            for it in items:
                row = self.rows.get(it.id)
                if row is None:
                    row = QTreeWidgetItem()
                    row.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable)
                    row.setData(0, ROLE_ID, it.id)
                    row.setFont(C_TARGET, mono_font())
                    row.setFont(C_OUT, mono_font())
                    self.groups[GROUP_OF.get(it.kind, "Raw")].addChild(row)
                    self.rows[it.id] = row
                self._fill(row, it, statuses.get(it.id), sizes.get(it.id))
            if not same:
                for iid in sel:
                    if iid in self.rows:
                        self.rows[iid].setSelected(True)
                if cur in self.rows:
                    self.setCurrentItem(self.rows[cur], 0, QItemSelectionModel.NoUpdate)
        finally:
            self.blockSignals(blocked)
            self._filling = False
        self._refresh()

    def _fill(self, row, it, status, size) -> None:
        src = Path(it.source)
        row.setText(C_SRC, src.name)
        row.setToolTip(C_SRC, str(src))
        row.setCheckState(0, Qt.Checked if it.enabled else Qt.Unchecked)
        if it.kind == "scene":
            n = len(it.options.get("renames") or {})
            row.setText(C_TARGET, it.target_name)
            row.setText(C_PACK, "")
            row.setText(C_OUT, f"split · {n} new" if n else "split")
        else:
            row.setText(C_TARGET, it.target_name or "—")
            row.setText(C_PACK, it.target_pack or "any")
            row.setText(C_OUT, it.output_name)
        if not it.enabled:
            text, color, tip = "off", GREY, ""
        elif status is None:
            text, color, tip = PEND, GREY, ""
        else:
            text, color, tip = status
        row.setText(C_STATUS, text)
        row.setToolTip(C_STATUS, tip)
        row.setForeground(C_STATUS, QBrush(color))
        row.setText(C_SIZE, human_size(size) if size is not None else "—")
        dim = QBrush(GREY) if not it.enabled else QBrush()
        for c in (C_SRC, C_TARGET, C_PACK, C_SIZE):
            row.setForeground(c, dim)
        new = bool(it.new_name) and it.kind != "scene"
        row.setForeground(C_OUT, QBrush(NEW) if new and it.enabled else dim)
        row.setToolTip(C_OUT, "new" if new else "")

    def _refresh(self):
        total = 0
        for g, gi in self.groups.items():
            n = gi.childCount()
            total += n
            gi.setText(0, f"{g}  ({n})")
            gi.setHidden(n == 0)
        self.hint.setVisible(total == 0)
        self._place_hint()

    def apply_filter(self, text: str, mode: str) -> int:
        text = text.lower().strip()
        shown = 0
        for row in self.rows.values():
            hay = " ".join(row.text(c) for c in range(len(self.COLS))).lower()
            ok = not text or text in hay
            st = row.text(C_STATUS)
            if mode == "Problems":
                ok = ok and (st.startswith("✗") or st.startswith("⚠"))
            elif mode == "Off":
                ok = ok and row.checkState(0) != Qt.Checked
            row.setHidden(not ok)
            shown += ok
        return shown

    def current_id(self) -> str | None:
        it = self.currentItem()
        return it.data(0, ROLE_ID) if it is not None and it.parent() is not None else None

    def selected_ids(self) -> list[str]:
        return [it.data(0, ROLE_ID) for it in self.selectedItems() if it.parent() is not None]

    def select_id(self, iid: str) -> None:
        row = self.rows.get(iid)
        if row is not None:
            row.setHidden(False)
            self.setCurrentItem(row)
            self.scrollToItem(row)

    def _on_changed(self, row, col):
        if self._filling or col != 0 or row.parent() is None:
            return
        self.toggled.emit(row.data(0, ROLE_ID), row.checkState(0) == Qt.Checked)

    # ---- events --------------------------------------------------------------------------------------------
    def _place_hint(self):
        self.hint.setGeometry(self.viewport().rect().adjusted(40, 40, -40, -40))

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_hint()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Delete:
            self.deletePressed.emit()
            return
        super().keyPressEvent(e)

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            self.setStyleSheet("QTreeWidget { border: 2px solid palette(highlight); }")
            e.acceptProposedAction()

    def dragMoveEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dragLeaveEvent(self, e):
        self.setStyleSheet("")

    def dropEvent(self, e):
        self.setStyleSheet("")
        paths = [Path(u.toLocalFile()) for u in e.mimeData().urls() if u.isLocalFile()]
        e.acceptProposedAction()
        if paths:
            self.filesDropped.emit(paths)


# ---- target picker ---------------------------------------------------------------------------------------------

class PickDialog(QDialog):
    LIMIT = 500

    def __init__(self, ctx, type_id: int, text: str = "", parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.type_id = type_id
        self.choice: tuple[str, str] | None = None
        self.setWindowTitle("Pick")
        self.resize(560, 460)
        lay = QVBoxLayout(self)
        self.search = QLineEdit(text)
        self.search.setPlaceholderText("search…")
        self.search.setClearButtonEnabled(True)
        lay.addWidget(self.search)
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Pack"])
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(self.accept)
        lay.addWidget(self.table, 1)
        self.count = QLabel()
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        row = QHBoxLayout()
        row.addWidget(self.count, 1)
        row.addWidget(bb)
        lay.addLayout(row)
        self._deb = Debouncer(self._run, 250, self)
        self.search.textChanged.connect(self._deb.trigger)
        self._run()

    def _run(self):
        cat = self.ctx.catalog
        text = self.search.text()
        type_id, limit = self.type_id, self.LIMIT

        def job():
            gids = cat.search(text, types=[type_id])
            rows = []
            for g in gids[:limit].tolist():
                e, _ = cat.split(g)
                if not e.user:
                    rows.append((cat.name(g), e.label))
            return rows, len(gids)
        self.ctx.runner.submit("build.pick", job, on_done=self._show)

    def _show(self, res):
        rows, n = res
        self.table.setRowCount(len(rows))
        for r, (name, label) in enumerate(rows):
            self.table.setItem(r, 0, _cell(name))
            self.table.setItem(r, 1, _cell(label))
        self.count.setText(f"{n}" if n <= self.LIMIT else f"{self.LIMIT} / {n}")
        if rows:
            self.table.selectRow(0)

    def accept(self):
        r = self.table.currentRow()
        if r >= 0 and self.table.item(r, 0) is not None:
            self.choice = (self.table.item(r, 0).text(), self.table.item(r, 1).text())
        super().accept()


# ---- inspector -------------------------------------------------------------------------------------------------

class Inspector(QWidget):
    """Right side: edits the selected item in place; emits `edited` after every change."""

    edited = Signal()

    def __init__(self, ctx, state: BuildState, parent=None):
        super().__init__(parent)
        self.ctx, self.state = ctx, state
        self.item = None
        self._loading = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 0, 0, 0)
        self.title = QLabel()
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.sub = QLabel()
        self.sub.setWordWrap(True)
        lay.addWidget(self.title)
        lay.addWidget(self.sub)

        # target
        self.tgt_box = QGroupBox("Target")
        f = self.tgt_form = QFormLayout(self.tgt_box)
        row = QHBoxLayout()
        self.target = QLineEdit()
        self.target.setFont(mono_font())
        self.target.textEdited.connect(self._on_target)
        self.target.editingFinished.connect(self._fill_packs)
        self.pick_btn = QPushButton("Pick…")
        self.pick_btn.clicked.connect(self._pick)
        row.addWidget(self.target, 1)
        row.addWidget(self.pick_btn)
        f.addRow("Resource", row)
        self.pack = QComboBox()
        self.pack.currentIndexChanged.connect(self._on_pack)
        f.addRow("Pack", self.pack)
        nrow = QHBoxLayout()
        self.name_mode = QComboBox()
        self.name_mode.addItems(["Same", "New"])
        self.name_mode.currentIndexChanged.connect(self._on_name_mode)
        self.new_name = QLineEdit()
        self.new_name.setFont(mono_font())
        self.new_name.textEdited.connect(self._on_new_name)
        nrow.addWidget(self.name_mode)
        nrow.addWidget(self.new_name, 1)
        self.name_row = QWidget()
        self.name_row.setLayout(nrow)
        nrow.setContentsMargins(0, 0, 0, 0)
        f.addRow("Name", self.name_row)
        self.match = QLabel()
        f.addRow("Match", self.match)
        lay.addWidget(self.tgt_box)

        # texture
        self.tex_box = QGroupBox("Texture")
        tf = QFormLayout(self.tex_box)
        self.fmt = QComboBox()
        for label, key in (("Auto", "auto"), ("RGBA8", "rgba8"), ("R8", "r8"), ("Normal", "normal")):
            self.fmt.addItem(label, key)
        self.fmt.currentIndexChanged.connect(self._on_tex)
        self.mips = QComboBox()
        self.mips.addItem("Full", "")
        self.mips.addItem("None", "none")
        self.mips.currentIndexChanged.connect(self._on_tex)
        tf.addRow("Format", self.fmt)
        tf.addRow("Mips", self.mips)
        lay.addWidget(self.tex_box)

        # mesh / scene
        self.mesh_box = QGroupBox("Mesh")
        mf = QVBoxLayout(self.mesh_box)
        trow = QHBoxLayout()
        trow.addWidget(QLabel("Tolerance"))
        self.tol = QDoubleSpinBox()
        self.tol.setDecimals(6)
        self.tol.setRange(0.0, 1.0)
        self.tol.setSingleStep(1e-4)
        self.tol.setValue(1e-4)
        self.tol.valueChanged.connect(self._on_tol)
        trow.addWidget(self.tol)
        trow.addStretch(1)
        mf.addLayout(trow)
        self.kind = QComboBox()
        self.kind.addItem("Mesh", "mesh")
        self.kind.addItem("Scene", "scene")
        self.kind.currentIndexChanged.connect(self._on_kind)
        trow.insertWidget(0, self.kind)
        self.report_row = QWidget()
        rr = QHBoxLayout(self.report_row)
        rr.setContentsMargins(0, 0, 0, 0)
        rr.addWidget(QLabel("Report"))
        self.report = QLineEdit()
        self.report.setReadOnly(True)
        rr.addWidget(self.report, 1)
        rb = QPushButton("…")
        rb.setFixedWidth(28)
        rb.clicked.connect(self._pick_report)
        rr.addWidget(rb)
        self.lods = QCheckBox("LODs")
        self.lods.setToolTip("Replace / hide on every LOD")
        self.lods.toggled.connect(self._on_lods)
        rr.addWidget(self.lods)
        mf.addWidget(self.report_row)

        self.scene_tabs = QTabWidget()
        self.scene_tabs.setDocumentMode(True)
        self.obj_tbl = QTableWidget(0, 3)
        self.obj_tbl.setHorizontalHeaderLabels(["Object", "Target", "2-sided"])
        self.obj_tbl.itemChanged.connect(self._on_two_sided)
        self.tgt_tbl = QTableWidget(0, 3)
        self.tgt_tbl.setHorizontalHeaderLabels(["Target", "Objects", "Hide"])
        self.tgt_tbl.itemChanged.connect(self._on_hide)
        self.scene_tbl = QTableWidget(0, 2)
        self.scene_tbl.setHorizontalHeaderLabels(["Mesh", "New name"])
        self.scene_tbl.itemChanged.connect(self._on_scene)
        obj_page = QWidget()
        ol = QVBoxLayout(obj_page)
        ol.setContentsMargins(0, 2, 0, 0)
        self.skip_rest = QCheckBox("Skip rest")
        self.skip_rest.toggled.connect(self._on_skip_rest)
        ol.addWidget(self.skip_rest)
        ol.addWidget(self.obj_tbl)
        for t, title in ((self.obj_tbl, "Objects"), (self.tgt_tbl, "Targets"), (self.scene_tbl, "Meshes")):
            t.verticalHeader().hide()
            t.setMinimumHeight(160)
            t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
            self.scene_tabs.addTab(obj_page if t is self.obj_tbl else t, title)
        self.tgt_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.obj_tbl.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self._obj_wheel = WheelGuard(self.obj_tbl)
        self._objects: dict[str, list[str]] = {}
        mf.addWidget(self.scene_tabs)
        lay.addWidget(self.mesh_box)

        # before / after
        self.cmp_box = QGroupBox("Before / after")
        g = QGridLayout(self.cmp_box)
        self.before = self._thumb_label()
        self.after = self._thumb_label()
        g.addWidget(QLabel("Game"), 0, 0)
        g.addWidget(QLabel("New"), 0, 1)
        g.addWidget(self.before, 1, 0)
        g.addWidget(self.after, 1, 1)
        lay.addWidget(self.cmp_box)

        box = QGroupBox("Changes")
        dl = QVBoxLayout(box)
        self.diff = QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setFont(mono_font())
        self.diff.setMaximumHeight(80)
        dl.addWidget(self.diff)
        lay.addWidget(box)
        lay.addStretch(1)
        self.bind(None)

    @staticmethod
    def _thumb_label() -> QLabel:
        box = QLabel()
        box.setAlignment(Qt.AlignCenter)
        box.setMinimumSize(THUMB + 10, THUMB + 10)
        box.setFrameShape(QFrame.StyledPanel)
        box.setWordWrap(True)
        box.setStyleSheet("background: palette(base);")
        return box

    # ---- binding -------------------------------------------------------------------------------------------
    def bind(self, item) -> None:
        self.item = item
        self._loading = True
        try:
            for w in (self.tgt_box, self.tex_box, self.mesh_box, self.cmp_box):
                w.setVisible(False)
            if item is None:
                self.title.setText("<b>—</b>")
                self.sub.setText("")
                self.diff.setPlainText("")
                self.setEnabled(False)
                return
            self.setEnabled(True)
            src = Path(item.source)
            self.title.setText(f"<b>{src.name}</b>")
            self.sub.setText(f"{item.kind.capitalize()} · {src.parent.name}")
            self.sub.setToolTip(str(src))
            self.tgt_box.setVisible(True)
            self.target.setText(item.target_name)
            scene = item.kind == "scene"
            self.target.setReadOnly(scene)
            self.pick_btn.setVisible(not scene)
            for w in (self.pack, self.name_row, self.match):
                self.tgt_form.setRowVisible(w, not scene)
            self._fill_packs()
            self.name_mode.setCurrentIndex(1 if item.new_name else 0)
            self.new_name.setText(item.new_name or default_new_name(item))
            self.new_name.setEnabled(bool(item.new_name))
            if item.kind == "texture":
                self.tex_box.setVisible(True)
                self.fmt.setCurrentIndex(max(0, self.fmt.findData(item.options.get("format", "auto"))))
                self.mips.setCurrentIndex(1 if item.options.get("mips") == "none" else 0)
                dds = src.suffix.lower() == ".dds"
                self.fmt.setEnabled(not dds)
                self.fmt.setToolTip("DDS keeps its format" if dds else "")
            if item.kind in ("mesh", "scene"):
                self.mesh_box.setVisible(True)
                self.tol.setValue(float(item.options.get("tol", 1e-4)))
                self.kind.setCurrentIndex(1 if scene else 0)
                self.report_row.setVisible(scene)
                self.scene_tabs.setVisible(scene)
                if scene:
                    self.report.setText(Path(item.options.get("report", "")).name)
                    self.report.setToolTip(item.options.get("report", ""))
                    self.lods.setChecked(bool(item.options.get("lods", True)))
                    self._fill_scene()
                    self._fill_map()
            self.cmp_box.setVisible(True)
            self._previews()
            self.refresh()
        finally:
            self._loading = False

    def refresh(self) -> None:
        """Derived labels only (match, changes); safe while the user types."""
        it = self.item
        if it is None:
            return
        if it.kind != "scene":
            ok = bool(it.target_name) and bool(self.state.env.find(it.target_name, it.type, it.target_pack))
            color = (GREEN if ok else RED).name()
            self.match.setText(f"<span style='color:{color}'>{'exact' if ok else 'none'}</span>")
        self.diff.setPlainText(change_text(it))

    def _fill_packs(self) -> None:
        it = self.item
        if it is None or it.kind == "scene":
            return
        was = self._loading
        self._loading = True
        try:
            self.pack.clear()
            self.pack.addItem("Any", "")
            labels = []
            if it.target_name:
                for lb, _ in self.state.env.find(it.target_name, it.type):
                    if lb not in labels:
                        labels.append(lb)
            if it.target_pack and it.target_pack not in labels:
                labels.append(it.target_pack)
            for lb in labels:
                self.pack.addItem(lb, lb)
            self.pack.setCurrentIndex(max(0, self.pack.findData(it.target_pack)))
        finally:
            self._loading = was

    def _fill_scene(self) -> None:
        it = self.item
        only = set(it.options.get("meshes") or [])
        renames = it.options.get("renames") or {}
        self.scene_tbl.blockSignals(True)
        names = scene_meshes(it)
        self.scene_tbl.setRowCount(len(names))
        for r, n in enumerate(names):
            a = QTableWidgetItem(n)
            a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable)
            a.setCheckState(Qt.Checked if n in only else Qt.Unchecked)
            a.setFont(mono_font())
            b = QTableWidgetItem(renames.get(n, ""))
            b.setFont(mono_font())
            self.scene_tbl.setItem(r, 0, a)
            self.scene_tbl.setItem(r, 1, b)
        self.scene_tbl.blockSignals(False)

    def _previews(self) -> None:
        it = self.item
        runner = self.ctx.runner
        for lbl in (self.before, self.after):
            lbl.setPixmap(QPixmap())
        src = Path(it.source)
        size = human_size(src.stat().st_size) if src.is_file() else "missing"
        gid = target_gid(self.ctx, it)
        if it.kind != "texture":
            self.before.setText(f"{it.target_name}\n{self.ctx.catalog.entry(gid).label}" if gid is not None
                                else "—")
            self.after.setText(f"{src.name}\n{size}")
            return
        iid = it.id
        if src.is_file():
            self.after.setText(PEND)
            runner.submit("build.thumb.src", _thumb_source, str(src),
                          on_done=lambda img: self._set_thumb(iid, self.after, img),
                          on_error=lambda _e: self._set_text(iid, self.after, f"{src.name}\n{size}"))
        else:
            self.after.setText("missing")
        if gid is None:
            self.before.setText("—")
            return
        self.before.setText(PEND)
        e, index = self.ctx.catalog.split(gid)
        runner.submit("build.thumb.tgt", _thumb_target, e.pack, index,
                      on_done=lambda img: self._set_thumb(iid, self.before, img),
                      on_error=lambda err: self._set_text(iid, self.before, err.splitlines()[0][:80]))

    def _set_thumb(self, iid, lbl, img) -> None:
        if self.item is not None and self.item.id == iid:
            lbl.setText("")
            lbl.setPixmap(QPixmap.fromImage(img))

    def _set_text(self, iid, lbl, text) -> None:
        if self.item is not None and self.item.id == iid:
            lbl.setPixmap(QPixmap())
            lbl.setText(text)

    # ---- edits ---------------------------------------------------------------------------------------------
    def _changed(self) -> None:
        self.refresh()
        self.edited.emit()

    def _on_target(self, text) -> None:
        if self._loading or self.item is None:
            return
        self.item.target_name = text.strip()
        self._changed()

    def _on_pack(self, _i) -> None:
        if self._loading or self.item is None:
            return
        self.item.target_pack = self.pack.currentData() or ""
        self._changed()
        self._previews()

    def _on_name_mode(self, i) -> None:
        if self._loading or self.item is None:
            return
        new = i == 1
        self.new_name.setEnabled(new)
        if new and not self.new_name.text().strip():
            self.new_name.setText(default_new_name(self.item))
        self.item.new_name = (self.new_name.text().strip() or default_new_name(self.item)) if new else None
        self._changed()

    def _on_new_name(self, text) -> None:
        if self._loading or self.item is None or self.name_mode.currentIndex() != 1:
            return
        self.item.new_name = text.strip() or None
        self._changed()

    def _on_tex(self, _i) -> None:
        if self._loading or self.item is None:
            return
        o = self.item.options
        fmt = self.fmt.currentData()
        if fmt == "auto":
            o.pop("format", None)
        else:
            o["format"] = fmt
        if self.mips.currentData():
            o["mips"] = "none"
        else:
            o.pop("mips", None)
        self._changed()

    def _on_tol(self, v) -> None:
        if self._loading or self.item is None:
            return
        self.item.options["tol"] = float(v)
        self._changed()

    # ---- scene mapping ----------------------------------------------------------------------------------------
    def _on_kind(self, _i) -> None:
        if self._loading or self.item is None:
            return
        kind = self.kind.currentData()
        if kind == self.item.kind:
            return
        self.item.kind = kind
        if kind == "scene":
            self.item.new_name = None
            rep = self.item.options.get("report")
            self.item.target_name = (load_json(rep).get("model") if rep and Path(rep).is_file() else None) \
                or Path(self.item.source).stem
        self._changed()
        self.bind(self.item)

    def _pick_report(self, path: str | None = None) -> None:
        it = self.item
        if it is None:
            return
        if path is None:
            start = str(Path(it.options.get("report") or it.source).parent)
            path, _ = QFileDialog.getOpenFileName(self, "Report", start, "Export report (*.cast.json)")
        if not path:
            return
        try:
            rep = load_json(path)
        except Exception as exc:  # noqa: BLE001
            self.ctx.status.emit(f"not a report: {exc}")
            return
        if not schema_startswith(rep.get("format"), "nightrunner.model_cast"):
            self.ctx.status.emit("not a model export report")
            return
        if it.options.get("report") and Path(it.options["report"]) != Path(path):
            for k in ("assign", "hide", "meshes", "renames"):       # targets of another export don't apply
                it.options.pop(k, None)
        it.options["report"] = path
        it.target_name = rep.get("model") or it.target_name
        self._changed()
        self.bind(it)

    def _on_lods(self, on) -> None:
        if self._loading or self.item is None:
            return
        if on:
            self.item.options.pop("lods", None)
        else:
            self.item.options["lods"] = False
        self._changed()

    def _fill_map(self) -> None:
        it = self.item
        src = it.source
        objs = self._objects.get(src)
        if objs is None:
            self.obj_tbl.setRowCount(0)
            if Path(src).is_file():
                self.ctx.runner.submit(("build.objects", src), scene_objects, src,
                                       on_done=lambda res, s=src: self._objects_ready(s, res),
                                       on_error=lambda err, s=src: self._objects_ready(s, []))
        targets = []
        rep = it.options.get("report")
        if rep and Path(rep).is_file():
            try:
                targets = scene_targets(rep)
            except Exception:  # noqa: BLE001
                targets = []
        self._targets = targets
        assign = it.options.get("assign") or {}
        hide = set(it.options.get("hide") or [])
        self.skip_rest.setChecked(bool(it.options.get("skip_rest")))
        # objects
        two = set(it.options.get("double_sided") or [])
        if objs is not None:
            self.obj_tbl.blockSignals(True)
            self.obj_tbl.setRowCount(len(objs))
            for r, o in enumerate(objs):
                a = QTableWidgetItem(o)
                a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                a.setFont(mono_font())
                self.obj_tbl.setItem(r, 0, a)
                c = QComboBox()
                c.addItem("—", None)
                c.addItem("Skip", "")
                for t in targets:
                    c.addItem(t["name"], t["name"])
                key = o if o in assign else _SUFFIX.sub("", o)
                cur = assign.get(key, None)
                c.setCurrentIndex(max(0, c.findData(cur)) if cur is not None else 0)
                if cur is None and any(t["name"] == key for t in targets):
                    c.setItemText(0, "same")
                c.currentIndexChanged.connect(lambda _i, obj=key, cb=c: self._on_assign(obj, cb.currentData()))
                self._obj_wheel.guard(c)
                self.obj_tbl.setCellWidget(r, 1, c)
                t2 = QTableWidgetItem()
                t2.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
                t2.setCheckState(Qt.Checked if key in two or o in two else Qt.Unchecked)
                t2.setData(Qt.UserRole, key)
                self.obj_tbl.setItem(r, 2, t2)
            self.obj_tbl.blockSignals(False)
        # targets
        users: dict[str, list[str]] = {}
        for o, t in assign.items():
            if t:
                users.setdefault(t, []).append(o)
        self.tgt_tbl.blockSignals(True)
        self.tgt_tbl.setRowCount(len(targets))
        for r, t in enumerate(targets):
            a = QTableWidgetItem(t["name"])
            a.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            a.setFont(mono_font())
            a.setToolTip(t.get("material", ""))
            b = QTableWidgetItem(", ".join(users.get(t["name"], [])))
            b.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            b.setForeground(QBrush(NEW))
            h = QTableWidgetItem()
            h.setFlags(Qt.ItemIsEnabled | Qt.ItemIsUserCheckable)
            h.setCheckState(Qt.Checked if t["name"] in hide else Qt.Unchecked)
            h.setData(Qt.UserRole, t["name"])
            self.tgt_tbl.setItem(r, 0, a)
            self.tgt_tbl.setItem(r, 1, b)
            self.tgt_tbl.setItem(r, 2, h)
        self.tgt_tbl.blockSignals(False)

    def _objects_ready(self, src: str, objs) -> None:
        self._objects[src] = list(objs or [])
        if self.item is not None and self.item.source == src and self.item.kind == "scene":
            was = self._loading
            self._loading = True
            try:
                self._fill_map()
            finally:
                self._loading = was

    def _on_assign(self, obj: str, target) -> None:
        if self._loading or self.item is None:
            return
        o = self.item.options
        assign = dict(o.get("assign") or {})
        if target is None:
            assign.pop(obj, None)
        else:
            assign[obj] = target
        if assign:
            o["assign"] = assign
        else:
            o.pop("assign", None)
        self._changed()
        was = self._loading
        self._loading = True
        try:
            self._fill_map()
        finally:
            self._loading = was

    def _on_two_sided(self, cell) -> None:
        if self._loading or self.item is None or cell.column() != 2:
            return
        two = []
        for r in range(self.obj_tbl.rowCount()):
            c = self.obj_tbl.item(r, 2)
            if c is not None and c.checkState() == Qt.Checked:
                two.append(c.data(Qt.UserRole))
        if two:
            self.item.options["double_sided"] = two
        else:
            self.item.options.pop("double_sided", None)
        self._changed()

    def _on_skip_rest(self, on) -> None:
        if self._loading or self.item is None:
            return
        if on:
            self.item.options["skip_rest"] = True
        else:
            self.item.options.pop("skip_rest", None)
        self._changed()

    def _on_hide(self, cell) -> None:
        if self._loading or self.item is None or cell.column() != 2:
            return
        hide = []
        for r in range(self.tgt_tbl.rowCount()):
            h = self.tgt_tbl.item(r, 2)
            if h is not None and h.checkState() == Qt.Checked:
                hide.append(h.data(Qt.UserRole))
        if hide:
            self.item.options["hide"] = hide
        else:
            self.item.options.pop("hide", None)
        self._changed()

    def _on_scene(self, _cell) -> None:
        if self._loading or self.item is None or self.item.kind != "scene":
            return
        only, renames = [], {}
        for r in range(self.scene_tbl.rowCount()):
            a, b = self.scene_tbl.item(r, 0), self.scene_tbl.item(r, 1)
            if a is None:
                continue
            if a.checkState() == Qt.Checked:
                only.append(a.text())
            nn = b.text().strip() if b is not None else ""
            if nn:
                renames[a.text()] = nn
        o = self.item.options
        for key, val in (("meshes", only), ("renames", renames)):
            if val:
                o[key] = val
            else:
                o.pop(key, None)
        self._changed()

    def _pick(self) -> None:
        it = self.item
        if it is None:
            return
        dlg = PickDialog(self.ctx, it.type, Path(it.target_name).stem, self)
        if dlg.exec() == QDialog.Accepted and dlg.choice:
            it.target_name, it.target_pack = dlg.choice
            self.bind(it)
            self.edited.emit()


# ---- the tab ---------------------------------------------------------------------------------------------------

class Tab(QWidget):
    TITLE = TITLE

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.state = BuildState(ctx, self)
        self.problems: list = []
        self.statuses: dict[str, tuple] = {}
        self._validated = False
        self._cancel: threading.Event | None = None
        self._last_out: Path | None = None
        self.models_page = None
        self._lines = _Lines(self)
        self._lines.line.connect(self.log)
        root = QVBoxLayout(self)

        root.addLayout(self._toolbar())

        # ---- filter row ------------------------------------------------------------------------------------
        frow = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("filter…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        frow.addWidget(self.search, 1)
        self.show_mode = QComboBox()
        self.show_mode.addItems(["All", "Problems", "Off"])
        self.show_mode.currentIndexChanged.connect(self._filter)
        frow.addWidget(self.show_mode)
        self.count = QLabel()
        frow.addWidget(self.count)
        root.addLayout(frow)

        # ---- pages -----------------------------------------------------------------------------------------
        self.pages = QTabWidget()
        self.pages.setTabPosition(QTabWidget.West)
        vsplit = QSplitter(Qt.Vertical)
        hsplit = QSplitter(Qt.Horizontal)
        self.tree = DropTree()
        self.tree.filesDropped.connect(self.add_paths)
        self.tree.deletePressed.connect(self._remove_selected)
        self.tree.toggled.connect(self._on_toggled)
        self.tree.customContextMenuRequested.connect(self._menu)
        self.tree.itemSelectionChanged.connect(self._on_sel)
        hsplit.addWidget(self.tree)
        self.insp = Inspector(ctx, self.state)
        self.insp.edited.connect(lambda: self.state.touch("items"))
        insp_scroll = QScrollArea()                     # the scene inspector is taller than a 1080p window
        insp_scroll.setWidgetResizable(True)
        insp_scroll.setFrameShape(QFrame.NoFrame)
        insp_scroll.setWidget(self.insp)
        hsplit.addWidget(insp_scroll)
        hsplit.setStretchFactor(0, 3)
        hsplit.setStretchFactor(1, 2)
        hsplit.setSizes([950, 520])
        self.pages.addTab(hsplit, "Resources")
        self.pages.addTab(self._models_page(), "Models")
        vsplit.addWidget(self.pages)
        vsplit.addWidget(self._bottom())
        vsplit.setSizes([640, 230])
        root.addWidget(vsplit, 1)

        # ---- footer ----------------------------------------------------------------------------------------
        foot = QHBoxLayout()
        self.summary = QLabel()
        foot.addWidget(self.summary, 1)
        self.prog = QProgressBar()
        self.prog.setMaximumWidth(260)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        self.prog.setFormat("idle")
        foot.addWidget(self.prog)
        root.addLayout(foot)

        # ---- wiring ----------------------------------------------------------------------------------------
        st = self.state
        self._auto = Debouncer(self.validate, 400, self)
        st.itemsChanged.connect(self._on_items)
        st.modelsChanged.connect(self._on_counts)
        st.settingsChanged.connect(self._on_counts)
        for sig in (st.itemsChanged, st.modelsChanged, st.settingsChanged):
            sig.connect(self._auto.trigger)
        st.projectReplaced.connect(self._on_replaced)
        st.dirtyChanged.connect(lambda _d: self._update_name())
        st.problemsChanged.connect(self._on_problems)
        cat = getattr(ctx, "catalog", None)
        if cat is not None:
            cat.ready.connect(self._on_catalog)

        last = str(ctx.settings.value("build/last_project", "") or "")
        opened = False
        if last and Path(last).is_file():
            try:
                st.open(last)
                opened = True
            except Exception:  # noqa: BLE001
                pass
        if not opened:
            self._on_replaced()

    # ---- construction --------------------------------------------------------------------------------------
    def _toolbar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        proj = QToolButton()
        proj.setText("Project ▾")
        proj.setPopupMode(QToolButton.InstantPopup)
        pm = QMenu(proj)
        pm.addAction("New", self.new_project)
        pm.addAction("Open…", self.open_project)
        pm.addAction("Save", self.save_project)
        pm.addAction("Save as…", lambda: self.save_project(as_new=True))
        pm.addSeparator()
        self.recent_menu = pm.addMenu("Recent")
        self.recent_menu.aboutToShow.connect(self._fill_recent)
        pm.addSeparator()
        pm.addAction("Import folder…", self._add_folder)
        proj.setMenu(pm)
        bar.addWidget(proj)
        self.proj_name = QLabel()
        self.proj_name.setMinimumWidth(80)
        bar.addWidget(self.proj_name)
        bar.addSpacing(12)

        bar.addWidget(QLabel("rpack"))
        self.rpack_edit = QLineEdit()
        self.rpack_edit.setMinimumWidth(160)
        self.rpack_edit.setClearButtonEnabled(True)
        self.rpack_edit.textEdited.connect(self._on_rpack)
        bar.addWidget(self.rpack_edit)
        bar.addWidget(QLabel("PAK"))
        self.pak_combo = QComboBox()
        self.pak_combo.setEditable(True)
        self.pak_combo.setInsertPolicy(QComboBox.NoInsert)
        self.pak_combo.setMinimumWidth(120)
        self.pak_combo.lineEdit().textEdited.connect(self._on_pak)
        self.pak_combo.activated.connect(lambda _i: self._on_pak(self.pak_combo.currentText()))
        bar.addWidget(self.pak_combo)
        bar.addSpacing(8)
        bar.addWidget(QLabel("Output"))
        self.out = QLineEdit()
        self.out.setMinimumWidth(200)
        self.out.textEdited.connect(self._on_out)
        bar.addWidget(self.out, 1)
        browse = QPushButton("…")
        browse.setFixedWidth(28)
        browse.clicked.connect(self._browse_out)
        bar.addWidget(browse)
        bar.addSpacing(12)
        add = QToolButton()
        add.setText("Add ▾")
        add.setPopupMode(QToolButton.InstantPopup)
        am = QMenu(add)
        am.addAction("Files…", self._add_files)
        am.addAction("Folder…", self._add_folder)
        add.setMenu(am)
        bar.addWidget(add)
        self.validate_btn = QPushButton("Validate")
        self.validate_btn.clicked.connect(self.validate)
        bar.addWidget(self.validate_btn)
        self.build_btn = QPushButton("Build")
        self.build_btn.setDefault(True)
        self.build_btn.clicked.connect(self._build_clicked)
        self._style_build(False)
        bar.addWidget(self.build_btn)
        self.open_btn = QPushButton("Open folder")
        self.open_btn.setEnabled(False)
        self.open_btn.clicked.connect(self._open_out)
        bar.addWidget(self.open_btn)
        return bar

    def _style_build(self, running: bool) -> None:
        bg = "#8a3a2a" if running else "#2f6f3e"
        self.build_btn.setText("Cancel" if running else "Build")
        self.build_btn.setStyleSheet(f"QPushButton {{ background:{bg}; color:white; font-weight:bold; "
                                     f"padding:4px 18px; }} QPushButton:disabled {{ background:#555; }}")

    def _models_page(self) -> QWidget:
        try:
            from .build_models import ModelOverridesPage
            self.models_page = ModelOverridesPage(self.ctx, self.state)
            return self.models_page
        except Exception:  # noqa: BLE001 - the Models page must not take the Resources page down
            self.models_page = None
            w = QPlainTextEdit()
            w.setReadOnly(True)
            w.setPlainText(f"Models page failed to load:\n\n{traceback.format_exc()}")
            return w

    def _bottom(self) -> QTabWidget:
        t = self.bottom = QTabWidget()
        t.setDocumentMode(True)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(mono_font())
        self.log_view.setMaximumBlockCount(5000)
        t.addTab(self.log_view, "Log")

        self.probs = QTableWidget(0, 3)
        self.probs.setHorizontalHeaderLabels(["", "Item", "Message"])
        self.probs.horizontalHeader().setStretchLastSection(True)
        self.probs.setColumnWidth(0, 28)
        self.probs.setColumnWidth(1, 240)
        self.probs.verticalHeader().hide()
        self.probs.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.probs.cellDoubleClicked.connect(self._problem_clicked)
        t.addTab(self.probs, "Problems")

        stw = QWidget()
        g = QGridLayout(stw)
        g.addWidget(QLabel("field08"), 0, 0)
        self.f08 = QComboBox()
        for label, v in (("Auto", None), ("0x0", 0), ("0x1000", 0x1000)):
            self.f08.addItem(label, v)
        self.f08.currentIndexChanged.connect(self._on_settings)
        g.addWidget(self.f08, 0, 1)
        self.opt_checks: dict[str, QCheckBox] = {}
        for r, (key, label) in enumerate((("validate", "Validate"), ("report", "Report"),
                                          ("keep_work", "Keep temp"))):
            cb = QCheckBox(label)
            cb.toggled.connect(self._on_settings)
            g.addWidget(cb, 1 + r, 1)
            self.opt_checks[key] = cb
        g.setColumnStretch(2, 1)
        g.setRowStretch(4, 1)
        t.addTab(stw, "Settings")

        self.hist = QTableWidget(0, 5)
        self.hist.setHorizontalHeaderLabels(["When", "Project", "Result", "Output", "Size"])
        self.hist.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.hist.verticalHeader().hide()
        self.hist.setColumnWidth(0, 130)
        self.hist.setColumnWidth(1, 140)
        self.hist.setColumnWidth(2, 60)
        t.addTab(self.hist, "History")
        self._fill_history()
        return t

    # ---- project -------------------------------------------------------------------------------------------
    def _update_name(self) -> None:
        p = self.state.project
        self.proj_name.setText(f"<b>{p.name}{' *' if self.state.dirty else ''}</b>")
        self.proj_name.setToolTip(p.path or "")

    def _on_replaced(self) -> None:
        p = self.state.project
        self.statuses = {}
        self.problems = []
        self._validated = False
        self.insp.bind(None)
        self._update_name()
        self._fill_outputs()
        self._fill_settings()
        self._on_items()
        self._on_problems_table([])
        if p.path:
            self._remember(p.path)
        self._auto.trigger()

    def _fill_outputs(self) -> None:
        st = self.state
        p = st.project
        try:
            rp_def = st.default_rpack()
        except Exception:  # noqa: BLE001
            rp_def = ""
        try:
            pak_def = st.default_pak()
        except Exception:  # noqa: BLE001
            pak_def = ""
        self.rpack_edit.setPlaceholderText(rp_def)
        self.rpack_edit.setText(p.rpack_name)
        self.pak_combo.blockSignals(True)
        self.pak_combo.clear()
        src = st.env.source
        existing = {q.name.lower() for q in src.iterdir()} if src and src.is_dir() else set()
        stock = {n.lower() for n in st.env.profile.stock_paks}
        for n in PAK_RANGE:
            name = f"data{n}.pak"
            self.pak_combo.addItem(name)
            if name in stock:                                # the game's own archives
                self.pak_combo.model().item(self.pak_combo.count() - 1).setEnabled(False)
            elif name in existing:
                self.pak_combo.setItemText(self.pak_combo.count() - 1, name)
                self.pak_combo.setItemData(self.pak_combo.count() - 1, "installed", Qt.ToolTipRole)
        self.pak_combo.setCurrentIndex(-1)
        self.pak_combo.setEditText(p.pak_name)
        self.pak_combo.lineEdit().setPlaceholderText(pak_def)
        self.pak_combo.blockSignals(False)
        self.out.setText(p.output_dir)
        self.out.setPlaceholderText(str(st.output_dir()))

    def _fill_settings(self) -> None:
        p = self.state.project
        widgets = [self.f08, *self.opt_checks.values()]
        for w in widgets:
            w.blockSignals(True)
        self.f08.setCurrentIndex(0 if p.field08 is None else max(0, self.f08.findData(p.field08)))
        defaults = {"validate": True, "report": True, "keep_work": False}
        for k, cb in self.opt_checks.items():
            cb.setChecked(bool(p.options.get(k, defaults[k])))
        for w in widgets:
            w.blockSignals(False)

    def _on_rpack(self, text: str) -> None:
        self.state.project.rpack_name = text.strip()
        self.state.touch("settings")

    def _on_pak(self, text: str) -> None:
        self.state.project.pak_name = text.strip()
        self.state.touch("settings")

    def _on_out(self, text: str) -> None:
        self.state.project.output_dir = text.strip()
        self.out.setPlaceholderText(str(self.state.output_dir()))
        self.state.touch("settings")

    def _browse_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output", str(self.state.output_dir()))
        if d:
            self.out.setText(d)
            self._on_out(d)

    def _on_settings(self, *_):
        p = self.state.project
        p.field08 = self.f08.currentData()
        for k, cb in self.opt_checks.items():
            p.options[k] = cb.isChecked()
        self.state.touch("settings")

    def new_project(self) -> None:
        self.state.new()

    def open_project(self, path: str | None = None) -> bool:
        if not path:
            start = str(Path(self.state.project.path).parent) if self.state.project.path else ""
            path, _ = QFileDialog.getOpenFileName(self, "Open", start, "Project (*.nrproj *.bpproj)")
            if not path:
                return False
        try:
            self.state.open(path)
        except Exception as exc:  # noqa: BLE001
            self.log(f"✗ open {path}: {exc}")
            QMessageBox.warning(self, "Open", str(exc))
            return False
        return True

    def save_project(self, as_new: bool = False, path: str | None = None) -> bool:
        p = self.state.project
        if path is None:
            path = p.path
            if as_new or not path:
                start = path or str(Path.cwd() / f"{p.name}.nrproj")
                path, _ = QFileDialog.getSaveFileName(self, "Save", start, "Project (*.nrproj)")
                if not path:
                    return False
        if p.name == "untitled" or (as_new and p.path):
            p.name = Path(path).stem
        try:
            saved = self.state.save(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Save", str(exc))
            return False
        self._remember(str(saved))
        self._update_name()
        self.out.setPlaceholderText(str(self.state.output_dir()))
        self.ctx.status.emit(f"saved {saved}")
        return True

    def _recent(self) -> list[str]:
        return [str(x) for x in _as_list(self.ctx.settings.value("build/recent", []))]

    def _remember(self, path: str) -> None:
        path = str(path)
        rec = [path] + [r for r in self._recent() if r != path]
        self.ctx.settings.setValue("build/recent", rec[:RECENT_MAX])

    def _fill_recent(self) -> None:
        m = self.recent_menu
        m.clear()
        rec = self._recent()
        for r in rec:
            a = m.addAction(r, lambda r=r: self.open_project(r))
            a.setEnabled(Path(r).is_file())
        if not rec:
            m.addAction("—").setEnabled(False)

    # ---- items ---------------------------------------------------------------------------------------------
    def add_paths(self, paths) -> list:
        try:
            new = add_files(self.state.project, [Path(p) for p in paths], self.state.env)
        except Exception as exc:  # noqa: BLE001
            self.log(f"✗ add: {exc}")
            return []
        if new:
            self.state.touch("items")
            self.tree.clearSelection()
            self.tree.select_id(new[0].id)
        else:
            self.ctx.status.emit("nothing to add")
        return new

    def _add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(self, "Add", "",
                                                "Resources (*.png *.dds *.cast *.glb *.gltf *.bin);;All (*)")
        if files:
            self.add_paths(files)

    def _add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Add folder")
        if d:
            self.add_paths([d])

    def _sizes(self) -> dict:
        out = {}
        for it in self.state.project.items:
            try:
                out[it.id] = Path(it.source).stat().st_size
            except OSError:
                out[it.id] = None
        return out

    def _on_items(self) -> None:
        items = self.state.project.items
        self.tree.set_items(items, self.statuses, self._sizes())
        self._filter()
        cur = self.insp.item
        if cur is not None and (cur not in items or self.tree.current_id() != cur.id):
            self._on_sel()
        else:
            self.insp.refresh()
        self._on_counts()

    def _on_counts(self) -> None:
        p = self.state.project
        self.pages.setTabText(0, f"Resources ({len(p.items)})")
        self.pages.setTabText(1, f"Models ({len(p.models)})")
        self._summary()

    def _filter(self, *_):
        shown = self.tree.apply_filter(self.search.text(), self.show_mode.currentText())
        total = len(self.tree.rows)
        self.count.setText(f"{shown} / {total}" if shown != total else str(total))

    def _find(self, iid):
        return next((it for it in self.state.project.items if it.id == iid), None)

    def _on_sel(self) -> None:
        it = self._find(self.tree.current_id())
        if it is not self.insp.item:
            self.insp.bind(it)

    def _on_toggled(self, iid: str, on: bool) -> None:
        ids = self.tree.selected_ids()
        self._set_enabled(ids if iid in ids else [iid], on)

    def _set_enabled(self, ids, on: bool) -> None:
        for t in ids:
            it = self._find(t)
            if it is not None:
                it.enabled = on
        self.state.touch("items")

    def _remove_selected(self) -> None:
        self._remove_ids(self.tree.selected_ids())

    def _remove_ids(self, ids) -> None:
        ids = set(ids)
        if not ids:
            return
        self.state.project.items = [x for x in self.state.project.items if x.id not in ids]
        self.state.touch("items")

    def _menu(self, pos) -> None:
        row = self.tree.itemAt(pos)
        if row is None or row.parent() is None:
            return
        it = self._find(row.data(0, ROLE_ID))
        if it is None:
            return
        ids = self.tree.selected_ids()
        if it.id not in ids:
            ids = [it.id]
        m = QMenu(self)
        rev = m.addAction("Reveal target", lambda: self._reveal(it))
        rev.setEnabled(it.kind != "scene" and target_gid(self.ctx, it) is not None)
        m.addAction("Show in Explorer", lambda: QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(Path(it.source).parent))))
        m.addSeparator()
        if all(x.enabled for x in map(self._find, ids) if x is not None):
            m.addAction("Exclude", lambda: self._set_enabled(ids, False))
        else:
            m.addAction("Include", lambda: self._set_enabled(ids, True))
        m.addAction("Duplicate", lambda: self._duplicate(it))
        m.addAction("Remove", lambda: self._remove_ids(ids))
        m.exec(self.tree.viewport().mapToGlobal(pos))

    def _duplicate(self, it) -> None:
        import uuid
        dup = copy.deepcopy(it)
        dup.id = uuid.uuid4().hex[:8]
        items = self.state.project.items
        items.insert(items.index(it) + 1, dup)
        self.state.touch("items")

    def _reveal(self, it) -> None:
        gid = target_gid(self.ctx, it)
        if gid is None:
            return
        sig = {"texture": self.ctx.openTexture, "mesh": self.ctx.openMesh}.get(it.kind, self.ctx.openRaw)
        sig.emit(gid)

    # ---- validation ----------------------------------------------------------------------------------------
    def _on_catalog(self) -> None:
        self._fill_outputs()
        if self.insp.item is not None:
            self.insp.bind(self.insp.item)
        self.validate()

    def validate(self) -> None:
        proj = copy.deepcopy(self.state.project)
        self.ctx.runner.submit("build.validate", validate, proj, self.state.env,
                               on_done=self.state.problemsChanged.emit,
                               on_error=lambda e: self.log(f"✗ validate: {e.splitlines()[0]}"))

    def _on_problems(self, problems: list) -> None:
        self.problems = list(problems)
        self._validated = True
        by: dict[str, list] = {it.id: [] for it in self.state.project.items}
        for p in problems:
            if p.where in by:
                by[p.where].append(p)
        self.statuses = {iid: item_status(ps) for iid, ps in by.items()}
        self.tree.set_items(self.state.project.items, self.statuses, self._sizes())
        self._filter()
        self._on_problems_table(problems)
        self._summary()

    def _where_label(self, where: str) -> str:
        it = self._find(where)
        if it is not None:
            return Path(it.source).name
        mo = next((m for m in self.state.project.models if m.id == where), None)
        return mo.label if mo is not None else where

    def _on_problems_table(self, problems: list) -> None:
        order = sorted(problems, key=lambda p: p.level != "error")
        self.probs.setRowCount(len(order))
        for r, p in enumerate(order):
            err = p.level == "error"
            ic = _cell("✗" if err else "⚠", RED if err else AMBER)
            ic.setData(ROLE_ID, p.where)
            self.probs.setItem(r, 0, ic)
            self.probs.setItem(r, 1, _cell(self._where_label(p.where)))
            self.probs.setItem(r, 2, _cell(p.message, tip=p.message))
        self.bottom.setTabText(1, f"Problems ({len(order)})" if order else "Problems")

    def _problem_clicked(self, row: int, _col: int) -> None:
        c = self.probs.item(row, 0)
        if c is None:
            return
        where = c.data(ROLE_ID)
        if where in self.tree.rows:
            self.pages.setCurrentIndex(0)
            self.search.clear()
            self.show_mode.setCurrentIndex(0)
            self.tree.clearSelection()
            self.tree.select_id(where)
        elif any(m.id == where for m in self.state.project.models):
            self.pages.setCurrentIndex(1)

    def _summary(self) -> None:
        p = self.state.project
        on = sum(it.enabled for it in p.items)
        parts = [f"{len(p.items)} items" if on == len(p.items) else f"{on}/{len(p.items)} items",
                 f"{len(p.models)} models"]
        if self._validated:
            w = sum(x.level != "error" for x in self.problems)
            e = len(self.problems) - w
            if w:
                parts.append(f"<span style='color:{AMBER.name()}'>{w} ⚠</span>")
            if e:
                parts.append(f"<span style='color:{RED.name()}'>{e} ✗</span>")
            if not self.problems:
                parts.append(f"<span style='color:{GREEN.name()}'>✓</span>")
        size = sum(s for s in self._sizes().values() if s)
        if size:
            parts.append(human_size(size))
        self.summary.setText(" · ".join(parts))

    # ---- build ---------------------------------------------------------------------------------------------
    def log(self, line: str) -> None:
        self.log_view.appendPlainText(f"[{time.strftime('%H:%M:%S')}] {line}")

    @property
    def running(self) -> bool:
        return self._cancel is not None

    def _build_clicked(self) -> None:
        if self.running:
            self._cancel.set()
            self.build_btn.setEnabled(False)
            self.log("cancelling…")
            return
        self.build()

    def build(self) -> bool:
        if self.running:
            return False
        st = self.state
        if not st.project.path and not self.save_project():
            return False
        out = st.output_dir()
        proj = copy.deepcopy(st.project)
        self._cancel = threading.Event()
        self._style_build(True)
        self.validate_btn.setEnabled(False)
        self.open_btn.setEnabled(False)
        self.prog.setRange(0, 0)
        self.prog.setFormat("building")
        self.bottom.setCurrentIndex(0)
        self.log(f"build {proj.name} → {out}")
        self.ctx.runner.submit("build.run", _build_job, proj, st.env, self._lines.line.emit, self._cancel, out,
                               on_done=lambda res: self._built(res, proj, out),
                               on_error=lambda err: self._build_failed(err, proj))
        return True

    def _finish(self) -> None:
        self._cancel = None
        self._style_build(False)
        self.build_btn.setEnabled(True)
        self.validate_btn.setEnabled(True)
        self.prog.setRange(0, 1)
        self.prog.setValue(0)
        self.prog.setFormat("idle")

    def _built(self, res: dict, proj, out: Path) -> None:
        self._finish()
        if not res.get("ok"):
            if res.get("cancelled"):
                self.log("cancelled")
                self.ctx.status.emit("build cancelled")
                return
            msg = res.get("error", "failed")
            self.log(f"✗ {msg}")
            self._on_problems_table(self.problems + [Problem("error", "project", msg)])
            self._add_history(proj, None, False)
            self.ctx.status.emit("build failed")
            return
        rep = res["report"]
        self.state.last_report = rep
        for w in rep.get("warnings", []):
            self.log(f"⚠ {w}")
        size = 0
        for kind, o in rep.get("outputs", {}).items():
            size += int(o.get("size") or 0)
            self.log(f"✓ {kind} {o.get('path')} ({human_size(o.get('size'))})")
        for kind, dest in (rep.get("install") or {}).items():
            if dest:
                self.log(f"  → {dest}")
        self.log(f"done · {rep.get('seconds', 0)} s")
        # pin auto names: once this build is installed, "next free" would move the project to another name
        pinned = False
        cur = self.state.project
        for key, kind in (("rpack_name", "rpack"), ("pak_name", "pak")):
            o = rep.get("outputs", {}).get(kind)
            if o and not getattr(cur, key):
                setattr(cur, key, Path(o["path"]).name)
                pinned = True
        if pinned:
            self.state.touch("settings")
        self._last_out = out
        self.open_btn.setEnabled(True)
        self._add_history(proj, rep, True)
        self.state.built.emit(rep)
        self.ctx.status.emit(f"built {proj.name} · {human_size(size)}")
        self._fill_outputs()

    def _build_failed(self, err: str, proj) -> None:
        self._finish()
        self.log(f"✗ {err.splitlines()[0] if err else 'failed'}")
        self._add_history(proj, None, False)
        QMessageBox.warning(self, "Build", err[:2000])

    def _open_out(self) -> None:
        if self._last_out is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._last_out)))

    # ---- history -------------------------------------------------------------------------------------------
    def history(self) -> list[dict]:
        raw = self.ctx.settings.value("build/history", "")
        try:
            h = json.loads(raw) if isinstance(raw, str) and raw else []
        except ValueError:
            h = []
        return [x for x in h if isinstance(x, dict)] if isinstance(h, list) else []

    def _add_history(self, proj, rep, ok: bool) -> None:
        outs = (rep or {}).get("outputs", {})
        rec = {"when": time.strftime("%Y-%m-%d %H:%M"), "project": proj.name,
               "rpack": Path(outs["rpack"]["path"]).name if "rpack" in outs else "",
               "pak": Path(outs["pak"]["path"]).name if "pak" in outs else "",
               "ok": ok, "size": sum(int(o.get("size") or 0) for o in outs.values())}
        h = (self.history() + [rec])[-HISTORY_MAX:]
        self.ctx.settings.setValue("build/history", json.dumps(h))
        self._fill_history()

    def _fill_history(self) -> None:
        h = list(reversed(self.history()))
        self.hist.setRowCount(len(h))
        for r, x in enumerate(h):
            ok = bool(x.get("ok"))
            outs = " + ".join(n for n in (x.get("rpack"), x.get("pak")) if n) or "—"
            vals = (x.get("when", ""), x.get("project", ""), "✓" if ok else "✗", outs,
                    human_size(x.get("size")) if ok else "—")
            for c, v in enumerate(vals):
                self.hist.setItem(r, c, _cell(str(v), (GREEN if ok else RED) if c == 2 else None))

    # ---- lifecycle -----------------------------------------------------------------------------------------
    def shutdown(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
        self.ctx.runner.cancel("build.validate")
        fn = getattr(self.models_page, "shutdown", None)
        if fn is not None:
            try:
                fn()
            except Exception:  # noqa: BLE001
                pass
