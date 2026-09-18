"""Build → Model overrides page.

One *override* = one character look: the `.model` documents of its roles (TPP / FPP / LodCC / UI, as routed by
scripts/playerappearances.scr), edited together and written into the project PAK (nightrunner/project.py).

Left: overrides, role members, output options. Center: slot grid over the union of the roles' slots (enable +
mesh per role, Link edits every role), Diff / Checks. Right: 3D preview through the Models tab pipeline
(modelresolve.resolve_model + models.load_preview, same material / rttiValues / SDB handling) and the material
overrides of the selected slot. Meshes / textures that only the project provides become visible after a build: the
built rpack is copied to a temp file, loaded as a user pack and preferred for the preview's lookups.
"""
from __future__ import annotations

import copy
import itertools
import os
import shutil
import tempfile
import uuid
from pathlib import Path

from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QBrush, QColor, QImage, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (QAbstractItemView, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QMenu, QMessageBox, QPlainTextEdit, QPushButton,
                               QSplitter, QStyle, QStyledItemDelegate, QTableWidget, QTableWidgetItem, QTabWidget,
                               QToolButton, QVBoxLayout, QWidget)

from ... import project as P
from ..modelresolve import read_mesh_info, resolve_model
from ..tasks import Debouncer
from ..widgets import WheelGuard, mono_font
from .models import add_preview_items, load_preview

NEW = QColor(110, 170, 255)      # project-provided asset
CHANGED = QColor(220, 170, 60)
OFF = QColor(130, 130, 130)
ERROR = QColor(230, 90, 90)

STAR = "★ "
NONE = "—"
BROWSE = "Browse…"
ROLE_LABEL = {"tpp": "TPP", "fpp": "FPP", "lodcc": "LodCC", "ui": "UI"}
ADD_PARAMS = ("dif_0_tex", "nrm_0_tex", "msk_1_tex", "rgh_0_tex", "spc_0_tex", "ocl_tex", None, "dif_0_val")
APPEARANCES = P.APPEARANCES_SCRIPT          # DLTB default; the live one comes from the game profile


def _profile(ctx):
    from ...games import profile
    game = getattr(ctx, "game", None)
    return game.profile if game is not None else profile(None)
PREVIEW_PREFIX = "nightrunner_preview_"
LIMIT = 1000
ROLE_SUB = Qt.UserRole + 1
ROLE_ID = Qt.UserRole + 2

_seq = itertools.count(1)


# ---- small helpers ------------------------------------------------------------------------------------------------

def bare(member: str) -> str:
    return member.replace("\\", "/").rsplit("/", 1)[-1]


def stem(mesh: str) -> str:
    return mesh[:-4] if mesh.lower().endswith(".msh") else mesh


def unstar(text: str) -> str:
    return text[len(STAR):] if text.startswith(STAR) else text


def slot_or_none(doc: dict | None, name: str) -> dict | None:
    for s in P.slots(doc or {}):
        if s.get("name") == name:
            return s
    return None


def selected_entry(s: dict | None) -> dict | None:
    return next((r for r in P.slot_resources(s or {}) if r.get("selected")), None)


def fmt_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return ", ".join(f"{float(x):.3g}" for x in v)
    if isinstance(v, float):
        return f"{v:.4g}"
    return "" if v is None else str(v)


def parse_value(text: str):
    parts = [p for p in text.replace(";", ",").split(",") if p.strip()]
    if len(parts) == 3:
        return [float(p) for p in parts]
    if len(parts) == 1:
        return float(parts[0])
    raise ValueError(text)


def tex_values(doc: dict | None) -> set:
    out = set()
    for s in P.slots(doc or {}):
        for r in P.slot_resources(s):
            for g in r.get("materialsResources") or []:
                for e in g.get("resources") or []:
                    for v in e.get("rttiValues") or []:
                        if v.get("type") == 7:
                            out.add((s.get("name"), r.get("name"), g.get("number"), e.get("name"), v.get("name"),
                                     v.get("val_str")))
    return out


def change_counts(mo: P.ModelOverride) -> tuple[int, int]:
    """(changed slots, new/changed texture overrides) over the override's unique members."""
    names, n_tex = set(), 0
    for r in mo.members().values():
        if r.doc is None or r.original is None:
            continue
        for s in P.slots(r.doc):
            if s != slot_or_none(r.original, s.get("name")):
                names.add(s.get("name"))
        n_tex += len(tex_values(r.doc) - tex_values(r.original))
    return len(names), n_tex


def find_model_rec(ctx, name: str) -> dict | None:
    """First provider (base game first) of a .model by member path or basename."""
    key = name.replace("\\", "/").lower()
    base = bare(key)
    if not base.endswith(".model"):
        base += ".model"
    recs = ctx.paks.models()
    hits = [m for m in recs if m["name"].replace("\\", "/").lower() == key] or \
        [m for m in recs if m["basename"] == base]
    return hits[0] if hits else None


def read_appearances(ctx) -> list[dict]:
    script = _profile(ctx).appearances_script
    if not script:
        return []
    for p in ctx.paks.paths:
        ix = ctx.paks.index(p)
        if ix is None:
            continue
        try:
            text = ix.read(script).decode("utf-8", "replace")
        except Exception:  # noqa: BLE001 - member missing in this archive
            continue
        return P.parse_appearances(text)
    return []


def appearance_label(a: dict) -> str:
    return f"{a.get('character') or '?'} · {a.get('appearance')}"


def _words(text: str):
    return [w for w in text.lower().split() if w]


def filter_names(names, text: str, limit: int = LIMIT) -> list[str]:
    ws = _words(text)
    out = [n for n in names if all(w in n.lower() for w in ws)]
    return out[:limit]


def catalog_names(catalog, text: str, type_id: int, limit: int = LIMIT) -> list[str]:
    seen, out = set(), []
    for g in catalog.search(text, types=[type_id]).tolist():
        n = catalog.name(g)
        if n.lower() not in seen:
            seen.add(n.lower())
            out.append(n)
            if len(out) >= limit:
                break
    return out


def mesh_materials(view, mesh: str) -> list[str] | None:
    """Worker: submesh material names of a catalog mesh (first provider), None when the mesh is not found."""
    gids = view.catalog.lookup(mesh, 0x10)
    if not gids:
        return None
    info = read_mesh_info(view.catalog, gids[0])
    names = [s.get("material") for s in info.get("submeshes") or []] or list(info.get("materials") or [])
    return list(dict.fromkeys(n for n in names if n and not n.lower().startswith(
        ("auto_shadow_caster", "shadowcaster", "shadow_caster"))))


def preview_job(view, docs: list[tuple[str, dict]], variant: str | None, ghost: bool, wait: list[int]) -> dict:
    """Worker: resolve + load every (tag, doc) the way the Models tab does. Off slots (nothing selected) are
    dropped, or kept and flagged when *ghost*."""
    if wait:
        view.catalog.wait_packs(wait)
    out = {"views": []}
    for tag, doc in docs:
        res = resolve_model(view, doc, variant, name=tag)
        off = {s.get("name") for s in P.slots(doc) if P.slot_mesh(s) is None}
        if not ghost:
            res = dict(res, slots=[s for s in res["slots"] if s["name"] not in off])
        pv = load_preview(view, res, True)
        out["views"].append({"tag": tag, "resolution": res, "off": off, "preview": pv})
    return out


def validate_job(project, env) -> list:
    return P.validate(project, env)


class _View:
    """ctx-like for the preview: preferred catalog view + the shared SDB."""

    def __init__(self, ctx, catalog):
        self.catalog, self.sdb, self.paks = catalog, ctx.sdb, ctx.paks


# ---- dialogs / delegates ----------------------------------------------------------------------------------------

class PickDialog(QDialog):
    """Search box + list. `fetch(text) -> [str]` runs in the task runner."""

    def __init__(self, parent, runner, title: str, fetch, free: bool = False, text: str = ""):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(460, 520)
        self.runner, self.fetch, self.free = runner, fetch, free
        self.key = ("build_models", "pick", next(_seq))
        lay = QVBoxLayout(self)
        self.edit = QLineEdit(text)
        self.edit.setPlaceholderText("search…")
        self.edit.setClearButtonEnabled(True)
        lay.addWidget(self.edit)
        self.list = QListWidget()
        self.list.setFont(mono_font())
        self.list.itemDoubleClicked.connect(lambda _i: self.accept())
        lay.addWidget(self.list, 1)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._deb = Debouncer(self._run, 150, self)
        self.edit.textChanged.connect(self._deb.trigger)
        self._run()

    def _run(self):
        self.runner.submit(self.key, self.fetch, self.edit.text(), on_done=self._fill)

    def _fill(self, names):
        self.list.clear()
        self.list.addItems(names)
        if names:
            self.list.setCurrentRow(0)

    def value(self) -> str | None:
        """Selected entry; with *free*, the typed text when it matches nothing listed (new names)."""
        it = self.list.currentItem()
        typed = self.edit.text().strip()
        if self.free and typed:
            exact = self.list.findItems(typed, Qt.MatchFixedString)
            if exact:
                return exact[0].text()
            if self.list.count() == 0:
                return typed
        return it.text() if it is not None else None

    def done(self, r):
        self.runner.cancel(self.key)
        super().done(r)


class TwoLineDelegate(QStyledItemDelegate):
    """Override list rows: label + grey sub line."""

    def paint(self, painter, option, index):
        self.initStyleOption(option, index)
        sub = index.data(ROLE_SUB) or ""
        widget = option.widget
        style = widget.style()
        text = option.text
        option.text = ""
        style.drawControl(QStyle.CE_ItemViewItem, option, painter, widget)
        r = style.subElementRect(QStyle.SE_ItemViewItemText, option, widget).adjusted(2, 2, -4, -2)
        h = r.height() // 2
        selected = bool(option.state & QStyle.State_Selected)
        fg = index.data(Qt.ForegroundRole)
        pal = option.palette
        pen = pal.color(QPalette.HighlightedText if selected else QPalette.Text)
        if isinstance(fg, QBrush) and not selected:
            pen = fg.color()
        painter.save()
        painter.setPen(pen)
        painter.drawText(r.adjusted(0, 0, 0, -h), Qt.AlignLeft | Qt.AlignVCenter,
                         option.fontMetrics.elidedText(text, Qt.ElideRight, r.width()))
        painter.setPen(pen if selected else OFF)
        painter.drawText(r.adjusted(10, h, 0, 0), Qt.AlignLeft | Qt.AlignVCenter,
                         option.fontMetrics.elidedText(sub, Qt.ElideRight, r.width() - 10))
        painter.restore()

    def sizeHint(self, option, index):
        s = super().sizeHint(option, index)
        return QSize(s.width(), option.fontMetrics.height() * 2 + 10)


def _item(text, color=None, mono=False, bold=False):
    it = QTableWidgetItem(text)
    it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
    if color is not None:
        it.setForeground(QBrush(color))
    if mono or bold:
        f = mono_font() if mono else it.font()
        f.setBold(bold)
        it.setFont(f)
    return it


def _centered(w: QWidget) -> QWidget:
    box = QWidget()
    lay = QHBoxLayout(box)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setAlignment(Qt.AlignCenter)
    lay.addWidget(w)
    return box


def _color_combo(c: QComboBox, color: QColor | None) -> None:
    c.setStyleSheet(f"QComboBox {{ color: {color.name()}; }}" if color is not None else "")


def _menu_button(text: str, entries, handler, parent) -> QToolButton:
    b = QToolButton(parent)
    b.setText(text)
    b.setPopupMode(QToolButton.InstantPopup)
    m = QMenu(b)
    for t in entries:
        if t is None:
            m.addSeparator()
        else:
            m.addAction(t, lambda t=t: handler(t))
    b.setMenu(m)
    return b


# ---- the page -----------------------------------------------------------------------------------------------------

class ModelOverridesPage(QWidget):
    def __init__(self, ctx, state, parent=None):
        super().__init__(parent)
        self.ctx, self.state = ctx, state
        self.mo: P.ModelOverride | None = None
        self.roles: list[str] = []                 # roles shown in the grid (distinct members)
        self.slot_names: list[str] = []
        self.sel_slot: str | None = None
        self.cells: dict[tuple[str, str], tuple[QCheckBox, QComboBox]] = {}
        self.problems: list[P.Problem] = []
        self.preview_packs: list[tuple[Path, int]] = []
        self.preview_out: dict | None = None
        self._stash: dict[str, dict[str, P.ModelRole]] = {}
        self._mat_cache: dict[str, list[str] | None] = {}
        self._mat_rows: list[tuple[str, str | None]] = []
        self._appearances: list[dict] | None = None
        self._touching = False
        self._filter = "all"
        self._ghost_img = QImage(4, 4, QImage.Format_RGBA8888)
        self._ghost_img.fill(QColor(190, 190, 200, 60))
        self._key = ("build_models", next(_seq))
        self._sweep_temp()
        self._preview_deb = Debouncer(self._run_preview, 300, self)
        self._check_deb = Debouncer(self._run_checks, 400, self)

        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)
        split.addWidget(self._left())
        split.addWidget(self._center())
        split.addWidget(self._right())
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 3)
        split.setStretchFactor(2, 2)
        split.setSizes([290, 760, 550])

        state.modelsChanged.connect(self._on_models_changed)
        state.projectReplaced.connect(self._on_project_replaced)
        state.itemsChanged.connect(self._on_items_changed)
        state.settingsChanged.connect(self._sync_output)
        state.problemsChanged.connect(self._on_problems)
        state.built.connect(self._on_built)
        if state.last_report:
            self._on_built(state.last_report)
        self.rebuild()

    # ================================================================================================ layout ====
    def _left(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 4, 0)
        row = QHBoxLayout()
        prof = _profile(self.ctx)
        new_items = ("Player…",) if prof.appearances_script else ()
        new_items += ("Pick .model…", None, "Duplicate", "Rename", "Remove")
        row.addWidget(_menu_button("New ▾", new_items,
                                   self._on_new_menu, w))
        row.addStretch(1)
        lay.addLayout(row)

        self.list = QListWidget()
        self.list.setItemDelegate(TwoLineDelegate(self.list))
        self.list.setEditTriggers(QAbstractItemView.EditKeyPressed | QAbstractItemView.DoubleClicked)
        self.list.currentItemChanged.connect(self._on_current)
        self.list.itemChanged.connect(self._on_item_changed)
        lay.addWidget(self.list, 1)

        box = QGroupBox("Models")
        f = QFormLayout(box)
        f.setContentsMargins(6, 6, 6, 6)
        self.role_rows: dict[str, tuple[QCheckBox, QLineEdit, QPushButton]] = {}
        for role in P.ROLES:
            r = QHBoxLayout()
            cb = QCheckBox()
            cb.toggled.connect(lambda on, role=role: self._on_role_toggled(role, on))
            e = QLineEdit()
            e.setFont(mono_font())
            e.editingFinished.connect(lambda role=role: self._on_role_edited(role))
            b = QPushButton("…")
            b.setFixedWidth(28)
            b.clicked.connect(lambda _=False, role=role: self._pick_role(role))
            r.addWidget(cb)
            r.addWidget(e, 1)
            r.addWidget(b)
            f.addRow(ROLE_LABEL[role], r)
            self.role_rows[role] = (cb, e, b)
        prow = QHBoxLayout()
        self.btn_pair = QPushButton("Pair")
        self.btn_pair.clicked.connect(lambda: self.pair())
        prow.addWidget(self.btn_pair)
        prow.addStretch(1)
        f.addRow("", prow)
        self.source = QLabel("—")
        f.addRow("Source", self.source)
        lay.addWidget(box)

        out = QGroupBox("Output")
        f2 = QFormLayout(out)
        f2.setContentsMargins(6, 6, 6, 6)
        self.pak = QComboBox()
        self.pak.setEditable(True)
        self.pak.setInsertPolicy(QComboBox.NoInsert)
        self.pak.lineEdit().editingFinished.connect(self._on_pak_edited)
        f2.addRow("PAK", self.pak)
        self.path_mode = QComboBox()
        for m in P.PATH_MODES:
            self.path_mode.addItem(m.capitalize(), m)
        self.path_mode.currentIndexChanged.connect(self._on_path_mode)
        f2.addRow("Path", self.path_mode)
        self.no_gear = QCheckBox("No gear")
        self.no_gear.setToolTip("Gear stops replacing slots (empty player_outfit_slots.scr)")
        self.no_gear.setVisible(bool(_profile(self.ctx).outfit_script))
        self.no_gear.toggled.connect(self._on_no_gear)
        f2.addRow("", self.no_gear)
        lay.addWidget(out)
        self.left_boxes = (box, out)
        return w

    def _center(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 4, 0)
        bar = QHBoxLayout()
        self.link = QCheckBox("Link")
        self.link.setChecked(False)
        bar.addWidget(self.link)
        bar.addSpacing(8)
        self.filter_bar = QHBoxLayout()
        self.filter_bar.setSpacing(2)
        bar.addLayout(self.filter_bar)
        self.filter_group = QButtonGroup(w)
        self.filter_group.setExclusive(True)
        self.filter_group.buttonClicked.connect(self._on_filter)
        bar.addStretch(1)
        self.slot_menu = _menu_button("Slot ▾", ("Move to…", "New slot", None, "Copy TPP → FPP", "Copy FPP → TPP", None, "TPP → FPP (all)", None, "Revert slot",
                                                 "Revert all"), self._on_slot_menu, w)
        bar.addWidget(self.slot_menu)
        lay.addLayout(bar)

        self.grid = QTableWidget(0, 2)
        self._grid_wheel = WheelGuard(self.grid)
        self.grid.verticalHeader().hide()
        self.grid.verticalHeader().setDefaultSectionSize(26)
        self.grid.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.grid.setSelectionMode(QAbstractItemView.SingleSelection)
        self.grid.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.grid.setAlternatingRowColors(True)
        self.grid.currentCellChanged.connect(self._on_grid_row)
        lay.addWidget(self.grid, 1)

        legend = QLabel(f"<span style='color:{CHANGED.name()}'>●</span> changed &nbsp; "
                        f"<span style='color:{NEW.name()}'>★</span> project &nbsp; "
                        f"<span style='color:{OFF.name()}'>grey</span> off")
        lay.addWidget(legend)

        self.bottom = QTabWidget()
        self.bottom.setDocumentMode(True)
        self.diff = QPlainTextEdit()
        self.diff.setReadOnly(True)
        self.diff.setFont(mono_font())
        self.diff.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.bottom.addTab(self.diff, "Diff")
        self.checks = QPlainTextEdit()
        self.checks.setReadOnly(True)
        self.checks.setFont(mono_font())
        self.bottom.addTab(self.checks, "Checks")
        self.bottom.setMinimumHeight(120)
        self.bottom.setMaximumHeight(200)
        lay.addWidget(self.bottom)
        return w

    def _right(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(4, 0, 0, 0)
        bar = QHBoxLayout()
        bar.setSpacing(4)
        self.view_group = QButtonGroup(w)
        self.view_btns = {}
        for i, (key, t) in enumerate((("tpp", "TPP"), ("fpp", "FPP"), ("side", "Side by side"))):
            b = QToolButton()
            b.setText(t)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.view_group.addButton(b)
            self.view_btns[key] = b
            bar.addWidget(b)
        self.view_group.buttonClicked.connect(lambda _b: self._preview_deb.trigger())
        bar.addStretch(1)
        self.ghost = QCheckBox("Ghost off")
        self.ghost.toggled.connect(self._preview_deb.trigger)
        bar.addWidget(self.ghost)
        self.original = QCheckBox("Original")
        self.original.toggled.connect(self._preview_deb.trigger)
        bar.addWidget(self.original)
        self.variant = QComboBox()
        self.variant.addItem("Default")
        self.variant.setMinimumContentsLength(10)
        self.variant.currentTextChanged.connect(self._preview_deb.trigger)
        bar.addWidget(self.variant)
        self.btn_open = QPushButton("Open in Models")
        self.btn_open.clicked.connect(self.open_in_models)
        bar.addWidget(self.btn_open)
        lay.addLayout(bar)

        self.meshview = None
        try:
            from ..meshview import MeshView
            self.meshview = MeshView()
            self.meshview.setMinimumHeight(220)
            lay.addWidget(self.meshview, 3)
        except Exception as exc:  # noqa: BLE001 - viewer optional
            ph = QLabel(f"3D preview unavailable ({type(exc).__name__})")
            ph.setAlignment(Qt.AlignCenter)
            ph.setMinimumHeight(220)
            lay.addWidget(ph, 3)
        self.preview_status = QLabel("")
        self.preview_status.setStyleSheet(f"color: {OFF.name()};")
        lay.addWidget(self.preview_status)

        self.mat_box = QGroupBox("Materials")
        ml = QVBoxLayout(self.mat_box)
        ml.setContentsMargins(6, 6, 6, 6)
        side = QHBoxLayout()
        self.mat_roles = QComboBox()
        self.mat_roles.currentIndexChanged.connect(lambda _i: self._refresh_materials())
        side.addWidget(self.mat_roles)
        side.addStretch(1)
        self.btn_base = QPushButton("Base…")
        self.btn_base.clicked.connect(self._pick_base)
        side.addWidget(self.btn_base)
        self.btn_add = _menu_button("Add ▾", list(ADD_PARAMS) + ["Other…"], self._add_param, w)
        side.addWidget(self.btn_add)
        self.btn_del = QToolButton()
        self.btn_del.setText("Remove")
        self.btn_del.clicked.connect(self._delete_param)
        side.addWidget(self.btn_del)
        ml.addLayout(side)
        self.mats = QTableWidget(0, 4)
        self.mats.setHorizontalHeaderLabels(["Material", "Param", "Value", "Src"])
        self.mats.verticalHeader().hide()
        self.mats.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.mats.setSelectionMode(QAbstractItemView.SingleSelection)
        self.mats.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed)
        self.mats.setWordWrap(False)
        h = self.mats.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Interactive)
        self.mats.setColumnWidth(0, 230)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.mats.itemChanged.connect(self._on_mat_item)
        QShortcut(QKeySequence.Delete, self.mats, self._delete_param, context=Qt.WidgetWithChildrenShortcut)
        ml.addWidget(self.mats)
        lay.addWidget(self.mat_box, 2)
        return w

    # ============================================================================================ project data ====
    def project_meshes(self) -> set[str]:
        return {n.lower() for n in self.project_mesh_names()}

    def project_mesh_names(self) -> list[str]:
        out = []
        for it in self.state.project.items:
            if it.enabled and it.kind == "mesh":
                out.append(it.output_name)
            elif it.enabled and it.kind == "scene":
                out += list((it.options.get("renames") or {}).values())
        return list(dict.fromkeys(out))

    def project_textures(self) -> list[str]:
        return list(dict.fromkeys(it.output_name for it in self.state.project.items
                                  if it.enabled and it.kind == "texture"))

    def view(self) -> _View:
        ids = [i for _, i in reversed(self.preview_packs)]
        return _View(self.ctx, self.ctx.catalog.prefer(ids))

    # ============================================================================================== rebuild ====
    def rebuild(self) -> None:
        """Everything from state.project.models (keeps the current override when it still exists)."""
        cur = self.mo.id if self.mo is not None else None
        for mo in self.state.project.models:
            if any(r.original is None for r in mo.roles.values()):
                try:
                    P.ensure_originals(mo, self.state.env)
                except Exception as exc:  # noqa: BLE001
                    self.ctx.status.emit(f"{mo.label}: {exc}")
        self.list.blockSignals(True)
        self.list.clear()
        for mo in self.state.project.models:
            it = QListWidgetItem(mo.label)
            it.setData(ROLE_ID, mo.id)
            it.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if mo.enabled else Qt.Unchecked)
            self.list.addItem(it)
        ids = [m.id for m in self.state.project.models]
        row = ids.index(cur) if cur in ids else (0 if ids else -1)
        self.list.setCurrentRow(row)
        self.list.blockSignals(False)
        self.select(self._override(ids[row]) if row >= 0 else None)
        for i in range(self.list.count()):
            self._update_list_item(self.list.item(i))

    def _override(self, oid: str) -> P.ModelOverride | None:
        return next((m for m in self.state.project.models if m.id == oid), None)

    def _on_current(self, cur, _prev=None) -> None:
        self.select(self._override(cur.data(ROLE_ID)) if cur is not None else None)

    def select(self, mo: P.ModelOverride | None) -> None:
        self.mo = mo
        if mo is not None:
            try:
                P.ensure_originals(mo, self.state.env)
            except Exception as exc:  # noqa: BLE001
                self.ctx.status.emit(f"{mo.label}: {exc}")
        for b in self.left_boxes:
            b.setEnabled(mo is not None)
        self._sync_roles_box()
        self._sync_output()
        self._rebuild_grid()
        self._refresh_diff()
        self._refresh_checks()
        self._refresh_materials()
        self._check_deb.trigger()
        self._preview_deb.trigger()

    def _update_list_item(self, it: QListWidgetItem | None) -> None:
        if it is None:
            return
        mo = self._override(it.data(ROLE_ID))
        if mo is None:
            return
        ns, nt = change_counts(mo)
        parts = []
        if not mo.roles:
            parts.append("empty")
        if ns:
            parts.append(f"{ns} slot{'s' * (ns != 1)}")
        if nt:
            parts.append(f"{nt} tex")
        if not ns and not nt and mo.roles:
            parts.append("unchanged")
        errs = [p for p in self.problems if p.where == mo.id]
        n_err = sum(p.level == "error" for p in errs)
        if n_err:
            parts.append(f"✗ {n_err}")
        elif len(errs) == 1:
            parts.append(f"⚠ {errs[0].message}")
        elif errs:
            parts.append(f"⚠ {len(errs)}")
        self.list.blockSignals(True)
        it.setData(ROLE_SUB, " · ".join(parts))
        it.setData(Qt.ForegroundRole, QBrush(ERROR) if n_err else QBrush(CHANGED) if errs else None)
        it.setText(mo.label)
        self.list.blockSignals(False)
        self.list.viewport().update()

    def _current_item(self) -> QListWidgetItem | None:
        if self.mo is None:
            return None
        for i in range(self.list.count()):
            if self.list.item(i).data(ROLE_ID) == self.mo.id:
                return self.list.item(i)
        return None

    # ---- state signals ------------------------------------------------------------------------------------------
    def _touch(self) -> None:
        self._touching = True
        try:
            self.state.touch("models")
        finally:
            self._touching = False

    def _on_models_changed(self) -> None:
        if not self._touching:
            self.rebuild()

    def _on_project_replaced(self) -> None:
        self._stash.clear()
        self.problems = []
        self.mo = None
        self.rebuild()

    def _on_items_changed(self) -> None:
        for name in self.slot_names:
            self._sync_row(name)
        self._refresh_materials()
        self._check_deb.trigger()
        self._preview_deb.trigger()

    def _on_problems(self, probs: list) -> None:
        self.problems = list(probs)
        self._refresh_checks()
        for i in range(self.list.count()):
            self._update_list_item(self.list.item(i))

    # ---- edits: common tail ------------------------------------------------------------------------------------
    def after_edit(self, slots_changed=None, materials: bool = True) -> None:
        self._touch()
        for name in (self.slot_names if slots_changed is None else slots_changed):
            self._sync_row(name)
        self._apply_filter()
        self._update_list_item(self._current_item())
        self._refresh_diff()
        if materials:
            self._refresh_materials()
        self._check_deb.trigger()
        self._preview_deb.trigger()

    # ============================================================================================= left box ====
    def _on_new_menu(self, t: str) -> None:
        if t == "Player…":
            apps = self.appearances()
            labels = [appearance_label(a) for a in apps]
            v = self._pick("Player", lambda text: filter_names(labels, text))
            if v in labels:
                self.new_player(apps[labels.index(v)])
        elif t == "Pick .model…":
            v = self._pick(".model", lambda text: filter_names(self._model_names(), text))
            if v:
                self.new_model(v)
        elif t == "Duplicate" and self.mo is not None:
            mo = copy.deepcopy(self.mo)
            mo.id = uuid.uuid4().hex[:8]
            mo.label = f"{self.mo.label} copy"
            self._add(mo)
        elif t == "Rename":
            it = self._current_item()
            if it is not None:
                self.list.editItem(it)
        elif t == "Remove" and self.mo is not None:
            if QMessageBox.question(self, "Remove", f"Remove {self.mo.label}?") == QMessageBox.Yes:
                self.remove_current()

    def _pick(self, title: str, fetch, free: bool = False, text: str = "") -> str | None:
        """Modal search dialog (tests replace this)."""
        dlg = PickDialog(self, self.ctx.runner, title, fetch, free=free, text=text)
        return dlg.value() if dlg.exec() == QDialog.Accepted else None

    def appearances(self) -> list[dict]:
        if self._appearances is None:
            try:
                self._appearances = read_appearances(self.ctx)
            except Exception as exc:  # noqa: BLE001
                self.ctx.status.emit(f"{_profile(self.ctx).appearances_script}: {exc}")
                return []
        return self._appearances

    def _model_names(self) -> list[str]:
        return list(dict.fromkeys(r["name"] for r in self.ctx.paks.models()))

    def _load_roles(self, members: dict[str, str]) -> dict[str, P.ModelRole]:
        """{role: member name} -> loaded roles (first provider of each). Unknown names are reported and skipped."""
        by_pak: dict[Path, dict[str, str]] = {}
        for role, name in members.items():
            if not name:
                continue
            rec = find_model_rec(self.ctx, name)
            if rec is None:
                self.ctx.status.emit(f"{name} not found")
                continue
            by_pak.setdefault(Path(rec["pak"]), {})[role] = rec["name"]
        out = {}
        for pak, mem in by_pak.items():
            try:
                out.update(P.load_override("", mem, pak).roles)
            except Exception as exc:  # noqa: BLE001
                self.ctx.status.emit(f"{pak.name}: {exc}")
        return out

    def new_player(self, app: dict) -> P.ModelOverride | None:
        roles = self._load_roles({r: app.get(r) for r in P.ROLES})
        if not roles:
            return None
        mo = P.ModelOverride(label=appearance_label(app), roles=roles)
        self._add(mo)
        return mo

    def new_model(self, name: str) -> P.ModelOverride | None:
        role = "fpp" if "fpp" in bare(name).lower() else "tpp"
        roles = self._load_roles({role: name})
        if not roles:
            return None
        mo = P.ModelOverride(label=bare(name).rsplit(".", 1)[0], roles=roles)
        self._add(mo)
        if role == "tpp":
            self.pair(quiet=True)
        return mo

    def _add(self, mo: P.ModelOverride) -> None:
        mo.no_gear = mo.is_player() and bool(_profile(self.ctx).outfit_script)
        self.state.project.models.append(mo)
        self.mo = mo
        self._touch()
        self.rebuild()

    def remove_current(self) -> None:
        if self.mo is None:
            return
        self.state.project.models = [m for m in self.state.project.models if m.id != self.mo.id]
        self.mo = None
        self._touch()
        self.rebuild()

    def _on_item_changed(self, it: QListWidgetItem) -> None:
        mo = self._override(it.data(ROLE_ID))
        if mo is None:
            return
        label = it.text().strip() or mo.label
        on = it.checkState() == Qt.Checked
        if label != mo.label or on != mo.enabled:
            mo.label, mo.enabled = label, on
            self._touch()
            self._check_deb.trigger()
        self._update_list_item(it)

    # ---- roles ----------------------------------------------------------------------------------------------------
    def _sync_roles_box(self) -> None:
        mo = self.mo
        for role, (cb, e, b) in self.role_rows.items():
            r = mo.roles.get(role) if mo else None
            for x in (cb, e):
                x.blockSignals(True)
            cb.setChecked(r is not None)
            e.setText(r.member if r else "")
            e.setCursorPosition(0)
            e.setToolTip(r.member if r else "")
            e.setCursorPosition(len(e.text()))        # the file name end is the informative part
            e.setToolTip(r.member if r else "")
            e.setEnabled(r is not None)
            for x in (cb, e):
                x.blockSignals(False)
        srcs = list(dict.fromkeys(r.source_pak for r in mo.roles.values())) if mo else []
        self.source.setText(", ".join(srcs) or "—")
        self.btn_pair.setEnabled(bool(mo and "tpp" in mo.roles))

    def _on_role_toggled(self, role: str, on: bool) -> None:
        mo = self.mo
        if mo is None:
            return
        stash = self._stash.setdefault(mo.id, {})
        if not on:
            r = mo.roles.pop(role, None)
            if r is not None:
                stash[role] = r
        elif role in stash:
            mo.roles[role] = stash.pop(role)
        else:
            self._pick_role(role)
            return
        self._roles_changed()

    def _on_role_edited(self, role: str) -> None:
        mo = self.mo
        e = self.role_rows[role][1]
        if mo is None or not e.isEnabled():
            return
        name = e.text().strip()
        cur = mo.roles.get(role)
        if cur is not None and name == cur.member:
            return
        if not name:
            self._sync_roles_box()
            return
        self.set_role(role, name)

    def _pick_role(self, role: str) -> None:
        if self.mo is None:
            return
        cur = self.mo.roles.get(role)
        v = self._pick(ROLE_LABEL[role], lambda text: filter_names(self._model_names(), text),
                       text=bare(cur.member).rsplit(".", 1)[0] if cur else "")
        if v:
            self.set_role(role, v)
        else:
            self._sync_roles_box()

    def set_role(self, role: str, name: str) -> bool:
        if self.mo is None:
            return False
        got = self._load_roles({role: name})
        if role not in got:
            self._sync_roles_box()
            return False
        self.mo.roles[role] = got[role]
        self._stash.get(self.mo.id, {}).pop(role, None)
        self._roles_changed()
        return True

    def _roles_changed(self) -> None:
        self._touch()
        self._sync_roles_box()
        self._rebuild_grid()
        self._update_list_item(self._current_item())
        self._refresh_diff()
        self._refresh_materials()
        self._check_deb.trigger()
        self._preview_deb.trigger()

    def pair(self, quiet: bool = False) -> int:
        """Fill the other roles from the appearance script entry that uses the current TPP model."""
        mo = self.mo
        if mo is None or "tpp" not in mo.roles:
            return 0
        tpp = bare(mo.roles["tpp"].member).lower()
        app = next((a for a in self.appearances() if (a.get("tpp") or "").lower() == tpp), None)
        if app is None:
            if not quiet:
                self.ctx.status.emit(f"no appearance uses {bare(mo.roles['tpp'].member)}")
            return 0
        want = {r: app[r] for r in ("fpp", "lodcc", "ui")
                if app.get(r) and (r not in mo.roles or bare(mo.roles[r].member).lower() != app[r].lower())}
        got = self._load_roles(want)
        mo.roles.update(got)
        if got:
            self._roles_changed()
        return len(got)

    # ---- output -----------------------------------------------------------------------------------------------------
    def _sync_output(self) -> None:
        self.pak.blockSignals(True)
        self.pak.clear()
        self.pak.addItem(self.state.pak_name())
        self.pak.setCurrentIndex(0)
        self.pak.blockSignals(False)
        self.path_mode.blockSignals(True)
        self.path_mode.setCurrentIndex(max(0, self.path_mode.findData(self.mo.path_mode if self.mo else "root")))
        self.path_mode.blockSignals(False)
        self.no_gear.blockSignals(True)
        self.no_gear.setChecked(bool(self.mo and self.mo.no_gear))
        self.no_gear.setEnabled(self.mo is not None and bool(_profile(self.ctx).outfit_script))
        self.no_gear.blockSignals(False)

    def _on_no_gear(self, on: bool) -> None:
        if self.mo is not None and bool(on) != self.mo.no_gear:
            self.mo.no_gear = bool(on)
            self._touch()
            self._check_deb.trigger()

    def _on_pak_edited(self) -> None:
        val = self.pak.currentText().strip()
        if val == self.state.default_pak():
            val = ""
        if val != self.state.project.pak_name:
            self.state.project.pak_name = val
            self.state.touch("settings")

    def _on_path_mode(self, _i: int) -> None:
        if self.mo is not None and self.path_mode.currentData() != self.mo.path_mode:
            self.mo.path_mode = self.path_mode.currentData()
            self._touch()
            self._check_deb.trigger()

    # ================================================================================================ grid ====
    def shown_roles(self) -> list[str]:
        """Roles with a distinct member (LodCC / UI only when they differ from TPP / FPP)."""
        if self.mo is None:
            return []
        seen, out = set(), []
        for role in P.ROLES:
            r = self.mo.roles.get(role)
            if r is None or r.doc is None or r.member.lower() in seen:
                continue
            seen.add(r.member.lower())
            out.append(role)
        return out

    def doc(self, role: str) -> dict | None:
        r = self.mo.roles.get(role) if self.mo else None
        return r.doc if r else None

    def _rebuild_grid(self) -> None:
        self.roles = self.shown_roles()
        names: list[str] = []
        filters: list[str] = []
        filt: dict[str, str] = {}
        for role in self.roles:
            for s in P.slots(self.doc(role)):
                n = s.get("name")
                ft = (s.get("filterText") or "").lower()
                if n not in names:
                    names.append(n)
                filt.setdefault(n, ft)
                if ft and ft not in filters:
                    filters.append(ft)
        self.slot_names = names
        g = self.grid
        g.blockSignals(True)
        g.clearContents()
        headers = ["Slot", "Filter"]
        for role in self.roles:
            headers += [ROLE_LABEL[role], f"{ROLE_LABEL[role]} mesh"]
        g.setColumnCount(len(headers))
        g.setHorizontalHeaderLabels(headers)
        g.setRowCount(len(names))
        self.cells.clear()
        for r, name in enumerate(names):
            g.setItem(r, 0, _item(name, mono=True))
            g.setItem(r, 1, _item(filt.get(name, ""), OFF))
            for k, role in enumerate(self.roles):
                cb = QCheckBox()
                cb.toggled.connect(lambda on, n=name, ro=role: self.set_enabled(n, ro, on))
                combo = QComboBox()
                combo.setEditable(True)
                combo.setInsertPolicy(QComboBox.NoInsert)
                combo.setFont(mono_font())
                combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
                combo.setMinimumContentsLength(12)
                combo.activated.connect(lambda i, n=name, ro=role, c=combo: self._on_mesh_combo(n, ro, c.itemText(i)))
                combo.lineEdit().editingFinished.connect(
                    lambda n=name, ro=role, c=combo: self._on_mesh_combo(n, ro, c.currentText()))
                for wdg in (cb, combo, combo.lineEdit()):
                    wdg.installEventFilter(self)
                    wdg.setProperty("slot_row", name)
                self._grid_wheel.guard(cb, combo, combo.lineEdit())
                g.setCellWidget(r, 2 + 2 * k, _centered(cb))
                g.setCellWidget(r, 3 + 2 * k, combo)
                self.cells[(name, role)] = (cb, combo)
            self._sync_row(name)
        h = g.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        for k in range(len(self.roles)):
            h.setSectionResizeMode(2 + 2 * k, QHeaderView.ResizeToContents)
            h.setSectionResizeMode(3 + 2 * k, QHeaderView.Stretch)
        g.blockSignals(False)
        self._rebuild_filters(filters)
        if self.sel_slot not in names:
            self.sel_slot = names[0] if names else None
        if self.sel_slot is not None:
            g.blockSignals(True)
            g.selectRow(names.index(self.sel_slot))
            g.blockSignals(False)
        has2 = "tpp" in self.roles and "fpp" in self.roles
        acts = self.slot_menu.menu().actions()
        acts[0].setEnabled(has2)
        acts[1].setEnabled(has2)
        self.view_btns["tpp"].setEnabled("tpp" in self.roles)
        self.view_btns["fpp"].setEnabled("fpp" in self.roles)
        self.view_btns["side"].setEnabled(has2)
        cur = self.view_group.checkedButton()
        if self.roles and (cur is None or not cur.isEnabled()):
            first = next((k for k in ("tpp", "fpp") if k in self.roles), None)
            if first:
                self.view_btns[first].setChecked(True)
        self._fill_mat_roles()

    def _rebuild_filters(self, filters: list[str]) -> None:
        for b in list(self.filter_group.buttons()):
            self.filter_group.removeButton(b)
            b.deleteLater()
        keys = ["all"] + filters + ["changed"] if self.slot_names else []
        if self._filter not in keys:
            self._filter = "all"
        for k in keys:
            b = QToolButton()
            b.setText(k)
            b.setCheckable(True)
            b.setChecked(k == self._filter)
            b.setProperty("filter", k)
            self.filter_group.addButton(b)
            self.filter_bar.addWidget(b)
        self._apply_filter()

    def _on_filter(self, b) -> None:
        self._filter = b.property("filter")
        self._apply_filter()

    def set_filter(self, key: str) -> None:
        for b in self.filter_group.buttons():
            if b.property("filter") == key:
                b.setChecked(True)
        self._filter = key
        self._apply_filter()

    def _apply_filter(self) -> None:
        f = self._filter
        for r, name in enumerate(self.slot_names):
            ft = (self.grid.item(r, 1).text() if self.grid.item(r, 1) else "")
            show = f == "all" or (f == "changed" and self.slot_changed(name)) or ft == f
            self.grid.setRowHidden(r, not show)

    def slot_changed(self, name: str) -> bool:
        for role in self.roles:
            r = self.mo.roles[role]
            if r.original is not None and slot_or_none(r.doc, name) != slot_or_none(r.original, name):
                return True
        return False

    @staticmethod
    def mesh_text(mesh: str | None, proj: set[str]) -> str:
        if mesh is None:
            return NONE
        return (STAR if stem(mesh).lower() in proj else "") + mesh

    def _sync_row(self, name: str) -> None:
        if name not in self.slot_names:
            return
        r = self.slot_names.index(name)
        changed = self.slot_changed(name)
        it = self.grid.item(r, 0)
        if it is not None:
            it.setText(("● " if changed else "   ") + name)
            it.setForeground(QBrush(CHANGED) if changed else QBrush())
        proj_names = self.project_mesh_names()
        proj = {n.lower() for n in proj_names}
        for role in self.roles:
            cb, combo = self.cells[(name, role)]
            s = slot_or_none(self.doc(role), name)
            cb.blockSignals(True)
            combo.blockSignals(True)
            combo.clear()
            if s is None:
                cb.setChecked(False)
                cb.setEnabled(False)
                combo.setEnabled(False)
                _color_combo(combo, None)
            else:
                mesh = P.slot_mesh(s)
                stash = self.mo.roles[role].stash
                alts = P.slot_alternatives(s, stash)
                cb.setEnabled(bool(alts))
                cb.setChecked(mesh is not None)
                combo.setEnabled(True)
                combo.addItems([self.mesh_text(a, proj) for a in alts])
                combo.addItem(NONE)
                low = {stem(a).lower() for a in alts}
                combo.addItems([STAR + n for n in proj_names if n.lower() not in low])
                combo.addItem(BROWSE)
                cur = self.mesh_text(mesh, proj)
                combo.setCurrentIndex(combo.findText(cur))
                combo.setEditText(cur)
                combo.lineEdit().setCursorPosition(0)
                _color_combo(combo, NEW if mesh and stem(mesh).lower() in proj else OFF if mesh is None else None)
            cb.blockSignals(False)
            combo.blockSignals(False)

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.FocusIn:
            name = obj.property("slot_row")
            if name and name in self.slot_names and name != self.sel_slot:
                self.grid.selectRow(self.slot_names.index(name))
        return super().eventFilter(obj, ev)

    def _on_grid_row(self, row, _c, _pr, _pc) -> None:
        if 0 <= row < len(self.slot_names):
            name = self.slot_names[row]
            if name != self.sel_slot:
                self.sel_slot = name
                self._refresh_materials()

    def select_slot(self, name: str) -> None:
        self.grid.selectRow(self.slot_names.index(name))

    def targets(self, name: str, role: str) -> list[str]:
        if self.link.isChecked():
            return [r for r in self.roles if slot_or_none(self.doc(r), name) is not None]
        return [role] if slot_or_none(self.doc(role), name) is not None else []

    def set_enabled(self, name: str, role: str, on: bool) -> None:
        for r in self.targets(name, role):
            P.set_slot_enabled(self.doc(r), name, on, self.mo.roles[r].stash)
        self.after_edit([name])

    def _on_mesh_combo(self, name: str, role: str, text: str) -> None:
        s = slot_or_none(self.doc(role), name)
        if s is None:
            return
        cur = P.slot_mesh(s)
        text = text.strip()
        if text == BROWSE:
            v = self._pick("Mesh", lambda t: catalog_names(self.ctx.catalog, t, 0x10), free=True,
                           text=stem(cur or ""))
            if not v:
                self._sync_row(name)
                return
            text = v
        mesh = None if text in (NONE, "") else unstar(text)
        if mesh is None and cur is None:
            return
        if mesh is not None and cur is not None and P._msh(mesh).lower() == cur.lower():
            return
        self.set_mesh(name, role, mesh)

    def set_mesh(self, name: str, role: str, mesh: str | None) -> None:
        for r in self.targets(name, role):
            P.set_slot_mesh(self.doc(r), name, mesh, self.mo.roles[r].stash)
        self.after_edit([name])

    def _on_slot_menu(self, t: str) -> None:
        if self.mo is None:
            return
        name = self.sel_slot
        if t.startswith("Copy") and name:
            src, dst = ("tpp", "fpp") if t == "Copy TPP → FPP" else ("fpp", "tpp")
            a, b = slot_or_none(self.doc(src), name), slot_or_none(self.doc(dst), name)
            if a is None or b is None:
                return
            b["meshResources"] = copy.deepcopy(a.get("meshResources") or {"resources": []})
            sa, sb = self.mo.roles[src].stash, self.mo.roles[dst].stash
            if name in sa:
                sb[name] = copy.deepcopy(sa[name])
            else:
                sb.pop(name, None)
            self.after_edit([name])
        elif t == "Revert slot" and name:
            self.revert_slot(name)
        elif t == "Move to…" and name:
            self.move_to(name)
        elif t == "New slot":
            self.new_slot()
        elif t == "TPP → FPP (all)":
            self.mirror_tpp()
        elif t == "Revert all":
            self.revert_all()

    def _slot_role(self) -> str:
        return "tpp" if "tpp" in self.roles else (self.roles[0] if self.roles else "tpp")

    def move_to(self, name: str, dst: str | None = None) -> None:
        """Move the slot's mesh (with its texture swaps) to another slot, in every linked role."""
        role = self._slot_role()
        doc = self.doc(role)
        if doc is None:
            return
        if dst is None:
            names = [x.get("name") for x in P.slots(doc) if x.get("name") != name] + [P.next_torso_slot(doc)]
            dst = self._pick("Slot", lambda t: [n for n in names if t.lower() in n.lower()], free=True,
                             text=P.next_torso_slot(doc))
            if not dst:
                return
        roles = [r for r in self.roles if slot_or_none(self.doc(r), name) is not None] if self.link.isChecked() \
            else [role]
        for r in roles:
            P.move_slot_mesh(self.doc(r), name, dst.strip().upper(), self.mo.roles[r].stash)
        self._touch()
        self._rebuild_grid()

    def new_slot(self, name: str | None = None) -> None:
        role = self._slot_role()
        doc = self.doc(role)
        if doc is None:
            return
        name = name or P.next_torso_slot(doc)
        roles = self.roles if self.link.isChecked() else [role]
        for r in roles:
            P.add_slot(self.doc(r), name)
        self._touch()
        self._rebuild_grid()

    def mirror_tpp(self) -> None:
        """Every slot of the FPP document takes the TPP slot's meshes (and stash); slots only in TPP are added."""
        tr, fr = self.mo.roles.get("tpp"), self.mo.roles.get("fpp")
        if tr is None or fr is None or tr.doc is None or fr.doc is None:
            return
        P.mirror_role(tr, fr)
        self._touch()
        self._rebuild_grid()

    def revert_slot(self, name: str) -> None:
        for role in self.roles:
            r = self.mo.roles[role]
            orig = slot_or_none(r.original, name)
            ss = P.slots(r.doc)
            for i, s in enumerate(ss):
                if s.get("name") == name and orig is not None:
                    ss[i] = copy.deepcopy(orig)
                    r.stash.pop(name, None)
        self.after_edit([name])

    def revert_all(self) -> None:
        for r in self.mo.roles.values():
            if r.original is not None:
                r.doc = copy.deepcopy(r.original)
                r.stash.clear()
        self._touch()
        self._rebuild_grid()
        self.after_edit()

    # ============================================================================================ diff / checks ====
    def _refresh_diff(self) -> None:
        lines = []
        if self.mo is not None:
            for member, r in self.mo.members().items():
                lines.append(bare(member))
                if r.original is None:
                    lines.append("  original n/a")
                    continue
                d = P.diff_docs(r.original, r.doc)
                lines += [f"  {x}" for x in d] or ["  —"]
        self.diff.setPlainText("\n".join(lines))

    def _run_checks(self) -> None:
        try:
            proj = copy.deepcopy(self.state.project)
        except Exception:  # noqa: BLE001
            return
        self.ctx.runner.submit((self._key, "validate"), validate_job, proj, self.state.env,
                               on_done=self._on_problems, on_error=lambda _e: None)

    def _refresh_checks(self) -> None:
        mine = [p for p in self.problems if self.mo is not None and p.where == self.mo.id]
        members = sorted({r.member for r in self.mo.roles.values()}, key=len, reverse=True) if self.mo else []

        def short(msg: str) -> str:
            for m in members:
                msg = msg.replace(m, bare(m))
            return msg
        self.checks.setPlainText("\n".join(f"{'✗' if p.level == 'error' else '⚠'} {short(p.message)}" for p in mine))
        self.bottom.setTabText(1, f"Checks ({len(mine)})" if mine else "Checks")

    # ============================================================================================== preview ====
    def preview_roles(self) -> list[str]:
        b = self.view_group.checkedButton()
        key = next((k for k, v in self.view_btns.items() if v is b), "tpp")
        if key == "side":
            return [r for r in ("tpp", "fpp") if r in self.roles]
        if key in self.roles:
            return [key]
        return self.roles[:1]

    def needs_build(self) -> bool:
        if self.preview_packs or self.mo is None:
            return False
        meshes, texs = self.project_meshes(), {t.lower() for t in self.project_textures()}
        if not meshes and not texs:
            return False
        for role in self.roles:
            doc = self.doc(role)
            for s in P.slots(doc):
                m = P.slot_mesh(s)
                if m and stem(m).lower() in meshes:
                    return True
            if any((v[-1] or "").lower() in texs for v in tex_values(doc)):
                return True
        return False

    def _preview_docs(self) -> list[tuple[str, dict]]:
        out = []
        for role in self.preview_roles():
            r = self.mo.roles[role]
            d = r.original if self.original.isChecked() and r.original is not None else r.doc
            out.append((bare(r.member), copy.deepcopy(d)))
        return out

    def _run_preview(self) -> None:
        if self.mo is None or not self.roles:
            if self.meshview is not None:
                self.meshview.clear()
            self.preview_status.setText("")
            self.preview_out = None
            return
        v = self.variant.currentText()
        wait = [i for _, i in self.preview_packs]
        self.preview_status.setText("…")
        self.ctx.runner.submit((self._key, "preview"), preview_job, self.view(), self._preview_docs(),
                               None if v in ("", "Default") else v, self.ghost.isChecked(), wait,
                               on_done=self._on_preview,
                               on_error=lambda e: self.preview_status.setText(e.splitlines()[0]))

    def _on_preview(self, out: dict) -> None:
        self.preview_out = out
        names = ["Default"]
        for vw in out["views"]:
            names += [n for n in vw["resolution"].get("variants") or [] if n not in names]
        self.variant.blockSignals(True)
        cur = self.variant.currentText()
        self.variant.clear()
        self.variant.addItems(names)
        self.variant.setCurrentIndex(max(0, self.variant.findText(cur)))
        self.variant.setEnabled(len(names) > 1)
        self.variant.blockSignals(False)
        parts = sum(len(vw["preview"].get("items") or []) for vw in out["views"])
        errors = [e for vw in out["views"] for e in vw["preview"].get("errors") or []]
        mv = self.meshview
        if mv is not None:
            mv.clear()
            x = None
            for vw in out["views"]:
                items = [it for it in vw["preview"].get("items") or [] if len(it["geom"].positions)]
                off = vw["off"]
                slots = [s["name"] for s in vw["resolution"]["slots"]]
                if not items:
                    continue
                lo = min(float(it["geom"].positions[:, 0].min()) for it in items)
                hi = max(float(it["geom"].positions[:, 0].max()) for it in items)
                shift = 0.0 if x is None else x - lo
                ghost = self._ghost_img

                def style(it, off=off, ghost=ghost):
                    if it["slot"] in off:
                        return {"texture": ghost, "alpha_mode": "blend", "color": (0.6, 0.6, 0.65)}
                    return None
                for it in items:
                    it["key"] = f"{vw['tag']}|{it['key']}"
                add_preview_items(mv, items, slots, errors, offset=(shift, 0.0, 0.0) if shift else None,
                                  style=style)
                x = hi + shift + 0.25 * max(hi - lo, 0.2)
            mv.frame_all()
        text = f"{parts} parts"
        if errors:
            text += f" · {len(errors)} ⚠"
        if self.needs_build() and not self.original.isChecked():
            text += f" · <span style='color:{NEW.name()}'>Build to preview ★</span>"
        self.preview_status.setText(text)
        self.preview_status.setToolTip("\n".join(errors[:10]))

    def open_in_models(self) -> None:
        if self.mo is None or not self.roles:
            return
        roles = self.preview_roles()
        role = roles[-1] if self.view_btns["fpp"].isChecked() else roles[0]
        r = self.mo.roles[role]
        d = r.original if self.original.isChecked() and r.original is not None else r.doc
        self.ctx.openModelDoc.emit(bare(r.member), copy.deepcopy(d), [p for p, _ in self.preview_packs])

    # ---- preview packs --------------------------------------------------------------------------------------------
    @staticmethod
    def _sweep_temp() -> None:
        mine = f"{PREVIEW_PREFIX}{os.getpid()}_"
        for p in Path(tempfile.gettempdir()).glob(f"{PREVIEW_PREFIX}*.rpack"):
            if not p.name.startswith(mine):
                try:
                    p.unlink()
                except OSError:
                    pass                          # in use by another running explorer

    def _on_built(self, report: dict) -> None:
        rp = (((report or {}).get("outputs") or {}).get("rpack") or {}).get("path")
        if not rp or not Path(rp).is_file():
            return
        tmp = Path(tempfile.gettempdir())
        while True:
            dst = tmp / f"{PREVIEW_PREFIX}{os.getpid()}_{next(_seq)}.rpack"
            if not dst.exists():
                break
        try:
            shutil.copyfile(rp, dst)
        except OSError as exc:
            self.ctx.status.emit(f"preview copy failed: {exc}")
            return
        self.ctx.catalog.load([dst], user=True)
        ids = self.ctx.catalog.pack_ids([dst])
        if ids:
            self.preview_packs.append((dst, ids[0]))
        self._mat_cache.clear()
        self._refresh_materials()
        self._preview_deb.trigger()

    def shutdown(self) -> None:
        for key in ((self._key, "preview"), (self._key, "validate")):
            self.ctx.runner.cancel(key)
        self.ctx.runner.pool.waitForDone(3000)
        cat = self.ctx.catalog
        for path, pid in self.preview_packs:
            try:
                e = cat.packs[pid]
                if e.pack is not None:
                    e.pack.close()
            except Exception:  # noqa: BLE001
                pass
            try:
                path.unlink()
            except OSError:
                pass
        self.preview_packs = []

    # ============================================================================================= materials ====
    def _fill_mat_roles(self) -> None:
        self.mat_roles.blockSignals(True)
        cur = self.mat_roles.currentText()
        self.mat_roles.clear()
        if len(self.roles) > 1:
            self.mat_roles.addItem(" + ".join(ROLE_LABEL[r] for r in self.roles), list(self.roles))
        for r in self.roles:
            self.mat_roles.addItem(ROLE_LABEL[r], [r])
        self.mat_roles.setCurrentIndex(max(0, self.mat_roles.findText(cur)))
        self.mat_roles.blockSignals(False)

    def mat_targets(self) -> tuple[str | None, list[str]]:
        """(display role, roles edited). Other roles are edited only when they select the same mesh."""
        roles = self.mat_roles.currentData() or []
        name = self.sel_slot
        if self.mo is None or not name:
            return None, []
        shown = next((r for r in roles if selected_entry(slot_or_none(self.doc(r), name)) is not None), None)
        if shown is None:
            return None, []
        mesh = selected_entry(slot_or_none(self.doc(shown), name))["name"].lower()
        out = [r for r in roles
               if (selected_entry(slot_or_none(self.doc(r), name)) or {}).get("name", "").lower() == mesh]
        return shown, out

    def material_names(self, mres: dict) -> list[str]:
        mesh = mres.get("name", "")
        key = mesh.lower()
        listed = [m.get("name") for m in mres.get("materialsData") or [] if m.get("name")]
        if key not in self._mat_cache:
            self._mat_cache[key] = None
            self.ctx.runner.submit((self._key, "mats", key), mesh_materials, self.view(), mesh,
                                   on_done=lambda names, k=key: self._on_mat_names(k, names),
                                   on_error=lambda _e: None)
        names = self._mat_cache.get(key)
        if not names:
            return listed
        low = {n.lower() for n in names}
        return names + [n for n in listed if n.lower() not in low
                        and not n.lower().startswith(("auto_shadow_caster", "shadowcaster"))]

    def _on_mat_names(self, key: str, names) -> None:
        if not names:
            return
        self._mat_cache[key] = names
        shown, _ = self.mat_targets()
        if shown:
            ent = selected_entry(slot_or_none(self.doc(shown), self.sel_slot))
            if ent and ent.get("name", "").lower() == key:
                self._refresh_materials()

    def _refresh_materials(self) -> None:
        t = self.mats
        sel = self._current_mat_row()
        t.blockSignals(True)
        t.clearContents()
        t.setRowCount(0)
        self._mat_rows = []
        shown, _ = self.mat_targets()
        for b in (self.btn_add, self.btn_base, self.btn_del):
            b.setEnabled(shown is not None)
        if shown is None:
            self.mat_box.setTitle(f"{self.sel_slot} · off" if self.sel_slot else "Materials")
            t.blockSignals(False)
            return
        mres = selected_entry(slot_or_none(self.doc(shown), self.sel_slot))
        self.mat_box.setTitle(f"{self.sel_slot} · {mres.get('name')}")
        proj = {n.lower() for n in self.project_textures()}
        for m in self.material_names(mres):
            ent = P.material_entry(mres, m, create=False)
            base = ent.get("name") if ent else m
            vals = list((ent or {}).get("rttiValues") or [])
            label = m if (base or "").lower() == m.lower() else f"{m}  → {base}"
            for k, v in enumerate(vals or [None]):
                r = t.rowCount()
                t.insertRow(r)
                mi = _item(label if k == 0 else "", mono=True)
                mi.setToolTip(label)
                t.setItem(r, 0, mi)
                if v is None:
                    self._mat_rows.append((m, None))
                    for c in (1, 2, 3):
                        t.setItem(r, c, _item(""))
                    continue
                self._mat_rows.append((m, v.get("name")))
                t.setItem(r, 1, _item(v.get("name", ""), mono=True))
                val = P.rtti_value(v)
                if v.get("type") == 7:
                    is_proj = (val or "").lower() in proj
                    t.setItem(r, 2, _item(""))
                    t.setCellWidget(r, 2, self._tex_combo(m, v.get("name"), val or "", is_proj))
                    t.setItem(r, 3, _item("rpack" if is_proj else "game", NEW if is_proj else OFF))
                else:
                    it = QTableWidgetItem(fmt_value(val))
                    it.setFont(mono_font())
                    t.setItem(r, 2, it)
                    t.setItem(r, 3, _item(""))
        t.blockSignals(False)
        if sel is not None and sel in self._mat_rows:
            t.selectRow(self._mat_rows.index(sel))

    def _tex_combo(self, material: str, param: str, value: str, is_proj: bool) -> QComboBox:
        c = QComboBox()
        c.setEditable(True)
        c.setInsertPolicy(QComboBox.NoInsert)
        c.setFont(mono_font())
        WheelGuard(self.mats, c).guard(c, c.lineEdit())
        c.setMinimumContentsLength(10)
        c.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        cur = (STAR if is_proj else "") + value
        items = [cur] + [STAR + n for n in self.project_textures() if n.lower() != value.lower()] + [BROWSE]
        c.addItems(items)
        c.setCurrentIndex(0)
        c.lineEdit().setCursorPosition(0)
        _color_combo(c, NEW if is_proj else None)
        c.activated.connect(lambda i, c=c: self._on_tex_value(material, param, c.itemText(i), value))
        c.lineEdit().editingFinished.connect(lambda c=c: self._on_tex_value(material, param, c.currentText(), value))
        return c

    def _on_tex_value(self, material: str, param: str, text: str, old: str) -> None:
        text = text.strip()
        if text == BROWSE:
            v = self._pick("Texture", lambda t: catalog_names(self.ctx.catalog, t, 0x20), free=True,
                           text=Path(old).stem)
            if not v:
                self._refresh_materials()
                return
            text = v
        val = unstar(text)
        if not val or val == old:
            return
        self.set_param(material, param, val)

    def _current_mat_row(self) -> tuple[str, str | None] | None:
        r = self.mats.currentRow()
        if 0 <= r < len(self._mat_rows):
            return self._mat_rows[r]
        return None

    def _on_mat_item(self, it: QTableWidgetItem) -> None:
        if it.column() != 2 or not 0 <= it.row() < len(self._mat_rows):
            return
        material, param = self._mat_rows[it.row()]
        if param is None:
            return
        try:
            val = parse_value(it.text())
        except ValueError:
            self._refresh_materials()
            return
        self.set_param(material, param, val)

    def _entries(self, material: str, base: str | None = None, create: bool = True) -> list[dict]:
        _shown, roles = self.mat_targets()
        out = []
        for r in roles:
            mres = selected_entry(slot_or_none(self.doc(r), self.sel_slot))
            ent = P.material_entry(mres, material, base=base, create=create)
            if ent is not None:
                out.append(ent)
        return out

    def set_param(self, material: str, param: str, value) -> None:
        """Set (None = remove) one rttiValues entry on the selected slot's mesh in the targeted roles."""
        for ent in self._entries(material, create=value is not None):
            P.set_rtti(ent, param, value)
        self.after_edit([self.sel_slot])

    def set_base(self, material: str, base: str) -> None:
        self._entries(material, base=base)
        self.after_edit([self.sel_slot])

    def _mat_for_action(self) -> str | None:
        cur = self._current_mat_row()
        if cur is not None:
            return cur[0]
        return self._mat_rows[0][0] if self._mat_rows else None

    def _default_value(self, param: str, material: str):
        if param.endswith("_tex"):
            texs = self.project_textures()
            if texs:
                return texs[0]
            for vw in (self.preview_out or {}).get("views", []):
                for s in vw["resolution"]["slots"]:
                    if s["name"] != self.sel_slot:
                        continue
                    for m in s["meshes"]:
                        for sub in m["submeshes"] if m["chosen"] else []:
                            if (sub.get("embedded_material") or "").lower() != material.lower():
                                continue
                            for t in sub["textures"]:
                                if (t.get("param") or "").lower() == param.lower() and t.get("texture"):
                                    return t["texture"]
            return ""
        if param.endswith("_val"):
            return [1.0, 1.0, 1.0]
        return 1.0

    def add_param(self, param: str, value=None, material: str | None = None) -> None:
        material = material or self._mat_for_action()
        if material is None:
            return
        self.set_param(material, param, self._default_value(param, material) if value is None else value)
        if (material, param) in self._mat_rows:
            self.mats.selectRow(self._mat_rows.index((material, param)))

    def _add_param(self, param: str) -> None:
        if param == "Other…":
            param, ok = QInputDialog.getText(self, "Param", "Name")
            param = param.strip()
            if not ok or not param:
                return
        self.add_param(param)

    def _delete_param(self) -> None:
        cur = self._current_mat_row()
        if cur is None or cur[1] is None:
            return
        self.set_param(cur[0], cur[1], None)

    def _pick_base(self) -> None:
        material = self._mat_for_action()
        if material is None:
            return

        def fetch(text):
            return filter_names(self.ctx.sdb.material_names(), text)
        v = self._pick("Base", fetch, free=True, text=Path(material).stem)
        if v:
            self.set_base(material, v if v.lower().endswith(".mat") else v + ".mat")
