"""Meshes tab: every type-0x10 resource of every pack — search, check, export (Cast + mesh.json / raw parts),
3D preview with per-submesh visibility and LOD selection, and details (overview, submeshes, SDB materials →
textures, skeleton, `_SKIN_` variants, raw parts).

All decoding runs in `ctx.runner` workers; the GUI thread only builds widgets from finished results.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import shiboken6
from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QSizePolicy,
                               QLabel, QMenu, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QTableView, QTabWidget, QToolButton, QTreeWidget, QTreeWidgetItem, QVBoxLayout,
                               QWidget)

from .. import meshdata
from ..meshview import MeshView, palette_color
from ..tasks import Debouncer
from ..widgets import Column, RefListModel, SearchBar, human_size, mono_font, unique_path

TITLE = "Meshes"
MESH_TYPE = 0x10
TEXTURE_TYPE = 0x20
GREY = (0.72, 0.72, 0.72)
DISPLAY_MODES = [("Lit · material colours", 0, True), ("Lit · grey", 0, False), ("Normals", 1, False),
                 ("Unlit · material colours", 2, True)]
MAX_TREE_NODES = 20000
TEXTURE_MAX_DIM = 1024


# ---- worker functions (no widgets) ---------------------------------------------------------------------------

def material_report(ctx, names: list[str]) -> list[dict]:
    """For each mesh material: SDB presence, preset, bound textures (parameter name when known) and the catalog
    gids providing each texture. Worker-thread safe (SDB + catalog lookups only)."""
    out = []
    for slot, name in enumerate(names):
        rec: dict[str, Any] = {"slot": slot, "name": name, "in_sdb": False, "preset": "", "textures": [],
                               "diffuse": None, "diffuse_gids": []}
        info = ctx.sdb.material_info(name) if name else None
        params: dict[str, str] = {}
        if info:
            rec["in_sdb"] = True
            presets = []
            for r in info.get("routes", []):
                if r.get("preset") and r["preset"] not in presets:
                    presets.append(r["preset"])
                for p in r.get("parameters", []):
                    pn, val = p.get("name") or "", p.get("value")
                    if pn.endswith("_tex") and isinstance(val, str) and val:
                        params.setdefault(val.lower(), pn)
                        if pn == "dif_0_tex" and rec["diffuse"] is None:
                            rec["diffuse"] = val
            rec["preset"] = ", ".join(presets)
        texs = ctx.sdb.textures_for(name) if info else []
        seen = set()
        for t in list(texs) + [v for v in params if v not in {x.lower() for x in texs}]:
            if t.lower() in seen:
                continue
            seen.add(t.lower())
            gids = ctx.catalog.lookup(t, TEXTURE_TYPE)
            rec["textures"].append({"texture": t, "param": params.get(t.lower(), ""), "gids": gids})
        if rec["diffuse"] is None:
            for t in rec["textures"]:
                if t["param"].startswith("dif") or "_dif." in t["texture"].lower():
                    rec["diffuse"] = t["texture"]
                    break
        if rec["diffuse"]:
            rec["diffuse_gids"] = ctx.catalog.lookup(rec["diffuse"], TEXTURE_TYPE)
        out.append(rec)
    return out


_LAYOUT_NAMES = {"dltb": "Dying Light: The Beast", "dl2": "Dying Light 2 (read-only)"}


def load_preview(ctx, gid: int) -> dict:
    entry, idx = ctx.catalog.split(gid)
    t0 = time.perf_counter()
    geoms, info = meshdata.load_mesh_geometry(entry.pack, idx)
    names = list(info["materials"])
    v = info.get("variants")
    if isinstance(v, dict) and isinstance(v.get("material_table"), list):
        extra = [str(x) for x in v["material_table"][len(names):]]
        if [str(x) for x in v["material_table"][:len(names)]] == names:
            names += extra
    mats = []
    if not info["error"]:
        try:
            mats = material_report(ctx, names)
        except Exception as exc:  # noqa: BLE001 - an SDB problem must not hide the geometry (e.g. another game's SDB)
            info.setdefault("warnings", []).append(f"material report unavailable: {type(exc).__name__}: {exc}")
            mats = [{"slot": k, "name": n, "in_sdb": False, "preset": "", "textures": [], "diffuse": None,
                     "diffuse_gids": []} for k, n in enumerate(names)]
    for m in mats:
        m["variant_only"] = m["slot"] >= len(info["materials"])
    return {"gid": gid, "geoms": geoms, "info": info, "materials": mats, "ms": (time.perf_counter() - t0) * 1000}


def load_textures(ctx, gid: int, wanted) -> dict:
    """material name → (QImage | None, message, alpha_mode, cutoff, hidden). Each material goes through the
    material-aware preview (gui/matpreview.py: eye layers, hair cut-out, opacity blend, plain diffuse); textures
    are decoded once per call, downscaled."""
    try:
        from .. import matpreview
    except Exception as exc:  # noqa: BLE001 - texture preview is optional
        return {"gid": gid, "error": f"texture preview unavailable: {exc}", "images": {}}
    fetch = matpreview.TextureFetcher(ctx.catalog, max_dim=TEXTURE_MAX_DIM)
    images = {}
    for mat in wanted:
        try:
            sp = matpreview.preview_for_material(ctx, mat, fetch)
        except Exception as exc:  # noqa: BLE001
            images[mat] = (None, f"{type(exc).__name__}: {exc}", "opaque", 0.5, False)
            continue
        images[mat] = (sp.image, sp.summary(), sp.alpha_mode, sp.cutoff, sp.hidden)
    return {"gid": gid, "error": None, "images": images}


def variant_overrides(variants: dict | None, index: int) -> dict[str, tuple[str, int]]:
    """{"e<entry>/s<sub>": (material name, material table index)} for variant *index* of a decoded 0x12 part."""
    out: dict[str, tuple[str, int]] = {}
    if not isinstance(variants, dict) or not isinstance(variants.get("variants"), list):
        return out
    if not 0 <= index < len(variants["variants"]):
        return out
    for mm in variants["variants"][index].get("material_map") or []:
        try:
            mat, name = int(mm["material"]), str(mm.get("material_name") or "")
            for e, sm in mm.get("submeshes") or []:
                out[f"e{int(e)}/s{int(sm)}"] = (name, mat)
        except (KeyError, TypeError, ValueError):
            continue
    return out


class _ExportBridge(QObject):
    progress = Signal(int, int, str)


SCENE_KINDS = ("cast", "glb", "gltf")


def run_export(catalog, jobs: list[tuple[int, Path]], kind: str, overwrite: bool, cancel: threading.Event,
               progress=None) -> dict:
    """Export *jobs* [(gid, target)] as 'cast' / 'glb' / 'gltf' (target = scene file, + <stem>.mesh.json) or
    'raw' (target = folder)."""
    done, failed, skipped = [], [], []
    for n, (gid, target) in enumerate(jobs):
        if cancel.is_set():
            skipped.extend(g for g, _ in jobs[n:])
            break
        name = catalog.name(gid)
        if progress:
            progress(n, len(jobs), name)
        try:
            if not overwrite:
                target = unique_path(target)
            entry, idx = catalog.split(gid)
            if kind in SCENE_KINDS:
                meshdata.export_cast_files(entry.pack, idx, target)
            else:
                meshdata.export_raw_parts(entry.pack, idx, target)
            done.append(str(target))
        except Exception as exc:  # noqa: BLE001
            failed.append(f"{name}: {type(exc).__name__}: {exc}")
    if progress:
        progress(len(jobs), len(jobs), "")
    return {"done": done, "failed": failed, "skipped": skipped, "cancelled": cancel.is_set()}


def export_targets(catalog, gids, dest: Path, kind: str) -> list[tuple[int, Path]]:
    """[(gid, target)]: <dest>/<pack label without .rpack>/<name>.<cast|glb|gltf> (or /<name>/ for raw parts)."""
    out = []
    for g in gids:
        e = catalog.entry(g)
        label = meshdata.safe_name(e.label[:-6] if e.label.lower().endswith(".rpack") else e.label)
        base = dest / label / meshdata.safe_name(catalog.name(g))
        out.append((g, base.with_name(base.name + "." + kind) if kind in SCENE_KINDS else base))
    return out


def _target_exists(p: Path, kind: str) -> bool:
    if kind in SCENE_KINDS:
        return p.exists() or p.with_name(p.stem + ".mesh.json").exists()
    return p.exists() and any(p.iterdir()) if p.is_dir() else p.exists()


# ---- the tab ---------------------------------------------------------------------------------------------------

class Tab(QWidget):
    TITLE = TITLE

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        cat = ctx.catalog
        self._cur: int | None = None
        self._pending: int | None = None
        self._result: dict | None = None
        self._packs_listed: set[int] = set()
        self._export_cancel: threading.Event | None = None
        self._updating_tree = False
        self._effective: dict[str, tuple[str, int]] = {}

        # ---- left: list ----
        self.search = SearchBar("search meshes")
        self.model = RefListModel([
            Column("Name", cat.name),
            Column("Pack", lambda g: cat.entry(g).label),
            Column("Parts", lambda g: cat.pack(g).logicals[g - cat.entry(g).base].part_count, align_right=True),
            Column("Bytes", lambda g: human_size(cat.part_size(g)), align_right=True),
        ], self)
        self.table = QTableView()
        self.table.setModel(self.model)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(20)
        self.table.setWordWrap(False)
        hh = self.table.horizontalHeader()
        hh.setSectionResizeMode(0, QHeaderView.Stretch)
        for c in (1, 2, 3):
            hh.setSectionResizeMode(c, QHeaderView.Interactive)
        hh.resizeSection(1, 170)
        hh.resizeSection(2, 45)
        hh.resizeSection(3, 80)

        self.btn_all = QPushButton("Check visible")
        self.btn_none = QPushButton("None")
        self.btn_export = QToolButton()
        self.btn_export.setText("Export Checked ▾")
        self.btn_export.setPopupMode(QToolButton.InstantPopup)
        menu = QMenu(self.btn_export)
        self.act_cast = menu.addAction("Cast + mesh.json…", lambda: self.export_checked("cast"))
        self.act_glb = menu.addAction("glTF binary (.glb) + mesh.json…", lambda: self.export_checked("glb"))
        self.act_gltf = menu.addAction("glTF (.gltf + .bin) + mesh.json…", lambda: self.export_checked("gltf"))
        self.act_raw = menu.addAction("Raw parts…", lambda: self.export_checked("raw"))
        self.btn_export.setMenu(menu)
        self.btn_export.setEnabled(False)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._row_menu)
        self.export_prog = QProgressBar()
        self.export_prog.setFormat("%v / %m  %p%")
        self.btn_cancel = QPushButton("Cancel")
        self.export_label = QLabel()
        self.export_label.setWordWrap(True)
        for w in (self.export_prog, self.btn_cancel):
            w.hide()

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(4, 4, 4, 4)
        ll.addWidget(self.search)
        row = QHBoxLayout()
        row.addWidget(self.btn_all)
        row.addWidget(self.btn_none)
        row.addStretch(1)
        row.addWidget(self.btn_export)
        ll.addLayout(row)
        ll.addWidget(self.table, 1)
        prow = QHBoxLayout()
        prow.addWidget(self.export_prog, 1)
        prow.addWidget(self.btn_cancel)
        ll.addLayout(prow)
        ll.addWidget(self.export_label)

        # ---- right: viewer + info ----
        self.view = MeshView()
        self.title = QLabel("Select a mesh")
        self.title.setStyleSheet("font-weight: bold;")
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.lod_combo = QComboBox()
        self.lod_combo.setToolTip("Level of detail (position in the geometry array)")
        self.display_combo = QComboBox()
        for label, _m, _c in DISPLAY_MODES:
            self.display_combo.addItem(label)
        self.wire_check = QCheckBox("Wireframe")
        self.variant_combo = QComboBox()
        self.variant_combo.setToolTip("_SKIN_ material variant (part 0x12) applied to colours / textures")
        self.variant_combo.setMinimumContentsLength(14)
        self.variant_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.tex_check = QCheckBox("SDB textures")
        self.tex_check.setToolTip("Apply each material's diffuse (dif_0_tex) texture, downscaled to "
                                  f"{TEXTURE_MAX_DIM} px")
        self.tex_check.setChecked(bool(ctx.settings.value("meshes/textures", False, type=bool)))
        self.flip_check = QCheckBox("Flip V")
        self.flip_check.setToolTip("Flip texture V coordinate")
        self.btn_frame = QPushButton("Frame (F)")
        self.btn_raw = QPushButton("Show in Raw")
        self.btn_raw.setEnabled(False)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)
        self.busy.setMaximumWidth(90)
        self.busy.setMaximumHeight(14)
        self.busy.setTextVisible(False)
        self.busy.hide()
        self.view_status = QLabel()
        self.view_status.setStyleSheet("color: #999;")

        head = QHBoxLayout()
        head.setContentsMargins(4, 2, 4, 0)
        head.addWidget(self.title, 1)
        head.addWidget(self.busy)
        bar = QHBoxLayout()
        bar.setContentsMargins(4, 0, 4, 2)
        bar.addWidget(QLabel("LOD"))
        bar.addWidget(self.lod_combo)
        bar.addWidget(self.display_combo)
        bar.addWidget(QLabel("Variant"))
        bar.addWidget(self.variant_combo)
        bar.addWidget(self.wire_check)
        bar.addWidget(self.tex_check)
        bar.addWidget(self.flip_check)
        bar.addWidget(self.btn_frame)
        bar.addWidget(self.btn_raw)
        bar.addStretch(1)
        top = QWidget()
        tl = QVBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        tl.addLayout(head)
        tl.addLayout(bar)
        tl.addWidget(self.view, 1)
        hint = QLabel("LMB orbit · MMB / Shift+LMB pan · wheel zoom · F frame · W wireframe · N normals · "
                      "R reset")
        hint.setStyleSheet("color: #888; font-size: 8pt; padding: 1px 4px;")
        hint.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        srow = QHBoxLayout()
        srow.addWidget(hint, 1)
        srow.addWidget(self.view_status)
        tl.addLayout(srow)

        self.info_tabs = QTabWidget()
        self.overview = QPlainTextEdit()
        self.overview.setReadOnly(True)
        self.overview.setFont(mono_font())
        self.overview.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.sub_tree = self._tree(["Entry / submesh", "Material", "Vertices", "Triangles", "Bones", "Format",
                                    "Bounds"])
        self.mat_tree = self._tree(["Material / texture", "SDB", "Preset / binding", "In catalog"])
        self.skel_tree = self._tree(["Bone", "Index", "Type", "Geometry", "Local position"])
        self.var_tree = self._tree(["Key", "Value"])
        self.var_text = QPlainTextEdit()
        self.var_text.setReadOnly(True)
        self.var_text.setFont(mono_font())
        self.var_text.setLineWrapMode(QPlainTextEdit.NoWrap)
        var_w = QSplitter(Qt.Vertical)
        var_w.addWidget(self.var_tree)
        var_w.addWidget(self.var_text)
        self.parts_tree = self._tree(["#", "Type", "Part", "Size", "Offset", "Direct", "Physical"])
        self.info_tabs.addTab(self.overview, "Overview")
        self.info_tabs.addTab(self.sub_tree, "Submeshes")
        self.info_tabs.addTab(self.mat_tree, "Materials")
        self.info_tabs.addTab(self.skel_tree, "Skeleton")
        self.info_tabs.addTab(var_w, "Variants")
        self.info_tabs.addTab(self.parts_tree, "Parts")

        right = QSplitter(Qt.Vertical)
        right.addWidget(top)
        right.addWidget(self.info_tabs)
        right.setStretchFactor(0, 3)
        right.setStretchFactor(1, 2)
        split = QSplitter(Qt.Horizontal)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 5)
        split.setSizes([480, 1100])
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(split)

        # ---- wiring ----
        self._search_deb = Debouncer(self._run_search, 200, self)
        self._select_deb = Debouncer(self._load_selected, 120, self)
        self.search.changed.connect(self._search_deb.trigger)
        self.btn_all.clicked.connect(lambda: self.model.check_all_visible(True))
        self.btn_none.clicked.connect(self.model.clear_checks)
        self.model.checkedChanged.connect(self._on_checked)
        self.btn_cancel.clicked.connect(self._cancel_export)
        self.table.selectionModel().currentRowChanged.connect(lambda *_: self._select_deb.trigger())
        self.table.doubleClicked.connect(lambda *_: self._select_deb.flush())
        self.lod_combo.currentIndexChanged.connect(self._apply_lod)
        self.variant_combo.currentIndexChanged.connect(self._apply_variant)
        self.display_combo.currentIndexChanged.connect(self._apply_display)
        self.wire_check.toggled.connect(self.view.set_wireframe)
        self.tex_check.toggled.connect(self._toggle_textures)
        self.flip_check.toggled.connect(self.view.set_flip_v)
        self.btn_frame.clicked.connect(self.view.frame_all)
        self.btn_raw.clicked.connect(self._show_raw)
        self.sub_tree.itemChanged.connect(self._sub_item_changed)
        self.mat_tree.itemDoubleClicked.connect(self._mat_double_clicked)
        cat.packAdded.connect(self._on_pack_added)
        cat.ready.connect(self._search_deb.trigger)
        for e in list(cat.packs):
            if e.pack is not None:
                self._on_pack_added(e.id)
        self._run_search()

    # ---- runner wrappers --------------------------------------------------------------------------------------
    def _drop_dead_job(self, key) -> None:
        """Work around TaskRunner calling tryTake() on a finished, auto-deleted job (RuntimeError)."""
        runner = self.ctx.runner
        job = getattr(runner, "_jobs", {}).get(key)
        if job is not None and not shiboken6.isValid(job):
            runner._jobs.pop(key, None)

    def _submit(self, key, fn, *args, **kwargs) -> int:
        self._drop_dead_job(key)
        return self.ctx.runner.submit(key, fn, *args, **kwargs)

    def _cancel(self, key) -> None:
        self._drop_dead_job(key)
        self.ctx.runner.cancel(key)

    @staticmethod
    def _tree(headers: list[str]) -> QTreeWidget:
        t = QTreeWidget()
        t.setHeaderLabels(headers)
        t.setUniformRowHeights(True)
        t.setAlternatingRowColors(True)
        t.header().setStretchLastSection(True)
        return t

    # ---- list / search ----------------------------------------------------------------------------------------
    def _on_pack_added(self, pid: int) -> None:
        e = self.ctx.catalog.packs[pid]
        if pid not in self._packs_listed and e.type_counts.get(MESH_TYPE):
            self._packs_listed.add(pid)
            self.search.add_pack(pid, f"{e.label} ({e.type_counts[MESH_TYPE]:,})")
        self._search_deb.trigger()

    def _run_search(self) -> None:
        self._submit("mesh-search", self.ctx.catalog.search, self.search.text(), types=(MESH_TYPE,),
                               packs=self.search.packs(), on_done=self._on_search)

    def _on_search(self, gids) -> None:
        cur = self.current_list_gid()
        self.model.set_gids(gids)
        total = self.ctx.catalog.type_counts().get(MESH_TYPE, 0)
        self.search.set_count(len(gids), total, len(self.model.checked))
        want = self._pending if self._pending is not None else cur
        if want is not None:
            row = self.model.row_of(want)
            if row >= 0:
                if want == self._pending:
                    self._pending = None
                self._select_row(row, load=False)
            elif self._pending is not None and (self.search.text() or self.search.packs()):
                self._clear_filters()

    def _clear_filters(self) -> None:
        self.search.edit.blockSignals(True)
        self.search.pack_combo.blockSignals(True)
        self.search.edit.clear()
        self.search.pack_combo.setCurrentIndex(0)
        self.search.edit.blockSignals(False)
        self.search.pack_combo.blockSignals(False)
        self._run_search()

    def current_list_gid(self) -> int | None:
        idx = self.table.currentIndex()
        return self.model.gid_at(idx.row()) if idx.isValid() and idx.row() < self.model.rowCount() else None

    def _select_row(self, row: int, load: bool = True) -> None:
        sm = self.table.selectionModel()
        sm.blockSignals(not load)
        try:
            self.table.selectRow(row)
            self.table.setCurrentIndex(self.model.index(row, 0))
        finally:
            sm.blockSignals(False)
        self.table.scrollTo(self.model.index(row, 0), QAbstractItemView.PositionAtCenter)

    def _on_checked(self, n: int) -> None:
        self.btn_export.setEnabled(n > 0 and self._export_cancel is None)
        total = self.ctx.catalog.type_counts().get(MESH_TYPE, 0)
        self.search.set_count(self.model.rowCount(), total, n)

    # ---- navigation ------------------------------------------------------------------------------------------
    def open_gid(self, gid: int) -> None:
        """Reveal *gid* in the list (clearing filters if needed), select it and load it."""
        gid = int(gid)
        row = self.model.row_of(gid)
        if row >= 0:
            self._select_row(row, load=False)
        else:
            self._pending = gid
            if self.search.text() or self.search.packs():
                self._clear_filters()
            else:
                self._run_search()
        self.load(gid)

    def sdb_changed(self) -> None:
        if self._cur is not None:
            self.load(self._cur, keep_camera=True)

    def shutdown(self) -> None:
        if self._export_cancel is not None:
            self._export_cancel.set()
        for k in ("mesh-preview", "mesh-textures", "mesh-search"):
            self._cancel(k)

    def _show_raw(self) -> None:
        if self._cur is not None:
            self.ctx.openRaw.emit(self._cur)

    # ---- loading ----------------------------------------------------------------------------------------------
    def _load_selected(self) -> None:
        g = self.current_list_gid()
        if g is not None and g != self._cur:
            self.load(g)

    def load(self, gid: int, keep_camera: bool = False) -> None:
        cat = self.ctx.catalog
        self._cur = gid
        self._keep_camera = keep_camera
        self.btn_raw.setEnabled(True)
        e, idx = cat.split(gid)
        self.title.setText(f"{cat.name(gid)}   <{e.label} #{idx}>")
        self.busy.show()
        self.view_status.setText("decoding…")
        self._cancel("mesh-textures")
        self._submit("mesh-preview", load_preview, self.ctx, gid, on_done=self._on_loaded,
                               on_error=self._on_load_error)

    def _on_load_error(self, err: str) -> None:
        self.busy.hide()
        self.view.clear()
        self.view.set_message("Could not load this mesh:\n" + err.splitlines()[0])
        self.overview.setPlainText(err)

    def _on_loaded(self, res: dict) -> None:
        if res["gid"] != self._cur:
            return
        self.busy.hide()
        self._result = res
        info, geoms = res["info"], res["geoms"]
        t0 = time.perf_counter()
        self.view.clear()
        self._fill_overview(res)
        self._fill_parts(info)
        self._fill_skeleton(info)
        self._fill_variants(info)
        self._fill_materials(res["materials"])
        if info["error"]:
            self.view.set_message(f"This resource could not be decoded:\n{info['error']}")
            self._fill_submeshes(info)
            self.lod_combo.clear()
            self._fill_variant_combo(info)
            self.view_status.setText("decode failed")
            return
        if not geoms:
            self.view.set_message(
                f"No geometry: skeleton-only mesh ({len(info['bones'])} bones, see the Skeleton tab)."
                if not info["entries"] else "No decodable geometry (see Overview).")
        else:
            self.view.set_message("")
        self._fill_variant_combo(info)
        self._effective = {g.key: (g.material, g.material_slot) for g in geoms}
        for g in geoms:
            self.view.add_mesh(g.key, g.positions, g.indices, g.normals, g.uv, color=GREY)
        self._apply_display()
        self.lod_combo.blockSignals(True)
        self.lod_combo.clear()
        for k in range(info["lod_count"]):
            self.lod_combo.addItem(f"LOD {k}", k)
        if info["lod_count"] > 1:
            self.lod_combo.addItem("All", -1)
        self.lod_combo.setEnabled(info["lod_count"] > 1)
        self.lod_combo.blockSignals(False)
        self._fill_submeshes(info)
        self._apply_lod()
        if not getattr(self, "_keep_camera", False):
            self.view.frame_all()
        add_ms = (time.perf_counter() - t0) * 1000
        tris = sum(len(g.indices) for g in geoms) // 3
        self.view_status.setText(f"{len(geoms)} parts · {tris:,} tris · decode {res['ms']:.0f} ms · "
                                 f"upload {add_ms:.0f} ms")
        if self.tex_check.isChecked():
            self._start_textures()

    # ---- textures ---------------------------------------------------------------------------------------------
    def _toggle_textures(self, on: bool) -> None:
        self.ctx.settings.setValue("meshes/textures", on)
        if on:
            self._start_textures()
        else:
            self._cancel("mesh-textures")
            if self._result:
                for g in self._result["geoms"]:
                    self.view.set_texture(g.key, None)
                self._reapply_visibility()
            self._mark_textures({})

    def _start_textures(self) -> None:
        res = self._result
        if not res or res["gid"] != self._cur or not res["geoms"]:
            return
        by_name = {m["name"].lower(): m for m in res["materials"]}
        wanted = []
        for g in res["geoms"]:
            mat = self._material_of(g)
            m = by_name.get(mat.lower())
            if m and m["in_sdb"] and mat not in wanted:
                wanted.append(mat)
        if not wanted:
            self.view_status.setText(self.view_status.text().split(" · textures")[0] + " · textures: none bound")
            return
        self.busy.show()
        self._submit("mesh-textures", load_textures, self.ctx, res["gid"], wanted,
                     on_done=self._on_textures, on_error=lambda e: self.busy.hide())

    def _on_textures(self, tr: dict) -> None:
        self.busy.hide()
        res = self._result
        if not res or tr["gid"] != self._cur or not self.tex_check.isChecked():
            return
        base = self.view_status.text().split(" · textures")[0]
        if tr["error"]:
            self.view_status.setText(base + " · textures: " + tr["error"])
            return
        hidden = 0
        for g in res["geoms"]:
            img, _msg, mode, cutoff, hide = tr["images"].get(self._material_of(g), (None, "", "opaque", 0.5, False))
            self.view.set_texture(g.key, img, mode, cutoff)
            if hide:
                self.view.set_visible(g.key, False)
                hidden += 1
        ok = sum(1 for v in tr["images"].values() if v[0] is not None)
        extra = f", {hidden} non-rendering hidden" if hidden else ""
        self.view_status.setText(base + f" · textures {ok}/{len(tr['images'])} materials{extra}")
        self._mark_textures(tr["images"])

    def _mark_textures(self, images: dict) -> None:
        for i in range(self.mat_tree.topLevelItemCount()):
            it = self.mat_tree.topLevelItem(i)
            name = it.data(0, Qt.UserRole + 1)
            v = images.get(name)
            it.setToolTip(0, "" if v is None else ("preview: " + v[1]))

    # ---- display ----------------------------------------------------------------------------------------------
    def _apply_display(self) -> None:
        _label, mode, colored = DISPLAY_MODES[self.display_combo.currentIndex()]
        self.view.set_shading(mode)
        if self._result:
            for g in self._result["geoms"]:
                idx = self._effective.get(g.key, (g.material, g.material_slot))[1]
                self.view.set_color(g.key, palette_color(idx) if colored else GREY)

    def _material_of(self, g) -> str:
        return self._effective.get(g.key, (g.material, g.material_slot))[0]

    def _fill_variant_combo(self, info: dict) -> None:
        self.variant_combo.blockSignals(True)
        self.variant_combo.clear()
        self.variant_combo.addItem("(mesh materials)", -1)
        v = info.get("variants")
        if isinstance(v, dict) and isinstance(v.get("variants"), list):
            for k, rec in enumerate(v["variants"]):
                name = rec.get("name") if isinstance(rec, dict) else None
                mod = " (modifier)" if isinstance(rec, dict) and rec.get("modifier") else ""
                self.variant_combo.addItem(f"{name or f'#{k}'}{mod}", k)
        self.variant_combo.setEnabled(self.variant_combo.count() > 1)
        self.variant_combo.blockSignals(False)

    def _apply_variant(self) -> None:
        """Map submeshes to the selected variant's materials (colours, and textures when enabled)."""
        res = self._result
        if not res:
            return
        k = self.variant_combo.currentData()
        base = {g.key: (g.material, g.material_slot) for g in res["geoms"]}
        if k is not None and k >= 0:
            base.update(variant_overrides(res["info"].get("variants"), int(k)))
        self._effective = base
        self._apply_display()
        if self.tex_check.isChecked():
            self._start_textures()

    def _apply_lod(self) -> None:
        lod = self.lod_combo.currentData()
        self._updating_tree = True
        try:
            for i in range(self.sub_tree.topLevelItemCount()):
                top = self.sub_tree.topLevelItem(i)
                elod = top.data(0, Qt.UserRole + 1)
                on = lod is None or lod == -1 or elod == lod
                for k in range(top.childCount()):
                    ch = top.child(k)
                    if ch.flags() & Qt.ItemIsUserCheckable:
                        ch.setCheckState(0, Qt.Checked if on else Qt.Unchecked)
                        self.view.set_visible(ch.data(0, Qt.UserRole), on)
        finally:
            self._updating_tree = False

    def _reapply_visibility(self) -> None:
        """Visibility back to the submesh checkboxes (undoes non-rendering hiding)."""
        for i in range(self.sub_tree.topLevelItemCount()):
            top = self.sub_tree.topLevelItem(i)
            for k in range(top.childCount()):
                ch = top.child(k)
                if ch.flags() & Qt.ItemIsUserCheckable and ch.data(0, Qt.UserRole):
                    self.view.set_visible(ch.data(0, Qt.UserRole), ch.checkState(0) == Qt.Checked)

    def _sub_item_changed(self, item: QTreeWidgetItem, col: int) -> None:
        if self._updating_tree or col != 0:
            return
        key = item.data(0, Qt.UserRole)
        if key:
            self.view.set_visible(key, item.checkState(0) == Qt.Checked)

    # ---- info panes -------------------------------------------------------------------------------------------
    def _fill_overview(self, res: dict) -> None:
        info = res["info"]
        cat = self.ctx.catalog
        gid = res["gid"]
        e, idx = cat.split(gid)
        L = [f"name            {info['name']}",
             f"pack            {e.label}   (logical #{idx}, gid {gid})",
             f"file            {e.path}"]
        if info["error"]:
            L += ["", f"ERROR           {info['error']}"]
        else:
            b = info["bounds"]
            size = [hi - lo for lo, hi in zip(*b)] if b else None
            L += [f"embedded name   {info.get('embedded_name', '')}",
                  f"layout          {_LAYOUT_NAMES.get(info.get('layout'), info.get('layout') or '-')}",
                  f"entities        {info['entities']}",
                  f"geometry        {len(info['entries'])} entries, {info['lod_count']} LOD level(s)",
                  f"formats         {', '.join(meshdata.FORMAT_NAMES.get(f, str(f)) for f in info['formats']) or '-'}",
                  f"vertices        {info['vertices']:,}",
                  f"triangles       {info['triangles']:,}",
                  f"skinned         {'yes' if info['skinned'] else 'no'}",
                  f"materials       {len(info['materials'])}",
                  f"variants part   {human_size(info['skin_size']) if info['skin_size'] is not None else '-'}",
                  f"cloth part      {human_size(info['cloth_size']) if info['cloth_size'] is not None else '-'}",
                  f"bounds min      {_vec(b[0]) if b else '-'}",
                  f"bounds max      {_vec(b[1]) if b else '-'}",
                  f"size            {_vec(size) if size else '-'}",
                  f"decode          {info['decode_ms']:.1f} ms (+ SDB {res['ms'] - info['decode_ms']:.1f} ms)",
                  "", "entry  lod  owner                          fmt  verts      indices    submeshes"]
            for er in info["entries"]:
                own = (er["owner_name"] or "-")[:30]
                L.append(f"{er['index']:>5}  {er['lod']:>3}  {own:<30} {er['format']:>3}  {er['vertex_count']:>9,}  "
                         f"{er['index_count']:>9,}  {len(er['submeshes'])}")
                for s in er["submeshes"]:
                    L.append(f"         s{s['index']:<3} {s['vertices']:>9,} v {s['indices']:>9,} i  "
                             f"{s['material']}" + (f"   [{s['error']}]" if s["error"] else ""))
        for w in info["warnings"]:
            L.append(f"warning: {w}")
        for w in info["errors"]:
            L.append(f"error: {w}")
        self.overview.setPlainText("\n".join(L))

    def _fill_submeshes(self, info: dict) -> None:
        self._updating_tree = True
        t = self.sub_tree
        t.clear()
        try:
            for er in info["entries"]:
                top = QTreeWidgetItem([f"entry {er['index']}  ·  LOD {er['lod']}  ·  {er['owner_name'] or '-'}",
                                       "", f"{er['vertex_count']:,}", f"{er['index_count'] // 3:,}", "",
                                       str(er["format"]), ""])
                top.setData(0, Qt.UserRole + 1, er["lod"])
                top.setFlags(top.flags() | Qt.ItemIsAutoTristate | Qt.ItemIsUserCheckable)
                for s in er["submeshes"]:
                    b = s["bounds"]
                    ch = QTreeWidgetItem([f"s{s['index']}", s["material"], f"{s['vertices']:,}",
                                          f"{s['triangles']:,}", str(s["bones"]), str(er["format"]),
                                          f"{_vec(b[0])} … {_vec(b[1])}" if b else (s["error"] or "")])
                    ch.setData(0, Qt.UserRole, s["key"])
                    if s["error"] is None:
                        ch.setFlags(ch.flags() | Qt.ItemIsUserCheckable)
                        ch.setCheckState(0, Qt.Checked)
                        ch.setIcon(1, _swatch(palette_color(s["material_slot"])))
                    else:
                        ch.setForeground(0, QBrush(QColor("#c77")))
                        ch.setToolTip(0, s["error"])
                    top.addChild(ch)
                t.addTopLevelItem(top)
                top.setExpanded(True)
            for c in range(6):
                t.resizeColumnToContents(c)
        finally:
            self._updating_tree = False

    def _fill_materials(self, mats: list[dict]) -> None:
        t = self.mat_tree
        t.clear()
        sdb_ok = self.ctx.sdb.available()
        for m in mats:
            sdb = "yes" if m["in_sdb"] else ("no" if sdb_ok else "no SDB")
            vo = "   (variant only)" if m.get("variant_only") else ""
            top = QTreeWidgetItem([f"[{m['slot']}] {m['name']}{vo}", sdb, m["preset"], ""])
            top.setData(0, Qt.UserRole, ("material", m["name"]))
            top.setData(0, Qt.UserRole + 1, m["name"])
            top.setIcon(0, _swatch(palette_color(m["slot"])))
            if not m["in_sdb"]:
                top.setForeground(1, QBrush(QColor("#c77")))
            for tx in m["textures"]:
                found = f"yes ({len(tx['gids'])})" if tx["gids"] else "missing"
                label = tx["texture"] + ("   ◀ diffuse" if tx["texture"] == m["diffuse"] else "")
                ch = QTreeWidgetItem([label, "", tx["param"] or "(resolved binding)", found])
                ch.setData(0, Qt.UserRole, ("texture", tx["gids"][0] if tx["gids"] else None))
                if tx["gids"]:
                    ch.setToolTip(3, "\n".join(f"{self.ctx.catalog.entry(g).label}" for g in tx["gids"][:20]))
                else:
                    ch.setForeground(3, QBrush(QColor("#c77")))
                top.addChild(ch)
            t.addTopLevelItem(top)
            top.setExpanded(True)
        if not mats:
            t.addTopLevelItem(QTreeWidgetItem(["(no materials)"]))
        t.resizeColumnToContents(0)
        t.resizeColumnToContents(1)
        t.resizeColumnToContents(2)

    def _mat_double_clicked(self, item: QTreeWidgetItem, _col: int) -> None:
        d = item.data(0, Qt.UserRole)
        if not d:
            return
        kind, val = d
        if kind == "material" and val:
            self.ctx.openMaterial.emit(val)
        elif kind == "texture" and val is not None:
            self.ctx.openTexture.emit(int(val))

    def _fill_skeleton(self, info: dict) -> None:
        t = self.skel_tree
        t.clear()
        bones = info["bones"]
        items = []
        for b in bones:
            it = QTreeWidgetItem([b["name"], str(b["index"]), str(b["type"]), str(b["geometry_count"] or ""),
                                  _vec(b["position"])])
            items.append(it)
        for b, it in zip(bones, items):
            p = b["parent"]
            if 0 <= p < len(items) and p != b["index"]:
                items[p].addChild(it)
            else:
                t.addTopLevelItem(it)
        t.expandToDepth(3 if len(bones) < 400 else 1)
        t.resizeColumnToContents(0)
        self.info_tabs.setTabText(3, f"Skeleton ({len(bones)})")

    def _fill_variants(self, info: dict) -> None:
        self.var_tree.clear()
        v = info.get("variants")
        raw = info.get("skin_raw")
        lines = []
        if isinstance(v, dict) and v.get("error"):
            info = dict(info, variants_error=f"decoder error: {v['error']}")
            v = None
        if v is not None:
            budget = [MAX_TREE_NODES]
            _dict_tree(self.var_tree.invisibleRootItem(), v, budget, depth=0)
            self.var_tree.expandToDepth(1)
            self.var_tree.resizeColumnToContents(0)
            if budget[0] <= 0:
                lines.append(f"(tree truncated at {MAX_TREE_NODES:,} nodes)")
        elif raw is None:
            lines.append("This mesh has no _SKIN_ (0x12) part.")
        else:
            if info.get("variants_error"):
                lines.append(f"Variant decoder: {info['variants_error']}")
            strs = meshdata.skin_strings(raw)
            lines.append(f"part 0x12: {len(raw):,} bytes; {len(strs)} strings: " + ", ".join(strs[:200]))
            lines.append("")
            lines.append(_hexdump(raw[:1 << 16]))
            if len(raw) > 1 << 16:
                lines.append(f"… {len(raw) - (1 << 16):,} more bytes")
        self.var_text.setPlainText("\n".join(lines))
        self.var_text.setVisible(bool(lines))
        self.var_tree.setVisible(v is not None)

    def _fill_parts(self, info: dict) -> None:
        t = self.parts_tree
        t.clear()
        for p in info["parts"]:
            t.addTopLevelItem(QTreeWidgetItem([str(p["ordinal"]), f"0x{p['type']:02X}", p["type_name"],
                                               f"{p['size']:,}", f"0x{p['offset']:X}",
                                               "yes" if p["direct"] else "no", str(p["physical"])]))
        for c in range(6):
            t.resizeColumnToContents(c)

    # ---- export -----------------------------------------------------------------------------------------------
    def _row_menu(self, pos) -> None:
        idx = self.table.indexAt(pos)
        if not idx.isValid():
            return
        m = self.item_menu(self.model.gid_at(idx.row()))
        m.exec(self.table.viewport().mapToGlobal(pos))

    def item_menu(self, gid: int) -> "QMenu":
        """The context menu for one mesh: exports just *gid*, without touching the check boxes."""
        name = self.ctx.catalog.name(gid)
        m = QMenu(self)
        m.addAction("Export Cast + mesh.json…", lambda: self.export_checked("cast", gids=[gid]))
        m.addAction("Export glTF binary (.glb)…", lambda: self.export_checked("glb", gids=[gid]))
        m.addAction("Export glTF (.gltf + .bin)…", lambda: self.export_checked("gltf", gids=[gid]))
        m.addAction("Export raw parts…", lambda: self.export_checked("raw", gids=[gid]))
        m.addSeparator()
        m.addAction("Copy name", lambda: QGuiApplication.clipboard().setText(name))
        checked = gid in self.model.checked
        m.addAction("Uncheck" if checked else "Check", lambda: self.model.set_checked([gid], not checked))
        return m

    def export_checked(self, kind: str, dest: str | None = None, overwrite: bool | None = None,
                       gids: list[int] | None = None) -> bool:
        """Export *gids*, or every checked mesh when none are given. *dest*/*overwrite* given → no dialogs."""
        gids = sorted(self.model.checked if gids is None else gids)
        if not gids or self._export_cancel is not None:
            return False
        if dest is None:
            dest = QFileDialog.getExistingDirectory(self, "Export meshes to…", self.ctx.export_dir())
            if not dest:
                return False
            self.ctx.set_export_dir(dest)
        jobs = export_targets(self.ctx.catalog, gids, Path(dest), kind)
        if overwrite is None:
            clash = sum(1 for _, p in jobs if _target_exists(p, kind))
            overwrite = False
            if clash:
                box = QMessageBox(QMessageBox.Question, "Export meshes",
                                  f"{clash:,} of {len(jobs):,} targets already exist in {dest}.", parent=self)
                b_over = box.addButton("Overwrite", QMessageBox.DestructiveRole)
                b_keep = box.addButton("Keep both (rename new)", QMessageBox.AcceptRole)
                box.addButton(QMessageBox.Cancel)
                box.setDefaultButton(b_keep)
                box.exec()
                if box.clickedButton() is b_over:
                    overwrite = True
                elif box.clickedButton() is not b_keep:
                    return False
        self._export_cancel = threading.Event()
        bridge = _ExportBridge(self)
        bridge.progress.connect(self._on_export_progress)
        self._bridge = bridge
        self.export_prog.setRange(0, len(jobs))
        self.export_prog.setValue(0)
        self.export_prog.show()
        self.btn_cancel.show()
        self.btn_cancel.setEnabled(True)
        self.btn_export.setEnabled(False)
        self.export_label.setText(f"exporting {len(jobs):,} mesh(es) to {dest}…")
        self._submit("mesh-export", run_export, self.ctx.catalog, jobs, kind, overwrite,
                               self._export_cancel, bridge.progress.emit,
                               on_done=lambda r: self._on_export_done(r, dest),
                               on_error=lambda e: self._on_export_done({"done": [], "failed": [e.splitlines()[0]],
                                                                        "skipped": [], "cancelled": False}, dest))
        return True

    def _on_export_progress(self, n: int, total: int, name: str) -> None:
        self.export_prog.setMaximum(total)
        self.export_prog.setValue(n)
        if name:
            self.export_label.setText(f"{n + 1:,} / {total:,}  {name}")

    def _cancel_export(self) -> None:
        if self._export_cancel is not None:
            self._export_cancel.set()
            self.btn_cancel.setEnabled(False)

    def _on_export_done(self, r: dict, dest: str) -> None:
        self._export_cancel = None
        self.export_prog.hide()
        self.btn_cancel.hide()
        self.btn_export.setEnabled(bool(self.model.checked))
        msg = f"exported {len(r['done']):,} to {dest}"
        if r["failed"]:
            msg += f"; {len(r['failed'])} failed"
        if r["cancelled"]:
            msg += f"; cancelled ({len(r['skipped'])} not exported)"
        self.export_label.setText(msg)
        self.export_label.setToolTip("\n".join(r["failed"][:50]))
        self.last_export = r
        self.ctx.status.emit(msg)
        if r["failed"]:
            box = QMessageBox(QMessageBox.Warning, "Export meshes", msg, QMessageBox.Close, self)
            box.setDetailedText("\n".join(r["failed"][:500]))
            box.setModal(False)
            box.setAttribute(Qt.WA_DeleteOnClose)
            box.show()


# ---- helpers ---------------------------------------------------------------------------------------------------

def _vec(v) -> str:
    return "(" + ", ".join(f"{float(x):.4g}" for x in v) + ")"


def _swatch(rgb):
    from PySide6.QtGui import QIcon, QPixmap
    pm = QPixmap(10, 10)
    pm.fill(QColor.fromRgbF(*rgb))
    return QIcon(pm)


def _short(v) -> str:
    if isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        return f"<{len(b)} bytes> {b[:32].hex(' ')}" + (" …" if len(b) > 32 else "")
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= 300 else s[:300] + "…"


def _dict_tree(parent: QTreeWidgetItem, obj, budget: list[int], depth: int) -> None:
    """Generic key/value tree of nested dicts/lists (node budget shared across the whole tree)."""
    if isinstance(obj, dict):
        items = list(obj.items())
    elif isinstance(obj, (list, tuple)):
        items = [(f"[{i}]", v) for i, v in enumerate(obj)]
    else:
        return
    for k, v in items:
        if budget[0] <= 0:
            return
        budget[0] -= 1
        if isinstance(v, (dict, list, tuple)) and depth < 12:
            label = _summary(v)
            it = QTreeWidgetItem([str(k), label])
            parent.addChild(it)
            _dict_tree(it, v, budget, depth + 1)
        else:
            parent.addChild(QTreeWidgetItem([str(k), _short(v)]))


def _summary(v) -> str:
    if isinstance(v, dict):
        name = v.get("name") if isinstance(v.get("name"), str) else None
        return (f"{name}  " if name else "") + f"{{{len(v)}}}"
    return f"[{len(v)}]"


def _hexdump(data: bytes, width: int = 16) -> str:
    out = []
    for off in range(0, len(data), width):
        chunk = data[off:off + width]
        asc = "".join(chr(c) if 32 <= c < 127 else "." for c in chunk)
        out.append(f"{off:08X}  {chunk.hex(' '):<{width * 3}} {asc}")
    return "\n".join(out)


__all__ = ["Tab", "TITLE", "material_report", "load_preview", "load_textures", "run_export", "export_targets"]
