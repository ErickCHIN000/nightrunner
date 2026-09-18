"""Models tab: pick a `.model` document (dataN.pak) and see everything it pulls in.

Left: every .model member (PakService.models(), with override info), debounced search, export check boxes.
Right: the full resolution (gui/modelresolve.py) as a tree — slots → meshes → submeshes → material → overrides →
SDB texture bindings → catalog providers — plus a mesh-variant selector, a 3D preview of the found meshes
(LOD 0, optional diffuse textures), the raw JSON and copyable unique texture / material lists.
All reading, resolving, decoding and exporting runs in `ctx.runner` / a worker thread.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QGuiApplication
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton, QSplitter,
                               QTableView, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ..modelresolve import json_safe, resolve_model
from ..tasks import Debouncer
from ..widgets import Column, RefListModel, SearchBar, mono_font, unique_path

TITLE = "Models"

C_ITEM, C_VALUE, C_SOURCE, C_STATUS = range(4)
ROLE_KIND = Qt.UserRole + 1          # "texture" | "mesh" | "material" | None
ROLE_REF = Qt.UserRole + 2           # gid (int) or material name
ROLE_FLAGS = Qt.UserRole + 3         # {"missing": bool, "override": bool}

GREEN, RED, AMBER = QColor(90, 180, 90), QColor(220, 90, 90), QColor(220, 170, 60)


# ---- worker helpers -----------------------------------------------------------------------------------------------

def load_and_resolve(ctx, rec: dict, variant: str | None) -> dict:
    """Worker: read the .model JSON and resolve it. Errors become {"error": …} (shown in the tab)."""
    out: dict[str, Any] = {"rec": rec, "doc": None, "resolution": None, "error": None}
    try:
        out["doc"] = rec["doc"] if rec.get("doc") is not None else ctx.paks.load_model(rec)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"cannot read {rec['name']}: {type(exc).__name__}: {exc}"
        return out
    view = rec_ctx(ctx, rec)
    try:
        out["resolution"] = resolve_model(view, out["doc"], variant, name=rec["name"])
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"resolution failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return out


def rec_ctx(ctx, rec: dict | None):
    """The ctx-like a record resolves against: in-memory documents (open_model_doc) may prefer extra packs."""
    view = (rec or {}).get("view")
    if view is None:
        return ctx
    if (rec or {}).get("wait"):
        view.catalog.wait_packs(rec["wait"])
    return view


def diffuse_texture(sub: dict) -> dict | None:
    """The texture row used for the 3D preview: dif_0_tex, else the first *_dif texture."""
    rows = sub.get("textures") or []
    for t in rows:
        if (t.get("param") or "").casefold() == "dif_0_tex":
            return t
    for t in rows:
        if "_dif" in (t.get("texture") or "").casefold():
            return t
    return None


def load_preview(ctx, resolution: dict, with_textures: bool, max_dim: int = 512) -> dict:
    """Worker: LOD-0 geometry of every found mesh (first provider) + optional diffuse QImages."""
    try:
        from .. import meshdata
    except Exception as exc:  # noqa: BLE001
        return {"error": f"gui.meshdata unavailable: {exc}", "items": []}
    from .. import matpreview
    items, errors, tex_cache = [], [], {}
    fetch = None
    done: set[tuple[str, int]] = set()
    for slot in resolution.get("slots", []):
        for m in slot.get("meshes", []):
            if not m.get("gids") or not m.get("chosen"):
                continue
            gid = m["gids"][0]
            if (slot["name"], gid) in done:
                continue
            done.add((slot["name"], gid))
            try:
                entry, idx = ctx.catalog.split(gid)
                try:
                    geoms, info = meshdata.load_mesh_geometry(entry.pack, idx, lods=[0])
                except TypeError:
                    geoms, info = meshdata.load_mesh_geometry(entry.pack, idx)
                if info.get("error"):
                    errors.append(f"{m['name']}: {info['error']}")
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{m['name']}: {type(exc).__name__}: {exc}")
                continue
            subs = {(s.get("entry"), s.get("submesh")): s for s in m.get("submeshes", [])}
            for g in geoms:
                if getattr(g, "lod", 0) != 0:
                    continue
                sub = subs.get((g.entry, g.submesh))
                base = (sub.get("base_material") if sub else None) or g.material
                if (base or "").casefold().startswith(("auto_shadow_caster", "shadowcaster", "shadow_caster", "null")):
                    continue
                image, tex_name, mode, cutoff, recipe = None, None, "opaque", 0.5, ""
                if with_textures:
                    # material-aware preview (eyes, hair cut-outs, opacity) with the model's texture overrides
                    override = {t["param"]: t["texture"] for t in (sub or {}).get("textures") or []
                                if t.get("param") and t.get("texture") and t.get("source") == "model override"}
                    ck = (base.casefold(), tuple(sorted(override.items())))
                    if ck not in tex_cache:
                        try:
                            if fetch is None:
                                fetch = matpreview.TextureFetcher(ctx.catalog, max_dim=max_dim)
                            tex_cache[ck] = matpreview.preview_for_material(ctx, base, fetch, override)
                        except Exception as exc:  # noqa: BLE001
                            tex_cache[ck] = None
                            errors.append(f"{base}: {type(exc).__name__}: {exc}")
                    sp = tex_cache[ck]
                    if sp is not None:
                        if sp.hidden:
                            continue
                        image, mode, cutoff, recipe = sp.image, sp.alpha_mode, sp.cutoff, sp.recipe
                        tex_name = sp.textures.get("diffuse") or sp.textures.get("veins")
                        if sp.image is None and sp.warnings:
                            errors.append(f"{base}: {'; '.join(sp.warnings[:2])}")
                items.append({"key": f"{slot['name']}|{m['name']}|{g.key}", "slot": slot["name"], "geom": g,
                              "texture": image, "texture_name": tex_name, "alpha_mode": mode, "cutoff": cutoff,
                              "recipe": recipe, "material": base})
    return {"error": None, "items": items, "errors": errors}


PREVIEW_PALETTE = [(0.75, 0.62, 0.50), (0.55, 0.65, 0.80), (0.70, 0.75, 0.55), (0.80, 0.55, 0.55),
                   (0.65, 0.55, 0.80), (0.55, 0.78, 0.75)]


def add_preview_items(mv, items: list[dict], slots: list[str], errors: list[str], offset=None,
                      style=None) -> tuple[dict[str, list[str]], int]:
    """Add load_preview() items to a MeshView (colour per slot). *offset* shifts positions (side-by-side views);
    *style(item)* may return {"color", "texture", "alpha_mode", "cutoff"} overrides or False to skip the item.
    Returns ({slot: [keys]}, textured count)."""
    keys: dict[str, list[str]] = {}
    n_tex = 0
    for it in items:
        g = it["geom"]
        col = PREVIEW_PALETTE[(slots.index(it["slot"]) if it["slot"] in slots else 0) % len(PREVIEW_PALETTE)]
        kw = {"texture": it["texture"], "alpha_mode": it.get("alpha_mode", "opaque"),
              "cutoff": it.get("cutoff", 0.5), "color": col}
        if style is not None:
            st = style(it)
            if st is False:
                continue
            kw.update(st or {})
        pos = g.positions if offset is None else np.asarray(g.positions, dtype=np.float32) + np.asarray(
            offset, dtype=np.float32)
        try:
            mv.add_mesh(it["key"], pos, g.indices, normals=g.normals, uv=g.uv, color=kw["color"],
                        texture=kw["texture"], alpha_mode=kw["alpha_mode"], alpha_cutoff=kw["cutoff"])
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{it['key']}: {exc}")
            continue
        n_tex += kw["texture"] is not None
        keys.setdefault(it["slot"], []).append(it["key"])
    return keys, n_tex


EXPORT_MODES = (("split", "Split files (one scene per mesh + DDS)"),
                ("cast", "Single file (rig + meshes + textures)"),
                ("both", "Both"))
EXPORT_FORMATS = (("cast", "Cast"), ("glb", "glTF binary (.glb)"), ("gltf", "glTF (.gltf + .bin)"))


def export_models(ctx, recs: list[dict], out_root: Path, progress=None, cancel: threading.Event | None = None,
                  mode: str = "split", variants: dict[str, str] | None = None, fmt: str = "cast") -> dict:
    """Export each model to `<out_root>/<model stem>/`: the .model JSON as stored and mapping.json (the resolution),
    then per *mode*:
      split  meshes (Cast + mesh.json + raw parts) and textures (DDS + sidecar) of every found reference;
      cast   one `<model>.cast` with the merged skeleton, all chosen meshes and PNG material maps
             (gui/modelcast.py) + `nightrunner_blender_materials.py`;
      both   both of the above.
    *fmt* ("cast" | "glb" | "gltf") is the scene format of both layouts (per-mesh files keep their mesh.json;
    re-import accepts all three). *variants* maps a model member name to the mesh variant to apply. Missing references are listed in
    export_log.txt. `progress(done, total, text)`; `cancel` stops between files."""
    from ..meshdata import export_cast_files, export_raw_parts, safe_name
    from ...codecs import ExtractContext
    from ...texture.codec import TextureCodec
    out_root = Path(out_root)
    summary = {"models": [], "errors": [], "cancelled": False}
    plans = []
    for rec in recs:
        try:
            doc = rec["doc"] if rec.get("doc") is not None else ctx.paks.load_model(rec)
            res = resolve_model(rec_ctx(ctx, rec), doc, (variants or {}).get(rec["name"]), name=rec["name"])
            plans.append((rec, doc, res))
        except Exception as exc:  # noqa: BLE001
            summary["errors"].append(f"{rec['name']}: {type(exc).__name__}: {exc}")
    split = mode in ("split", "both")
    single = mode in ("cast", "both")
    total = sum(1 + (len(r["unique_meshes"]) + len(r["unique_textures"]) if split else 0)
                + (1 if single else 0) for _, _, r in plans) or 1
    done = 0

    def step(text: str) -> bool:
        nonlocal done
        done += 1
        if progress:
            progress(done, total, text)
        return bool(cancel and cancel.is_set())

    for rec, doc, res in plans:
        stem = rec["basename"].rsplit(".", 1)[0]
        mdir = unique_path(out_root / safe_name(stem))
        mdir.mkdir(parents=True)
        log = [f"model {rec['name']} from {Path(rec['pak']).name}"
               + (f" (overridden by {', '.join(rec['overridden_by'])})" if rec.get("overridden_by") else "")]
        entry = {"name": rec["name"], "dir": str(mdir), "meshes": 0, "textures": 0, "missing": 0, "failed": 0}
        try:
            ix = ctx.paks.index(rec["pak"])
            (mdir / rec["basename"]).write_bytes(ix.read(rec["member"]))
        except Exception as exc:  # noqa: BLE001
            (mdir / rec["basename"]).write_text(json.dumps(doc, indent=2), encoding="utf-8")
            log.append(f"raw member read failed ({exc}); JSON re-serialised instead")
        (mdir / "mapping.json").write_text(json.dumps(json_safe(res), indent=1), encoding="utf-8")
        if step(f"{stem}: model"):
            summary["cancelled"] = True
        for m in (res["unique_meshes"] if split else []):
            if summary["cancelled"]:
                break
            if not m["gids"]:
                log.append(f"MISSING mesh {m['name']}")
                entry["missing"] += 1
            else:
                try:
                    e, idx = ctx.catalog.split(m["gids"][0])
                    base = safe_name(m["name"].removesuffix(".msh"))
                    d = mdir / "meshes" / base
                    d.mkdir(parents=True, exist_ok=True)
                    export_cast_files(e.pack, idx, d / f"{base}.{fmt}")
                    export_raw_parts(e.pack, idx, d / "raw")
                    entry["meshes"] += 1
                    log.append(f"mesh {m['name']} <- {e.label}")
                except Exception as exc:  # noqa: BLE001
                    entry["failed"] += 1
                    log.append(f"FAILED mesh {m['name']}: {type(exc).__name__}: {exc}")
            if step(f"{stem}: {m['name']}"):
                summary["cancelled"] = True
        for t in (res["unique_textures"] if split else []):
            if summary["cancelled"]:
                break
            if not t["gids"]:
                log.append(f"MISSING texture {t['name']}")
                entry["missing"] += 1
            else:
                try:
                    e, idx = ctx.catalog.split(t["gids"][0])
                    d = mdir / "textures"
                    d.mkdir(parents=True, exist_ok=True)
                    TextureCodec().extract(ExtractContext(pack=e.pack, rpx_dir=d), e.pack.resource(idx), d)
                    entry["textures"] += 1
                    log.append(f"texture {t['name']} <- {e.label}")
                except Exception as exc:  # noqa: BLE001
                    entry["failed"] += 1
                    log.append(f"FAILED texture {t['name']}: {type(exc).__name__}: {exc}")
            if step(f"{stem}: {t['name']}"):
                summary["cancelled"] = True
        if single and not summary["cancelled"]:
            try:
                from ..modelcast import export_model_cast
                cr = export_model_cast(ctx, res, mdir, stem, cancel=cancel, formats=(fmt,),
                                       progress=lambda t: progress and progress(done, total, f"{stem}: {t}"))
                entry["cast"] = cr.get("cast")
                if cr.get("cancelled"):
                    summary["cancelled"] = True
                if cr.get("cast"):
                    log.append(f"single {fmt} {Path(cr['cast']).name}: {cr.get('meshes', 0)} submeshes, "
                               f"{(cr.get('skeleton') or {}).get('merged_bones', 0)} bones, "
                               f"{len(cr.get('materials') or {})} materials (variant {res.get('variant')})")
                    if not split:
                        entry["meshes"] += sum(1 for p in cr.get("parts") or [])
                        entry["textures"] += sum(bool(v.get("albedo")) + bool(v.get("normal"))
                                                 for v in (cr.get("materials") or {}).values())
                for pr in cr.get("problems") or []:
                    log.append(f"cast: {pr}")
                    if "missing" in pr and not split:
                        entry["missing"] += 1
            except Exception as exc:  # noqa: BLE001
                entry["failed"] += 1
                log.append(f"FAILED single Cast: {type(exc).__name__}: {exc}")
            if step(f"{stem}: single Cast"):
                summary["cancelled"] = True
        if summary["cancelled"]:
            log.append("CANCELLED")
        (mdir / "export_log.txt").write_text("\n".join(log) + "\n", encoding="utf-8")
        summary["models"].append(entry)
        if summary["cancelled"]:
            break
    return summary


class _ExportSignals(QObject):
    progress = Signal(int, int, str)
    finished = Signal(object)


class _ExportJob(QRunnable):
    def __init__(self, ctx, recs, out_root, cancel, mode="split", variants=None, fmt="cast"):
        super().__init__()
        self.sig = _ExportSignals()
        self.ctx, self.recs, self.out_root, self.cancel = ctx, recs, out_root, cancel
        self.mode, self.variants, self.fmt = mode, variants, fmt

    def run(self):
        try:
            res = export_models(self.ctx, self.recs, self.out_root, self.sig.progress.emit, self.cancel,
                                mode=self.mode, variants=self.variants, fmt=self.fmt)
        except Exception as exc:  # noqa: BLE001
            res = {"models": [], "errors": [f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"],
                   "cancelled": False}
        self.sig.finished.emit(res)


# ---- the tab ------------------------------------------------------------------------------------------------------

class Tab(QWidget):
    TITLE = TITLE
    exportFinished = Signal(object)

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.records: list[dict] = []
        self.current: dict | None = None          # record
        self.result: dict | None = None           # load_and_resolve output
        self._pending_open: str | None = None
        self._export_job: _ExportJob | None = None
        self._cancel = threading.Event()
        self._slot_keys: dict[str, list[str]] = {}
        self._build()
        self._search_deb = Debouncer(self._run_search, 200, self)
        self.search.changed.connect(self._search_deb.trigger)
        self.ctx.runner.submit(("models", "list"), self._list_models, on_done=self._on_models,
                               on_error=self._on_list_error)

    # ---- layout ---------------------------------------------------------------------------------------------
    def _build(self) -> None:
        root = QHBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        root.addWidget(split)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.search = SearchBar("search .model members")
        self.search.pack_combo.hide()
        ll.addWidget(self.search)
        self.list_model = RefListModel([
            Column("Model", lambda i: self.records[i]["name"]),
            Column("PAK", lambda i: Path(self.records[i]["pak"]).name),
            Column("Overridden by", lambda i: ", ".join(self.records[i]["overridden_by"])),
        ], self)
        self.view = QTableView()
        self.view.setModel(self.list_model)
        self.view.setSelectionBehavior(QTableView.SelectRows)
        self.view.setSelectionMode(QTableView.SingleSelection)
        self.view.verticalHeader().hide()
        self.view.setTextElideMode(Qt.ElideLeft)
        self.view.setWordWrap(False)
        self.view.verticalHeader().setDefaultSectionSize(20)
        self.view.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.view.selectionModel().currentRowChanged.connect(self._on_row)
        self.list_model.checkedChanged.connect(self._update_count)
        ll.addWidget(self.view, 1)
        row = QHBoxLayout()
        self.btn_check = QPushButton("Check visible")
        self.btn_uncheck = QPushButton("Clear checks")
        self.btn_export_checked = QPushButton("Export Checked…")
        self.btn_export_sel = QPushButton("Export Selected…")
        for b in (self.btn_check, self.btn_uncheck, self.btn_export_checked, self.btn_export_sel):
            row.addWidget(b)
        ll.addLayout(row)
        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Export as"))
        self.export_mode = QComboBox()
        for key, label in EXPORT_MODES:
            self.export_mode.addItem(label, key)
        i = self.export_mode.findData(self.ctx.settings.value("models/export_mode", "split"))
        self.export_mode.setCurrentIndex(max(0, i))
        self.export_mode.setToolTip("Split: per-mesh Casts (re-importable to packs) + original DDS.\n"
                                    "Single file: one scene with the merged rig, every chosen mesh and PNG materials "
                                    "(run nightrunner_blender_materials.py after importing for hair/eye alpha).")
        self.export_mode.currentIndexChanged.connect(
            lambda _: self.ctx.settings.setValue("models/export_mode", self.export_mode.currentData()))
        mrow.addWidget(self.export_mode, 1)
        self.export_fmt = QComboBox()
        for key, label in EXPORT_FORMATS:
            self.export_fmt.addItem(label, key)
        self.export_fmt.setCurrentIndex(max(0, self.export_fmt.findData(
            self.ctx.settings.value("models/export_format", "cast"))))
        self.export_fmt.setToolTip("Scene format. glTF / GLB carry hair / eye alpha natively and open in any "
                                   "viewer; all three re-import (per-mesh files and Split edited…).")
        self.export_fmt.currentIndexChanged.connect(
            lambda _: self.ctx.settings.setValue("models/export_format", self.export_fmt.currentData()))
        mrow.addWidget(self.export_fmt)
        self.btn_split = QPushButton("Split edited…")
        self.btn_split.setToolTip("Turn a Single-Cast export (e.g. edited in Blender and exported with the Cast "
                                  "add-on) back into per-mesh model.cast folders for `nr build`")
        self.btn_split.clicked.connect(self._ask_split)
        mrow.addWidget(self.btn_split)
        ll.addLayout(mrow)
        erow = QHBoxLayout()
        self.progress = QProgressBar()
        self.progress.hide()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.hide()
        erow.addWidget(self.progress, 1)
        erow.addWidget(self.btn_cancel)
        ll.addLayout(erow)
        self.btn_check.clicked.connect(lambda: self.list_model.check_all_visible(True))
        self.btn_uncheck.clicked.connect(self.list_model.clear_checks)
        self.btn_export_checked.clicked.connect(lambda: self._ask_export(checked=True))
        self.btn_export_sel.clicked.connect(lambda: self._ask_export(checked=False))
        self.btn_cancel.clicked.connect(self._cancel.set)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.header = QLabel("Select a model.")
        self.header.setTextFormat(Qt.RichText)
        self.header.setWordWrap(True)
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        rl.addWidget(self.header)
        bar = QHBoxLayout()
        bar.addWidget(QLabel("Mesh variant:"))
        self.variant = QComboBox()
        self.variant.addItem("Default")
        self.variant.setMinimumContentsLength(16)
        self.variant.currentTextChanged.connect(self._on_variant)
        bar.addWidget(self.variant)
        self.only_missing = QCheckBox("only missing")
        self.only_overrides = QCheckBox("only overrides")
        self.only_missing.toggled.connect(self._apply_filter)
        self.only_overrides.toggled.connect(self._apply_filter)
        bar.addWidget(self.only_missing)
        bar.addWidget(self.only_overrides)
        bar.addStretch(1)
        self.btn_expand = QPushButton("Expand all")
        self.btn_expand.clicked.connect(lambda: self.tree.expandAll())
        bar.addWidget(self.btn_expand)
        rl.addLayout(bar)

        self.tabs = QTabWidget()
        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["Item", "Value", "Source", "Status"])
        self.tree.setUniformRowHeights(True)
        self.tree.header().setSectionResizeMode(C_ITEM, QHeaderView.Interactive)
        self.tree.setColumnWidth(C_ITEM, 280)
        self.tree.setColumnWidth(C_VALUE, 330)
        self.tree.setColumnWidth(C_SOURCE, 150)
        self.tree.itemDoubleClicked.connect(self._on_double)
        self.tabs.addTab(self.tree, "Mapping")

        v3 = QWidget()
        vl = QHBoxLayout(v3)
        vl.setContentsMargins(0, 0, 0, 0)
        side = QVBoxLayout()
        self.chk_textures = QCheckBox("diffuse textures")
        self.chk_textures.setChecked(True)
        self.chk_textures.toggled.connect(self._load_preview)
        self.slot_list = QListWidget()
        self.slot_list.setMaximumWidth(220)
        self.slot_list.itemChanged.connect(self._on_slot_toggled)
        self.preview_status = QLabel("")
        self.preview_status.setWordWrap(True)
        self.preview_status.setMaximumWidth(220)
        side.addWidget(self.chk_textures)
        side.addWidget(QLabel("Slots:"))
        side.addWidget(self.slot_list, 1)
        side.addWidget(self.preview_status)
        vl.addLayout(side)
        self.meshview = None
        try:
            from ..meshview import MeshView
            self.meshview = MeshView()
            vl.addWidget(self.meshview, 1)
        except Exception as exc:  # noqa: BLE001 - viewer module optional / GL may be unavailable
            ph = QLabel(f"3D preview unavailable ({type(exc).__name__}: {exc})")
            ph.setAlignment(Qt.AlignCenter)
            ph.setWordWrap(True)
            vl.addWidget(ph, 1)
        self.tabs.addTab(v3, "3D")

        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.raw.setFont(mono_font())
        self.raw.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.tabs.addTab(self.raw, "Raw JSON")

        summ = QWidget()
        sl = QHBoxLayout(summ)
        self.tex_text = QPlainTextEdit()
        self.mat_text = QPlainTextEdit()
        for title, w in (("Unique textures", self.tex_text), ("Unique materials / meshes", self.mat_text)):
            w.setReadOnly(True)
            w.setFont(mono_font())
            w.setLineWrapMode(QPlainTextEdit.NoWrap)
            col = QVBoxLayout()
            hr = QHBoxLayout()
            hr.addWidget(QLabel(title))
            hr.addStretch(1)
            b = QPushButton("Copy")
            b.clicked.connect(lambda _=False, w=w: QGuiApplication.clipboard().setText(w.toPlainText()))
            hr.addWidget(b)
            col.addLayout(hr)
            col.addWidget(w, 1)
            sl.addLayout(col)
        self.tabs.addTab(summ, "Summary")
        rl.addWidget(self.tabs, 1)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        split.addWidget(right)
        split.setSizes([480, 1100])

    # ---- model list -----------------------------------------------------------------------------------------
    def _list_models(self) -> list[dict]:
        return self.ctx.paks.models()

    def _on_list_error(self, err: str) -> None:
        self.header.setText(f"<b>Cannot list .model members:</b> {err.splitlines()[0]}")

    def _on_models(self, recs: list[dict]) -> None:
        self.records = list(recs)
        if not self.records:
            self.header.setText("No .model documents found (no dataN.pak in the game folder?).")
        self._run_search()
        if self._pending_open:
            name, self._pending_open = self._pending_open, None
            self.open_model(name)

    def _run_search(self) -> None:
        text = self.search.text()
        names = [r["name"].lower() for r in self.records]
        self.ctx.runner.submit(("models", "search"), _filter, names, text, on_done=self._on_search)

    def _on_search(self, rows: np.ndarray) -> None:
        cur = self.current
        self.list_model.set_gids(rows)
        self._update_count()
        if cur is not None:
            i = self._index_of(cur)
            r = self.list_model.row_of(i) if i is not None else -1
            if r >= 0:
                self.view.selectionModel().blockSignals(True)
                self.view.selectRow(r)
                self.view.selectionModel().blockSignals(False)

    def _update_count(self, *_):
        self.search.set_count(len(self.list_model.gids), len(self.records), len(self.list_model.checked))

    def _index_of(self, rec: dict) -> int | None:
        for i, r in enumerate(self.records):
            if r is rec:
                return i
        return None

    def _on_row(self, cur, _prev=None) -> None:
        if not cur.isValid():
            return
        self._select_record(self.records[self.list_model.gid_at(cur.row())])

    def _select_record(self, rec: dict) -> None:
        if rec is self.current:
            return
        self.current = rec
        self.variant.blockSignals(True)
        self.variant.clear()
        self.variant.addItem("Default")
        self.variant.blockSignals(False)
        self._resolve()

    # ---- resolution -----------------------------------------------------------------------------------------
    def _resolve(self) -> None:
        if self.current is None:
            return
        v = self.variant.currentText() or None
        self.header.setText(f"Resolving <b>{self.current['name']}</b>…")
        self.ctx.runner.submit(("models", "resolve"), load_and_resolve, self.ctx, self.current, v,
                               on_done=self._on_resolved, on_error=self._on_resolve_error)

    def _on_resolve_error(self, err: str) -> None:
        self.header.setText(f"<b>Error:</b> {err.splitlines()[0]}")

    def _on_resolved(self, out: dict) -> None:
        if out["rec"] is not self.current:
            return
        self.result = out
        doc = out.get("doc")
        self.raw.setPlainText(json.dumps(doc, indent=2) if doc is not None else "")
        if out.get("error"):
            self.header.setText(f"<b>{out['rec']['name']}</b>: <span style='color:#d66'>"
                                f"{out['error'].splitlines()[0]}</span>")
            self.tree.clear()
            return
        res = out["resolution"]
        self._fill_variants(res)
        self._fill_header(out["rec"], res)
        self._fill_tree(res)
        self._fill_summary(res)
        self._fill_slots(res)
        if self.tabs.currentIndex() == 1:
            self._load_preview()
        else:
            self._preview_dirty = True

    def _fill_variants(self, res: dict) -> None:
        names = res.get("variants") or ["Default"]
        cur = res.get("variant") or "Default"
        self.variant.blockSignals(True)
        self.variant.clear()
        self.variant.addItems(names)
        i = self.variant.findText(cur)
        self.variant.setCurrentIndex(max(0, i))
        self.variant.blockSignals(False)
        self.variant.setEnabled(len(names) > 1)

    def _fill_header(self, rec: dict, res: dict) -> None:
        s = res["summary"]

        def frac(a, b):
            color = "#6b6" if a == b else "#d66" if a == 0 else "#db4"
            return f"<span style='color:{color}'>{a}/{b}</span>"

        ov = ""
        if rec.get("overridden_by"):
            ov = f" &nbsp;<span style='color:#db4'>overridden by {', '.join(rec['overridden_by'])}</span>"
        sk = res["skeleton"]
        self.header.setText(
            f"<b>{rec['name']}</b> <i>({Path(rec['pak']).name})</i>{ov}<br>"
            f"meshes {frac(s['meshes_found'], s['meshes'])} found &nbsp;·&nbsp; "
            f"textures {frac(s['textures_found'], s['textures'])} found &nbsp;·&nbsp; "
            f"materials in SDB {frac(s['materials_in_sdb'], s['materials'])} &nbsp;·&nbsp; "
            f"overrides {s['overrides']} ({s['texture_overrides']} texture) &nbsp;·&nbsp; "
            f"material remaps {s['material_remaps']} &nbsp;·&nbsp; slots {s['slots']}, submeshes {s['submeshes']}"
            f"<br>skeleton {sk['name'] or '—'}: {sk['status']} &nbsp;·&nbsp; variant: {res['variant']}")

    # ---- tree ---------------------------------------------------------------------------------------------------
    def _item(self, parent, item: str, value: Any = "", source: str = "", status: str = "", kind: str | None = None,
              ref: Any = None, missing: bool = False, override: bool = False) -> QTreeWidgetItem:
        it = QTreeWidgetItem(parent, [str(item), "" if value is None else str(value), source, status])
        it.setToolTip(C_VALUE, str(value) if value is not None else "")
        it.setToolTip(C_STATUS, status)
        if kind:
            it.setData(0, ROLE_KIND, kind)
            it.setData(0, ROLE_REF, ref)
        it.setData(0, ROLE_FLAGS, {"missing": missing, "override": override})
        if status:
            col = RED if missing or status.startswith(("missing", "not in SDB")) else \
                GREEN if status.startswith(("found", "in SDB", "applied")) or _all_found(status) else AMBER
            it.setForeground(C_STATUS, QBrush(col))
        if override:
            it.setForeground(C_SOURCE, QBrush(AMBER))
        return it

    def _fill_tree(self, res: dict) -> None:
        t = self.tree
        t.setUpdatesEnabled(False)
        t.clear()
        sk = res["skeleton"]
        if sk["name"]:
            self._item(t, "skeleton (preset)", sk["name"], "model", sk["status"], "mesh",
                       sk["gids"][0] if sk["gids"] else None, missing=not sk["gids"])
        for n in res.get("notes") or []:
            self._item(t, "note", n, "", "warning")
        if any("not available" in (m.get("variants_error") or "") for sl in res["slots"] for m in sl["meshes"]):
            self._item(t, "note", "mesh variant decoder (nightrunner.mesh.variants) not available: variants not listed",
                       "", "warning")
        for slot in res["slots"]:
            n_found = sum(1 for m in slot["meshes"] if m["gids"])
            st = f"{n_found}/{len(slot['meshes'])} meshes found" if slot["meshes"] else "empty slot"
            si = self._item(t, f"slot {slot['name']}", f"uid {slot['slotUid']}  filter {slot['filterText']!r}",
                            "model", st,
                            missing=n_found == 0 and bool(slot["meshes"]))
            if slot.get("problem"):
                self._item(si, "problem", slot["problem"], "model", "warning")
            for m in slot["meshes"]:
                self._fill_mesh(si, m)
        t.expandToDepth(1)
        t.setUpdatesEnabled(True)
        self._apply_filter()

    def _fill_mesh(self, parent, m: dict) -> None:
        role = "selected" if m["selected"] else ("chosen (first)" if m["chosen"] else "alternate")
        mi = self._item(parent, f"mesh {m['name']}", role, "model", m["status"], "mesh",
                        m["gids"][0] if m["gids"] else None, missing=not m["gids"])
        for g, p in zip(m["gids"][1:], m["providers"][1:]):
            self._item(mi, "also provided by", p, "catalog", "found", "mesh", g)
        verr = m.get("variants_error")
        if verr and "not available" in verr:
            verr = None                                    # reported once at the top of the tree
        for msg in [m.get("error"), verr, *m.get("problems", [])]:
            if msg:
                self._item(mi, "problem", msg, "mesh", "warning")
        if m.get("variants"):
            vi = self._item(mi, "variants (part 0x12)", ", ".join(v["name"] for v in m["variants"]), "mesh")
            for v in m["variants"]:
                mp = v.get("map") or {}
                self._item(vi, v["name"], "; ".join(f"{a} → {b}" for a, b in mp.items()) or "(no change)",
                           "variant")
        for s in m["submeshes"]:
            self._fill_submesh(mi, s)

    def _fill_submesh(self, parent, s: dict) -> None:
        if s["entry"] is not None:
            label = f"submesh e{s['entry']}/s{s['submesh']}"
            where = f"material slot {s['slot']}, {s.get('triangles') or 0} tris"
        else:
            label = f"material row #{s['number']}"
            where = "mesh missing: from model materialsData"
        overridden = bool(s["overrides"]) or s["material_source"] in ("model (remap)", "variant")
        sdb = s["sdb"]
        si = self._item(parent, label, s["base_material"], s["material_source"],
                        "in SDB" if sdb["found"] else "not in SDB", "material", s["base_material"],
                        missing=not sdb["found"], override=overridden)
        si.setToolTip(C_ITEM, where)
        self._item(si, "embedded material", s["embedded_material"], s["embedded_source"], where)
        if s.get("variant_material"):
            self._item(si, "variant material", s["variant_material"], "variant", "", "material",
                       s["variant_material"], override=True)
        if s["number"] is not None:
            self._item(si, "join", f"materialsData #{s['number']} → {s['base_material']}"
                       + (f"  (loadFlags {s['loadFlags']})" if s.get("loadFlags") else ""), s["material_source"],
                       "", override=s["material_source"] != "model")
        else:
            self._item(si, "join", "no materialsData row: embedded material applies", "mesh default")
        if s.get("join_error"):
            self._item(si, "join problem", s["join_error"], "model", "warning")
        for alt in s.get("alternatives") or []:
            self._item(si, "alternative (unselected)", alt, "model", "", "material", alt)
        if sdb["found"]:
            self._item(si, "SDB preset", sdb.get("preset"), "sdb",
                       "non-rendering" if sdb.get("non_rendering") else "")
        for o in s["overrides"]:
            val = o["value"]
            if o["type"] == 7:
                txt = f"{o.get('original') or '∅'} → {val}"
            else:
                txt = f"{o.get('original')} → {val}"
            st = "applied" if o.get("applied") else o.get("note", "not applied")
            self._item(si, f"override {o['name']} ({o['kind']})", txt, "override", st, override=True)
        if sdb["found"] and sdb["parameters"]:
            pi = self._item(si, f"parameters ({len(sdb['parameters'])})", "", "sdb")
            ov = {(o["name"] or "").casefold(): o for o in s["overrides"] if o["type"] != 7}
            for p in sdb["parameters"]:
                o = ov.get((p["name"] or "").casefold())
                if o is not None:
                    self._item(pi, p["name"], f"{p['value']} → {o['value']}", "override",
                               p["type"] or "", override=True)
                else:
                    self._item(pi, p["name"], p["string"] if p["type"] == "string" else p["value"], "sdb",
                               p["type"] or "")
        ti = self._item(si, f"textures ({len(s['textures'])})", "", "sdb")
        for tex in s["textures"]:
            is_ov = tex["source"] == "model override"
            ref = tex["gids"][0] if tex["gids"] else None
            it = self._item(ti, tex["param"], tex["texture"], tex["source"], tex["status"], "texture", ref,
                            missing=not tex["gids"], override=is_ov)
            if tex.get("original"):
                self._item(it, "original (base material)", tex["original"], "sdb",
                           tex.get("original_status", ""), "texture",
                           (tex.get("original_gids") or [None])[0], missing=not tex.get("original_gids"))
            for g, p in zip(tex["gids"][1:], tex["providers"][1:]):
                self._item(it, "also provided by", p, "catalog", "found", "texture", g)
            for o in tex.get("others") or []:
                self._item(it, "other shader variant binds", o, "sdb", "")

    def _apply_filter(self) -> None:
        miss, ovr = self.only_missing.isChecked(), self.only_overrides.isChecked()

        def visit(it: QTreeWidgetItem) -> bool:
            f = it.data(0, ROLE_FLAGS) or {}
            own = (not miss or f.get("missing")) and (not ovr or f.get("override"))
            child_vis = False
            for k in range(it.childCount()):
                child_vis |= visit(it.child(k))
            vis = bool(own) or child_vis
            if (miss or ovr) and own and not child_vis:
                for k in range(it.childCount()):      # show the matching node's details
                    _unhide(it.child(k))
            it.setHidden(not vis)
            return vis

        for i in range(self.tree.topLevelItemCount()):
            visit(self.tree.topLevelItem(i))
        if miss or ovr:
            self.tree.expandToDepth(3)

    def _on_double(self, it: QTreeWidgetItem, _col: int) -> None:
        kind, ref = it.data(0, ROLE_KIND), it.data(0, ROLE_REF)
        if not kind or ref is None:
            return
        if kind == "texture":
            self.ctx.openTexture.emit(int(ref))
        elif kind == "mesh":
            self.ctx.openMesh.emit(int(ref))
        elif kind == "material":
            self.ctx.openMaterial.emit(str(ref))

    def _fill_summary(self, res: dict) -> None:
        lines = [f"# {res['summary']['textures_found']}/{res['summary']['textures']} textures found"]
        for t in res["unique_textures"]:
            flag = "found  " if t["found"] else "MISSING"
            extra = "  (only as overridden original)" if t["only_as_overridden_original"] else ""
            lines.append(f"{flag}  {t['name']}{extra}")
        self.tex_text.setPlainText("\n".join(lines))
        ml = [f"# {res['summary']['materials_in_sdb']}/{res['summary']['materials']} materials in SDB"]
        for m in res["unique_materials"]:
            ml.append(f"{'sdb    ' if m['in_sdb'] else 'NO SDB '}  {m['name']}  [{m.get('preset') or '-'}]"
                      f"  ×{m['users']}")
        ml.append("")
        ml.append(f"# {res['summary']['meshes_found']}/{res['summary']['meshes']} meshes found")
        for m in res["unique_meshes"]:
            ml.append(f"{'found  ' if m['gids'] else 'MISSING'}  {m['name']}"
                      + (f"  <- {', '.join(m['providers'])}" if m["providers"] else ""))
        self.mat_text.setPlainText("\n".join(ml))

    # ---- 3D ---------------------------------------------------------------------------------------------------
    def _fill_slots(self, res: dict) -> None:
        self.slot_list.blockSignals(True)
        self.slot_list.clear()
        for slot in res["slots"]:
            found = any(m["gids"] and m["chosen"] for m in slot["meshes"])
            it = QListWidgetItem(slot["name"] or "?")
            it.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled if found else Qt.NoItemFlags)
            it.setCheckState(Qt.Checked if found else Qt.Unchecked)
            if not found:
                it.setToolTip("no mesh of this slot is available")
            self.slot_list.addItem(it)
        self.slot_list.blockSignals(False)

    def _on_tab_changed(self, i: int) -> None:
        if i == 1 and getattr(self, "_preview_dirty", False):
            self._load_preview()

    def _load_preview(self, *_):
        self._preview_dirty = False
        if self.meshview is None or not self.result or not self.result.get("resolution"):
            return
        self.preview_status.setText("loading meshes…")
        cx = self.ctx if self.result["rec"].get("view") is None else self.result["rec"]["view"]
        self.ctx.runner.submit(("models", "preview"), load_preview, cx, self.result["resolution"],
                               self.chk_textures.isChecked(), on_done=self._on_preview,
                               on_error=lambda e: self.preview_status.setText(e.splitlines()[0]))

    def _on_preview(self, out: dict) -> None:
        mv = self.meshview
        if mv is None:
            return
        mv.clear()
        self._slot_keys = {}
        if out.get("error"):
            self.preview_status.setText(out["error"])
            return
        slots = [self.slot_list.item(i).text() for i in range(self.slot_list.count())]
        self._slot_keys, n_tex = add_preview_items(mv, out["items"], slots, out["errors"])
        for i in range(self.slot_list.count()):
            item = self.slot_list.item(i)
            if item.checkState() != Qt.Checked:
                for k in self._slot_keys.get(item.text(), []):
                    mv.set_visible(k, False)
        mv.frame_all()
        msg = f"{len(out['items'])} submeshes, {n_tex} textured"
        if out.get("errors"):
            msg += f"; {len(out['errors'])} problem(s): " + "; ".join(out["errors"][:3])
        self.preview_status.setText(msg)
        self.preview_info = out

    def _on_slot_toggled(self, item: QListWidgetItem) -> None:
        if self.meshview is None:
            return
        on = item.checkState() == Qt.Checked
        for k in self._slot_keys.get(item.text(), []):
            self.meshview.set_visible(k, on)

    def _on_variant(self, text: str) -> None:
        if text and self.current is not None:
            self._resolve()

    # ---- export -------------------------------------------------------------------------------------------------
    def _ask_export(self, checked: bool) -> None:
        if checked:
            recs = [self.records[i] for i in sorted(self.list_model.checked)]
        else:
            recs = [self.current] if self.current is not None else []
        if not recs:
            QMessageBox.information(self, TITLE, "Nothing to export: check or select a model first.")
            return
        d = QFileDialog.getExistingDirectory(self, "Export models to", self.ctx.export_dir())
        if d:
            self.ctx.set_export_dir(d)
            self.start_export(recs, Path(d))

    def start_export(self, recs: list[dict], out_root: Path, mode: str | None = None, fmt: str | None = None) -> bool:
        """Run `export_models` in the thread pool with progress + cancel. False if an export is already running.
        The selected model is exported with the mesh variant currently chosen in the tab."""
        if self._export_job is not None:
            return False
        self._cancel = threading.Event()
        mode = mode or self.export_mode.currentData() or "split"
        variants = {}
        if self.current is not None and self.variant.currentText() not in ("", "Default"):
            variants[self.current["name"]] = self.variant.currentText()
        fmt = fmt or self.export_fmt.currentData() or "cast"
        job = _ExportJob(self.ctx, recs, Path(out_root), self._cancel, mode, variants, fmt)
        job.setAutoDelete(False)
        job.sig.progress.connect(self._on_export_progress)
        job.sig.finished.connect(self._on_export_done)
        self._export_job = job
        self.progress.setRange(0, 0)
        self.progress.show()
        self.btn_cancel.show()
        self.btn_export_checked.setEnabled(False)
        self.btn_export_sel.setEnabled(False)
        self.ctx.runner.pool.start(job)
        return True

    def _on_export_progress(self, done: int, total: int, text: str) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"%v / %m  {text}")

    def _on_export_done(self, res: dict) -> None:
        self._export_job = None
        self.progress.hide()
        self.btn_cancel.hide()
        self.btn_export_checked.setEnabled(True)
        self.btn_export_sel.setEnabled(True)
        n = res["models"]
        msg = (f"exported {len(n)} model(s): {sum(m['meshes'] for m in n)} meshes, "
               f"{sum(m['textures'] for m in n)} textures, {sum(m['missing'] for m in n)} missing "
               f"(see export_log.txt)")
        if res.get("cancelled"):
            msg += " — cancelled"
        if res.get("errors"):
            msg += f"; {len(res['errors'])} error(s): {res['errors'][0].splitlines()[0]}"
        self.ctx.status.emit(msg)
        self.last_export = res
        self.exportFinished.emit(res)

    # ---- split an edited single Cast ------------------------------------------------------------------------
    def _ask_split(self) -> None:
        edited, _ = QFileDialog.getOpenFileName(self, "Edited single-file model (Cast / glTF / GLB)",
                                                self.ctx.export_dir(), "Scenes (*.cast *.glb *.gltf)")
        if not edited:
            return
        folder = Path(edited).parent
        reports = sorted(folder.glob("*.cast.json"))
        if len(reports) == 1:
            report = reports[0]
        else:
            r, _ = QFileDialog.getOpenFileName(self, "Export report (<model>.cast.json) of that model",
                                               str(folder), "Export report (*.cast.json)")
            if not r:
                return
            report = Path(r)
        out = QFileDialog.getExistingDirectory(self, "Write the per-mesh folders to", str(report.parent))
        if not out:
            return
        rpx = None
        if QMessageBox.question(self, TITLE, "Also copy the results into an extracted .rpx tree (ready for "
                                "`nr build`)?") == QMessageBox.Yes:
            rpx = QFileDialog.getExistingDirectory(self, "Extracted .rpx tree (folder with pack.json)", out) or None
        self.start_split(Path(edited), report, Path(out), Path(rpx) if rpx else None)

    def start_split(self, edited: Path, report: Path, out: Path, rpx: Path | None = None) -> None:
        from ...cast.split import split_model_cast
        self.btn_split.setEnabled(False)
        self.progress.setRange(0, 0)
        self.progress.setFormat("splitting…")
        self.progress.show()

        def work():
            return split_model_cast(edited, report, out, rpx=rpx)

        self.ctx.runner.submit("models-split", work, on_done=self._on_split,
                               on_error=lambda e: self._on_split({"error": e.splitlines()[0]}))

    def _on_split(self, res: dict) -> None:
        self.btn_split.setEnabled(True)
        self.progress.hide()
        self.last_split = res
        if res.get("error"):
            QMessageBox.warning(self, TITLE, f"Split failed: {res['error']}")
            return
        lines = []
        for m in res["meshes"]:
            subs = m["submeshes"]
            chk = m.get("check") or {}
            state = ("unchanged" if chk.get("unchanged") else "encodes OK" if chk.get("ok")
                     else f"FAILED: {chk.get('error')}" if chk else "not checked")
            lines.append(f"{m['mesh']}: moved {sum(x.get('moved', 0) for x in subs)}, "
                         f"new {sum(x.get('new', 0) for x in subs)} — {state}"
                         + (f"\n    → {m['rpx']}" if m.get("rpx") else ""))
            lines += [f"    ! {p}" for p in m["problems"]]
        lines += [f"! {p}" for p in res.get("problems", [])]
        box = QMessageBox(QMessageBox.Information, TITLE, f"Split into {len(res['meshes'])} mesh folder(s).")
        box.setDetailedText("\n".join(lines) + "\n\n(split_report.json has everything)")
        box.show()
        self._split_box = box
        self.ctx.status.emit(f"split {len(res['meshes'])} meshes; see split_report.json")

    # ---- slots called by the main window ----------------------------------------------------------------------
    def open_model(self, name: str) -> bool:
        """Select a model by member path (any case, either slash) or basename. Returns False if not found yet."""
        if not self.records:
            self._pending_open = name
            return False
        key = name.replace("\\", "/").casefold()
        base = key.rsplit("/", 1)[-1]
        if not base.endswith((".model", ".models")):
            key, base = key + ".model", base + ".model"
        hits = [i for i, r in enumerate(self.records) if r["name"].replace("\\", "/").casefold() == key]
        if not hits:
            hits = [i for i, r in enumerate(self.records) if r["basename"] == base]
        if not hits:
            self.ctx.status.emit(f"model {name!r} not found")
            return False
        # prefer the overriding provider (last), as the listing documents (precedence not native-verified)
        i = hits[-1]
        row = self.list_model.row_of(i)
        if row < 0:
            self.search.edit.clear()
            self.list_model.set_gids(np.arange(len(self.records)))
            self._update_count()
            row = i
        self.view.selectRow(row)
        self.view.scrollTo(self.list_model.index(row, 0))
        self._select_record(self.records[i])
        return True

    def open_model_doc(self, name: str, doc: dict, extra_packs: list[Path] | None = None) -> bool:
        """Show an in-memory .model document (e.g. a Build override) like a game model. *extra_packs* are loaded
        as user packs and preferred for its lookups (a built mod rpack)."""
        if not isinstance(doc, dict):
            return False
        cat = self.ctx.catalog
        paths = [Path(p) for p in extra_packs or []]
        if paths:
            cat.load(paths, user=True)
        ids = cat.pack_ids(paths)
        view = None
        if ids:
            from types import SimpleNamespace
            view = SimpleNamespace(catalog=cat.prefer(list(reversed(ids))), sdb=self.ctx.sdb, paks=self.ctx.paks)
        base = name.replace("\\", "/").rsplit("/", 1)[-1]
        rec = {"name": name, "member": None, "pak": Path("project"), "basename": base.lower(),
               "overridden_by": [], "doc": doc, "view": view, "wait": ids}
        sm = self.view.selectionModel()
        sm.blockSignals(True)
        self.view.clearSelection()
        sm.blockSignals(False)
        self._select_record(rec)
        return True

    def sdb_changed(self) -> None:
        self._resolve()

    def shutdown(self) -> None:
        self._cancel.set()


def _all_found(status: str) -> bool:
    head = status.split(" ", 1)[0]
    a, _, b = head.partition("/")
    return a.isdigit() and a == b


def _unhide(it: QTreeWidgetItem) -> None:
    it.setHidden(False)
    for k in range(it.childCount()):
        _unhide(it.child(k))


def _filter(names: list[str], text: str) -> np.ndarray:
    words = [w for w in text.lower().split() if w]
    if not words:
        return np.arange(len(names), dtype=np.int64)
    return np.asarray([i for i, n in enumerate(names) if all(w in n for w in words)], dtype=np.int64)
