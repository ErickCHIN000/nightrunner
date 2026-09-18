"""Audio tab: find a sound, see where its audio actually lives, export it.

Left: banks, or the registry's named events, with a search box. Right: the sounds of the selected bank, each with
the container member its audio resolves from, and an export button.

Read-only. Nothing here writes into the game, and there is no injection path yet — see `docs/roadmap.md` for what
that still needs (chiefly whether `mods/audio/` is a real engine path).

Everything that touches a container runs in `ctx.runner`; the GUI thread only builds widgets from finished
results, the same contract as the other tabs (`notes/GUI.md`).
"""
from __future__ import annotations

import tempfile
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QFileDialog, QHBoxLayout, QHeaderView, QLabel,
                               QMenu, QMessageBox, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
                               QVBoxLayout, QWidget)

from ...audio import preview
from ...audio.resolve import SOURCE_BANK, SOURCE_MISSING, AudioIndex
from ..widgets import SearchBar, human_size, mono_font

TITLE = "Audio"

BANK_COLUMNS = ("Bank", "Sounds", "Size")
SOUND_COLUMNS = ("#", "Source id", "Stream", "Audio from", "Size", "Event")


def _cell(text, mono: bool = False) -> QTableWidgetItem:
    it = QTableWidgetItem(str(text))
    it.setFlags(it.flags() & ~Qt.ItemIsEditable)
    if mono:
        it.setFont(mono_font())
    return it


def load_index(directory: Path) -> dict:
    """Worker: open every container and list the banks with their sound counts."""
    idx = AudioIndex(directory)
    rows = []
    for name in idx.bank_names():
        b = idx.bank(name)
        rows.append({"name": name, "sounds": len(b.sounds) if b else 0, "size": len(b.data) if b else 0})
    rows.sort(key=lambda r: -r["sounds"])
    return {"index": idx, "banks": rows, "summary": idx.summary()}


def decode_wem(data: bytes, source_id: int) -> Path:
    """Worker: decode a wem to a wav in a temp folder, for playback. Raises UnsupportedError without a decoder."""
    out = Path(tempfile.gettempdir()) / "nightrunner-audio"
    return preview.decode_to_wav(data, out / f"{source_id}.wav")


def resolve_bank(idx: AudioIndex, name: str) -> dict:
    """Worker: resolve one bank's sounds to where their audio lives, and name them from the registry."""
    return {"bank": name, "rows": idx.resolve_bank(name), "events": idx.event_names_by_sound(name)}


class Tab(QWidget):
    TITLE = TITLE

    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.index: AudioIndex | None = None
        self.banks: list[dict] = []
        self.rows: list = []
        self._export_lock = threading.Lock()
        self._build_ui()
        self._start_load()

    # ---- ui -------------------------------------------------------------------------------------------------
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        head = QHBoxLayout()
        self.header = QLabel("Loading audio containers…")
        self.header.setTextInteractionFlags(Qt.TextSelectableByMouse)
        head.addWidget(self.header, 1)
        self.btn_play = QPushButton("Play")
        self.btn_play.setEnabled(False)
        self.btn_play.clicked.connect(self._play_selected)
        head.addWidget(self.btn_play)
        self.btn_export = QPushButton("Export sound…")
        self.btn_export.setEnabled(False)
        self.btn_export.clicked.connect(self._export_selected)
        head.addWidget(self.btn_export)
        root.addLayout(head)

        split = QSplitter(Qt.Horizontal)
        root.addWidget(split, 1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        self.search = SearchBar("search banks and events")
        self.search.edit.textChanged.connect(self._filter)
        ll.addWidget(self.search)
        self.mode = QComboBox()
        self.mode.addItem("Banks", "banks")
        self.mode.addItem("Events (registry)", "events")
        self.mode.currentIndexChanged.connect(self._filter)
        ll.addWidget(self.mode)
        self.left_table = QTableWidget(0, len(BANK_COLUMNS))
        self.left_table.setHorizontalHeaderLabels(BANK_COLUMNS)
        self.left_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.left_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.left_table.verticalHeader().hide()
        self.left_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        self.left_table.itemSelectionChanged.connect(self._left_selected)
        ll.addWidget(self.left_table, 1)
        self.left_count = QLabel("")
        ll.addWidget(self.left_count)
        split.addWidget(left)

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self.bank_label = QLabel("Select a bank.")
        rl.addWidget(self.bank_label)
        self.sound_table = QTableWidget(0, len(SOUND_COLUMNS))
        self.sound_table.setHorizontalHeaderLabels(SOUND_COLUMNS)
        self.sound_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sound_table.verticalHeader().hide()
        self.sound_table.horizontalHeader().setSectionResizeMode(len(SOUND_COLUMNS) - 1, QHeaderView.Stretch)
        self.sound_table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.sound_table.customContextMenuRequested.connect(self._sound_menu)
        self.sound_table.itemSelectionChanged.connect(self._sound_selected)
        rl.addWidget(self.sound_table, 1)
        self.sound_count = QLabel("")
        rl.addWidget(self.sound_count)
        split.addWidget(right)
        split.setSizes([420, 900])

    # ---- loading ---------------------------------------------------------------------------------------------
    def _audio_dir(self) -> Path | None:
        """`<data>/work/data/audio` of the current install, or None when it is not there."""
        g = getattr(self.ctx, "game", None)
        if g is None:
            return None
        d = Path(g.data) / "work" / "data" / "audio"
        return d if d.is_dir() else None

    def _start_load(self) -> None:
        d = self._audio_dir()
        if d is None:
            self.header.setText("No audio folder for this install.")
            return
        self.ctx.runner.submit(("audio", "load"), load_index, d,
                               on_done=self._on_loaded, on_error=self._on_failed)

    def _on_loaded(self, res: dict) -> None:
        self.index = res["index"]
        self.banks = res["banks"]
        s = res["summary"]
        members = sum(s["containers"].values())
        self.header.setText(f"{s['banks']} banks · {members:,} container members · "
                            f"{s['registry_objects']:,} named objects in the registry")
        self._filter()

    def _on_failed(self, err) -> None:
        self.header.setText(f"Audio unavailable: {err}")

    # ---- left list -------------------------------------------------------------------------------------------
    def _filter(self) -> None:
        q = self.search.edit.text().casefold().strip()
        self.left_table.setSortingEnabled(False)
        if self.mode.currentData() == "events":
            self._fill_events(q)
        else:
            self._fill_banks(q)
        self.left_table.setSortingEnabled(True)

    def _fill_banks(self, q: str) -> None:
        self.left_table.setHorizontalHeaderLabels(BANK_COLUMNS)
        rows = [b for b in self.banks if not q or q in b["name"].casefold()]
        self.left_table.setRowCount(len(rows))
        for r, b in enumerate(rows):
            self.left_table.setItem(r, 0, _cell(b["name"]))
            self.left_table.setItem(r, 1, _cell(f"{b['sounds']:,}"))
            self.left_table.setItem(r, 2, _cell(human_size(b["size"])))
        self.left_count.setText(f"{len(rows):,} of {len(self.banks):,} banks")

    def _fill_events(self, q: str) -> None:
        """The registry's named events, and which preload (bank) each belongs to."""
        self.left_table.setHorizontalHeaderLabels(("Event", "Preload id", "Duration"))
        ph = self.index.registry if self.index else None
        objs = [o for o in (ph.objects if ph else []) if o.kind == "Event" and (not q or q in o.name.casefold())]
        objs = objs[:5000]
        self.left_table.setRowCount(len(objs))
        for r, o in enumerate(objs):
            self.left_table.setItem(r, 0, _cell(o.name))
            self.left_table.setItem(r, 1, _cell(o.attrs.get("preload_id", "")))
            self.left_table.setItem(r, 2, _cell(o.attrs.get("duration", "")))
        total = sum(1 for o in (ph.objects if ph else []) if o.kind == "Event")
        self.left_count.setText(f"{len(objs):,} of {total:,} events" + ("  (capped)" if len(objs) == 5000 else ""))

    def _left_selected(self) -> None:
        items = self.left_table.selectedItems()
        if not items or self.index is None:
            return
        name = self.left_table.item(items[0].row(), 0).text()
        if self.mode.currentData() == "events":
            pid = self.left_table.item(items[0].row(), 1).text()
            bank = self._bank_of_preload(pid)
            if bank is None:
                self.bank_label.setText(f"{name} — preload {pid}, no bank of that name in meta.aesp")
                self.sound_table.setRowCount(0)
                return
            name = bank
        self.ctx.runner.submit(("audio", "resolve"), resolve_bank, self.index, name,
                               on_done=self._on_resolved, on_error=self._on_failed)

    def _bank_of_preload(self, preload_id: str) -> str | None:
        ph = self.index.registry if self.index else None
        if ph is None:
            return None
        for p in ph.preloads:
            if str(p.id) == preload_id:
                m = self.index.containers.get("meta")
                hit = m.find(p.name) if m else None
                return hit.name if hit else None
        return None

    # ---- sounds ----------------------------------------------------------------------------------------------
    def _on_resolved(self, res: dict) -> None:
        self.rows = res["rows"]
        name = res["bank"]
        from_bank = sum(1 for r in self.rows if r.source == SOURCE_BANK)
        missing = sum(1 for r in self.rows if r.source == SOURCE_MISSING)
        named = len(res.get("events") or {})
        self.bank_label.setText(f"<b>{name}</b> — {len(self.rows):,} sounds, {from_bank:,} baked into the bank, "
                                f"{missing:,} not in this install, {named:,} named by an event")
        self.bank_label.setTextFormat(Qt.RichText)
        events = res.get("events") or {}
        self.sound_table.setRowCount(len(self.rows))
        for r, row in enumerate(self.rows):
            self.sound_table.setItem(r, 0, _cell(row.sound_index))
            self.sound_table.setItem(r, 1, _cell(row.source_id, mono=True))
            self.sound_table.setItem(r, 2, _cell(row.stream_type_name))
            self.sound_table.setItem(r, 3, _cell("bank DIDX" if row.source == SOURCE_BANK else
                                                 "—" if row.source == SOURCE_MISSING else f"{row.source}.aesp"))
            self.sound_table.setItem(r, 4, _cell(human_size(row.size) if row.size else ""))
            self.sound_table.setItem(r, 5, _cell(", ".join(events.get(row.sound_id, []))))
        self.sound_table.resizeColumnsToContents()
        self.sound_count.setText(f"{len(self.rows):,} sounds")

    def _sound_selected(self) -> None:
        on = bool(self.sound_table.selectedItems())
        self.btn_export.setEnabled(on)
        self.btn_play.setEnabled(on and preview.available())
        if on and not preview.available():
            self.btn_play.setToolTip(preview.INSTALL_HINT)
        else:
            self.btn_play.setToolTip("Decode and play this sound")

    def _selected_row(self):
        items = self.sound_table.selectedItems()
        if not items or not self.rows:
            return None
        i = items[0].row()
        return self.rows[i] if 0 <= i < len(self.rows) else None

    def _sound_menu(self, pos) -> None:
        idx = self.sound_table.indexAt(pos)
        if not idx.isValid() or not self.rows:
            return
        row = self.rows[idx.row()]
        m = QMenu(self)
        play = m.addAction("Play", lambda: self._play(row))
        play.setEnabled(row.found and preview.available())
        if not preview.available():
            play.setToolTip(preview.INSTALL_HINT)
        act = m.addAction("Export .wem…", lambda: self._export(row))
        act.setEnabled(row.found)
        m.addAction("Copy source id", lambda: QGuiApplication.clipboard().setText(str(row.source_id)))
        m.exec(self.sound_table.viewport().mapToGlobal(pos))

    # ---- playback --------------------------------------------------------------------------------------------
    def _player(self):
        """QtMultimedia is created on first use: it pulls in a backend, and most sessions never press Play."""
        if getattr(self, "_media", None) is None:
            from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
            self._audio_out = QAudioOutput()
            self._media = QMediaPlayer()
            self._media.setAudioOutput(self._audio_out)
            self._media.errorOccurred.connect(
                lambda _e, msg: self.ctx.status.emit(f"Playback failed: {msg}"))
        return self._media

    def _play_selected(self) -> None:
        row = self._selected_row()
        if row is not None:
            self._play(row)

    def _play(self, row) -> None:
        if not row.found:
            QMessageBox.information(self, TITLE, "That sound's audio is not in this install's containers.")
            return
        data = self.index.read_audio(row)
        if data is None:
            QMessageBox.warning(self, TITLE, "Could not read that sound's audio.")
            return
        self.btn_play.setEnabled(False)
        self.ctx.runner.submit(("audio", "decode"), decode_wem, data, row.source_id,
                               on_done=self._on_decoded,
                               on_error=lambda exc: self._on_decode_failed(exc))

    def _on_decoded(self, wav: Path) -> None:
        from PySide6.QtCore import QUrl
        self.btn_play.setEnabled(bool(self.sound_table.selectedItems()) and preview.available())
        p = self._player()
        p.stop()
        p.setSource(QUrl.fromLocalFile(str(wav)))
        p.play()
        self.ctx.status.emit(f"Playing {wav.name}")

    def _on_decode_failed(self, exc) -> None:
        self.btn_play.setEnabled(bool(self.sound_table.selectedItems()) and preview.available())
        QMessageBox.information(self, TITLE, str(exc))

    # ---- export ----------------------------------------------------------------------------------------------
    def _export_selected(self) -> None:
        row = self._selected_row()
        if row is not None:
            self._export(row)

    def _export(self, row) -> None:
        if not row.found:
            QMessageBox.information(self, TITLE, "That sound's audio is not in this install's containers.")
            return
        start = str(Path(self.ctx.export_dir() or ".") / f"{row.source_id}.wem")
        dest, _ = QFileDialog.getSaveFileName(self, "Export .wem", start, "Wwise audio (*.wem)")
        if not dest:
            return
        self.ctx.set_export_dir(str(Path(dest).parent))
        data = self.index.read_audio(row)
        if data is None:
            QMessageBox.warning(self, TITLE, "Could not read that sound's audio.")
            return
        Path(dest).write_bytes(data)
        self.ctx.status.emit(f"Exported {row.source_id}.wem ({human_size(len(data))}) to {dest}")

    # ---- lifecycle -------------------------------------------------------------------------------------------
    def shutdown(self) -> None:
        if getattr(self, "_media", None) is not None:
            from PySide6.QtCore import QUrl
            self._media.stop()
            self._media.setSource(QUrl())          # drops the wav handle; Windows keeps it open otherwise
            self._media = None
            self._audio_out = None
        if self.index is not None:
            self.index.close()
            self.index = None
