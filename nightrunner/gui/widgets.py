"""Reusable pieces for the asset tabs: a virtual, checkable list over gids, a search bar, raw-part writing."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PySide6.QtCore import QAbstractTableModel, QEvent, QModelIndex, QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QLineEdit, QWidget

from ..container.rp6l import Pack


def human_size(n: int | None) -> str:
    if n is None:
        return ""
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{int(f):,} B" if unit == "B" else f"{f:,.1f} {unit}"
        f /= 1024
    return str(n)


def mono_font(size: int = 9) -> QFont:
    f = QFont("Consolas", size)
    f.setStyleHint(QFont.Monospace)
    return f


@dataclass
class Column:
    title: str
    value: Callable[[int], object]          # gid -> display value (called only for visible rows)
    align_right: bool = False
    mono: bool = False


class RefListModel(QAbstractTableModel):
    """A flat table over an array of gids. Rows are materialised only when Qt paints them, so 100k+ rows scroll
    smoothly. Column 0 is checkable; the check set survives re-filtering (it is keyed by gid)."""
    checkedChanged = Signal(int)            # number of checked gids

    def __init__(self, columns: list[Column], parent=None):
        super().__init__(parent)
        self.columns = columns
        self.gids = np.zeros(0, dtype=np.int64)
        self.checked: set[int] = set()
        self._mono = mono_font()
        self._cache: dict[tuple[int, int], object] = {}

    def set_gids(self, gids) -> None:
        self.beginResetModel()
        self.gids = np.asarray(gids, dtype=np.int64)
        self._cache.clear()
        self.endResetModel()

    def gid_at(self, row: int) -> int:
        return int(self.gids[row])

    def row_of(self, gid: int) -> int:
        hits = np.flatnonzero(self.gids == gid)
        return int(hits[0]) if len(hits) else -1

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self.gids)

    def columnCount(self, parent=QModelIndex()):
        return len(self.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self.columns[section].title
        return None

    def flags(self, index):
        f = Qt.ItemIsEnabled | Qt.ItemIsSelectable
        if index.column() == 0:
            f |= Qt.ItemIsUserCheckable
        return f

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
        gid = int(self.gids[index.row()])
        col = self.columns[index.column()]
        if role == Qt.DisplayRole:
            key = (gid, index.column())
            v = self._cache.get(key)
            if v is None:
                try:
                    v = col.value(gid)
                except Exception as exc:  # noqa: BLE001
                    v = f"<{type(exc).__name__}>"
                v = "" if v is None else str(v)
                if len(self._cache) > 20000:
                    self._cache.clear()
                self._cache[key] = v
            return v
        if role == Qt.CheckStateRole and index.column() == 0:
            return Qt.Checked if gid in self.checked else Qt.Unchecked
        if role == Qt.TextAlignmentRole and col.align_right:
            return int(Qt.AlignRight | Qt.AlignVCenter)
        if role == Qt.FontRole and col.mono:
            return self._mono
        if role == Qt.UserRole:
            return gid
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.CheckStateRole and index.column() == 0:
            gid = int(self.gids[index.row()])
            if Qt.CheckState(value) == Qt.Checked:
                self.checked.add(gid)
            else:
                self.checked.discard(gid)
            self.dataChanged.emit(index, index, [Qt.CheckStateRole])
            self.checkedChanged.emit(len(self.checked))
            return True
        return False

    def set_checked(self, gids, on: bool) -> None:
        g = [int(x) for x in gids]
        if on:
            self.checked.update(g)
        else:
            self.checked.difference_update(g)
        if len(self.gids):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.gids) - 1, 0), [Qt.CheckStateRole])
        self.checkedChanged.emit(len(self.checked))

    def check_all_visible(self, on: bool = True) -> None:
        self.set_checked(self.gids.tolist(), on)

    def clear_checks(self) -> None:
        self.checked.clear()
        if len(self.gids):
            self.dataChanged.emit(self.index(0, 0), self.index(len(self.gids) - 1, 0), [Qt.CheckStateRole])
        self.checkedChanged.emit(0)


class SearchBar(QWidget):
    """Text box + pack combo + result counter. Emits `changed` (debounced by the owner)."""
    changed = Signal()

    def __init__(self, placeholder: str = "search…", parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder + "   (space-separated words must all match)")
        self.edit.setClearButtonEnabled(True)
        self.pack_combo = QComboBox()
        self.pack_combo.addItem("all packs", None)
        self.pack_combo.setMinimumContentsLength(18)
        self.pack_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.count = QLabel()
        lay.addWidget(self.edit, 1)
        lay.addWidget(self.pack_combo)
        lay.addWidget(self.count)
        self.edit.textChanged.connect(self.changed)
        self.pack_combo.currentIndexChanged.connect(self.changed)

    def text(self) -> str:
        return self.edit.text()

    def packs(self):
        pid = self.pack_combo.currentData()
        return None if pid is None else [pid]

    def add_pack(self, pack_id: int, label: str) -> None:
        self.pack_combo.addItem(label, pack_id)

    def set_count(self, shown: int, total: int, checked: int = 0) -> None:
        s = f"{shown:,} / {total:,}"
        if checked:
            s += f"   ✓ {checked:,}"
        self.count.setText(s)


def write_span(pk: Pack, off: int, size: int, dest: Path, chunk: int = 8 << 20) -> int:
    """Stream a byte range of a pack to *dest* (parents created). Returns bytes written."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    view = memoryview(pk._data)
    written = 0
    tmp = dest.with_name(dest.name + ".partial")
    try:
        with open(tmp, "wb") as fh:
            pos, end = off, min(pk.size, off + size)
            while pos < end:
                n = min(chunk, end - pos)
                part = view[pos:pos + n]
                fh.write(part)
                part.release()
                pos += n
                written += n
        tmp.replace(dest)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    finally:
        view.release()
    return written


def unique_path(p: Path) -> Path:
    if not p.exists():
        return p
    for k in range(1, 10000):
        q = p.with_name(f"{p.stem}.{k}{p.suffix}")
        if not q.exists():
            return q
    raise OSError(f"no free name for {p}")


class WheelGuard(QObject):
    """Event filter: the wheel never changes a combo / spin box / check box inside a scroll area; it scrolls the
    area instead (install on the cell widgets of a table)."""

    def __init__(self, area, parent=None):
        super().__init__(parent or area)
        self.area = area

    def eventFilter(self, obj, ev):
        if ev.type() == QEvent.Wheel:
            from PySide6.QtWidgets import QApplication
            bar = self.area.verticalScrollBar()
            if bar is not None and bar.isVisible():
                QApplication.sendEvent(bar, ev)
            return True
        return False

    def guard(self, *widgets) -> None:
        for w in widgets:
            if w is None:
                continue
            w.installEventFilter(self)
            if w.focusPolicy() == Qt.WheelFocus:
                w.setFocusPolicy(Qt.StrongFocus)
