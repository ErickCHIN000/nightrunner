"""Update indicator on the tab bar and the update dialog behind it.

`UpdateCorner` sits in the tab widget's top-right corner: `v0.1.0 (+3) 89ef0b6`. It checks on a worker thread at
start and then on a long timer, and never blocks the GUI. Clicking it opens `UpdateDialog`, which lists what
changed and offers to apply the update.

Applying is a fast-forward of the git checkout (`nightrunner.update.apply_update`). When that is not possible --
an archive install, a dirty tree, no git -- the dialog says exactly why and offers the repository link instead.
Nothing is ever applied without the user pressing the button.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog, QFrame, QHBoxLayout, QLabel, QMessageBox, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from .. import __version__
from .. import update as up

CHECK_MS = 6 * 60 * 60 * 1000          # re-check every six hours; the indicator is not a live feed


class UpdateCorner(QWidget):
    """The tab-bar corner indicator. Emits nothing; owns its own dialog."""

    checked = Signal(object)           # Status, for tests and for anyone who wants to react

    def __init__(self, ctx, parent=None):
        super().__init__(parent)
        self.ctx = ctx
        self.status: up.Status | None = None
        self._changelog: dict | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 0, 8, 0)
        lay.setSpacing(0)
        self.label = QLabel(f"v{__version__}")
        self.label.setCursor(Qt.PointingHandCursor)
        self.label.setToolTip("Checking for updates...")
        self.label.mouseReleaseEvent = self._clicked            # a label is enough; no button chrome wanted here
        lay.addWidget(self.label)
        self._paint("")
        self.timer = QTimer(self)
        self.timer.setInterval(CHECK_MS)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        QTimer.singleShot(1500, self.refresh)                   # let the window finish opening first

    # ---- checking -------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        runner = getattr(self.ctx, "runner", None)
        if runner is None:                                      # no thread pool (tests): check inline
            self._done(up.check(token=up.default_token()))
            return
        runner.submit("update-check", lambda: up.check(token=up.default_token()),
                      on_done=self._done, on_error=lambda exc: self._done(
                          up.Status(state="error", local=up.local_commit(), error=type(exc).__name__)))

    def _done(self, st: up.Status) -> None:
        self.status = st
        self._changelog = None
        extra = f" (+{st.behind})" if st.state == "behind" and st.behind else ""
        self.label.setText(f"v{__version__}{extra} <span style='color:#7a7a7a'>{st.short}</span>")
        self.label.setTextFormat(Qt.RichText)
        self.label.setToolTip(self._tooltip(st))
        self._paint("#d08a3a" if st.state == "behind" else "")
        self.checked.emit(st)

    @staticmethod
    def _tooltip(st: up.Status) -> str:
        if st.state == "behind":
            return f"{st.behind} new commit(s) on {st.branch}. Click to update."
        if st.state == "up-to-date":
            return f"Up to date with {st.repo}@{st.branch}."
        if st.state == "ahead":
            return f"{st.ahead} local commit(s) not pushed."
        if st.state == "local-only":
            return "This commit is not on the remote yet."
        if st.state == "diverged":
            return "Local and remote have both moved on."
        return st.error or "Update state unknown."

    def _paint(self, colour: str) -> None:
        self.label.setStyleSheet(f"QLabel {{ color: {colour}; }}" if colour else "")

    # ---- opening the dialog ---------------------------------------------------------------------------------
    def _clicked(self, event) -> None:
        if event.button() != Qt.LeftButton:
            return
        st = self.status
        if st is None:
            self.refresh()
            return
        if st.state != "behind":
            QDesktopServices.openUrl(QUrl(st.compare_url))
            return
        if self._changelog is None:
            try:
                self._changelog = up.changelog(st.repo, st.local, st.remote, up.default_token())
            except Exception as exc:                            # the dialog is still useful without the list
                self._changelog = {"groups": {}, "listed": 0, "total": st.behind,
                                   "more": st.behind, "error": type(exc).__name__}
        dlg = UpdateDialog(st, self._changelog, self)
        if dlg.exec() == QDialog.Accepted:
            self.refresh()


class UpdateDialog(QDialog):
    """New update available: what changed, and the two buttons."""

    def __init__(self, st: up.Status, log: dict, parent=None):
        super().__init__(parent)
        self.st = st
        self.setWindowTitle("Update")
        self.setMinimumWidth(460)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(28, 24, 28, 20)
        lay.setSpacing(4)

        title = QLabel("New update available")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 17px; font-weight: 600;")
        lay.addWidget(title)

        sub = QLabel(f"{st.behind} new commit(s) on {st.branch}.")
        sub.setAlignment(Qt.AlignCenter)
        sub.setStyleSheet("color: #9a9a9a;")
        lay.addWidget(sub)
        lay.addSpacing(14)

        body = QWidget()
        bl = QVBoxLayout(body)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(2)
        for heading, items in (log.get("groups") or {}).items():
            h = QLabel(heading)
            h.setStyleSheet("color: #8a8a8a; font-size: 10px; font-weight: 600; letter-spacing: 1px;")
            bl.addSpacing(8)
            bl.addWidget(h)
            for s in items:
                it = QLabel(f"•  {s}")
                it.setWordWrap(True)
                bl.addWidget(it)
        if log.get("error"):
            bl.addWidget(QLabel("Could not load the change list."))
        bl.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidget(body)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMaximumHeight(280)
        lay.addWidget(scroll)

        if log.get("more"):
            more = QLabel(f"+ {log['more']} more change(s) included.")
            more.setAlignment(Qt.AlignCenter)
            more.setStyleSheet("color: #7a7a7a; font-size: 11px;")
            lay.addSpacing(10)
            lay.addWidget(more)

        lay.addSpacing(14)
        ok, why = up.can_apply()
        self.update_btn = QPushButton("Update now" if ok else "Open on GitHub")
        self.update_btn.setMinimumHeight(34)
        self.update_btn.setDefault(True)
        self.update_btn.clicked.connect(self._apply if ok else self._open)
        lay.addWidget(self.update_btn)
        if not ok:
            reason = QLabel(why)
            reason.setAlignment(Qt.AlignCenter)
            reason.setWordWrap(True)
            reason.setStyleSheet("color: #b06a4a; font-size: 11px;")
            lay.addWidget(reason)

        later = QPushButton("Maybe later")
        later.setFlat(True)
        later.setStyleSheet("QPushButton { color: #9a9a9a; border: none; }")
        later.clicked.connect(self.reject)
        lay.addWidget(later)

    def _open(self) -> None:
        QDesktopServices.openUrl(QUrl(self.st.compare_url))
        self.accept()

    def _apply(self) -> None:
        self.update_btn.setEnabled(False)
        self.update_btn.setText("Updating...")
        try:
            sha = up.apply_update(branch=self.st.branch)
        except up.UpdateError as exc:
            self.update_btn.setEnabled(True)
            self.update_btn.setText("Update now")
            QMessageBox.warning(self, "Update failed", str(exc))
            return
        QMessageBox.information(self, "Updated", f"Now on {sha[:7]}. Restart Nightrunner to load it.")
        self.accept()
