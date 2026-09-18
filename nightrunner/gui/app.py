"""Application bootstrap: Qt app, crash log, exception hook (a .pyw has no console to print to)."""
from __future__ import annotations

import datetime
import os
import sys
import traceback
from pathlib import Path

from . import APP_NAME

LOG_DIR = Path(os.environ.get("LOCALAPPDATA") or Path.home()) / "nightrunner"
LOG_FILE = LOG_DIR / "nightrunner.log"


def _log(text: str) -> None:
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOG_FILE, "a", encoding="utf-8") as fh:
            fh.write(f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")
    except OSError:
        pass


def _install_excepthook() -> None:
    def hook(etype, value, tb):
        text = "".join(traceback.format_exception(etype, value, tb))
        _log(text)
        sys.__stderr__ and sys.__stderr__.write(text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            if QApplication.instance() is not None:
                box = QMessageBox(QMessageBox.Critical, APP_NAME, f"{etype.__name__}: {value}")
                box.setDetailedText(text + f"\n(logged to {LOG_FILE})")
                box.exec()
        except Exception:  # noqa: BLE001
            pass
    sys.excepthook = hook


APP_ID = "nightrunner.app"


def icon_path() -> Path | None:
    """icon.ico at the repo root (user-supplied), else the packaged copy."""
    here = Path(__file__).resolve().parent
    for p in (here.parents[1] / "icon.ico", here / "assets" / "nightrunner.ico"):
        if p.is_file():
            return p
    return None


def _set_app_id() -> None:
    """Own taskbar group + icon on Windows (otherwise pythonw.exe's group and icon are used)."""
    if os.name != "nt":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else argv
    _install_excepthook()
    try:
        from PySide6.QtWidgets import QApplication
    except ImportError as exc:
        msg = (f"{APP_NAME} needs PySide6 ({exc}).\n\nRun setup.bat once (creates .venv and installs "
               "requirements-gui.txt), then start Nightrunner.pyw with the venv's pythonw.")
        _log(msg)
        if os.name == "nt":
            import ctypes
            ctypes.windll.user32.MessageBoxW(None, msg, APP_NAME, 0x10)
        else:
            print(msg, file=sys.stderr)
        return 1
    from .mainwindow import MainWindow
    from .theme import apply_theme
    from PySide6.QtGui import QIcon

    _set_app_id()
    app = QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("nightrunner")
    ico = icon_path()
    icon = QIcon(str(ico)) if ico else QIcon()
    app.setWindowIcon(icon)
    win = MainWindow()
    win.setWindowIcon(icon)
    apply_theme(app, win.settings.value("view/dark", True, type=bool))
    extra = [Path(a) for a in argv[1:] if a.lower().endswith(".rpack") and Path(a).is_file()]
    if extra:
        win.ctx.open_user_packs(extra)
    win.show()
    return app.exec()
