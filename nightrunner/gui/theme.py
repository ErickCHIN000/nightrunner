"""Fusion style with an optional dark palette."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette


def apply_theme(app, dark: bool) -> None:
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        return
    p = QPalette()
    base, alt, text, mid = QColor(30, 30, 32), QColor(38, 38, 42), QColor(220, 220, 220), QColor(45, 45, 50)
    p.setColor(QPalette.Window, mid)
    p.setColor(QPalette.WindowText, text)
    p.setColor(QPalette.Base, base)
    p.setColor(QPalette.AlternateBase, alt)
    p.setColor(QPalette.ToolTipBase, mid)
    p.setColor(QPalette.ToolTipText, text)
    p.setColor(QPalette.Text, text)
    p.setColor(QPalette.Button, mid)
    p.setColor(QPalette.ButtonText, text)
    p.setColor(QPalette.BrightText, QColor(255, 80, 80))
    p.setColor(QPalette.Link, QColor(90, 160, 255))
    p.setColor(QPalette.Highlight, QColor(52, 104, 170))
    p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
    p.setColor(QPalette.PlaceholderText, QColor(130, 130, 130))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(120, 120, 120))
    app.setPalette(p)
