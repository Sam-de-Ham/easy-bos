"""Dark, calm visual theme: Fusion style, one restrained accent, clear typography."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtGui import QColor, QFont, QPalette
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class Colors:
    window: str = "#111315"
    panel: str = "#16181b"
    raised: str = "#1d2024"
    hover: str = "#23272c"
    border: str = "#2a2e33"
    border_strong: str = "#3a3f46"
    text: str = "#e3e5e8"
    muted: str = "#8d939a"
    faint: str = "#5f656c"
    accent: str = "#86b3d9"
    accent_dim: str = "#3d5a73"
    ok: str = "#86b98c"
    warn: str = "#d8b464"
    fail: str = "#d97f72"
    canvas: str = "#0c0d0f"


C = Colors()

FONT_FAMILY = "Segoe UI"
MONO_FAMILY = "Cascadia Mono"


def status_color(status: str) -> str:
    return {"ok": C.ok, "warn": C.warn, "fail": C.fail}.get(status, C.muted)


QSS = f"""
* {{ outline: none; }}
QWidget {{ color: {C.text}; font-family: "{FONT_FAMILY}"; font-size: 9.5pt; }}
QMainWindow, QDialog {{ background: {C.window}; }}
QToolTip {{ background: {C.raised}; color: {C.text}; border: 1px solid {C.border_strong};
            padding: 6px 8px; }}

#Panel, #Inspector {{ background: {C.panel}; }}
#Inspector {{ border-right: 1px solid {C.border}; }}
#Canvas {{ background: {C.canvas}; }}
#NavRail {{ background: {C.window}; border-right: 1px solid {C.border}; }}
#StatusBar {{ background: {C.window}; border-top: 1px solid {C.border}; }}

QLabel#H1 {{ font-size: 15pt; font-weight: 600; }}
QLabel#H2 {{ font-size: 10.5pt; font-weight: 600; color: {C.text}; }}
QLabel#Section {{ font-size: 8pt; font-weight: 600; color: {C.muted}; padding-top: 6px; }}
QLabel#Muted {{ color: {C.muted}; }}
QLabel#Faint {{ color: {C.faint}; }}
QLabel#Value {{ font-family: "{MONO_FAMILY}", "Consolas"; font-size: 9pt; }}
QLabel#BigValue {{ font-family: "{MONO_FAMILY}", "Consolas"; font-size: 13pt; }}

QPushButton {{ background: {C.raised}; border: 1px solid {C.border_strong}; border-radius: 4px;
               padding: 5px 12px; min-height: 18px; }}
QPushButton:hover {{ background: {C.hover}; }}
QPushButton:pressed {{ background: {C.border}; }}
QPushButton:disabled {{ color: {C.faint}; border-color: {C.border}; }}
QPushButton#Primary {{ background: {C.accent_dim}; border-color: {C.accent}; color: #f2f6fa; }}
QPushButton#Primary:hover {{ background: #4a6b88; }}
QPushButton#Primary:disabled {{ background: {C.raised}; border-color: {C.border}; color: {C.faint}; }}
QPushButton#Flat {{ background: transparent; border: none; color: {C.muted}; padding: 4px 6px; }}
QPushButton#Flat:hover {{ color: {C.text}; }}
QPushButton:checked {{ background: {C.accent_dim}; border-color: {C.accent}; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit {{
    background: {C.raised}; border: 1px solid {C.border}; border-radius: 4px;
    padding: 3px 6px; selection-background-color: {C.accent_dim}; min-height: 18px; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border-color: {C.accent}; }}
QSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::up-button,
QDoubleSpinBox::down-button {{ width: 0; border: none; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{ background: {C.raised}; border: 1px solid {C.border_strong};
                               selection-background-color: {C.accent_dim}; }}

QCheckBox {{ spacing: 8px; }}
QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {C.border_strong};
                        border-radius: 3px; background: {C.raised}; }}
QCheckBox::indicator:checked {{ background: {C.accent}; border-color: {C.accent}; }}
QRadioButton::indicator {{ width: 12px; height: 12px; border: 1px solid {C.border_strong};
                           border-radius: 7px; background: {C.raised}; }}
QRadioButton::indicator:checked {{ background: {C.accent}; border-color: {C.accent}; }}

QSlider::groove:horizontal {{ height: 3px; background: {C.border_strong}; border-radius: 1px; }}
QSlider::sub-page:horizontal {{ background: {C.accent}; border-radius: 1px; }}
QSlider::handle:horizontal {{ background: {C.text}; width: 12px; height: 12px; margin: -5px 0;
                              border-radius: 6px; }}

QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: {C.border_strong}; border-radius: 4px; min-height: 30px;
                               margin: 2px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: {C.border_strong}; border-radius: 4px; min-width: 30px;
                                 margin: 2px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

QTabBar::tab {{ background: transparent; color: {C.muted}; padding: 6px 12px;
                border-bottom: 2px solid transparent; }}
QTabBar::tab:selected {{ color: {C.text}; border-bottom-color: {C.accent}; }}
QTabWidget::pane {{ border: none; }}

QProgressBar {{ background: {C.raised}; border: none; border-radius: 2px; height: 4px;
                text-align: center; color: transparent; }}
QProgressBar::chunk {{ background: {C.accent}; border-radius: 2px; }}

QSplitter::handle {{ background: {C.border}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}

QTableWidget, QTreeWidget, QListWidget {{ background: {C.panel}; border: 1px solid {C.border};
    gridline-color: {C.border}; alternate-background-color: {C.raised}; }}
QHeaderView::section {{ background: {C.panel}; color: {C.muted}; border: none;
                        border-bottom: 1px solid {C.border}; padding: 4px 6px; }}
QMenu {{ background: {C.raised}; border: 1px solid {C.border_strong}; padding: 4px; }}
QMenu::item {{ padding: 5px 18px; }}
QMenu::item:selected {{ background: {C.accent_dim}; }}
QMenuBar {{ background: {C.window}; }}
QMenuBar::item:selected {{ background: {C.raised}; }}
QFrame#Rule {{ background: {C.border}; max-height: 1px; min-height: 1px; border: none; }}
QFrame#Card {{ background: {C.panel}; border: 1px solid {C.border}; border-radius: 6px; }}
"""


def apply(app: QApplication) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    role = QPalette.ColorRole
    for r, c in [
        (role.Window, C.window),
        (role.WindowText, C.text),
        (role.Base, C.raised),
        (role.AlternateBase, C.panel),
        (role.ToolTipBase, C.raised),
        (role.ToolTipText, C.text),
        (role.Text, C.text),
        (role.Button, C.raised),
        (role.ButtonText, C.text),
        (role.BrightText, "#ffffff"),
        (role.Highlight, C.accent_dim),
        (role.HighlightedText, "#ffffff"),
        (role.Link, C.accent),
        (role.PlaceholderText, C.faint),
        (role.Mid, C.border),
        (role.Dark, C.window),
        (role.Shadow, "#000000"),
    ]:
        pal.setColor(r, QColor(c))
    pal.setColor(QPalette.ColorGroup.Disabled, role.Text, QColor(C.faint))
    pal.setColor(QPalette.ColorGroup.Disabled, role.ButtonText, QColor(C.faint))
    pal.setColor(QPalette.ColorGroup.Disabled, role.WindowText, QColor(C.faint))
    app.setPalette(pal)
    font = QFont(FONT_FAMILY, 9)
    font.setHintingPreference(QFont.HintingPreference.PreferNoHinting)
    app.setFont(font)
    app.setStyleSheet(QSS)

    import pyqtgraph as pg

    pg.setConfigOptions(
        background=C.canvas, foreground=C.muted, imageAxisOrder="row-major", antialias=True
    )
