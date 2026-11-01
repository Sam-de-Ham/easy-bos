"""Full-screen background pattern on a chosen monitor, pixel-exact.

Qt scales logical coordinates by the screen's device pixel ratio (125 %, 150 % ... on Windows).
The pattern is rendered at the screen's *physical* resolution and tagged with that ratio, so one
pattern pixel lands on exactly one display pixel whatever the Windows scaling setting.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QRect, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QGuiApplication, QImage, QKeyEvent, QPainter, QScreen
from PySide6.QtWidgets import QWidget

from easy_bos.core.patterns import PatternSpec, generate, to_display_uint8


def physical_size(screen: QScreen) -> tuple[int, int]:
    g = screen.geometry()
    dpr = screen.devicePixelRatio()
    return round(g.width() * dpr), round(g.height() * dpr)


def screen_diagonal_inches(screen: QScreen) -> float | None:
    """Diagonal from the EDID physical size Windows reports (None if unknown/implausible)."""
    s = screen.physicalSize()  # millimetres
    d = (s.width() ** 2 + s.height() ** 2) ** 0.5 / 25.4
    return d if 5.0 < d < 150.0 else None


class PatternWindow(QWidget):
    closed = Signal()

    def __init__(
        self, screen: QScreen, spec: PatternSpec, contrast: tuple[float, float] = (0.0, 1.0)
    ):
        super().__init__(
            None, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.spec = spec
        self.contrast = contrast
        self._screen = screen
        self._image: QImage | None = None
        self._array: np.ndarray | None = None
        self._overlay = True
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self._render()
        QTimer.singleShot(2500, self._hide_overlay)

    def show_on_screen(self) -> None:
        self.show()
        handle = self.windowHandle()
        if handle is not None:
            handle.setScreen(self._screen)
        self.setGeometry(self._screen.geometry())
        self.showFullScreen()
        self.activateWindow()

    def _render(self) -> None:
        w, h = physical_size(self._screen)
        lin = generate(self.spec, w, h)
        g8 = to_display_uint8(lin, *self.contrast)
        self._array = np.ascontiguousarray(g8)
        img = QImage(self._array.data, w, h, w, QImage.Format.Format_Grayscale8)
        img.setDevicePixelRatio(self._screen.devicePixelRatio())
        self._image = img
        self.update()

    def set_spec(self, spec: PatternSpec) -> None:
        self.spec = spec
        self._render()
        self._flash_overlay()

    def _hide_overlay(self) -> None:
        self._overlay = False
        self.update()

    def _flash_overlay(self) -> None:
        self._overlay = True
        self.update()
        QTimer.singleShot(1500, self._hide_overlay)

    def paintEvent(self, _):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        if self._image is not None:
            p.drawImage(0, 0, self._image)
        if self._overlay:
            w, h = physical_size(self._screen)
            text = (
                f"{w} × {h} px · scale {self._screen.devicePixelRatio():.2f} · "
                f"element {self.spec.element_px} px · seed {self.spec.seed}\n"
                "Esc close  ·  ← → element size  ·  N new pattern  ·  I invert"
            )
            f = QFont("Segoe UI", 11)
            p.setFont(f)
            r = QRect(0, 0, 640, 64)
            r.moveCenter(self.rect().center())
            p.fillRect(r.adjusted(-12, -10, 12, 10), QColor(17, 19, 21, 235))
            p.setPen(QColor("#e3e5e8"))
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, text)

    def keyPressEvent(self, e: QKeyEvent):  # noqa: N802 - Qt API
        k = e.key()
        if k in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        elif k == Qt.Key.Key_Right:
            self.set_spec(self.spec.with_(element_px=min(64, self.spec.element_px + 1)))
        elif k == Qt.Key.Key_Left:
            self.set_spec(self.spec.with_(element_px=max(1, self.spec.element_px - 1)))
        elif k == Qt.Key.Key_N:
            self.set_spec(self.spec.with_(seed=self.spec.seed + 1))
        elif k == Qt.Key.Key_I:
            self.set_spec(self.spec.with_(invert=not self.spec.invert))
        elif k == Qt.Key.Key_Space:
            self._flash_overlay()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):  # noqa: N802 - Qt API
        self.closed.emit()
        super().closeEvent(e)


def screens() -> list[QScreen]:
    return list(QGuiApplication.screens())
