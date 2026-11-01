"""Vertical colour bar with physical units."""

from __future__ import annotations

import numpy as np
from matplotlib import colormaps
from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QSizePolicy, QWidget

from easy_bos.ui.theme import FONT_FAMILY, MONO_FAMILY, C


class ColorBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedWidth(86)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Expanding)
        self.cmap = "gray"
        self.vmin, self.vmax = 0.0, 1.0
        self.units = ""
        self.title = ""
        self.wheel = False

    def configure(
        self,
        cmap: str | None,
        vmin: float,
        vmax: float,
        units: str,
        title: str,
        wheel: bool = False,
    ) -> None:
        self.cmap = cmap or "gray"
        self.vmin, self.vmax = vmin, vmax
        self.units = units
        self.title = title
        self.wheel = wheel
        self.update()

    def paintEvent(self, _):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.fillRect(self.rect(), QColor(C.canvas))
        f = QFont(FONT_FAMILY, 8)
        p.setFont(f)
        p.setPen(QColor(C.muted))
        p.drawText(
            QRectF(4, 6, self.width() - 8, 30),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            self.title,
        )
        top, bottom = 44, self.height() - 30
        bar = QRectF(8, top, 14, max(10, bottom - top))
        if self.wheel:
            self._wheel(p, bar)
            return
        n = 256
        vals = np.linspace(1, 0, n)[:, None]
        rgb = (colormaps[self.cmap](vals)[..., :3] * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(np.repeat(rgb, 4, axis=1))
        img = QImage(rgb.data, 4, n, 12, QImage.Format.Format_RGB888)
        p.drawImage(bar, img)
        p.setPen(QColor(C.border_strong))
        p.drawRect(bar)
        p.setFont(QFont(MONO_FAMILY, 8))
        p.setPen(QColor(C.text))
        for t in np.linspace(0, 1, 5):
            v = self.vmax - t * (self.vmax - self.vmin)
            y = bar.top() + t * bar.height()
            p.drawLine(int(bar.right()), int(y), int(bar.right() + 4), int(y))
            p.drawText(
                QRectF(bar.right() + 6, y - 8, 60, 16),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                _fmt(v),
            )
        p.setFont(f)
        p.setPen(QColor(C.muted))
        p.drawText(
            QRectF(4, bottom + 6, self.width() - 8, 18), Qt.AlignmentFlag.AlignLeft, self.units
        )

    def _wheel(self, p: QPainter, bar: QRectF) -> None:
        size = 64
        yy, xx = np.mgrid[-1 : 1 : complex(size), -1 : 1 : complex(size)]
        ang = (np.arctan2(-yy, xx) / (2 * np.pi)) % 1.0
        r = np.clip(np.hypot(xx, yy), 0, 1)
        rgb = colormaps["hsv"](ang)[..., :3] * r[..., None]
        rgb[np.hypot(xx, yy) > 1] = np.array([12, 13, 15]) / 255
        a = np.ascontiguousarray((rgb * 255).astype(np.uint8))
        img = QImage(a.data, size, size, size * 3, QImage.Format.Format_RGB888)
        p.drawImage(QRectF(8, bar.top(), 64, 64), img)
        p.setPen(QColor(C.muted))
        p.drawText(
            QRectF(4, bar.top() + 70, self.width() - 8, 60),
            Qt.AlignmentFlag.AlignLeft | Qt.TextFlag.TextWordWrap,
            f"hue: direction\nsaturation: 0 – {_fmt(self.vmax)} {self.units}",
        )


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a >= 100:
        return f"{v:.0f}"
    if a >= 1:
        return f"{v:.2f}"
    if a >= 0.01:
        return f"{v:.3f}"
    return f"{v:.1e}"
