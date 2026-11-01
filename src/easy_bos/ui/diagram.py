"""Side-view schematic of the BOS setup, drawn to scale with the user's numbers."""

from __future__ import annotations

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

from easy_bos.core.geometry import Setup
from easy_bos.ui.theme import FONT_FAMILY, C


class SetupDiagram(QWidget):
    def __init__(self):
        super().__init__()
        self.setup: Setup | None = None
        self.setMinimumHeight(170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

    def set_setup(self, setup: Setup) -> None:
        self.setup = setup
        self.update()

    def paintEvent(self, _):  # noqa: N802 - Qt API
        if self.setup is None:
            return
        s = self.setup
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        left, right = 56.0, w - 40.0
        mid = h * 0.46
        total = s.z_b
        x_cam = left
        x_obj = left + (right - left) * s.z_a / total
        x_bg = right
        half_bg = h * 0.30
        # field-of-view cone
        cone = QPainterPath()
        cone.moveTo(x_cam + 18, mid)
        cone.lineTo(x_bg, mid - half_bg)
        cone.lineTo(x_bg, mid + half_bg)
        cone.closeSubpath()
        p.fillPath(cone, QColor(134, 179, 217, 22))
        pen = QPen(QColor(C.accent_dim), 1)
        p.setPen(pen)
        p.drawLine(QPointF(x_cam + 18, mid), QPointF(x_bg, mid - half_bg))
        p.drawLine(QPointF(x_cam + 18, mid), QPointF(x_bg, mid + half_bg))
        # camera body + lens
        p.setPen(QPen(QColor(C.text), 1.4))
        p.setBrush(QColor(C.raised))
        p.drawRoundedRect(QRectF(x_cam - 22, mid - 13, 30, 26), 3, 3)
        p.drawRect(QRectF(x_cam + 8, mid - 8, 10, 16))
        # background
        p.setPen(QPen(QColor(C.text), 2.5))
        p.drawLine(QPointF(x_bg, mid - half_bg - 6), QPointF(x_bg, mid + half_bg + 6))
        for i in range(9):
            y = mid - half_bg + i * (2 * half_bg) / 8
            p.setPen(QPen(QColor(C.muted), 1))
            p.drawLine(QPointF(x_bg + 3, y), QPointF(x_bg + 9, y))
        # flow (plume glyph)
        plume = QPainterPath()
        top = mid - half_bg * 0.9
        plume.moveTo(x_obj - 4, mid + half_bg * 0.6)
        plume.cubicTo(x_obj - 12, mid, x_obj + 10, mid - 20, x_obj - 6, top)
        plume.moveTo(x_obj + 4, mid + half_bg * 0.6)
        plume.cubicTo(x_obj + 14, mid, x_obj - 4, mid - 26, x_obj + 10, top)
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.setPen(QPen(QColor(C.warn), 1.6))
        p.drawPath(plume)
        # deflected ray
        p.setPen(QPen(QColor(C.warn), 1, Qt.PenStyle.DashLine))
        p.drawLine(QPointF(x_cam + 18, mid), QPointF(x_obj, mid - 8))
        p.drawLine(
            QPointF(x_obj, mid - 8),
            QPointF(x_bg, mid - 8 - 26 * (x_bg - x_obj) / max(1, x_bg - x_cam)),
        )
        # dimensions
        f = QFont(FONT_FAMILY, 8)
        p.setFont(f)
        y_dim = h - 26
        self._dim(p, x_cam, x_obj, y_dim, f"Z_A = {s.z_a:.2f} m")
        self._dim(p, x_obj, x_bg, y_dim, f"Z_D = {s.z_d:.2f} m")
        p.setPen(QColor(C.muted))
        p.drawText(QRectF(x_cam - 40, mid + 18, 80, 16), Qt.AlignmentFlag.AlignCenter, "camera")
        p.drawText(
            QRectF(x_obj - 40, mid + half_bg * 0.6 + 4, 80, 16),
            Qt.AlignmentFlag.AlignCenter,
            "flow",
        )
        p.drawText(
            QRectF(x_bg - 70, mid - half_bg - 24, 80, 16), Qt.AlignmentFlag.AlignRight, "background"
        )

    @staticmethod
    def _dim(p: QPainter, x0: float, x1: float, y: float, text: str) -> None:
        p.setPen(QPen(QColor(C.faint), 1))
        p.drawLine(QPointF(x0, y), QPointF(x1, y))
        p.drawLine(QPointF(x0, y - 4), QPointF(x0, y + 4))
        p.drawLine(QPointF(x1, y - 4), QPointF(x1, y + 4))
        p.setPen(QColor(C.muted))
        p.drawText(QRectF(x0, y + 3, x1 - x0, 16), Qt.AlignmentFlag.AlignCenter, text)
