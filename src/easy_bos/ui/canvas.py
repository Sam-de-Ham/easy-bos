"""Image canvas (pyqtgraph): zoom/pan, hover readout, editable regions, line profiles."""

from __future__ import annotations

import cv2
import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QPen
from PySide6.QtWidgets import QVBoxLayout, QWidget

from easy_bos.ui.theme import C

REGION_STYLES = {
    "roi": (C.accent, "Processing area"),
    "exclude": (C.fail, "Excluded (no background)"),
    "stable": (C.ok, "Stable reference (no flow)"),
    "anchor": (C.warn, "Ambient anchor"),
}


class ImageCanvas(QWidget):
    hovered = Signal(float, float)  # image x, y (pixels); NaN when outside
    regions_changed = Signal()
    line_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("Canvas")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.view = pg.GraphicsLayoutWidget()
        self.view.setBackground(C.canvas)
        lay.addWidget(self.view)
        self.vb = self.view.addViewBox(lockAspect=True, invertY=True, enableMenu=False)
        self.vb.setDefaultPadding(0.01)
        self.image = pg.ImageItem(axisOrder="row-major")
        self.vb.addItem(self.image)
        self._shape: tuple[int, int] | None = None
        self.regions: dict[str, list[pg.RectROI]] = {k: [] for k in REGION_STYLES}
        self.line: pg.LineSegmentROI | None = None
        self.axis_line: pg.InfiniteLine | None = None
        self.view.scene().sigMouseMoved.connect(self._mouse_moved)

    # ------------------------------------------------------------------ image
    def set_image(
        self, img: np.ndarray, levels: tuple[float, float] | None = None, keep_view: bool = True
    ) -> None:
        first = self._shape is None or self._shape != img.shape[:2]
        if img.ndim == 2:
            lv = levels if levels is not None else (float(np.nanmin(img)), float(np.nanmax(img)))
            self.image.setImage(img, levels=lv, autoLevels=False)
        else:
            self.image.setImage(img, levels=(0, 255), autoLevels=False)
        self._shape = img.shape[:2]
        if first or not keep_view:
            self.fit()

    def set_frame(self, img: np.ndarray, keep_view: bool = True) -> None:
        """Show a camera frame quickly: robust contrast from a sparse sample, converted to
        8 bits in one pass (pyqtgraph would otherwise map a 4K float image on every update)."""
        sample = img[::8, ::8]
        lo, hi = np.percentile(sample, [0.5, 99.5])
        scale = 255.0 / max(float(hi - lo), 1e-6)
        u8 = cv2.convertScaleAbs(img, alpha=scale, beta=-float(lo) * scale)
        self.set_image(u8, levels=(0, 255), keep_view=keep_view)

    def clear(self) -> None:
        self.image.clear()
        self._shape = None

    def fit(self) -> None:
        if self._shape:
            h, w = self._shape
            self.vb.setRange(xRange=(0, w), yRange=(0, h), padding=0.01)

    def _mouse_moved(self, pos) -> None:
        if self._shape is None:
            return
        p = self.vb.mapSceneToView(pos)
        h, w = self._shape
        if 0 <= p.x() < w and 0 <= p.y() < h:
            self.hovered.emit(p.x(), p.y())
        else:
            self.hovered.emit(float("nan"), float("nan"))

    # ------------------------------------------------------------------ regions
    def add_region(self, kind: str, rect: tuple[int, int, int, int] | None = None) -> pg.RectROI:
        color, _ = REGION_STYLES[kind]
        if rect is None:
            h, w = self._shape or (100, 100)
            rect = (w // 4, h // 4, w // 2, h // 2)
        x, y, rw, rh = rect
        pen = QPen(QColor(color), 0)
        pen.setCosmetic(True)
        pen.setWidthF(1.5)
        hover = QPen(QColor(color).lighter(130), 0)
        hover.setCosmetic(True)
        hover.setWidthF(2.0)
        roi = pg.RectROI(
            [x, y],
            [rw, rh],
            pen=pen,
            hoverPen=hover,
            removable=True,
            snapSize=1,
            scaleSnap=True,
            translateSnap=True,
        )
        roi.addScaleHandle([0, 0], [1, 1])
        roi.addScaleHandle([1, 0], [0, 1])
        roi.addScaleHandle([0, 1], [1, 0])
        roi.sigRegionChangeFinished.connect(lambda *_: self.regions_changed.emit())
        roi.sigRemoveRequested.connect(lambda r, k=kind: self.remove_region(k, r))
        self.vb.addItem(roi)
        if kind == "roi":
            for old in list(self.regions["roi"]):
                self.remove_region("roi", old, emit=False)
        self.regions[kind].append(roi)
        self.regions_changed.emit()
        return roi

    def remove_region(self, kind: str, roi: pg.RectROI, emit: bool = True) -> None:
        if roi in self.regions[kind]:
            self.regions[kind].remove(roi)
            self.vb.removeItem(roi)
            if emit:
                self.regions_changed.emit()

    def clear_regions(self, kind: str | None = None) -> None:
        for k in [kind] if kind else list(self.regions):
            for r in list(self.regions[k]):
                self.remove_region(k, r, emit=False)
        self.regions_changed.emit()

    def region_rects(self, kind: str) -> list[tuple[int, int, int, int]]:
        out = []
        h, w = self._shape or (0, 0)
        for r in self.regions[kind]:
            x, y = r.pos()
            rw, rh = r.size()
            x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
            x1, y1 = min(w, int(round(x + rw))), min(h, int(round(y + rh)))
            if x1 > x0 and y1 > y0:
                out.append((x0, y0, x1 - x0, y1 - y0))
        return out

    def set_regions_visible(self, visible: bool) -> None:
        for rois in self.regions.values():
            for r in rois:
                r.setVisible(visible)

    # ------------------------------------------------------------------ profile line
    def enable_line(self, on: bool) -> None:
        if on and self.line is None and self._shape:
            h, w = self._shape
            pen = QPen(QColor(C.warn), 0)
            pen.setCosmetic(True)
            pen.setWidthF(1.5)
            self.line = pg.LineSegmentROI(
                [QPointF(w * 0.2, h * 0.5), QPointF(w * 0.8, h * 0.5)], pen=pen
            )
            self.line.sigRegionChanged.connect(lambda *_: self.line_changed.emit())
            self.vb.addItem(self.line)
            self.line_changed.emit()
        elif not on and self.line is not None:
            self.vb.removeItem(self.line)
            self.line = None

    def line_endpoints(self) -> tuple[tuple[float, float], tuple[float, float]] | None:
        if self.line is None:
            return None
        handles = self.line.getSceneHandlePositions()
        pts = [self.vb.mapSceneToView(pos) for _, pos in handles]
        return (pts[0].x(), pts[0].y()), (pts[1].x(), pts[1].y())

    def show_axis(self, x: float | None, horizontal: bool = False) -> None:
        if x is None:
            if self.axis_line is not None:
                self.vb.removeItem(self.axis_line)
                self.axis_line = None
            return
        if self.axis_line is None:
            pen = QPen(QColor(C.accent), 0)
            pen.setCosmetic(True)
            pen.setStyle(Qt.PenStyle.DashLine)
            self.axis_line = pg.InfiniteLine(angle=0 if horizontal else 90, pen=pen)
            self.vb.addItem(self.axis_line)
        self.axis_line.setAngle(0 if horizontal else 90)
        self.axis_line.setPos(x)


def sample_line(field: np.ndarray, p0, p1, n: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear samples of a 2-D field along a segment; returns (distance_px, values)."""
    from scipy.ndimage import map_coordinates

    (x0, y0), (x1, y1) = p0, p1
    length = float(np.hypot(x1 - x0, y1 - y0))
    n = n or max(2, int(length) + 1)
    t = np.linspace(0, 1, n)
    xs, ys = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
    vals = map_coordinates(np.nan_to_num(field, nan=np.nan), [ys, xs], order=1, mode="nearest")
    return t * length, vals
