"""Step 3 — Process: quality, regions and frames; preview, compare and run."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from easy_bos.core import render
from easy_bos.core.flow import REGISTRY
from easy_bos.core.media import ArraySource
from easy_bos.core.pipeline import QUALITY, Processor, ReferenceMode
from easy_bos.ui import widgets as W
from easy_bos.ui.canvas import ImageCanvas
from easy_bos.ui.frames import FrameScrubber
from easy_bos.ui.state import AppState

#: Methods offered in the UI. Wavelet flow and RAFT remain in the library and the benchmark,
#: but neither beats variational flow on the benchmark while costing 10-50x the time.
UI_METHODS = ["variational", "correlation", "lucas_kanade", "farneback", "dis"]


class ProcessPage(QWidget):
    finished = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._preview: np.ndarray | None = None
        self._seconds_per_frame: float | None = None
        self._loading = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._inspector())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(12, 8, 12, 6)
        self.view_mode = W.Segmented(
            [
                ("Frame", "frame", "The captured frame, with the regions you marked"),
                ("Preview", "result", "Result of 'Preview this frame' (knife-edge view)"),
                ("Comparison", "compare", "All methods side by side on the current frame"),
            ]
        )
        self.view_mode.setFixedWidth(330)
        self.view_mode.changed.connect(self._view_mode)
        self.show_regions = QCheckBox("Show regions")
        self.show_regions.setChecked(True)
        self.show_regions.toggled.connect(lambda on: self.canvas.set_regions_visible(on))
        bar.addWidget(self.view_mode)
        bar.addSpacing(16)
        bar.addWidget(self.show_regions)
        bar.addStretch(1)
        self.hint = W.label("", "Faint")
        bar.addWidget(self.hint)
        rv.addLayout(bar)
        self.views = QStackedWidget()
        self.canvas = ImageCanvas()
        self.compare = CompareView()
        self.compare.selected.connect(self._use_method)
        self.views.addWidget(self.canvas)
        self.views.addWidget(self.compare)
        rv.addWidget(self.views, 1)
        self.scrubber = FrameScrubber(playable=False)
        rv.addWidget(self.scrubber)
        lay.addWidget(right, 1)
        self.view_mode.set_value("frame")

        self.scrubber.changed.connect(self._frame_changed)
        self.canvas.regions_changed.connect(self._regions_changed)
        state.source_changed.connect(self._source_changed)
        state.reference_changed.connect(self._source_changed)
        state.project_changed.connect(self._load)
        self._load()
        self._source_changed()

    # ------------------------------------------------------------------ inspector
    def _inspector(self) -> QWidget:
        p = W.ScrollPanel()
        p.add(W.label("Process", "H1"))
        p.add(W.label("Try one frame, then run the sequence.", "Muted", wrap=True))

        p.add(W.section("Quality"))
        self.quality = W.Segmented(
            [
                ("Fast", "fast", QUALITY["fast"][2]),
                ("Balanced", "balanced", QUALITY["balanced"][2]),
                ("Best", "best", QUALITY["best"][2]),
                ("Custom", "custom", "Choose the method and its parameters yourself."),
            ]
        )
        self.quality.changed.connect(self._quality_changed)
        p.add(self.quality)
        self.quality_info = W.label("", "Faint", wrap=True)
        p.add(self.quality_info)
        self.custom = QWidget()
        cv = QVBoxLayout(self.custom)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(8)
        self.method = W.combo([(REGISTRY[k].label, k) for k in UI_METHODS])
        self.method.currentIndexChanged.connect(self._method_changed)
        cv.addWidget(self.method)
        self.param_holder = QVBoxLayout()
        self.param_holder.setContentsMargins(0, 0, 0, 0)
        cv.addLayout(self.param_holder)
        self.param_form: W.ParamForm | None = None
        p.add(self.custom)

        p.add(W.section("Regions"))
        p.add(
            W.hbox(
                W.button(
                    "Area",
                    lambda: self._add_region("roi"),
                    tooltip="Process only this rectangle (faster).",
                ),
                W.button(
                    "Exclude",
                    lambda: self._add_region("exclude"),
                    tooltip="Areas without a usable background: objects, edges, the heat "
                    "source. Untextured and glowing areas are excluded automatically.",
                ),
                W.button(
                    "Stable",
                    lambda: self._add_region("stable"),
                    tooltip="Background that never sees flow; used to measure camera "
                    "motion exactly when the camera is not fixed.",
                ),
                W.button("Clear", self._clear_regions, flat=True),
                spacing=6,
            )
        )
        p.add(
            W.label(
                "Drag on the image to adjust; right-click a region to remove it.",
                "Faint",
                wrap=True,
            )
        )

        p.add(W.section("Frames"))
        self.first = W.integer(0, 0, 10**7)
        self.last = W.integer(0, 0, 10**7)
        self.step = W.integer(1, 1, 1000)
        p.add(
            W.hbox(
                W.label("From", "Muted"),
                self.first,
                W.label("to", "Muted"),
                self.last,
                W.label("every", "Muted"),
                self.step,
                spacing=6,
            )
        )
        self.count_label = W.label("", "Faint")
        p.add(self.count_label)

        more = W.Collapsible("More options")
        f = W.form()
        self.ref_mode = W.combo([(m.label, m.value) for m in ReferenceMode])
        self.binning = W.combo([("Automatic", 0), ("Full", 1), ("½ (2 × 2)", 2), ("¼ (4 × 4)", 4)])
        self.binning.setToolTip(
            "Average pixel blocks before processing. Automatic keeps frames at most ~2000 px "
            "wide: 4K is processed at half resolution, which is 4x faster and less noisy; the "
            "flow is blurred by the optics well beyond 2 px anyway."
        )
        self.tavg = W.integer(1, 1, 61, "frames")
        self.tavg.setToolTip("Moving average of the results over time (steady flows only).")
        self.stabilize = W.combo(
            [("Auto", "auto"), ("Always compensate", "on"), ("Off (fixed camera)", "off")]
        )
        self.stabilize.setToolTip(
            "Camera-motion compensation. Auto compensates only if the flow-free reference "
            "frames show the camera moving, or a frame jumps."
        )
        self.motion_model = W.combo(
            [
                ("Shift + rotation", "rigid"),
                ("Shift only", "translation"),
                ("Affine (needs stable region)", "affine"),
            ]
        )
        f.addRow(W.label("Reference", "Muted"), self.ref_mode)
        f.addRow(W.label("Resolution", "Muted"), self.binning)
        f.addRow(W.label("Time average", "Muted"), self.tavg)
        f.addRow(W.label("Camera motion", "Muted"), self.stabilize)
        f.addRow(W.label("Motion model", "Muted"), self.motion_model)
        more.add(f)
        self.auto_mask = QCheckBox("Ignore untextured and glowing areas")
        self.auto_mask.setToolTip(
            "Exclude areas without background texture (bezels, dark "
            "borders) and areas whose brightness changes (flames, hands)."
        )
        self.exposure = QCheckBox("Match exposure between frames")
        self.banding = QCheckBox("Remove flicker banding")
        for cb in (self.auto_mask, self.exposure, self.banding):
            more.add(cb)
        p.add(more)

        p.add(W.section("Run"))
        self.preview_btn = W.button("Preview this frame", self._preview_frame)
        self.compare_btn = W.button("Compare methods", self._compare)
        p.add(W.hbox(self.preview_btn, self.compare_btn, None))
        self.run_btn = W.button("Process frames", self._run, primary=True)
        self.cancel_btn = W.button("Cancel", self._cancel)
        self.cancel_btn.setEnabled(False)
        p.add(W.hbox(self.run_btn, self.cancel_btn, None))
        self.run_status = W.label("", "Muted", wrap=True)
        p.add(self.run_status)
        p.finish()

        for cb in (self.auto_mask, self.exposure, self.banding):
            cb.toggled.connect(self._store)
        for c in (self.motion_model, self.ref_mode, self.binning, self.stabilize):
            c.currentIndexChanged.connect(self._store)
        for s in (self.first, self.last, self.step, self.tavg):
            s.valueChanged.connect(self._store)
        return p

    # ------------------------------------------------------------------ state <-> widgets
    def _load(self) -> None:
        self._loading = True
        pr = self.state.project.process
        self.quality.set_value(pr.quality)
        self.method.setCurrentIndex(max(0, self.method.findData(pr.method)))
        self._rebuild_params(pr.method_params)
        self.stabilize.setCurrentIndex(max(0, self.stabilize.findData(pr.stabilize)))
        self.motion_model.setCurrentIndex(max(0, self.motion_model.findData(pr.motion_model)))
        self.auto_mask.setChecked(pr.auto_mask)
        self.exposure.setChecked(pr.exposure_correction)
        self.banding.setChecked(pr.banding_correction)
        self.ref_mode.setCurrentIndex(max(0, self.ref_mode.findData(pr.reference_mode)))
        self.binning.setCurrentIndex(max(0, self.binning.findData(pr.binning)))
        self.tavg.setValue(pr.temporal_average)
        self._loading = False
        self._show_quality()

    def _store(self, *_) -> None:
        if self._loading:
            return
        pr = self.state.project.process
        pr.quality = self.quality.value() or "best"
        pr.method = self.method.currentData()
        pr.method_params = self.param_form.values() if self.param_form else {}
        pr.stabilize = self.stabilize.currentData()
        pr.motion_model = self.motion_model.currentData()
        pr.auto_mask = self.auto_mask.isChecked()
        pr.exposure_correction = self.exposure.isChecked()
        pr.banding_correction = self.banding.isChecked()
        pr.reference_mode = self.ref_mode.currentData()
        pr.binning = self.binning.currentData()
        pr.temporal_average = self.tavg.value()
        if self.state.source is not None:
            pr.frame_range = (self.first.value(), self.last.value() + 1, self.step.value())
        self.motion_model.setEnabled(self.stabilize.currentData() != "off")
        self._invalidate_preview()
        self.state.mark_dirty()
        self._update_count()

    def _quality_changed(self, _key: str) -> None:
        self._seconds_per_frame = None
        self._store()
        self._show_quality()

    def _show_quality(self) -> None:
        q = self.quality.value() or "best"
        self.custom.setVisible(q == "custom")
        if q in QUALITY:
            self.quality_info.setText(QUALITY[q][2])
        else:
            cls = REGISTRY[self.method.currentData()]
            self.quality_info.setText(cls.summary)

    def _method_changed(self) -> None:
        if self._loading:
            return
        self._rebuild_params({})
        self._seconds_per_frame = None
        self._store()
        self._show_quality()

    def _rebuild_params(self, values: dict) -> None:
        if self.param_form is not None:
            self.param_form.setParent(None)
            self.param_form.deleteLater()
        self.param_form = W.ParamForm(REGISTRY[self.method.currentData()].Params, values)
        self.param_form.changed.connect(self._store)
        self.param_holder.addWidget(self.param_form)

    def _use_method(self, key: str) -> None:
        """A comparison tile was clicked: process with that method."""
        preset = next((q for q, (m, prm, _) in QUALITY.items() if m == key and not prm), None)
        if preset:
            self.quality.set_value(preset)
        else:
            self.quality.set_value("custom")
            self.method.setCurrentIndex(self.method.findData(key))
        self._quality_changed("")
        self.run_status.setText(f"Using {REGISTRY[key].label}.")

    def _source_changed(self) -> None:
        src = self.state.source
        enabled = src is not None
        for b in (self.preview_btn, self.compare_btn, self.run_btn):
            b.setEnabled(enabled)
        self.canvas.blockSignals(True)
        self.canvas.clear_regions()
        self.canvas.blockSignals(False)
        self._seconds_per_frame = None
        if not enabled:
            self.canvas.clear()
            self.hint.setText("Open a capture first.")
            return
        n = len(src)
        pr = self.state.project.process
        saved = [
            ("roi", [pr.roi] if pr.roi else []),
            ("exclude", list(pr.exclusions)),
            ("stable", list(pr.stable_regions)),
        ]
        a, b, c = pr.frame_range or (pr.reference_range[1] if n > 1 else 0, n, 1)
        self._loading = True
        for w in (self.first, self.last):
            w.setRange(0, n - 1)
        self.first.setValue(min(a, n - 1))
        self.last.setValue(min(b - 1, n - 1))
        self.step.setValue(c)
        self._loading = False
        self.scrubber.configure(n, src.info.fps)
        self.scrubber.set_frame(self.state.current_frame)
        self._frame_changed(self.state.current_frame, fit=True)
        self.canvas.blockSignals(True)
        for kind, rects in saved:
            for r in rects:
                self.canvas.add_region(kind, r)
        self.canvas.blockSignals(False)
        self.hint.setText("")
        self._update_count()

    def _frame_indices(self) -> list[int]:
        return list(range(self.first.value(), self.last.value() + 1, max(1, self.step.value())))

    def _update_count(self) -> None:
        if self.state.source is None:
            return
        n = len(self._frame_indices())
        text = f"{n} frame{'s' if n != 1 else ''}"
        if self._seconds_per_frame:
            text += f" · about {_fmt_time(n * self._seconds_per_frame)}"
        self.count_label.setText(text)
        self.run_btn.setText(f"Process {n} frame{'s' if n != 1 else ''}")

    def _invalidate_preview(self) -> None:
        """A preview belongs to the settings and regions it was computed with."""
        self._preview = None
        if self.view_mode.value() == "result":
            self.view_mode.set_value("frame", emit=True)

    def _regions_changed(self) -> None:
        self._invalidate_preview()
        pr = self.state.project.process
        roi = self.canvas.region_rects("roi")
        pr.roi = roi[0] if roi else None
        pr.exclusions = self.canvas.region_rects("exclude")
        pr.stable_regions = self.canvas.region_rects("stable")
        self._seconds_per_frame = None
        self._update_count()
        self.state.mark_dirty()

    def _add_region(self, kind: str) -> None:
        if self.state.source is not None:
            self.view_mode.set_value("frame", emit=True)
            self.canvas.add_region(kind)

    def _clear_regions(self) -> None:
        self.canvas.clear_regions()

    # ------------------------------------------------------------------ views
    def _frame_changed(self, index: int, fit: bool = False) -> None:
        self.state.set_frame(index)
        self._invalidate_preview()
        self._show_frame(fit)

    def _show_frame(self, fit: bool = False) -> None:
        img = self.state.frame(self.state.current_frame)
        if img is not None:
            self.canvas.set_frame(img, keep_view=not fit)

    def _view_mode(self, mode: str) -> None:
        self.views.setCurrentIndex(1 if mode == "compare" else 0)
        if mode == "frame":
            self._show_frame()
        elif mode == "result":
            if self._preview is None:
                self.hint.setText("Press 'Preview this frame' first.")
                self.view_mode.set_value("frame")
                self._show_frame()
            else:
                self._show_preview()

    def _show_preview(self) -> None:
        d = self._preview
        rgb = render.render(render.View.KNIFE_EDGE, d, render.ViewSettings(scale=0))
        img = self.state.frame(self.state.current_frame)
        pr = self.state.project.process
        x0, y0, w, h = pr.roi if pr.roi else (0, 0, img.shape[1], img.shape[0])
        b = pr.effective_binning(img.shape[1])
        full = render.dim_gray(img)
        if b > 1:
            rgb = np.repeat(np.repeat(rgb, b, axis=0), b, axis=1)
        hh, ww = min(rgb.shape[0], h), min(rgb.shape[1], w)
        full[y0 : y0 + hh, x0 : x0 + ww] = rgb[:hh, :ww]
        self.canvas.set_image(full)

    # ------------------------------------------------------------------ processing
    def _processor(self) -> Processor:
        src = self.state.source
        return Processor(
            src, self.state.project.process.settings((src.info.height, src.info.width))
        )

    def _set_busy(self, busy: bool) -> None:
        for b in (self.preview_btn, self.compare_btn, self.run_btn):
            b.setEnabled(not busy and self.state.source is not None)
        self.cancel_btn.setEnabled(busy)

    def _preview_frame(self) -> None:
        if self.state.source is None:
            return
        self._store()
        proc = self._processor()
        index = self.state.current_frame
        self._set_busy(True)
        self.run_status.setText("Processing the current frame…")

        def work(progress, cancelled):
            _ = proc.reference  # load once, outside the per-frame timing
            t0 = time.perf_counter()
            res = proc.process_frame(index)
            return res, time.perf_counter() - t0

        def done(out):
            res, secs = out
            self._set_busy(False)
            self._preview = res.displacement
            self._seconds_per_frame = secs
            peak = np.nanpercentile(np.hypot(*res.displacement), 99.5)
            motion = ""
            if res.motion is not None:
                tx, ty = res.motion.translation
                motion = f" Camera motion compensated: {np.hypot(tx, ty):.2f} px."
            self.run_status.setText(
                f"Frame {index}: peak displacement {peak:.2f} px, {secs:.1f} s.{motion}"
            )
            self._update_count()
            self.view_mode.set_value("result")
            self._show_preview()

        t = self.state.run_task(work, done, message="Preview")
        t.failed.connect(lambda *_: self._set_busy(False))

    def _compare(self) -> None:
        if self.state.source is None:
            return
        self._store()
        index = self.state.current_frame
        base = self._processor()
        self._set_busy(True)
        self.view_mode.set_value("compare")
        self.views.setCurrentIndex(1)
        self.compare.reset(UI_METHODS)

        def work(progress, cancelled):
            if base.settings.reference_mode is ReferenceMode.PREVIOUS:
                ref = base.load(max(0, index - 1))
            else:
                ref = base.reference
            cur = base.load(index)
            out = {}
            for i, key in enumerate(UI_METHODS):
                if cancelled():
                    break
                progress(i, len(UI_METHODS), REGISTRY[key].label)
                proc = Processor(
                    ArraySource(np.stack([ref, cur])),
                    base.settings.with_(
                        method=key,
                        method_params={},
                        roi=None,
                        binning=1,
                        reference_range=(0, 1),
                        reference_mode=ReferenceMode.AVERAGE,
                    ),
                )
                t0 = time.perf_counter()
                try:
                    res = proc.process_pair(ref, cur, index)
                    out[key] = (res.displacement, time.perf_counter() - t0, None)
                except Exception as exc:  # one failing method must not abort the comparison
                    out[key] = (None, 0.0, str(exc))
            return out

        def done(out):
            self._set_busy(False)
            self.compare.show_results(out)
            self.run_status.setText(
                "Same frame, every method, default parameters. Click a result to use that method."
            )

        t = self.state.run_task(work, done, message="Comparing methods")
        t.failed.connect(lambda *_: self._set_busy(False))

    def _run(self) -> None:
        if self.state.source is None or not self.state.ensure_idle("start a new run"):
            return
        self._store()
        proc = self._processor()
        store = self.state.new_results()
        st = self.state.processed_setup()
        pr = self.state.project.process
        store.set_metadata(
            project=self.state.project.to_json(),
            method=proc.settings.method,
            pixels_per_radian=st.pixels_per_radian,
            object_pixel_size_m=st.object_pixel_size_m,
            binning=proc.settings.binning,
            roi=list(pr.roi) if pr.roi else [],
            fps=self.state.source.info.fps or 0.0,
        )
        indices = proc.frame_indices()
        self._set_busy(True)
        t_start = time.perf_counter()

        def work(progress, cancelled):
            n = 0
            for res in proc.run(lambda d, t, r: progress(d, t, ""), cancelled):
                store.add(res)
                n += 1
            return n

        def on_progress(done, total, _):
            elapsed = time.perf_counter() - t_start
            eta = elapsed / max(done, 1) * (total - done)
            self.run_status.setText(f"Frame {done} of {total} · {_fmt_time(eta)} left")
            win = self.window()
            if hasattr(win, "set_progress"):
                win.set_progress(done, total, "Processing")

        def done(n):
            self._set_busy(False)
            dt = time.perf_counter() - t_start
            self.run_status.setText(
                f"Processed {n} of {len(indices)} frames in {_fmt_time(dt)}"
                f" ({dt / max(n, 1):.2f} s per frame)."
            )
            self.state.results_changed.emit()
            if n:
                self.finished.emit()

        t = self.state.run_task(work, done, on_progress, message="Processing")
        t.failed.connect(lambda *_: self._set_busy(False))

    def _cancel(self) -> None:
        self.state.cancel_tasks()
        self.run_status.setText("Cancelling…")

    def activated(self) -> None:
        if self.state.source is not None:
            self.scrubber.set_frame(self.state.current_frame)
            if self.view_mode.value() == "frame":
                self._show_frame()


class CompareView(QWidget):
    """Every method's result on the same frame: a steady grid of tiles with one shared,
    outlier-robust colour scale. Clicking a tile selects that method."""

    selected = Signal(str)

    def __init__(self):
        super().__init__()
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(16, 12, 16, 12)
        self.grid.setSpacing(14)
        self.tiles: dict[str, _Tile] = {}

    def reset(self, methods: list[str]) -> None:
        W.clear_layout(self.grid)
        self.tiles.clear()
        cols = 3
        for i, key in enumerate(methods):
            tile = _Tile(REGISTRY[key].label)
            tile.clicked.connect(lambda k=key: self.selected.emit(k))
            self.grid.addWidget(tile, i // cols, i % cols)
            self.tiles[key] = tile
        for c in range(cols):
            self.grid.setColumnStretch(c, 1)

    def show_results(self, out: dict) -> None:
        # Shared scale: the median over methods of each one's 99th percentile, so one method's
        # outliers cannot wash out the others.
        peaks = [
            float(np.nanpercentile(np.hypot(*d), 99))
            for d, _, _ in out.values()
            if d is not None and np.isfinite(d).any()
        ]
        scale = float(np.median(peaks)) if peaks else 1.0
        ref = out.get("variational", (None,))[0]
        for key, (d, secs, err) in out.items():
            tile = self.tiles[key]
            if d is None:
                tile.set_text(f"{REGISTRY[key].label} · failed", err or "")
                continue
            tile.set_image(
                render.render(render.View.KNIFE_EDGE, d, render.ViewSettings(scale=scale))
            )
            detail = f"{secs:.1f} s"
            if ref is not None and key != "variational":
                detail += f" · differs from Best by {np.nanmean(np.hypot(*(d - ref))):.3f} px"
            tile.set_text(REGISTRY[key].label, detail)


class _Tile(QWidget):
    clicked = Signal()

    def __init__(self, title: str):
        super().__init__()
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)
        self.title = W.label(title, "H2")
        self.detail = W.label("running…", "Faint")
        self.image = QLabel()
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.image.setMinimumSize(120, 80)
        lay.addWidget(self.title)
        lay.addWidget(self.detail)
        lay.addWidget(self.image, 1)
        self._pix: QPixmap | None = None

    def set_text(self, title: str, detail: str) -> None:
        self.title.setText(title)
        self.detail.setText(detail)

    def set_image(self, rgb: np.ndarray) -> None:
        a = np.ascontiguousarray(rgb)
        h, w = a.shape[:2]
        self._pix = QPixmap.fromImage(
            QImage(a.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()
        )
        self._rescale()

    def _rescale(self) -> None:
        if self._pix is not None:
            self.image.setPixmap(
                self._pix.scaled(
                    self.image.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    def resizeEvent(self, e):  # noqa: N802 - Qt API
        super().resizeEvent(e)
        self._rescale()

    def mousePressEvent(self, e):  # noqa: N802 - Qt API
        self.clicked.emit()


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f} s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"
