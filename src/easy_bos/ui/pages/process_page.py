"""Step 3 — Process: method, regions, corrections; preview, compare and run."""

from __future__ import annotations

import time

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from easy_bos.core import render
from easy_bos.core.flow import REGISTRY
from easy_bos.core.media import ArraySource
from easy_bos.core.pipeline import Processor, ReferenceMode
from easy_bos.ui import widgets as W
from easy_bos.ui.canvas import ImageCanvas
from easy_bos.ui.frames import FrameScrubber
from easy_bos.ui.state import AppState
from easy_bos.ui.theme import C

SPEED = {
    "instant": "instant",
    "fast": "fast",
    "moderate": "a few seconds",
    "slow": "slow, highest accuracy",
}
METHOD_ORDER = ["variational", "wavelet", "correlation", "lucas_kanade", "farneback", "dis", "raft"]


class ProcessPage(QWidget):
    finished = Signal()

    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._task = None
        self._preview: np.ndarray | None = None
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
        bar.setContentsMargins(12, 8, 12, 4)
        self.view_mode = W.combo(
            [("Frame", "frame"), ("Preview result", "result"), ("Method comparison", "compare")]
        )
        self.view_mode.currentIndexChanged.connect(self._view_mode)
        self.show_regions = QCheckBox("Show regions")
        self.show_regions.setChecked(True)
        self.show_regions.toggled.connect(lambda on: self.canvas.set_regions_visible(on))
        bar.addWidget(W.label("View", "Muted"))
        bar.addWidget(self.view_mode)
        bar.addSpacing(16)
        bar.addWidget(self.show_regions)
        bar.addStretch(1)
        self.hint = W.label("", "Faint")
        bar.addWidget(self.hint)
        rv.addLayout(bar)
        self.views = QStackedWidget()
        self.canvas = ImageCanvas()
        self.compare = CompareGrid()
        self.views.addWidget(self.canvas)
        self.views.addWidget(self.compare)
        rv.addWidget(self.views, 1)
        self.scrubber = FrameScrubber(playable=False)
        rv.addWidget(self.scrubber)
        lay.addWidget(right, 1)

        self.scrubber.changed.connect(self._frame_changed)
        self.canvas.regions_changed.connect(self._regions_changed)
        state.source_changed.connect(self._source_changed)
        state.project_changed.connect(self._load)
        self._load()
        self._source_changed()

    # ------------------------------------------------------------------ inspector
    def _inspector(self) -> QWidget:
        p = W.ScrollPanel()
        p.add(W.label("Process", "H1"))
        p.add(
            W.label(
                "Pick a method, try it on one frame, then run the sequence.", "Muted", wrap=True
            )
        )
        p.add(W.section("Method"))
        items = [(REGISTRY[k].label, k) for k in METHOD_ORDER if k in REGISTRY]
        self.method = W.combo(items)
        p.add(self.method)
        self.method_info = W.label("", "Faint", wrap=True)
        p.add(self.method_info)
        self.param_holder = QVBoxLayout()
        self.param_holder.setContentsMargins(0, 0, 0, 0)
        p.add(self.param_holder)
        self.param_form: W.ParamForm | None = None
        self.method.currentIndexChanged.connect(self._method_changed)

        p.add(W.section("Regions"))
        p.add(
            W.label(
                "Drag on the image to adjust; right-click a region to remove it.",
                "Faint",
                wrap=True,
            )
        )
        grid = QVBoxLayout()
        grid.setSpacing(6)
        grid.addLayout(
            W.hbox(
                W.button("Processing area", lambda: self._add_region("roi")),
                W.button(
                    "Exclude",
                    lambda: self._add_region("exclude"),
                    tooltip="Areas without a visible background (objects, "
                    "frame edges, the heat source itself).",
                ),
            )
        )
        grid.addLayout(
            W.hbox(
                W.button(
                    "Stable region",
                    lambda: self._add_region("stable"),
                    tooltip="Background that never sees flow; used to "
                    "measure camera motion exactly.",
                ),
                W.button("Clear all", self._clear_regions, flat=True),
            )
        )
        p.add(grid)

        p.add(W.section("Corrections"))
        self.stabilize = W.combo(
            [("Auto", "auto"), ("Always compensate", "on"), ("Off (fixed camera)", "off")]
        )
        self.stabilize.setToolTip(
            "Camera motion compensation. Auto compensates only when the flow-free reference "
            "frames show the camera moving, or when a frame jumps. If the flow fills the whole "
            "frame, a uniform shift cannot be told apart from camera motion: mark a stable "
            "region or use Off on a fixed camera."
        )
        self.motion_model = W.combo(
            [
                ("Shift + rotation", "rigid"),
                ("Shift only", "translation"),
                ("Affine (needs stable region)", "affine"),
            ]
        )
        self.exposure = QCheckBox("Match exposure between frames")
        self.banding = QCheckBox("Remove flicker banding")
        f = W.form()
        f.addRow(W.label("Camera motion", "Muted"), self.stabilize)
        f.addRow(W.label("Motion model", "Muted"), self.motion_model)
        p.add(f)
        p.add(self.exposure)
        p.add(self.banding)

        p.add(W.section("Frames"))
        f2 = W.form()
        self.ref_mode = W.combo([(m.label, m.value) for m in ReferenceMode])
        self.first = W.integer(0, 0, 10**7)
        self.last = W.integer(0, 0, 10**7)
        self.step = W.integer(1, 1, 1000)
        self.binning = W.combo([("Full resolution", 1), ("2 × 2", 2), ("4 × 4", 4)])
        self.binning.setToolTip(
            "Average pixel blocks first: 4x faster per step, less noise, "
            "lower resolution. 2 × 2 suits 4K video well."
        )
        self.tavg = W.integer(1, 1, 61, "frames")
        self.tavg.setToolTip("Moving average of the results over time. For steady flows only.")
        f2.addRow(W.label("Reference", "Muted"), self.ref_mode)
        f2.addRow(W.label("First frame", "Muted"), self.first)
        f2.addRow(W.label("Last frame", "Muted"), self.last)
        f2.addRow(W.label("Every", "Muted"), self.step)
        f2.addRow(W.label("Binning", "Muted"), self.binning)
        f2.addRow(W.label("Time average", "Muted"), self.tavg)
        p.add(f2)
        self.count_label = W.label("", "Faint")
        p.add(self.count_label)

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

        for wdg in (self.exposure, self.banding):
            wdg.toggled.connect(self._store)
        self.stabilize.currentIndexChanged.connect(self._store)
        for wdg in (self.motion_model, self.ref_mode, self.binning):
            wdg.currentIndexChanged.connect(self._store)
        for wdg in (self.first, self.last, self.step, self.tavg):
            wdg.valueChanged.connect(self._store)
        return p

    # ------------------------------------------------------------------ state <-> widgets
    def _load(self) -> None:
        self._loading = True
        pr = self.state.project.process
        self.method.setCurrentIndex(max(0, self.method.findData(pr.method)))
        self._rebuild_params(pr.method_params)
        self.stabilize.setCurrentIndex(max(0, self.stabilize.findData(pr.stabilize)))
        self.motion_model.setCurrentIndex(max(0, self.motion_model.findData(pr.motion_model)))
        self.exposure.setChecked(pr.exposure_correction)
        self.banding.setChecked(pr.banding_correction)
        self.ref_mode.setCurrentIndex(max(0, self.ref_mode.findData(pr.reference_mode)))
        self.binning.setCurrentIndex(max(0, self.binning.findData(pr.binning)))
        self.tavg.setValue(pr.temporal_average)
        self._loading = False

    def _store(self, *_) -> None:
        if self._loading:
            return
        pr = self.state.project.process
        pr.method = self.method.currentData()
        pr.method_params = self.param_form.values() if self.param_form else {}
        pr.stabilize = self.stabilize.currentData()
        pr.motion_model = self.motion_model.currentData()
        pr.exposure_correction = self.exposure.isChecked()
        pr.banding_correction = self.banding.isChecked()
        pr.reference_mode = self.ref_mode.currentData()
        pr.binning = self.binning.currentData()
        pr.temporal_average = self.tavg.value()
        if self.state.source is not None:
            pr.frame_range = (self.first.value(), self.last.value() + 1, self.step.value())
        self.motion_model.setEnabled(self.stabilize.currentData() != "off")
        self.state.mark_dirty()
        self._update_count()

    def _method_changed(self) -> None:
        if self._loading:
            return
        self._rebuild_params({})
        self._store()

    def _rebuild_params(self, values: dict) -> None:
        key = self.method.currentData()
        cls = REGISTRY[key]
        if self.param_form is not None:
            self.param_form.setParent(None)
            self.param_form.deleteLater()
        self.param_form = W.ParamForm(cls.Params, values)
        self.param_form.changed.connect(self._store)
        self.param_holder.addWidget(self.param_form)
        dev = "GPU" if cls.uses_gpu else "CPU"
        self.method_info.setText(f"{cls.summary} Speed: {SPEED[cls.speed]} ({dev}).")

    def _source_changed(self) -> None:
        src = self.state.source
        enabled = src is not None
        for b in (self.preview_btn, self.compare_btn, self.run_btn):
            b.setEnabled(enabled)
        self.canvas.blockSignals(True)
        self.canvas.clear_regions()
        self.canvas.blockSignals(False)
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

    def _update_count(self) -> None:
        if self.state.source is None:
            return
        n = len(range(self.first.value(), self.last.value() + 1, max(1, self.step.value())))
        self.count_label.setText(f"{n} frame{'s' if n != 1 else ''} to process")

    def _regions_changed(self) -> None:
        pr = self.state.project.process
        roi = self.canvas.region_rects("roi")
        pr.roi = roi[0] if roi else None
        pr.exclusions = self.canvas.region_rects("exclude")
        pr.stable_regions = self.canvas.region_rects("stable")
        self.state.mark_dirty()

    def _add_region(self, kind: str) -> None:
        if self.state.source is not None:
            self.view_mode.setCurrentIndex(0)
            self.canvas.add_region(kind)

    def _clear_regions(self) -> None:
        self.canvas.clear_regions()

    # ------------------------------------------------------------------ views
    def _frame_changed(self, index: int, fit: bool = False) -> None:
        self.state.set_frame(index)
        self._preview = None
        if self.view_mode.currentData() == "result":
            self.view_mode.setCurrentIndex(0)
        self._show_frame(fit)

    def _show_frame(self, fit: bool = False) -> None:
        img = self.state.frame(self.state.current_frame)
        if img is None:
            return
        lo, hi = np.percentile(img[::4, ::4], [0.5, 99.5])
        self.canvas.set_image(img, levels=(float(lo), float(hi)), keep_view=not fit)

    def _view_mode(self) -> None:
        mode = self.view_mode.currentData()
        self.views.setCurrentIndex(1 if mode == "compare" else 0)
        if mode == "frame":
            self._show_frame()
        elif mode == "result" and self._preview is not None:
            self._show_preview()

    def _show_preview(self) -> None:
        d = self._preview
        rgb = render.render(render.View.KNIFE_EDGE, d, render.ViewSettings(scale=0))
        # Show the processed region in place on the full frame.
        img = self.state.frame(self.state.current_frame)
        full = np.zeros((*img.shape, 3), np.uint8)
        lo, hi = np.percentile(img[::4, ::4], [0.5, 99.5])
        g = np.clip((img - lo) / max(hi - lo, 1e-6) * 255 * 0.35, 0, 255).astype(np.uint8)
        full[:] = g[..., None]
        pr = self.state.project.process
        x0, y0, w, h = pr.roi if pr.roi else (0, 0, img.shape[1], img.shape[0])
        b = max(1, pr.binning)
        if b > 1:
            rgb = np.repeat(np.repeat(rgb, b, axis=0), b, axis=1)
        hh, ww = min(rgb.shape[0], h), min(rgb.shape[1], w)
        full[y0 : y0 + hh, x0 : x0 + ww] = rgb[:hh, :ww]
        self.canvas.set_image(full)

    # ------------------------------------------------------------------ processing
    def _processor(self, method: str | None = None, params: dict | None = None) -> Processor:
        src = self.state.source
        pr = self.state.project.process
        settings = pr.settings((src.info.height, src.info.width))
        if method is not None:
            settings = settings.with_(method=method, method_params=params or {})
        return Processor(src, settings)

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
            return proc.process_frame(index)

        def done(res):
            self._set_busy(False)
            self._preview = res.displacement
            mot = ""
            if res.motion is not None:
                tx, ty = res.motion.translation
                mot = f" Camera motion {np.hypot(tx, ty):.2f} px, {res.motion.rotation_deg:.3f}°."
            peak = np.nanpercentile(np.hypot(*res.displacement), 99.5)
            self.run_status.setText(
                f"Frame {index}: {res.elapsed_s:.2f} s, peak displacement {peak:.3f} px.{mot}"
            )
            self.view_mode.setCurrentIndex(1)
            self._show_preview()

        t = self.state.run_task(work, done, message="Preview")
        t.failed.connect(lambda *_: self._set_busy(False))

    def _compare(self) -> None:
        if self.state.source is None:
            return
        self._store()
        index = self.state.current_frame
        base = self._processor()
        methods = [k for k in METHOD_ORDER if k in REGISTRY]
        self._set_busy(True)
        self.view_mode.setCurrentIndex(2)
        self.compare.reset(methods)

        def work(progress, cancelled):
            ref = (
                base.reference
                if base.settings.reference_mode is not ReferenceMode.PREVIOUS
                else base.load(max(0, index - 1))
            )
            cur = base.load(index)
            out = {}
            for i, key in enumerate(methods):
                if cancelled():
                    break
                progress(i, len(methods), REGISTRY[key].label)
                proc = Processor(
                    ArraySource(np.stack([ref, cur])),
                    base.settings.with_(
                        method=key,
                        method_params={},
                        reference_mode=ReferenceMode.AVERAGE,
                        reference_range=(0, 1),
                        roi=None,
                        binning=1,
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
            self.compare.show(out)
            self.run_status.setText(
                "Comparison on the current frame with default parameters. "
                "Δ is the mean difference from variational flow. Agreement "
                "between methods is a good sign; outliers deserve a closer "
                "look."
            )

        t = self.state.run_task(work, done, message="Comparing methods")
        t.failed.connect(lambda *_: self._set_busy(False))
        self._task = t

    def _run(self) -> None:
        if self.state.source is None:
            return
        self._store()
        proc = self._processor()
        store = self.state.new_results()
        st = self.state.setup
        pr = self.state.project.process
        store.set_metadata(
            project=self.state.project.to_json(),
            method=pr.method,
            pixels_per_radian=st.pixels_per_radian,
            object_pixel_size_m=st.object_pixel_size_m,
            binning=pr.binning,
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
            el = time.perf_counter() - t_start
            eta = el / max(done, 1) * (total - done)
            self.run_status.setText(f"Frame {done} of {total} · {_fmt_time(eta)} left")
            win = self.window()
            if hasattr(win, "set_progress"):
                win.set_progress(done, total, "Processing")

        def done(n):
            self._set_busy(False)
            dt = time.perf_counter() - t_start
            self.run_status.setText(f"Processed {n} of {len(indices)} frames in {_fmt_time(dt)}.")
            self.state.results_changed.emit()
            if n:
                self.finished.emit()

        t = self.state.run_task(work, done, on_progress, message="Processing")
        t.failed.connect(lambda *_: self._set_busy(False))
        self._task = t

    def _cancel(self) -> None:
        self.state.cancel_tasks()
        self.run_status.setText("Cancelling…")

    def activated(self) -> None:
        if self.state.source is not None:
            self.scrubber.set_frame(self.state.current_frame)
            self._show_frame()


class CompareGrid(pg.GraphicsLayoutWidget):
    """Small multiples of the knife-edge view, one per method, with a shared scale."""

    def __init__(self):
        super().__init__()
        self.setBackground(C.canvas)
        self._boxes: dict[str, tuple[pg.ViewBox, pg.ImageItem, pg.LabelItem]] = {}

    def reset(self, methods: list[str]) -> None:
        self.clear()
        self._boxes.clear()
        cols = 3
        first_vb = None
        for i, key in enumerate(methods):
            r, c = divmod(i, cols)
            lab = self.addLabel(
                f"{REGISTRY[key].label} · running…", row=2 * r, col=c, color=C.muted, size="8.5pt"
            )
            vb = self.addViewBox(
                row=2 * r + 1, col=c, lockAspect=True, invertY=True, enableMenu=False
            )
            img = pg.ImageItem(axisOrder="row-major")
            vb.addItem(img)
            if first_vb is None:
                first_vb = vb
            else:
                vb.setXLink(first_vb)
                vb.setYLink(first_vb)
            self._boxes[key] = (vb, img, lab)

    def show(self, out: dict) -> None:
        mags = [np.hypot(*d) for d, _, e in out.values() if d is not None]
        scale = (
            float(np.nanpercentile(np.concatenate([m.ravel() for m in mags]), 99.5))
            if mags
            else 1.0
        )
        ref = out.get("variational", (None,))[0]
        for key, (d, secs, err) in out.items():
            vb, img, lab = self._boxes[key]
            if d is None:
                lab.setText(f"{REGISTRY[key].label} · failed: {err[:40]}")
                continue
            rgb = render.render(render.View.KNIFE_EDGE, d, render.ViewSettings(scale=scale))
            img.setImage(rgb, levels=(0, 255))
            agree = ""
            if ref is not None and key != "variational":
                diff = np.nanmean(np.hypot(*(d - ref)))
                agree = f" · Δ {diff:.3f} px"
            lab.setText(f"{REGISTRY[key].label} · {secs:.1f} s{agree}")
            vb.autoRange(padding=0.02)


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.0f} s"
    m, sec = divmod(int(s), 60)
    if m < 60:
        return f"{m} min {sec:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"
