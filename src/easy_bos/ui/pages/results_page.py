"""Step 4 — Results: schlieren views, physical fields, profiles, time series and export."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from easy_bos.core import export, physics, render
from easy_bos.core.geometry import Camera, Setup
from easy_bos.core.reconstruct.density import FlowGeometry, Reconstruction, reconstruct
from easy_bos.ui import widgets as W
from easy_bos.ui.canvas import ImageCanvas, sample_line
from easy_bos.ui.colorbar import ColorBar
from easy_bos.ui.frames import FrameScrubber
from easy_bos.ui.state import AppState
from easy_bos.ui.theme import C

VIEWS = [
    render.View.KNIFE_EDGE,
    render.View.MAGNITUDE,
    render.View.COLOUR_SCHLIEREN,
    render.View.DIRECTION,
    render.View.HORIZONTAL,
    render.View.VERTICAL,
    render.View.DIVERGENCE,
    render.View.DENSITY,
    render.View.TEMPERATURE,
]


def processed_setup(state: AppState) -> Setup:
    """Setup describing the processed (cropped, binned) image grid."""
    s = state.project.setup
    b = max(1, state.project.process.binning)
    cam = Camera(
        s.focal_length_mm * 1e-3,
        s.sensor_width_mm * 1e-3,
        max(1, s.image_width_px // b),
        max(1, s.image_height_px // b),
        s.camera_name,
    )
    # Keep the pixel pitch exact even if the width is not divisible by the binning factor.
    cam = Camera(
        cam.focal_length_m,
        s.sensor_width_mm * 1e-3 / s.image_width_px * b * cam.image_width_px,
        cam.image_width_px,
        cam.image_height_px,
        cam.name,
    )
    return Setup(cam, s.z_a_m, s.z_d_m, s.f_number)


class ResultsPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.index: int | None = None
        self._d: np.ndarray | None = None
        self._mean_cache: np.ndarray | None = None
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._inspector())

        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        split = QSplitter(Qt.Orientation.Vertical)
        top = QWidget()
        tl = QHBoxLayout(top)
        tl.setContentsMargins(0, 0, 0, 0)
        tl.setSpacing(0)
        self.canvas = ImageCanvas()
        self.colorbar = ColorBar()
        tl.addWidget(self.canvas, 1)
        tl.addWidget(self.colorbar)
        split.addWidget(top)
        self.plots = QTabWidget()
        self.profile_plot = _plot("Distance along line", "")
        self.series_plot = _plot("Time", "")
        self.hist_plot = _plot("", "Pixels")
        self.plots.addTab(self.profile_plot, "Profile")
        self.plots.addTab(self.series_plot, "Time series")
        self.plots.addTab(self.hist_plot, "Histogram")
        split.addWidget(self.plots)
        split.setSizes([640, 220])
        rv.addWidget(split, 1)
        self.scrubber = FrameScrubber()
        rv.addWidget(self.scrubber)
        self.readout = W.label("", "Value")
        self.readout.setContentsMargins(14, 0, 14, 6)
        rv.addWidget(self.readout)
        lay.addWidget(right, 1)
        self.empty = W.label("No results yet. Process a capture first.", "Muted")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rv.insertWidget(0, self.empty)

        self.scrubber.changed.connect(self._frame)
        self.canvas.hovered.connect(self._hover)
        self.canvas.line_changed.connect(self._update_profile)
        self.plots.currentChanged.connect(lambda *_: self._update_plots())
        state.results_changed.connect(self._results_changed)
        state.project_changed.connect(self._load_recon_settings)
        self._load_recon_settings()
        self._results_changed()

    # ------------------------------------------------------------------ inspector
    def _inspector(self) -> QWidget:
        p = W.ScrollPanel()
        p.add(W.label("Results", "H1"))
        p.add(W.section("View"))
        self.view = W.combo([(v.label, v.value) for v in VIEWS])
        p.add(self.view)
        f = W.form()
        self.auto_scale = QCheckBox("Automatic")
        self.auto_scale.setChecked(True)
        self.scale = W.number(0.5, 1e-4, 1e4, 3)
        self.units = W.combo([("Pixels", "px"), ("Deflection (mrad)", "mrad")])
        self.cmap = W.combo([("Default", "auto")] + [(c, c) for c in render.COLORMAPS])
        self.knife = QSlider(Qt.Orientation.Horizontal)
        self.knife.setRange(0, 359)
        self.knife.setToolTip(
            "Knife-edge orientation: 0° shows horizontal density gradients "
            "(vertical knife edge), 90° vertical ones."
        )
        self.mean = QCheckBox("Average over all frames")
        self.mean.setToolTip("Time-averaged field: steady flows, much less noise.")
        f.addRow(W.label("Range", "Muted"), W.hbox(self.scale, self.auto_scale))
        f.addRow(W.label("Units", "Muted"), self.units)
        f.addRow(W.label("Colours", "Muted"), self.cmap)
        f.addRow(W.label("Knife edge", "Muted"), self.knife)
        p.add(f)
        p.add(self.mean)
        for wdg in (self.view, self.units, self.cmap):
            wdg.currentIndexChanged.connect(self._redraw)
        self.scale.valueChanged.connect(self._redraw)
        self.auto_scale.toggled.connect(self._redraw)
        self.knife.valueChanged.connect(self._redraw)
        self.mean.toggled.connect(lambda *_: self._frame(self.scrubber.frame()))

        p.add(W.section("Tools"))
        self.line_btn = W.button("Profile line")
        self.line_btn.setCheckable(True)
        self.line_btn.toggled.connect(self._toggle_line)
        p.add(
            W.hbox(self.line_btn, W.button("Fit view", lambda: self.canvas.fit(), flat=True), None)
        )

        p.add(W.section("Density and temperature"))
        p.add(
            W.label(
                "Physical fields need the flow's geometry and the setup distances from step 1.",
                "Faint",
                wrap=True,
            )
        )
        f2 = W.form()
        self.geom = W.combo([(g.label, g.value) for g in FlowGeometry])
        self.depth = W.number(100, 1, 10000, 0, "mm")
        self.orient = W.combo(
            [("Vertical axis (plume)", "vertical"), ("Horizontal axis (jet)", "horizontal")]
        )
        self.axis_auto = QCheckBox("Find axis automatically")
        self.axis_pos = W.number(0, 0, 100000, 1, "px")
        self.abel = W.combo(
            [
                ("Integrate, then invert (robust)", "indirect"),
                ("Invert deflection (sharp)", "direct"),
            ]
        )
        self.t_amb = W.number(20, -50, 60, 1, "°C")
        self.p_amb = W.number(101.325, 50, 120, 3, "kPa")
        self.ds = W.combo([("Full", 1), ("½", 2), ("¼", 4)])
        self.ds.setToolTip(
            "Reconstruct on a coarser grid: faster and less noisy. Blur in the "
            "flow plane usually limits resolution anyway."
        )
        anchors = QHBoxLayout()
        self.anchor = {}
        for k, t in (("l", "Left"), ("r", "Right"), ("t", "Top"), ("b", "Bottom")):
            cb = QCheckBox(t)
            self.anchor[k] = cb
            anchors.addWidget(cb)
        f2.addRow(W.label("Geometry", "Muted"), self.geom)
        f2.addRow(W.label("Depth", "Muted"), self.depth)
        f2.addRow(W.label("Axis", "Muted"), self.orient)
        f2.addRow(W.label("", "Muted"), W.hbox(self.axis_auto, self.axis_pos))
        f2.addRow(W.label("Method", "Muted"), self.abel)
        f2.addRow(W.label("Ambient", "Muted"), W.hbox(self.t_amb, self.p_amb))
        f2.addRow(W.label("Resolution", "Muted"), self.ds)
        p.add(f2)
        p.add(W.label("Ambient air at these image edges:", "Muted"))
        p.add(anchors)
        self.recon_btn = W.button("Reconstruct", self._reconstruct, primary=True)
        self.recon_all = W.button("All frames", self._reconstruct_all)
        p.add(W.hbox(self.recon_btn, self.recon_all, None))
        self.recon_status = W.label("", "Muted", wrap=True)
        p.add(self.recon_status)
        self.recon_warn = QVBoxLayout()
        p.add(self.recon_warn)
        for wdg in (self.geom, self.orient, self.abel, self.ds):
            wdg.currentIndexChanged.connect(self._store_recon)
        for wdg in (self.depth, self.axis_pos, self.t_amb, self.p_amb):
            wdg.valueChanged.connect(self._store_recon)
        self.axis_auto.toggled.connect(self._store_recon)
        for cb in self.anchor.values():
            cb.toggled.connect(self._store_recon)

        p.add(W.section("Export"))
        p.add(
            W.hbox(
                W.button("Image…", self._export_image), W.button("Video…", self._export_video), None
            )
        )
        p.add(
            W.hbox(
                W.button("Data (HDF5)…", self._export_data),
                W.button("Figure…", self._export_figure),
                None,
            )
        )
        p.add(W.hbox(W.button("Profile (CSV)…", self._export_profile), None))
        self.export_status = W.label("", "Faint", wrap=True)
        p.add(self.export_status)
        p.finish()
        return p

    # ------------------------------------------------------------------ data
    def _results_changed(self) -> None:
        store = self.state.results
        has = store is not None and len(store) > 0
        self.canvas.setVisible(has)
        self.colorbar.setVisible(has)
        self.plots.setVisible(has)
        self.scrubber.setVisible(has)
        self.empty.setVisible(not has)
        self._mean_cache = None
        if not has:
            self.index = None
            self._d = None
            return
        idx = store.indices()
        fps = store.metadata("fps", 0.0) or None
        self.scrubber.configure(idx[-1] + 1, fps, allowed=idx)
        self.scrubber.set_frame(idx[0])
        self._frame(idx[0], fit=True)

    def _frame(self, index: int, fit: bool = False) -> None:
        store = self.state.results
        if store is None or not len(store):
            return
        self.index = index
        if self.mean.isChecked():
            if self._mean_cache is None:
                self._mean_cache = store.mean_displacement()
            self._d = self._mean_cache
        else:
            self._mean_cache = None
            self._d = store.displacement(index)
        self._redraw(fit=fit)
        self._update_plots()

    def _scalar(self, view: render.View) -> tuple[np.ndarray | None, str, str]:
        """Scalar field for probes and plots: (data, units, title)."""
        d = self._d
        if d is None:
            return None, "", ""
        if view in (render.View.DENSITY, render.View.TEMPERATURE):
            rec = self._recon()
            if rec is None:
                return None, "", ""
            if view is render.View.DENSITY:
                return rec.density, "kg/m³", "Density"
            return physics.celsius(rec.temperature), "°C", "Temperature"
        factor, unit = self._unit_factor()
        if view is render.View.KNIFE_EDGE:
            a = np.radians(self.knife.value())
            return (d[0] * np.cos(a) + d[1] * np.sin(a)) * factor, unit, "Displacement ⟂ knife edge"
        if view in (render.View.MAGNITUDE, render.View.COLOUR_SCHLIEREN, render.View.DIRECTION):
            return np.hypot(d[0], d[1]) * factor, unit, "Magnitude"
        if view is render.View.HORIZONTAL:
            return d[0] * factor, unit, "Horizontal"
        if view is render.View.VERTICAL:
            return d[1] * factor, unit, "Vertical"
        if view is render.View.DIVERGENCE:
            return render.divergence(d) * factor, f"{unit}/px", "Divergence"
        return None, "", ""

    def _unit_factor(self) -> tuple[float, str]:
        if self.units.currentData() == "mrad":
            return 1e3 / processed_setup(self.state).pixels_per_radian, "mrad"
        return 1.0, "px"

    def _recon(self) -> Reconstruction | None:
        key = -1 if self.mean.isChecked() else self.index
        return self.state.reconstructions.get(key)

    def _redraw(self, *_, fit: bool = False) -> None:
        if self._d is None:
            return
        view = render.View(self.view.currentData())
        scalar, unit, title = self._scalar(view)
        cmap_name = self.cmap.currentData()
        if view.needs_reconstruction and scalar is None:
            self.canvas.set_image(np.zeros((*self._d.shape[1:], 3), np.uint8), keep_view=not fit)
            self.colorbar.configure("gray", 0, 1, "", "Press Reconstruct")
            return
        if self.auto_scale.isChecked() or scalar is None:
            if view.needs_reconstruction:
                lo, hi = np.nanpercentile(scalar, [0.5, 99.5])
            else:
                s = render.auto_scale(scalar)
                lo, hi = (
                    (0.0, s)
                    if view
                    in (render.View.MAGNITUDE, render.View.COLOUR_SCHLIEREN, render.View.DIRECTION)
                    else (-s, s)
                )
            self.scale.blockSignals(True)
            self.scale.setValue(float(max(abs(lo), abs(hi))))
            self.scale.blockSignals(False)
        else:
            s = self.scale.value()
            lo, hi = (
                (0.0, s)
                if view
                in (render.View.MAGNITUDE, render.View.COLOUR_SCHLIEREN, render.View.DIRECTION)
                else (-s, s)
            )
            if view.needs_reconstruction:
                c = float(np.nanmedian(scalar))
                lo, hi = c - s, c + s
        factor, _ = self._unit_factor()
        settings = render.ViewSettings(
            scale=(hi / factor) if not view.needs_reconstruction else hi,
            knife_angle_deg=float(self.knife.value()),
            colormap=cmap_name,
        )
        if view.needs_reconstruction:
            cm = render.DEFAULT_CMAPS[view] if cmap_name == "auto" else cmap_name
            rgb = render.colorize(scalar, cm, lo, hi)
        else:
            rgb = render.render(view, self._d, settings)
            cm = render.DEFAULT_CMAPS.get(view, "gray") if cmap_name == "auto" else cmap_name
        self.canvas.set_image(rgb, keep_view=not fit)
        rec = self._recon()
        if rec is not None and rec.axis_px is not None and view.needs_reconstruction:
            self.canvas.show_axis(rec.axis_px, rec.settings.axis_orientation == "horizontal")
        else:
            self.canvas.show_axis(None)
        wheel = view in (render.View.DIRECTION, render.View.COLOUR_SCHLIEREN)
        self.colorbar.configure(cm, lo, hi, unit, title, wheel=wheel)
        self._update_plots()

    # ------------------------------------------------------------------ probes and plots
    def _hover(self, x: float, y: float) -> None:
        if self._d is None or not np.isfinite(x):
            self.readout.setText("")
            return
        i, j = int(y), int(x)
        st = processed_setup(self.state)
        u, v = self._d[:, i, j]
        eps = np.hypot(u, v) / st.pixels_per_radian * 1e3
        txt = (
            f"x {j}  y {i}   ·   D = ({u:+.3f}, {v:+.3f}) px   |D| {np.hypot(u, v):.3f} px"
            f"   ·   ε {eps:.4f} mrad"
        )
        rec = self._recon()
        if rec is not None and np.isfinite(rec.density[i, j]):
            txt += (
                f"   ·   ρ {rec.density[i, j]:.4f} kg/m³   T "
                f"{physics.celsius(rec.temperature[i, j]):.1f} °C"
            )
        self.readout.setText(txt)

    def _toggle_line(self, on: bool) -> None:
        self.canvas.enable_line(on)
        if on:
            self.plots.setCurrentIndex(0)
        self._update_profile()

    def _update_plots(self) -> None:
        tab = self.plots.currentIndex()
        if tab == 0:
            self._update_profile()
        elif tab == 1:
            self._update_series()
        else:
            self._update_hist()

    def _update_profile(self) -> None:
        self.profile_plot.clear()
        ends = self.canvas.line_endpoints()
        view = render.View(self.view.currentData())
        scalar, unit, title = self._scalar(view)
        if ends is None or scalar is None:
            return
        dist, vals = sample_line(scalar, *ends)
        mm = processed_setup(self.state).object_pixel_size_m * 1e3
        self.profile_plot.plot(dist * mm, vals, pen=pg.mkPen(C.accent, width=1.5))
        self.profile_plot.setLabel("bottom", "Distance along line", units="mm")
        self.profile_plot.setLabel("left", title, units=None)
        self.profile_plot.getAxis("left").setLabel(f"{title} ({unit})")

    def _update_series(self) -> None:
        self.series_plot.clear()
        store = self.state.results
        if store is None:
            return
        idx = store.indices()
        if len(idx) > 400:
            idx = idx[:: len(idx) // 400 + 1]
        factor, unit = self._unit_factor()
        vals = []
        for i in idx:
            d = store.displacement(i)
            m = np.hypot(d[0], d[1])
            vals.append(float(np.nanpercentile(m, 99)) * factor)
        fps = store.metadata("fps", 0.0) or 0.0
        t = np.array(idx) / fps if fps else np.array(idx, float)
        self.series_plot.plot(t, vals, pen=pg.mkPen(C.accent, width=1.5))
        if self.index is not None:
            x = self.index / fps if fps else self.index
            self.series_plot.addLine(x=x, pen=pg.mkPen(C.warn, style=Qt.PenStyle.DashLine))
        self.series_plot.setLabel("bottom", "Time (s)" if fps else "Frame")
        self.series_plot.getAxis("left").setLabel(f"99th percentile |D| ({unit})")

    def _update_hist(self) -> None:
        self.hist_plot.clear()
        view = render.View(self.view.currentData())
        scalar, unit, title = self._scalar(view)
        if scalar is None:
            return
        v = scalar[np.isfinite(scalar)]
        if v.size == 0:
            return
        hist, edges = np.histogram(v, bins=120)
        self.hist_plot.plot(
            edges,
            hist,
            stepMode="center",
            fillLevel=0,
            brush=pg.mkBrush(C.accent_dim),
            pen=pg.mkPen(C.accent),
        )
        self.hist_plot.getAxis("bottom").setLabel(f"{title} ({unit})")

    # ------------------------------------------------------------------ reconstruction
    def _load_recon_settings(self) -> None:
        r = self.state.project.reconstruct
        widgets = (
            self.geom,
            self.orient,
            self.abel,
            self.ds,
            self.depth,
            self.axis_pos,
            self.t_amb,
            self.p_amb,
            self.axis_auto,
            *self.anchor.values(),
        )
        for w in widgets:
            w.blockSignals(True)
        self.geom.setCurrentIndex(max(0, self.geom.findData(r.geometry)))
        self.orient.setCurrentIndex(max(0, self.orient.findData(r.axis_orientation)))
        self.abel.setCurrentIndex(max(0, self.abel.findData(r.abel_method)))
        self.ds.setCurrentIndex(max(0, self.ds.findData(r.downsample)))
        self.depth.setValue(r.depth_mm)
        self.axis_pos.setValue(r.axis_px)
        self.t_amb.setValue(r.t_ambient_c)
        self.p_amb.setValue(r.p_ambient_kpa)
        self.axis_auto.setChecked(r.axis_auto)
        for k, cb in self.anchor.items():
            cb.setChecked(k in r.anchor_sides)
        for w in widgets:
            w.blockSignals(False)
        self._recon_enable()

    def _store_recon(self, *_) -> None:
        r = self.state.project.reconstruct
        r.geometry = self.geom.currentData()
        r.axis_orientation = self.orient.currentData()
        r.abel_method = self.abel.currentData()
        r.downsample = self.ds.currentData()
        r.depth_mm = self.depth.value()
        r.axis_px = self.axis_pos.value()
        r.t_ambient_c = self.t_amb.value()
        r.p_ambient_kpa = self.p_amb.value()
        r.axis_auto = self.axis_auto.isChecked()
        r.anchor_sides = "".join(k for k, cb in self.anchor.items() if cb.isChecked()) or "lr"
        self.state.mark_dirty()
        self._recon_enable()

    def _recon_enable(self) -> None:
        axi = self.geom.currentData() == FlowGeometry.AXISYMMETRIC.value
        self.depth.setEnabled(not axi)
        for w in (self.orient, self.axis_auto, self.abel):
            w.setEnabled(axi)
        self.axis_pos.setEnabled(axi and not self.axis_auto.isChecked())

    def _reconstruct(self) -> None:
        if self._d is None:
            return
        self._store_recon()
        d = self._d.copy()
        key = -1 if self.mean.isChecked() else self.index
        setup = processed_setup(self.state)
        settings = self.state.project.reconstruct.settings()
        self.recon_btn.setEnabled(False)
        self.recon_status.setText("Reconstructing…")

        def work(progress, cancelled):
            return reconstruct(d, setup, settings)

        def done(rec: Reconstruction):
            self.recon_btn.setEnabled(True)
            self.state.reconstructions[key] = rec
            self._show_recon_info(rec)
            if not render.View(self.view.currentData()).needs_reconstruction:
                self.view.setCurrentIndex(VIEWS.index(render.View.TEMPERATURE))
            else:
                self._redraw()

        t = self.state.run_task(work, done, message="Reconstructing")
        t.failed.connect(lambda *_: self.recon_btn.setEnabled(True))

    def _reconstruct_all(self) -> None:
        store = self.state.results
        if store is None:
            return
        self._store_recon()
        setup = processed_setup(self.state)
        settings = self.state.project.reconstruct.settings()
        idx = store.indices()
        self.recon_all.setEnabled(False)

        def work(progress, cancelled):
            out = {}
            for n, i in enumerate(idx):
                if cancelled():
                    break
                rec = reconstruct(store.displacement(i), setup, settings)
                out[i] = rec
                store.add_reconstruction(
                    i, rec.density, rec.temperature, {"geometry": settings.geometry.value}
                )
                progress(n + 1, len(idx), "")
            return out

        def done(out):
            self.recon_all.setEnabled(True)
            self.state.reconstructions.update(out)
            if self.index in out:
                self._show_recon_info(out[self.index])
            self.recon_status.setText(f"Reconstructed {len(out)} frames.")
            self._redraw()

        def prog(d, t, _):
            self.recon_status.setText(f"Reconstructing frame {d} of {t}…")

        t = self.state.run_task(work, done, prog, message="Reconstructing")
        t.failed.connect(lambda *_: self.recon_all.setEnabled(True))

    def _show_recon_info(self, rec: Reconstruction) -> None:
        t = physics.celsius(rec.temperature)
        parts = [f"Peak {np.nanmax(t):.1f} °C · minimum density {np.nanmin(rec.density):.4f} kg/m³"]
        if rec.axis_px is not None:
            parts.append(f"axis at {rec.axis_px:.1f} px")
        self.recon_status.setText(" · ".join(parts))
        W.clear_layout(self.recon_warn)
        for w in rec.warnings:
            self.recon_warn.addWidget(W.StatusRow("Check", "warn", "", w))

    # ------------------------------------------------------------------ export
    def _current_rgb(self) -> np.ndarray | None:
        img = self.canvas.image.image
        return None if img is None else np.asarray(img)

    def _export_image(self) -> None:
        if self._d is None:
            return
        path, filt = QFileDialog.getSaveFileName(
            self,
            "Export image",
            f"bos-{self.index:05d}.png",
            "Rendered view (*.png);;Data as 32-bit TIFF (*.tif)",
        )
        if not path:
            return
        if path.lower().endswith((".tif", ".tiff")):
            scalar, unit, _ = self._scalar(render.View(self.view.currentData()))
            export.write_tiff_float(path, scalar if scalar is not None else np.hypot(*self._d))
        else:
            export.write_png(path, self._current_rgb())
        self.export_status.setText(f"Saved {Path(path).name}")

    def _export_video(self) -> None:
        store = self.state.results
        if store is None:
            return
        opts = ";;".join(f"{v[0]} (*.{k})" for k, v in export.VIDEO_FORMATS.items())
        path, _ = QFileDialog.getSaveFileName(self, "Export video", "bos.mp4", opts)
        if not path:
            return
        view = render.View(self.view.currentData())
        if view.needs_reconstruction:
            self.state.error.emit(
                "Video export of density/temperature needs reconstruction of "
                "all frames: use 'All frames' first, then export."
            )
            return
        idx = store.indices()
        scalar, _, _ = self._scalar(view)
        factor, _ = self._unit_factor()
        s = (
            self.scale.value() / factor
            if not self.auto_scale.isChecked()
            else render.auto_scale(scalar / factor)
        )
        settings = render.ViewSettings(
            scale=s, knife_angle_deg=float(self.knife.value()), colormap=self.cmap.currentData()
        )
        fps = store.metadata("fps", 0.0) or 25.0

        def work(progress, cancelled):
            frames = (render.render(view, store.displacement(i), settings) for i in idx)
            export.write_video(path, frames, fps, lambda n: progress(n, len(idx), ""))
            return path

        self.state.run_task(
            work,
            lambda p: self.export_status.setText(f"Saved {Path(p).name}"),
            message="Encoding video",
        )

    def _export_data(self) -> None:
        store = self.state.results
        if store is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export data", "bos-results.h5", "HDF5 (*.h5)")
        if not path:
            return
        setup = processed_setup(self.state)

        def work(progress, cancelled):
            store.set_metadata(
                pixels_per_radian=setup.pixels_per_radian,
                object_pixel_size_m=setup.object_pixel_size_m,
                units_displacement="pixel",
                units_density="kg/m^3",
                units_temperature="K",
            )
            store.save_as(path)
            return path

        self.state.run_task(
            work,
            lambda p: self.export_status.setText(
                f"Saved {Path(p).name}: displacement per frame (px), reconstructions, setup."
            ),
            message="Writing data",
        )

    def _export_figure(self) -> None:
        if self._d is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export figure", "bos-figure.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if not path:
            return
        view = render.View(self.view.currentData())
        scalar, unit, title = self._scalar(view)
        rgb = self._current_rgb()
        mm = processed_setup(self.state).object_pixel_size_m * 1e3
        from easy_bos.core.figures import save_field_figure

        save_field_figure(
            path,
            rgb,
            mm,
            title=f"{view.label}",
            cbar_label=f"{title} ({unit})",
            vrange=(self.colorbar.vmin, self.colorbar.vmax),
            cmap=self.colorbar.cmap if not self.colorbar.wheel else None,
        )
        self.export_status.setText(f"Saved {Path(path).name}")

    def _export_profile(self) -> None:
        ends = self.canvas.line_endpoints()
        if ends is None:
            self.export_status.setText("Draw a profile line first (Tools → Profile line).")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export profile", "bos-profile.csv", "CSV (*.csv)"
        )
        if not path:
            return
        mm = processed_setup(self.state).object_pixel_size_m * 1e3
        cols = {}
        for v in (
            render.View.HORIZONTAL,
            render.View.VERTICAL,
            render.View.MAGNITUDE,
            render.View.DENSITY,
            render.View.TEMPERATURE,
        ):
            scalar, unit, title = self._scalar(v)
            if scalar is not None:
                dist, vals = sample_line(scalar, *ends)
                cols[f"{title} ({unit})"] = vals
        if not cols:
            self.export_status.setText("Nothing to export along this line.")
            return
        export.write_profile_csv(path, dist * mm, cols)
        self.export_status.setText(f"Saved {Path(path).name} (distance in mm)")

    def activated(self) -> None:
        if self.state.results is not None and len(self.state.results) and self._d is None:
            self._results_changed()


def _plot(xlabel: str, ylabel: str) -> pg.PlotWidget:
    w = pg.PlotWidget()
    w.setBackground(C.canvas)
    w.showGrid(x=True, y=True, alpha=0.15)
    for ax in ("left", "bottom"):
        w.getAxis(ax).setPen(pg.mkPen(C.border_strong))
        w.getAxis(ax).setTextPen(pg.mkPen(C.muted))
    if xlabel:
        w.setLabel("bottom", xlabel)
    if ylabel:
        w.setLabel("left", ylabel)
    return w
