"""Step 1 — Setup: camera, geometry and background, with a live report of what to expect."""

from __future__ import annotations

import math

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QRadioButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from easy_bos.core.export import write_png, write_print_pattern_pdf
from easy_bos.core.geometry import (
    CAMERA_PRESETS,
    DISPLAY_PRESETS,
    MoireRisk,
    analyze_display,
    recommend_f_number,
)
from easy_bos.core.patterns import PatternKind, generate
from easy_bos.ui import widgets as W
from easy_bos.ui.diagram import SetupDiagram
from easy_bos.ui.pattern_window import PatternWindow, physical_size, screen_diagonal_inches, screens
from easy_bos.ui.state import AppState

PAPERS = {
    "A4": (210, 297),
    "A3": (297, 420),
    "A2": (420, 594),
    "Letter": (216, 279),
    "Tabloid": (279, 432),
}


class SetupPage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self._pattern_window: PatternWindow | None = None
        self._loading = False
        self._emitting = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._build_inspector())
        lay.addWidget(self._build_report(), 1)
        state.project_changed.connect(self.load_from_project)
        state.setup_changed.connect(self.load_from_project)
        self.load_from_project()

    # ------------------------------------------------------------------ inspector
    def _build_inspector(self) -> QWidget:
        panel = W.ScrollPanel()
        panel.add(W.label("Setup", "H1"))
        panel.add(
            W.label(
                "Describe your camera and where things stand; everything on the right updates "
                "as you type. Rule of thumb: put the flow about halfway between camera and "
                "background, use the longest lens that still frames the flow, and stop down "
                "to f/8–f/11. Hover over any field for guidance.",
                "Muted",
                wrap=True,
            )
        )

        panel.add(W.section("Camera"))
        f = W.form()
        self.cam_preset = W.combo([("Custom", "")] + [(k, k) for k in CAMERA_PRESETS])
        self.focal = W.number(50, 1, 2000, 1, "mm")
        self.sensor_w = W.number(23.5, 1, 100, 2, "mm")
        self.img_w = W.integer(3840, 64, 20000, "px")
        self.img_h = W.integer(2160, 64, 20000, "px")
        self.fnum = W.number(8, 1, 32, 1)
        self.fnum.setPrefix("f/")
        f.addRow(W.label("Preset", "Muted"), self.cam_preset)
        f.addRow(W.label("Focal length", "Muted"), self.focal)
        f.addRow(W.label("Sensor width", "Muted"), self.sensor_w)
        f.addRow(W.label("Frame width", "Muted"), self.img_w)
        f.addRow(W.label("Frame height", "Muted"), self.img_h)
        f.addRow(W.label("Aperture", "Muted"), self.fnum)
        self.fnum.setToolTip(
            "Higher f-number: sharper flow (less depth-of-field blur) but less light and more "
            "diffraction blur of the pattern. f/8–f/11 is typical; the report shows the "
            "recommended range for your setup."
        )
        self.sensor_w.setToolTip(
            "Width of the sensor area that records the frame. Video modes "
            "may crop: use the cropped width."
        )
        self.focal.setToolTip(
            "Real focal length (not 35 mm equivalent). Longer is more sensitive and samples "
            "the pattern more finely at the same distances, but frames a smaller area: use the "
            "longest lens that still fits the flow."
        )
        panel.add(f)

        panel.add(W.section("Distances"))
        f2 = W.form()
        self.z_a = W.number(1.0, 0.05, 100, 3, "m")
        self.z_d = W.number(1.0, 0.01, 100, 3, "m")
        self.z_a.setToolTip(
            "From the camera lens to the flow (the hot air, jet, ...). Closer puts more pixels "
            "on the flow (finer detail) but frames a smaller area. Typical: 0.5–3 m."
        )
        self.z_d.setToolTip(
            "From the flow to the background pattern. Larger is more sensitive (displacement "
            "grows with it) but blurs the flow more, because the camera focuses on the "
            "background. Aim for about the same as camera-to-flow; go larger for weak flows and "
            "smaller for fine detail. Watch 'Displacement per mrad' and 'Blur at the flow'."
        )
        f2.addRow(W.label("Camera to flow", "Muted"), self.z_a)
        f2.addRow(W.label("Flow to background", "Muted"), self.z_d)
        panel.add(f2)

        panel.add(W.section("Background"))
        self.bg_group = QButtonGroup(self)
        row = QHBoxLayout()
        for i, (key, text) in enumerate(
            [("screen", "Screen"), ("print", "Print"), ("natural", "Natural")]
        ):
            rb = QRadioButton(text)
            rb.setProperty("key", key)
            self.bg_group.addButton(rb, i)
            row.addWidget(rb)
        row.addStretch(1)
        panel.add(row)
        self.bg_stack = QStackedWidget()
        self.bg_stack.addWidget(self._screen_controls())
        self.bg_stack.addWidget(self._print_controls())
        self.bg_stack.addWidget(
            W.label(
                "Natural backgrounds (foliage, brick, gravel, tarmac) work when they are finely "
                "textured, random, high-contrast and evenly lit. Focus on the background and keep "
                "it two to five times further away than the flow. The capture check measures "
                "whether "
                "the texture is fine and sharp enough.",
                "Muted",
                wrap=True,
            )
        )
        panel.add(self.bg_stack)

        panel.add(W.section("Pattern"))
        f3 = W.form()
        self.pattern = W.combo([(k.label, k.value) for k in PatternKind])
        self.pattern_desc = W.label("", "Faint", wrap=True)
        self.seed = W.integer(1, 0, 10**6)
        f3.addRow(W.label("Type", "Muted"), self.pattern)
        f3.addRow(W.label("Seed", "Muted"), self.seed)
        panel.add(f3)
        panel.add(self.pattern_desc)
        panel.finish()

        for wdg in (self.focal, self.sensor_w, self.fnum, self.z_a, self.z_d):
            wdg.valueChanged.connect(self._changed)
        for wdg in (self.img_w, self.img_h, self.seed):
            wdg.valueChanged.connect(self._changed)
        self.cam_preset.currentIndexChanged.connect(self._preset_selected)
        self.pattern.currentIndexChanged.connect(self._changed)
        self.bg_group.idClicked.connect(self._changed)
        return panel

    def _screen_controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        f = W.form()
        self.screen_combo = W.combo([])
        self.disp_w = W.integer(2560, 320, 16000, "px")
        self.disp_h = W.integer(1440, 240, 16000, "px")
        self.disp_diag = W.number(27, 5, 150, 1, "in")
        self.elem_auto = W.QCheckBox("Size automatically")
        self.elem = W.integer(3, 1, 64, "display px")
        f.addRow(W.label("Display", "Muted"), self.screen_combo)
        f.addRow(W.label("Resolution", "Muted"), W.hbox(self.disp_w, self.disp_h))
        f.addRow(W.label("Diagonal", "Muted"), self.disp_diag)
        f.addRow(W.label("Element", "Muted"), W.hbox(self.elem, self.elem_auto))
        v.addLayout(f)
        self.show_btn = W.button("Show pattern full screen", self._show_pattern, primary=True)
        v.addWidget(self.show_btn)
        v.addWidget(
            W.label(
                "Esc closes. Arrow keys change the element size, N draws a new "
                "pattern. Set the display to full brightness and turn off night "
                "light and auto-brightness.",
                "Faint",
                wrap=True,
            )
        )
        self._fill_screens()
        self.screen_combo.currentIndexChanged.connect(self._screen_selected)
        for wdg in (self.disp_w, self.disp_h, self.elem):
            wdg.valueChanged.connect(self._changed)
        self.disp_diag.valueChanged.connect(self._changed)
        self.elem_auto.toggled.connect(self._changed)
        return w

    def _print_controls(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(8)
        f = W.form()
        self.paper = W.combo([(k, k) for k in PAPERS])
        self.dpi = W.combo([("300 dpi", 300), ("600 dpi", 600), ("1200 dpi", 1200)], 600)
        self.print_elem = W.number(0.6, 0.05, 20, 2, "mm")
        self.print_auto = W.QCheckBox("Size automatically")
        f.addRow(W.label("Paper", "Muted"), self.paper)
        f.addRow(W.label("Printer", "Muted"), self.dpi)
        f.addRow(W.label("Element", "Muted"), W.hbox(self.print_elem, self.print_auto))
        v.addLayout(f)
        v.addLayout(
            W.hbox(
                W.button("Export PDF…", self._export_pdf, primary=True),
                W.button("Export PNG…", self._export_png),
                None,
            )
        )
        v.addWidget(
            W.label(
                "Print at 100 % (no fit-to-page) on matte paper and check the 50 mm "
                "scale bar. Mount it flat and light it evenly.",
                "Faint",
                wrap=True,
            )
        )
        for wdg in (self.paper, self.dpi):
            wdg.currentIndexChanged.connect(self._changed)
        self.print_elem.valueChanged.connect(self._changed)
        self.print_auto.toggled.connect(self._changed)
        return w

    # ------------------------------------------------------------------ report
    def _build_report(self) -> QWidget:
        outer = W.QScrollArea()
        outer.setWidgetResizable(True)
        body = QWidget()
        body.setObjectName("Panel")
        outer.setWidget(body)
        v = QVBoxLayout(body)
        v.setContentsMargins(28, 22, 28, 22)
        v.setSpacing(14)
        self.diagram = SetupDiagram()
        v.addWidget(self.diagram)
        v.addWidget(W.rule())

        grid = QGridLayout()
        grid.setHorizontalSpacing(36)
        grid.setVerticalSpacing(4)
        self.kv: dict[str, W.KeyValue] = {}
        cols = [
            (
                "Sensitivity",
                [
                    (
                        "ppmrad",
                        "Displacement per mrad",
                        "Image shift for 1 mrad of ray deflection. Higher is more sensitive; 1–5 "
                        "px is typical.",
                    ),
                    (
                        "candle",
                        "Candle plume (~100 K)",
                        "Expected peak displacement. Aim for 0.3–3 px: below ~0.1 px the plume "
                        "drowns in noise.",
                    ),
                    (
                        "breath",
                        "Breath (~10 K)",
                        "Expected peak displacement. Weak flows need ≥ 0.05 px to stand out.",
                    ),
                    (
                        "mindt",
                        "Smallest visible rise",
                        "Smallest plume temperature rise 3x above a 0.02 px noise floor. Lower is"
                        " better.",
                    ),
                ],
            ),
            (
                "Resolution",
                [
                    (
                        "hpx",
                        "Pixel size at the flow",
                        "Size of one camera pixel in the flow plane. Smaller resolves finer "
                        "detail.",
                    ),
                    (
                        "fov",
                        "Field of view at the flow",
                        "Area covered at the flow. The flow, plus some calm air around it, should"
                        " fit.",
                    ),
                    (
                        "blur",
                        "Blur at the flow",
                        "Ray-cone footprint in the flow plane; finer "
                        "detail than this is smeared, whatever the algorithm. Lower is sharper; "
                        "keep it well below the size of what you want to see.",
                    ),
                    (
                        "airy",
                        "Diffraction spot",
                        "Airy disk of a background point on the sensor. Keep below ~4 px (stop "
                        "down less if larger).",
                    ),
                ],
            ),
            (
                "Background",
                [
                    (
                        "bgfov",
                        "Field of view at background",
                        "Area of background the camera sees: the pattern must cover all of it "
                        "behind the flow.",
                    ),
                    (
                        "elem",
                        "Pattern element on sensor",
                        "How large one pattern dot appears to the camera. Aim for 3–5 px; below "
                        "2.5 px it aliases.",
                    ),
                    (
                        "disppx",
                        "Display pixel on sensor",
                        "How large one screen pixel appears to the camera. Values between ~0.5 "
                        "and 3 px risk moiré.",
                    ),
                    (
                        "aperture",
                        "Recommended aperture",
                        "Small enough for sharp flow detail, large "
                        "enough to avoid diffraction blur of the pattern.",
                    ),
                ],
            ),
        ]
        for c, (title, rows) in enumerate(cols):
            grid.addWidget(W.section(title), 0, c)
            for r, (key, text, tip) in enumerate(rows, start=1):
                kv = W.KeyValue(text, "", tip)
                self.kv[key] = kv
                grid.addWidget(kv, r, c)
        v.addLayout(grid)
        v.addWidget(W.rule())
        v.addWidget(W.section("Advice"))
        self.advice_box = QVBoxLayout()
        self.advice_box.setSpacing(2)
        v.addLayout(self.advice_box)
        v.addWidget(W.rule())
        v.addWidget(W.section("Pattern as the camera will see it"))
        prev = QHBoxLayout()
        prev.setSpacing(18)
        self.prev_display = QLabel()
        self.prev_camera = QLabel()
        for lab, cap in (
            (self.prev_display, "On the background (1:1 display pixels)"),
            (self.prev_camera, "On the sensor (1:1 camera pixels)"),
        ):
            col = QVBoxLayout()
            lab.setFixedSize(256, 256)
            lab.setStyleSheet("background: #000;")
            col.addWidget(lab)
            col.addWidget(W.label(cap, "Faint"))
            prev.addLayout(col)
        prev.addStretch(1)
        v.addLayout(prev)
        v.addStretch(1)
        return outer

    # ------------------------------------------------------------------ data flow
    def _fill_screens(self) -> None:
        self.screen_combo.blockSignals(True)
        self.screen_combo.clear()
        for i, sc in enumerate(screens()):
            w, h = physical_size(sc)
            diag = screen_diagonal_inches(sc)
            txt = f"Screen {i + 1}: {sc.name()} · {w}×{h}"
            self.screen_combo.addItem(txt, ("screen", i, w, h, diag))
        for name, d in DISPLAY_PRESETS.items():
            self.screen_combo.addItem(
                name, ("preset", name, d.width_px, d.height_px, d.diagonal_m / 0.0254)
            )
        self.screen_combo.blockSignals(False)

    def _screen_selected(self) -> None:
        data = self.screen_combo.currentData()
        if not data:
            return
        _, _, w, h, diag = data
        self._loading = True
        self.disp_w.setValue(w)
        self.disp_h.setValue(h)
        if diag:
            self.disp_diag.setValue(round(diag, 1))
        self._loading = False
        self._changed()

    def _preset_selected(self) -> None:
        name = self.cam_preset.currentData()
        if not name:
            return
        self.state.project.setup.apply_preset(name)
        self.load_from_project()
        self._changed()

    def load_from_project(self) -> None:
        if self._emitting:
            return
        self._loading = True
        s, b = self.state.project.setup, self.state.project.background
        idx = self.cam_preset.findData(s.camera_name)
        self.cam_preset.blockSignals(True)
        self.cam_preset.setCurrentIndex(max(0, idx))
        self.cam_preset.blockSignals(False)
        self.focal.setValue(s.focal_length_mm)
        self.sensor_w.setValue(s.sensor_width_mm)
        self.img_w.setValue(s.image_width_px)
        self.img_h.setValue(s.image_height_px)
        self.fnum.setValue(s.f_number)
        self.z_a.setValue(s.z_a_m)
        self.z_d.setValue(s.z_d_m)
        self.bg_group.button({"screen": 0, "print": 1, "natural": 2}[b.kind]).setChecked(True)
        self.bg_stack.setCurrentIndex({"screen": 0, "print": 1, "natural": 2}[b.kind])
        self.disp_w.setValue(b.display_width_px)
        self.disp_h.setValue(b.display_height_px)
        self.disp_diag.setValue(b.display_diagonal_in)
        self.elem.setValue(b.element_px)
        self.elem_auto.setChecked(b.auto_element)
        self.print_auto.setChecked(b.auto_element)
        self.print_elem.setValue(b.print_element_mm)
        self.paper.setCurrentIndex(max(0, self.paper.findData(b.paper)))
        self.dpi.setCurrentIndex(max(0, self.dpi.findData(b.print_dpi)))
        self.pattern.setCurrentIndex(max(0, self.pattern.findData(b.pattern)))
        self.seed.setValue(b.seed)
        self._loading = False
        self._refresh()

    def _changed(self, *_) -> None:
        if self._loading:
            return
        s, b = self.state.project.setup, self.state.project.background
        s.focal_length_mm = self.focal.value()
        s.sensor_width_mm = self.sensor_w.value()
        s.image_width_px = self.img_w.value()
        s.image_height_px = self.img_h.value()
        s.f_number = self.fnum.value()
        s.z_a_m = self.z_a.value()
        s.z_d_m = self.z_d.value()
        btn = self.bg_group.checkedButton()
        b.kind = btn.property("key") if btn else "screen"
        self.bg_stack.setCurrentIndex({"screen": 0, "print": 1, "natural": 2}[b.kind])
        b.display_width_px = self.disp_w.value()
        b.display_height_px = self.disp_h.value()
        b.display_diagonal_in = self.disp_diag.value()
        b.display_name = self.screen_combo.currentText()
        auto = self.elem_auto if b.kind == "screen" else self.print_auto
        b.auto_element = auto.isChecked()
        b.element_px = self.elem.value()
        b.print_element_mm = self.print_elem.value()
        b.paper = self.paper.currentData()
        b.print_dpi = self.dpi.currentData()
        b.pattern = self.pattern.currentData()
        b.seed = self.seed.value()
        self.state.mark_dirty()
        self._refresh()
        self._emitting = True
        self.state.setup_changed.emit()
        self._emitting = False

    def _refresh(self) -> None:
        st = self.state.setup
        b = self.state.project.background
        self.diagram.set_setup(st)
        self.pattern_desc.setText(PatternKind(b.pattern).description)
        ppr = st.pixels_per_radian
        self.kv["ppmrad"].set(f"{ppr * 1e-3:.2f} px")
        self.kv["candle"].set(f"{st.displacement_for_temperature_rise(100):.2f} px")
        self.kv["breath"].set(f"{st.displacement_for_temperature_rise(10):.3f} px")
        self.kv["mindt"].set(f"{_min_rise(st, 0.02):.1f} K")
        self.kv["hpx"].set(f"{st.object_pixel_size_m * 1e3:.3f} mm")
        fw, fh = st.object_field_of_view_m
        self.kv["fov"].set(f"{fw * 100:.1f} × {fh * 100:.1f} cm")
        blur_status = "warn" if st.object_blur_m > 0.05 * fw else None
        self.kv["blur"].set(
            f"{st.object_blur_m * 1e3:.1f} mm · {st.object_blur_px:.0f} px", blur_status
        )
        self.kv["airy"].set(
            f"{st.diffraction_spot_px:.1f} px", "warn" if st.diffraction_spot_px > 4 else None
        )
        bw, bh = st.background_field_of_view_m
        self.kv["bgfov"].set(f"{bw * 100:.1f} × {bh * 100:.1f} cm")
        lo, hi = recommend_f_number(st, resolution_m=max(0.002, 0.02 * fw))
        self.kv["aperture"].set(
            f"f/{lo:.1f} – f/{hi:.1f}" if lo <= hi else "conflict", None if lo <= hi else "warn"
        )
        advice: list[tuple[str, str, str]] = []
        if b.kind == "screen":
            a = analyze_display(st, b.display())
            if b.auto_element and self.elem.value() != a.block_size_display_px:
                self._loading = True
                self.elem.setValue(a.block_size_display_px)
                b.element_px = a.block_size_display_px
                self._loading = False
            elem_cam = b.element_px * a.display_pixel_in_camera_px
            self.kv["elem"].set(f"{elem_cam:.1f} px", None if 2.5 <= elem_cam <= 5 else "warn")
            risk_status = {MoireRisk.NONE: "ok", MoireRisk.LOW: "warn", MoireRisk.HIGH: "fail"}
            self.kv["disppx"].set(
                f"{a.display_pixel_in_camera_px:.2f} px · moiré {a.moire_risk.value}",
                risk_status[a.moire_risk],
            )
            advice += [("warn", "Display", n) for n in a.notes]
            if a.moire_risk is MoireRisk.HIGH:
                advice.append(("fail", "Moiré risk", "See the display note above."))
        else:
            size_bg = b.print_element_mm * 1e-3
            if b.kind == "print" and b.auto_element:
                size_bg = st.background_feature_size_m(3.5)
                self._loading = True
                self.print_elem.setValue(round(size_bg * 1e3, 2))
                b.print_element_mm = round(size_bg * 1e3, 2)
                self._loading = False
            elem_cam = size_bg / st.background_pixel_size_m
            self.kv["elem"].set(
                f"{elem_cam:.1f} px" if b.kind == "print" else "—",
                None if 2.5 <= elem_cam <= 5 or b.kind != "print" else "warn",
            )
            self.kv["disppx"].set("—")
            if b.kind == "print":
                dots = b.print_element_mm / 25.4 * b.print_dpi
                if dots < 3:
                    advice.append(
                        (
                            "warn",
                            "Printer resolution",
                            f"Each element is only "
                            f"{dots:.1f} printer dots: use a higher dpi or larger "
                            "elements.",
                        )
                    )
        if lo > hi:
            advice.append(
                (
                    "warn",
                    "Aperture",
                    "No aperture gives both sharp flow detail and a "
                    "sharp pattern: move the flow closer to the background (smaller Z_D) "
                    "or use a shorter focal length.",
                )
            )
        if st.z_d < 0.3 * st.z_a:
            advice.append(
                (
                    "warn",
                    "Sensitivity",
                    "The flow is close to the background: "
                    "sensitivity scales with the flow-to-background distance.",
                )
            )
        if st.object_blur_m > 0.05 * fw:
            advice.append(
                (
                    "warn",
                    "Depth of field",
                    "The flow is blurred over more than 5 % of "
                    "the field of view; stop down or move the flow towards the "
                    "background.",
                )
            )
        advice.append(
            (
                "ok",
                "Before you shoot",
                "Manual focus on the background, manual "
                "exposure at base ISO, all stabilisation off (IBIS/OIS/EIS), highest "
                "bitrate or RAW, tripod, a few seconds without flow for the reference.",
            )
        )
        W.clear_layout(self.advice_box)
        for status, title, text in advice:
            self.advice_box.addWidget(W.StatusRow(title, status, "", text))
        self._update_preview()

    def _update_preview(self) -> None:
        st = self.state.setup
        b = self.state.project.background
        if b.kind == "screen":
            a = analyze_display(st, b.display())
            spec = b.spec(b.element_px)
            disp = generate(spec, 256, 256)
            factor = a.display_pixel_in_camera_px
        elif b.kind == "print":
            dot_m = 25.4e-3 / b.print_dpi
            e = max(1, round(b.print_element_mm * 1e-3 / dot_m))
            spec = b.spec(e)
            disp = generate(spec, 256, 256)
            factor = dot_m / st.background_pixel_size_m
        else:
            self.prev_display.clear()
            self.prev_camera.clear()
            return
        self.prev_display.setPixmap(_pixmap(disp))
        # Simulated sensor view: area-resample by the imaged pixel size, then a light PSF.
        n = max(8, int(256 / max(factor, 1e-3)))
        big = generate(spec, n, n)
        cam = cv2.resize(
            big, (max(2, int(n * factor)), max(2, int(n * factor))), interpolation=cv2.INTER_AREA
        )
        sigma = max(0.4, 0.42 * st.f_number * st.wavelength_m / st.p)
        cam = cv2.GaussianBlur(cam, (0, 0), sigma)[:256, :256]
        self.prev_camera.setPixmap(_pixmap(cam))

    # ------------------------------------------------------------------ actions
    def _show_pattern(self) -> None:
        data = self.screen_combo.currentData()
        scr = screens()
        index = data[1] if data and data[0] == "screen" else (1 if len(scr) > 1 else 0)
        screen = scr[min(index, len(scr) - 1)]
        if self._pattern_window is not None:
            self._pattern_window.close()
        b = self.state.project.background
        self._pattern_window = PatternWindow(screen, b.spec(b.element_px))
        self._pattern_window.closed.connect(self._pattern_closed)
        self._pattern_window.show_on_screen()

    def _pattern_closed(self) -> None:
        if self._pattern_window is not None:
            spec = self._pattern_window.spec
            b = self.state.project.background
            b.element_px, b.seed, b.invert = spec.element_px, spec.seed, spec.invert
            self._pattern_window = None
            self.load_from_project()

    def _print_image(self) -> tuple[np.ndarray, int]:
        b = self.state.project.background
        w_mm, h_mm = PAPERS[b.paper]
        margin = 12
        dpi = b.print_dpi
        w = int((w_mm - 2 * margin) / 25.4 * dpi)
        h = int((h_mm - 2 * margin - 12) / 25.4 * dpi)
        e = max(1, round(b.print_element_mm / 25.4 * dpi))
        return generate(b.spec(e), w, h), dpi

    def _export_pdf(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export printable pattern", "bos-pattern.pdf", "PDF (*.pdf)"
        )
        if not path:
            return
        b = self.state.project.background
        img, dpi = self._print_image()
        title = f"{PatternKind(b.pattern).label}, {b.print_element_mm:.2f} mm, seed {b.seed}"
        self.state.run_task(
            lambda *_: write_print_pattern_pdf(path, img, dpi, title),
            lambda _: self.state.status.emit(f"Pattern saved to {path}"),
            message="Writing PDF",
        )

    def _export_png(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export pattern image", "bos-pattern.png", "PNG (*.png)"
        )
        if not path:
            return
        img, dpi = self._print_image()
        rgb = np.repeat((img * 255).astype(np.uint8)[..., None], 3, axis=2)
        write_png(path, rgb)
        self.state.status.emit(f"Pattern saved to {path} ({dpi} dpi, print at 100 %)")


def _pixmap(img: np.ndarray) -> QPixmap:
    a = np.ascontiguousarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    h, w = a.shape
    qi = QImage(a.data, w, h, w, QImage.Format.Format_Grayscale8).copy()
    return QPixmap.fromImage(qi).scaled(
        256, 256, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation
    )


def _min_rise(st, noise_px: float) -> float:
    lo, hi = 0.0, 3000.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if st.displacement_for_temperature_rise(mid) < 3 * noise_px:
            lo = mid
        else:
            hi = mid
    return hi if not math.isinf(hi) else float("nan")
