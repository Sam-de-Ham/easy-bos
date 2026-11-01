"""Step 2 — Capture: load footage, choose flow-free reference frames, and check quality."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QVBoxLayout, QWidget

from easy_bos.core import quality
from easy_bos.core.media import IMAGE_EXTS, RAW_EXTS, VIDEO_EXTS
from easy_bos.ui import widgets as W
from easy_bos.ui.canvas import ImageCanvas
from easy_bos.ui.frames import FrameScrubber
from easy_bos.ui.state import AppState

GUIDE = [
    (
        "Stabilisation off",
        "IBIS, OIS and electronic stabilisation all off. They warp frames by "
        "far more than the effect you are measuring.",
    ),
    (
        "Manual everything",
        "Manual focus on the background, manual exposure, base ISO, fixed white balance.",
    ),
    (
        "Quality",
        "RAW stills, or video at the highest bitrate and bit depth your camera offers "
        "(10-bit, All-I if available).",
    ),
    ("Reference", "Record a few seconds without the flow at the start of each clip."),
    (
        "Flicker",
        "Shutter time a multiple of the mains period (1/50 or 1/100 s at 50 Hz; 1/60 or "
        "1/120 s at 60 Hz); display at full brightness.",
    ),
]


class CapturePage(QWidget):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setAcceptDrops(True)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self._inspector())
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)
        rv.setSpacing(0)
        self.canvas = ImageCanvas()
        self.empty = W.label(
            "Drop a video, photos or RAW files here, or use Open capture.", "Muted"
        )
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rv.addWidget(self.empty, 1)
        rv.addWidget(self.canvas, 1)
        self.scrubber = FrameScrubber()
        rv.addWidget(self.scrubber)
        self.readout = W.label("", "Faint")
        self.readout.setContentsMargins(14, 0, 14, 6)
        rv.addWidget(self.readout)
        lay.addWidget(right, 1)
        self.scrubber.changed.connect(self._show_frame)
        self.canvas.hovered.connect(self._hover)
        state.source_changed.connect(self._source_changed)
        state.project_changed.connect(self._source_changed)
        self._source_changed()

    def _inspector(self) -> QWidget:
        p = W.ScrollPanel()
        p.add(W.label("Capture", "H1"))
        p.add(
            W.label(
                "Load what you filmed. A few flow-free frames are enough to check the "
                "setup; the same frames become the reference.",
                "Muted",
                wrap=True,
            )
        )
        p.add(W.hbox(W.button("Open capture…", self.open_capture, primary=True), None))
        p.add(W.section("Source"))
        self.info = {
            k: W.KeyValue(t)
            for k, t in [
                ("name", "File"),
                ("size", "Frame size"),
                ("frames", "Frames"),
                ("fps", "Frame rate"),
                ("depth", "Bit depth"),
                ("codec", "Codec"),
                ("bitrate", "Bitrate"),
            ]
        }
        for kv in self.info.values():
            p.add(kv)
        f = W.form()
        self.channel = W.combo(
            [
                ("Green (recommended)", "green"),
                ("Luminance", "luma"),
                ("Red", "red"),
                ("Blue", "blue"),
            ]
        )
        self.channel.setToolTip(
            "Which colour channel to analyse. Green has the most samples on "
            "a Bayer sensor and the sharpest optics."
        )
        f.addRow(W.label("Channel", "Muted"), self.channel)
        p.add(f)
        self.channel.currentIndexChanged.connect(self._channel_changed)

        p.add(W.section("Reference (no flow)"))
        f2 = W.form()
        self.ref_start = W.integer(0, 0, 10**7)
        self.ref_end = W.integer(8, 1, 10**7)
        f2.addRow(W.label("From frame", "Muted"), self.ref_start)
        f2.addRow(W.label("To frame (excl.)", "Muted"), self.ref_end)
        p.add(f2)
        use_current = W.button("Use current frame as start", self._ref_from_current, flat=True)
        p.add(W.hbox(use_current, None))
        self.ref_start.valueChanged.connect(self._ref_changed)
        self.ref_end.valueChanged.connect(self._ref_changed)

        p.add(W.section("Capture check"))
        self.check_btn = W.button("Check capture", self._run_checks, primary=True)
        p.add(W.hbox(self.check_btn, None))
        self.summary = W.label("Checks use the reference frames.", "Faint", wrap=True)
        p.add(self.summary)
        self.rows: dict[str, W.StatusRow] = {}
        self.rows_box = QVBoxLayout()
        self.rows_box.setSpacing(0)
        p.add(self.rows_box)

        p.add(W.section("Shooting checklist"))
        for title, text in GUIDE:
            p.add(W.StatusRow(title, "none", "", text))
        p.finish()
        return p

    # ------------------------------------------------------------------ source
    def open_capture(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(VIDEO_EXTS | IMAGE_EXTS | RAW_EXTS))
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open capture", "", f"Video, images or RAW ({exts});;All files (*)"
        )
        if paths:
            self._open(paths)

    def _open(self, paths: list[str]) -> None:
        try:
            self.state.open_capture(paths)
        except Exception as exc:
            self.state.error.emit(f"Could not open the capture: {exc}")

    def dragEnterEvent(self, e):  # noqa: N802 - Qt API
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):  # noqa: N802 - Qt API
        paths = [u.toLocalFile() for u in e.mimeData().urls() if u.isLocalFile()]
        if len(paths) == 1 and Path(paths[0]).is_dir():
            paths = [
                str(f)
                for f in sorted(Path(paths[0]).iterdir())
                if f.suffix.lower() in IMAGE_EXTS | RAW_EXTS
            ]
        if paths:
            self._open(paths)

    def _source_changed(self) -> None:
        src = self.state.source
        has = src is not None
        self.canvas.setVisible(has)
        self.scrubber.setVisible(has)
        self.empty.setVisible(not has)
        self.check_btn.setEnabled(has)
        self._clear_checks()
        idx = self.channel.findData(self.state.project.capture.channel)
        self.channel.blockSignals(True)
        self.channel.setCurrentIndex(max(0, idx))
        self.channel.blockSignals(False)
        if not has:
            for kv in self.info.values():
                kv.set("—")
            return
        i = src.info
        self.info["name"].set(_ellipsis(i.name, 30))
        self.info["size"].set(f"{i.width} × {i.height}")
        self.info["frames"].set(str(i.frame_count))
        self.info["fps"].set(f"{i.fps:.2f} fps" if i.fps else "—")
        self.info["depth"].set(f"{i.bit_depth}-bit")
        self.info["codec"].set(str(i.details.get("codec", i.kind)))
        br = i.details.get("bitrate_mbps")
        self.info["bitrate"].set(
            f"{br:.0f} Mbit/s" if br else "—",
            "warn" if br and br < 50 and i.width >= 3000 else None,
        )
        n = len(src)
        a, b = self.state.project.process.reference_range
        for w in (self.ref_start, self.ref_end):
            w.blockSignals(True)
        self.ref_start.setRange(0, max(0, n - 1))
        self.ref_end.setRange(1, n)
        self.ref_start.setValue(a)
        self.ref_end.setValue(b)
        for w in (self.ref_start, self.ref_end):
            w.blockSignals(False)
        self.scrubber.configure(n, i.fps)
        self.scrubber.set_frame(min(self.state.current_frame, n - 1))
        self._show_frame(self.scrubber.frame(), fit=True)

    def _channel_changed(self) -> None:
        ch = self.channel.currentData()
        if ch == self.state.project.capture.channel:
            return
        self.state.project.capture.channel = ch
        if self.state.project.capture.paths:
            self._open(self.state.project.capture.paths)

    def _show_frame(self, index: int, fit: bool = False) -> None:
        img = self.state.frame(index)
        if img is None:
            return
        self.state.set_frame(index)
        lo, hi = np.percentile(img[::4, ::4], [0.5, 99.5])
        self.canvas.set_image(img, levels=(float(lo), float(hi)), keep_view=not fit)
        a, b = self.state.project.process.reference_range
        self.readout.setText("Reference frame" if a <= index < b else "")

    def _hover(self, x: float, y: float) -> None:
        img = self.state.frame(self.scrubber.frame())
        if img is None or not np.isfinite(x):
            return
        v = img[int(y), int(x)]
        self.readout.setText(f"x {int(x)}  y {int(y)}  ·  value {v * 100:.1f}%")

    def _ref_changed(self) -> None:
        a, b = self.ref_start.value(), self.ref_end.value()
        if b <= a:
            b = a + 1
            self.ref_end.blockSignals(True)
            self.ref_end.setValue(b)
            self.ref_end.blockSignals(False)
        self.state.project.process.reference_range = (a, b)
        self.state.mark_dirty()

    def _ref_from_current(self) -> None:
        cur = self.scrubber.frame()
        span = self.ref_end.value() - self.ref_start.value()
        self.ref_start.setValue(cur)
        self.ref_end.setValue(cur + max(span, 2))

    # ------------------------------------------------------------------ checks
    def _clear_checks(self) -> None:
        W.clear_layout(self.rows_box)
        self.rows.clear()
        self.summary.setText("Checks use the reference frames.")

    def _run_checks(self) -> None:
        src = self.state.source
        if src is None:
            return
        a, b = self.state.project.process.reference_range
        idx = list(range(a, min(b, len(src))))[:6]
        if len(idx) < 2 and len(src) > a + 1:
            idx = [a, a + 1]
        setup = self.state.setup
        self.check_btn.setEnabled(False)
        self.summary.setText("Checking…")

        def work(progress, cancelled):
            frames = [src.frame(i) for i in idx]
            return quality.run_checks(frames, setup)

        def done(checks):
            self.check_btn.setEnabled(True)
            self._clear_checks()
            for c in checks:
                row = W.StatusRow(c.title, c.status.value, c.value, c.detail)
                self.rows_box.addWidget(row)
            ov = quality.overall(checks)
            self.summary.setText(
                {
                    quality.Status.OK: "Everything looks good. Go ahead and process.",
                    quality.Status.WARN: "Usable, with caveats. See the notes below.",
                    quality.Status.FAIL: "Fix the failed items for reliable results.",
                }[ov]
            )

        task = self.state.run_task(work, done, message="Checking capture")
        task.failed.connect(lambda *_: self.check_btn.setEnabled(True))

    def activated(self) -> None:
        if self.state.source is not None:
            self.scrubber.set_frame(self.state.current_frame)


def _ellipsis(text: str, n: int) -> str:
    return text if len(text) <= n else text[: n // 2 - 1] + "…" + text[-(n // 2 - 1) :]
