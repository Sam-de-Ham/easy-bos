"""Frame scrubber: slider, frame number, time, and play/pause."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import QHBoxLayout, QSlider, QWidget

from easy_bos.ui import widgets as W


class FrameScrubber(QWidget):
    changed = Signal(int)

    def __init__(self, playable: bool = True):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(12, 6, 12, 6)
        lay.setSpacing(10)
        self.play_btn = W.button("Play", self._toggle_play, flat=True)
        self.play_btn.setFixedWidth(52)
        self.play_btn.setVisible(playable)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.spin = W.integer(0, 0, 0)
        self.spin.setFixedWidth(70)
        self.time = W.label("", "Value")
        self.time.setFixedWidth(120)
        lay.addWidget(self.play_btn)
        lay.addWidget(self.slider, 1)
        lay.addWidget(self.spin)
        lay.addWidget(self.time)
        self.fps: float | None = None
        self.allowed: list[int] | None = None
        # Decode only when the handle is released: seeking a long-GOP video on every pixel of
        # a drag queues dozens of decodes. While dragging, only the labels follow.
        self.slider.setTracking(False)
        self.slider.sliderMoved.connect(self._preview_position)
        self.slider.valueChanged.connect(self._slider)
        self.spin.valueChanged.connect(self._spin)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)

    def configure(self, count: int, fps: float | None, allowed: list[int] | None = None) -> None:
        """``allowed`` restricts the scrubber to these frame indices (e.g. processed frames)."""
        self.fps = fps
        self.allowed = allowed
        n = len(allowed) if allowed is not None else count
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.slider.setRange(0, max(0, n - 1))
        self.spin.setRange(0, max(0, (allowed[-1] if allowed else count - 1)))
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        self.setEnabled(n > 0)

    def set_frame(self, index: int) -> None:
        pos = self.allowed.index(index) if self.allowed and index in self.allowed else index
        for w in (self.slider, self.spin):
            w.blockSignals(True)
        self.slider.setValue(pos)
        self.spin.setValue(index)
        for w in (self.slider, self.spin):
            w.blockSignals(False)
        self._update_time(index)

    def frame(self) -> int:
        return self.spin.value()

    def _slider(self, pos: int) -> None:
        index = self.allowed[pos] if self.allowed else pos
        self.spin.blockSignals(True)
        self.spin.setValue(index)
        self.spin.blockSignals(False)
        self._update_time(index)
        self.changed.emit(index)

    def _preview_position(self, pos: int) -> None:
        index = self.allowed[pos] if self.allowed else pos
        self.spin.blockSignals(True)
        self.spin.setValue(index)
        self.spin.blockSignals(False)
        self._update_time(index)

    def _spin(self, index: int) -> None:
        if self.allowed:
            index = min(self.allowed, key=lambda a: abs(a - index))
            pos = self.allowed.index(index)
        else:
            pos = index
        self.slider.blockSignals(True)
        self.slider.setValue(pos)
        self.slider.blockSignals(False)
        self._update_time(index)
        self.changed.emit(index)

    def _update_time(self, index: int) -> None:
        self.time.setText(f"{index / self.fps:7.3f} s" if self.fps else f"frame {index}")

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_btn.setText("Play")
        else:
            interval = int(1000 / min(self.fps or 15, 30))
            self._timer.start(max(15, interval))
            self.play_btn.setText("Pause")

    def stop(self) -> None:
        if self._timer.isActive():
            self._toggle_play()

    def _tick(self) -> None:
        v = self.slider.value() + 1
        if v > self.slider.maximum():
            v = 0
        self.slider.setValue(v)
