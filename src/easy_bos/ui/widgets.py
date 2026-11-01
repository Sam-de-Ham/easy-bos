"""Reusable widgets: layout helpers, labelled fields, parameter forms, status rows."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from easy_bos.core.flow.base import MethodParams
from easy_bos.ui.theme import C, status_color


def label(text: str, kind: str | None = None, wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if kind:
        lab.setObjectName(kind)
    lab.setWordWrap(wrap)
    if wrap:
        lab.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    return lab


def section(text: str) -> QLabel:
    return label(text.upper(), "Section")


def rule() -> QFrame:
    f = QFrame()
    f.setObjectName("Rule")
    return f


def hbox(*widgets, spacing: int = 8, stretch_last: bool = False) -> QHBoxLayout:
    lay = QHBoxLayout()
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(spacing)
    for w in widgets:
        if w is None:
            lay.addStretch(1)
        elif isinstance(w, QWidget):
            lay.addWidget(w)
        else:
            lay.addLayout(w)
    if stretch_last:
        lay.addStretch(1)
    return lay


def button(
    text: str,
    on_click: Callable | None = None,
    primary: bool = False,
    flat: bool = False,
    tooltip: str | None = None,
) -> QPushButton:
    b = QPushButton(text)
    if primary:
        b.setObjectName("Primary")
    elif flat:
        b.setObjectName("Flat")
    if on_click:
        b.clicked.connect(on_click)
    if tooltip:
        b.setToolTip(tooltip)
    b.setCursor(Qt.CursorShape.PointingHandCursor)
    return b


def form() -> QFormLayout:
    f = QFormLayout()
    f.setContentsMargins(0, 0, 0, 0)
    f.setHorizontalSpacing(12)
    f.setVerticalSpacing(7)
    f.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    f.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return f


def number(
    value: float,
    minimum: float,
    maximum: float,
    decimals: int = 2,
    suffix: str = "",
    step: float | None = None,
) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(minimum, maximum)
    s.setDecimals(decimals)
    s.setValue(value)
    s.setKeyboardTracking(False)
    s.setAccelerated(True)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setSingleStep(step if step is not None else 10 ** (-decimals) * 10 if decimals else 1)
    s.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
    return s


def integer(value: int, minimum: int, maximum: int, suffix: str = "") -> QSpinBox:
    s = QSpinBox()
    s.setRange(minimum, maximum)
    s.setValue(value)
    s.setKeyboardTracking(False)
    if suffix:
        s.setSuffix(f" {suffix}")
    s.setButtonSymbols(QSpinBox.ButtonSymbols.NoButtons)
    return s


def combo(items: list[tuple[str, Any]], current: Any = None) -> QComboBox:
    c = QComboBox()
    for text, data in items:
        c.addItem(text, data)
    if current is not None:
        idx = c.findData(current)
        if idx >= 0:
            c.setCurrentIndex(idx)
    return c


class ScrollPanel(QScrollArea):
    """A vertically scrolling inspector panel with comfortable margins."""

    def __init__(self, width: int = 340):
        super().__init__()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setFixedWidth(width)
        self.setObjectName("Inspector")
        inner = QWidget()
        inner.setObjectName("Inspector")
        self.body = QVBoxLayout(inner)
        self.body.setContentsMargins(18, 16, 18, 18)
        self.body.setSpacing(10)
        self.setWidget(inner)

    def add(self, item) -> None:
        if isinstance(item, QWidget):
            self.body.addWidget(item)
        else:
            self.body.addLayout(item)

    def finish(self) -> None:
        self.body.addStretch(1)


class StatusDot(QWidget):
    def __init__(self, status: str = "none", size: int = 8):
        super().__init__()
        self._status = status
        self._size = size
        self.setFixedSize(size + 4, size + 4)

    def set_status(self, status: str) -> None:
        self._status = status
        self.update()

    def paintEvent(self, _):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = (
            QColor(status_color(self._status))
            if self._status != "none"
            else QColor(C.border_strong)
        )
        p.setBrush(color)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(2, 2, self._size, self._size)


class StatusRow(QWidget):
    """Status dot, title, value and an explanation that wraps underneath."""

    def __init__(self, title: str, status: str = "none", value: str = "", detail: str = ""):
        super().__init__()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 4)
        lay.setSpacing(2)
        self.dot = StatusDot(status)
        self.title = label(title, "H2")
        self.value = label(value, "Value")
        self.value.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        top = hbox(self.dot, self.title, None, self.value)
        lay.addLayout(top)
        self.detail = label(detail, "Muted", wrap=True)
        self.detail.setContentsMargins(20, 0, 0, 0)
        lay.addWidget(self.detail)
        self.detail.setVisible(bool(detail))

    def update_status(self, status: str, value: str, detail: str) -> None:
        self.dot.set_status(status)
        self.value.setText(value)
        self.detail.setText(detail)
        self.detail.setVisible(bool(detail))


class KeyValue(QWidget):
    """A compact 'label ........ value' line for readouts."""

    def __init__(self, key: str, value: str = "", hint: str = ""):
        super().__init__()
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 1, 0, 1)
        self.key = label(key, "Muted")
        self.val = label(value, "Value")
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.key)
        lay.addStretch(1)
        lay.addWidget(self.val)
        if hint:
            self.setToolTip(hint)

    def set(self, value: str, status: str | None = None) -> None:
        self.val.setText(value)
        self.val.setStyleSheet(f"color: {status_color(status)};" if status else "")


class ParamForm(QWidget):
    """A form generated from a ``MethodParams`` dataclass, with an advanced section."""

    changed = Signal()

    def __init__(self, params_cls: type[MethodParams], values: dict | None = None):
        super().__init__()
        self.params_cls = params_cls
        self.editors: dict[str, QWidget] = {}
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(6)
        basic, adv = form(), form()
        values = values or {}
        has_adv = False
        for name, meta, default in params_cls.describe():
            v = values.get(name, default)
            ed = self._editor(meta, default, v)
            ed.setToolTip(meta.help)
            lab = label(meta.label, "Muted")
            lab.setToolTip(meta.help)
            (adv if meta.advanced else basic).addRow(lab, ed)
            has_adv |= meta.advanced
            self.editors[name] = ed
        outer.addLayout(basic)
        if has_adv:
            self.adv_toggle = button("Advanced settings", flat=True)
            self.adv_toggle.setCheckable(True)
            self.adv_box = QWidget()
            self.adv_box.setLayout(adv)
            self.adv_box.setVisible(False)
            self.adv_toggle.toggled.connect(self._toggle_adv)
            outer.addWidget(self.adv_toggle, alignment=Qt.AlignmentFlag.AlignLeft)
            outer.addWidget(self.adv_box)

    def _toggle_adv(self, on: bool) -> None:
        self.adv_box.setVisible(on)
        self.adv_toggle.setText("Hide advanced settings" if on else "Advanced settings")

    def _editor(self, meta, default, value) -> QWidget:
        if meta.choices:
            items = [(c.capitalize() if c.isalpha() else c, c) for c in meta.choices]
            ed = combo(items, str(value))
            ed.currentIndexChanged.connect(self.changed.emit)
            return ed
        if isinstance(default, bool):
            ed = QCheckBox()
            ed.setChecked(bool(value))
            ed.toggled.connect(self.changed.emit)
            return ed
        if isinstance(default, int):
            ed = integer(int(value), int(meta.minimum or 0), int(meta.maximum or 10**6))
            ed.valueChanged.connect(self.changed.emit)
            return ed
        decimals = 3 if (meta.maximum or 1) <= 10 else 2
        ed = number(float(value), meta.minimum or 0.0, meta.maximum or 1e6, decimals)
        if meta.log:
            ed.setStepType(QDoubleSpinBox.StepType.AdaptiveDecimalStepType)
        ed.valueChanged.connect(self.changed.emit)
        return ed

    def values(self) -> dict:
        out = {}
        defaults = {n: d for n, _, d in self.params_cls.describe()}
        for name, ed in self.editors.items():
            d = defaults[name]
            if isinstance(ed, QComboBox):
                v = ed.currentData()
                out[name] = type(d)(v) if not isinstance(d, str) else v
            elif isinstance(ed, QCheckBox):
                out[name] = ed.isChecked()
            elif isinstance(ed, QSpinBox):
                out[name] = ed.value()
            else:
                out[name] = ed.value()
        return out
