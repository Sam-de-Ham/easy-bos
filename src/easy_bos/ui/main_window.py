"""Main window: navigation rail, the four workflow pages, status bar and menus."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QKeySequence, QPainter
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from easy_bos import __version__
from easy_bos.core.device import default_device, describe
from easy_bos.ui import widgets as W
from easy_bos.ui.pages.capture_page import CapturePage
from easy_bos.ui.pages.process_page import ProcessPage
from easy_bos.ui.pages.results_page import ResultsPage
from easy_bos.ui.pages.setup_page import SetupPage
from easy_bos.ui.state import AppState
from easy_bos.ui.theme import FONT_FAMILY, C

STEPS = [
    ("Setup", "Camera, distances and background"),
    ("Capture", "Check your footage"),
    ("Process", "Measure the displacement"),
    ("Results", "Views, physics and export"),
]


class NavItem(QWidget):
    clicked = Signal(int)

    def __init__(self, index: int, title: str, subtitle: str):
        super().__init__()
        self.index = index
        self.title = title
        self.subtitle = subtitle
        self.active = False
        self.state = "none"  # none, done, attention
        self.setFixedHeight(58)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._hover = False

    def set_active(self, on: bool) -> None:
        self.active = on
        self.update()

    def enterEvent(self, _):  # noqa: N802 - Qt API
        self._hover = True
        self.update()

    def leaveEvent(self, _):  # noqa: N802 - Qt API
        self._hover = False
        self.update()

    def mousePressEvent(self, _):  # noqa: N802 - Qt API
        self.clicked.emit(self.index)

    def paintEvent(self, _):  # noqa: N802 - Qt API
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.active or self._hover:
            p.fillRect(self.rect(), QColor(C.raised if self.active else C.panel))
        if self.active:
            p.fillRect(0, 10, 2, self.height() - 20, QColor(C.accent))
        # step number in a ring
        cx, cy, r = 30, self.height() / 2, 11
        ring = QColor(C.accent if self.active else C.border_strong)
        p.setPen(ring)
        if self.state == "done":
            p.setBrush(QColor(C.accent_dim))
        else:
            p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(int(cx - r), int(cy - r), 2 * r, 2 * r)
        f = QFont(FONT_FAMILY, 8)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(C.text if self.active else C.muted))
        p.drawText(
            int(cx - r),
            int(cy - r),
            2 * r,
            2 * r,
            Qt.AlignmentFlag.AlignCenter,
            str(self.index + 1),
        )
        f = QFont(FONT_FAMILY, 10)
        f.setWeight(QFont.Weight.DemiBold)
        p.setFont(f)
        p.setPen(QColor(C.text if self.active else C.muted))
        p.drawText(52, 12, self.width() - 60, 18, Qt.AlignmentFlag.AlignLeft, self.title)
        p.setFont(QFont(FONT_FAMILY, 8))
        p.setPen(QColor(C.faint if not self.active else C.muted))
        p.drawText(52, 31, self.width() - 60, 16, Qt.AlignmentFlag.AlignLeft, self.subtitle)


class MainWindow(QMainWindow):
    def __init__(self, state: AppState):
        super().__init__()
        self.state = state
        self.setWindowTitle("Easy BOS")
        self.resize(1480, 920)
        self.setMinimumSize(QSize(1100, 700))
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        body = QHBoxLayout()
        body.setSpacing(0)
        outer.addLayout(body, 1)

        rail = QWidget()
        rail.setObjectName("NavRail")
        rail.setFixedWidth(232)
        rl = QVBoxLayout(rail)
        rl.setContentsMargins(0, 18, 0, 12)
        rl.setSpacing(2)
        brand = W.label("Easy BOS", "H1")
        brand.setContentsMargins(20, 0, 0, 0)
        tagline = W.label("Background-oriented schlieren", "Faint")
        tagline.setContentsMargins(20, 0, 0, 16)
        rl.addWidget(brand)
        rl.addWidget(tagline)
        self.nav: list[NavItem] = []
        for i, (t, sub) in enumerate(STEPS):
            item = NavItem(i, t, sub)
            item.clicked.connect(self.go)
            rl.addWidget(item)
            self.nav.append(item)
        rl.addStretch(1)
        self.device_label = W.label("", "Faint", wrap=True)
        self.device_label.setContentsMargins(20, 0, 16, 0)
        rl.addWidget(self.device_label)
        body.addWidget(rail)

        self.stack = QStackedWidget()
        self.pages = [SetupPage(state), CapturePage(state), ProcessPage(state), ResultsPage(state)]
        for p in self.pages:
            self.stack.addWidget(p)
        body.addWidget(self.stack, 1)
        self.pages[2].finished.connect(lambda: self.go(3))

        bar = QWidget()
        bar.setObjectName("StatusBar")
        bar.setFixedHeight(28)
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(14, 0, 14, 0)
        self.message = QLabel("")
        self.message.setObjectName("Muted")
        self.progress = QProgressBar()
        self.progress.setFixedWidth(220)
        self.progress.setVisible(False)
        self.busy_label = W.label("", "Faint")
        bl.addWidget(self.message, 1)
        bl.addWidget(self.busy_label)
        bl.addWidget(self.progress)
        outer.addWidget(bar)

        self._menus()
        state.status.connect(self.show_message)
        state.error.connect(self.show_error)
        state.busy_changed.connect(self._busy)
        state.source_changed.connect(self._update_nav)
        state.results_changed.connect(self._update_nav)
        state.project_changed.connect(self._update_title)
        dev = default_device()
        self.device_label.setText(
            f"Compute: {describe(dev)}"
            + ("" if dev.type == "cuda" else "\nNo CUDA GPU found: processing runs on the CPU.")
        )
        self.go(0)
        self._update_title()

    # ------------------------------------------------------------------ navigation
    def go(self, index: int) -> None:
        self.stack.setCurrentIndex(index)
        for i, item in enumerate(self.nav):
            item.set_active(i == index)
        page = self.pages[index]
        if hasattr(page, "activated"):
            page.activated()

    def _update_nav(self) -> None:
        st = self.state
        self.nav[0].state = "done"
        self.nav[1].state = "done" if st.source is not None else "none"
        self.nav[2].state = "done" if st.results is not None and len(st.results) else "none"
        for item in self.nav:
            item.update()

    def _update_title(self) -> None:
        name = Path(self.state.project.path).stem if self.state.project.path else "Untitled"
        self.setWindowTitle(f"{name} — Easy BOS")

    # ------------------------------------------------------------------ status
    def show_message(self, text: str) -> None:
        self.message.setText(text)

    def show_error(self, text: str) -> None:
        first, _, rest = text.partition("\n")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle("Something went wrong")
        box.setText(first)
        if rest.strip():
            box.setDetailedText(rest.strip())
        box.exec()
        self.message.setText("Error: " + first)

    def _busy(self, busy: bool, message: str) -> None:
        self.progress.setVisible(busy)
        if busy:
            self.progress.setRange(0, 0)
            self.busy_label.setText(message)
        else:
            self.busy_label.setText("")

    def set_progress(self, done: int, total: int, message: str = "") -> None:
        self.progress.setVisible(True)
        self.progress.setRange(0, max(total, 1))
        self.progress.setValue(done)
        if message:
            self.busy_label.setText(message)

    # ------------------------------------------------------------------ menus
    def _menus(self) -> None:
        mb = self.menuBar()
        fm = mb.addMenu("&File")
        self._action(fm, "New project", self._new, QKeySequence.StandardKey.New)
        self._action(fm, "Open project…", self._open, QKeySequence.StandardKey.Open)
        self._action(fm, "Save project", self._save, QKeySequence.StandardKey.Save)
        self._action(fm, "Save project as…", self._save_as, QKeySequence.StandardKey.SaveAs)
        fm.addSeparator()
        self._action(
            fm,
            "Open capture…",
            lambda: (self.go(1), self.pages[1].open_capture()),
            QKeySequence("Ctrl+I"),
        )
        fm.addSeparator()
        self._action(fm, "Quit", self.close, QKeySequence.StandardKey.Quit)
        vm = mb.addMenu("&Go")
        for i, (t, _) in enumerate(STEPS):
            self._action(vm, t, lambda i=i: self.go(i), QKeySequence(f"Ctrl+{i + 1}"))
        hm = mb.addMenu("&Help")
        self._action(hm, "Open demo project", self.open_demo)
        hm.addSeparator()
        self._action(hm, "About Easy BOS", self._about)

    def _action(self, menu, text, fn, shortcut=None) -> QAction:
        a = QAction(text, self)
        if shortcut is not None:
            a.setShortcut(shortcut)
        a.triggered.connect(fn)
        menu.addAction(a)
        return a

    def _confirm_discard(self) -> bool:
        if not self.state.dirty:
            return True
        r = QMessageBox.question(
            self,
            "Unsaved changes",
            "Save changes to the current project?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if r == QMessageBox.StandardButton.Save:
            return self._save()
        return r == QMessageBox.StandardButton.Discard

    def _new(self) -> None:
        if self._confirm_discard():
            self.state.new_project()
            self.go(0)

    def _open(self) -> None:
        if not self._confirm_discard():
            return
        path, _ = QFileDialog.getOpenFileName(self, "Open project", "", "Easy BOS project (*.ebos)")
        if path:
            self.state.load_project(path)
            self._update_title()

    def _save(self) -> bool:
        if not self.state.project.path:
            return self._save_as()
        self.state.save_project(self.state.project.path)
        return True

    def _save_as(self) -> bool:
        path, _ = QFileDialog.getSaveFileName(
            self, "Save project", "experiment.ebos", "Easy BOS project (*.ebos)"
        )
        if not path:
            return False
        self.state.save_project(path)
        self._update_title()
        return True

    def open_demo(self) -> None:
        """Render the synthetic demo capture (once) and open it with a matching setup."""
        if not self._confirm_discard():
            return
        from easy_bos.bench.demo import write_demo
        from easy_bos.core.paths import data_dir

        path = data_dir() / "demo" / "plume-demo.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)

        def work(progress, cancelled):
            return write_demo(path) if not path.exists() else None

        def done(project):
            from easy_bos.bench.demo import demo_project

            self.state.new_project()
            self.state.project = project or demo_project()
            self.state.project_changed.emit()
            self.state.open_capture([str(path)])
            self.state.project.process.reference_range = (0, 10)
            self.state.source_changed.emit()
            self.state.dirty = False
            self.go(1)
            self.show_message(
                "Demo loaded: a synthetic hot plume. Check the capture, then process it."
            )

        self.state.run_task(work, done, message="Preparing demo")

    def _about(self) -> None:
        QMessageBox.about(
            self,
            "About Easy BOS",
            f"<b>Easy BOS {__version__}</b><p>Background-oriented schlieren: see heat, gas and "
            "air flow with an ordinary camera and a patterned background.</p>"
            f"<p>Compute device: {describe()}</p>",
        )

    def closeEvent(self, e):  # noqa: N802 - Qt API
        if not self._confirm_discard():
            e.ignore()
            return
        self.state.shutdown()
        super().closeEvent(e)
