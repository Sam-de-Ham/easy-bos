"""Render the app's pages to PNG for the README.

    uv run python scripts/screenshots.py <output folder> --clip <video> --frame <n>
    uv run python scripts/screenshots.py <output folder> --demo

With a clip: setup, capture check, method comparison and results for that footage (frames
around ``--frame`` are processed). With ``--demo``: the temperature reconstruction of the
synthetic demo plume, which is axisymmetric and steady as Abel inversion needs.
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QApplication

SIZE = (1600, 980)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--frame", type=int, default=0)
    ap.add_argument("--preset", default="Sony A6400 · 4K video · 50 mm")
    ap.add_argument("--display", default='27" 1440p monitor')
    ap.add_argument(
        "--line",
        type=float,
        nargs=4,
        metavar=("X0", "Y0", "X1", "Y1"),
        help="profile line on the results page, as fractions of the image size",
    )
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication([])

    from easy_bos.ui import theme
    from easy_bos.ui.main_window import MainWindow
    from easy_bos.ui.state import AppState

    theme.apply(app)
    state = AppState()
    win = MainWindow(state)
    win.resize(*SIZE)
    win.show()

    def pump(seconds: float = 0.3) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()

    def settle() -> None:
        while state._tasks:
            pump(0.05)
        pump(0.5)

    def shot(name: str) -> None:
        win.grab().save(str(args.out / f"{name}.png"))
        print(name, flush=True)

    process, results = win.pages[2], win.pages[3]
    if args.demo:
        from easy_bos.bench.demo import write_demo

        demo = Path(tempfile.mkdtemp()) / "demo.mp4"
        state.project = write_demo(demo)
        state.project_changed.emit()
        state.open_capture([str(demo)])
        settle()
        process.first.setValue(30)
        process.last.setValue(45)
        process.step.setValue(3)
        process._run()
        settle()
        win.go(3)
        pump()
        results._reconstruct()
        settle()
        results.line_btn.setChecked(True)
        pump()
        shot("results-temperature")
    else:
        state.project.setup.apply_preset(args.preset)
        state.project_changed.emit()
        screens = win.pages[0].screen_combo
        screens.setCurrentIndex(screens.findText(args.display))
        state.open_capture([str(args.clip)])
        settle()
        win.go(0)
        pump()
        shot("setup")
        win.go(1)
        win.pages[1]._run_checks()
        settle()
        shot("capture")
        win.go(2)
        state.set_frame(args.frame)
        process.scrubber.set_frame(args.frame)
        process._frame_changed(args.frame)
        process._compare()
        settle()
        shot("compare")
        process.first.setValue(args.frame - 30)
        process.last.setValue(args.frame + 30)
        process.step.setValue(2)
        process._run()
        settle()
        win.go(3)
        results.view.setCurrentIndex(2)  # colour schlieren
        results.scrubber.set_frame(args.frame)
        results._frame(args.frame)
        results.line_btn.setChecked(True)
        if args.line:
            h, w = results.canvas._shape
            x0, y0, x1, y1 = args.line
            results.canvas.line.movePoint(0, QPointF(x0 * w, y0 * h))
            results.canvas.line.movePoint(1, QPointF(x1 * w, y1 * h))
        pump()
        shot("results")
    state.shutdown()


if __name__ == "__main__":
    main()
