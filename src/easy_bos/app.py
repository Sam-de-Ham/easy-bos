"""Application entry point (``uv run easy-bos``)."""

from __future__ import annotations

import logging
import os
import sys


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv if argv is None else argv)
    from easy_bos.core.paths import log_dir

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_dir() / "easy-bos.log", encoding="utf-8")],
    )
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Keep fractional Windows scale factors exact (125 %, 150 %): the pattern window relies on
    # knowing the true device pixel ratio to draw one pattern pixel per display pixel.
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv)
    app.setApplicationName("Easy BOS")
    app.setOrganizationName("EasyBOS")

    from easy_bos.ui import theme
    from easy_bos.ui.main_window import MainWindow
    from easy_bos.ui.state import AppState

    theme.apply(app)
    state = AppState()
    win = MainWindow(state)
    project = next((a for a in argv[1:] if a.lower().endswith(".ebos")), None)
    if project:
        state.load_project(project)
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
