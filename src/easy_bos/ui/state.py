"""Application state shared by all pages, and background tasks."""

from __future__ import annotations

import logging
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal

from easy_bos.core.geometry import Camera, Setup
from easy_bos.core.media import Source, open_source
from easy_bos.core.project import Project
from easy_bos.core.reconstruct.density import Reconstruction
from easy_bos.core.results import ResultStore

log = logging.getLogger(__name__)


class Task(QThread):
    """Runs ``fn(progress, cancelled)`` off the UI thread.

    ``progress(done, total, message)`` and ``cancelled()`` are thread-safe callbacks.
    """

    progressed = Signal(int, int, str)
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, fn: Callable, parent: QObject | None = None):
        super().__init__(parent)
        self._fn = fn
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:  # noqa: D401 - Qt API
        try:
            out = self._fn(
                lambda d, t, m="": self.progressed.emit(int(d), int(t), str(m)),
                lambda: self._cancel,
            )
            self.succeeded.emit(out)
        except Exception as exc:  # surfaced to the user, never swallowed
            log.exception("task failed")
            self.failed.emit(f"{exc}\n\n{traceback.format_exc(limit=4)}")


class AppState(QObject):
    project_changed = Signal()
    setup_changed = Signal()
    source_changed = Signal()
    results_changed = Signal()
    reconstruction_changed = Signal()
    frame_changed = Signal(int)
    busy_changed = Signal(bool, str)
    status = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.project = Project()
        self.source: Source | None = None
        self.results: ResultStore | None = None
        self.reconstructions: dict[int, Reconstruction] = {}
        self.current_frame = 0
        self._tasks: list[Task] = []
        self.dirty = False

    # ------------------------------------------------------------------ project
    @property
    def setup(self) -> Setup:
        return self.project.setup.setup()

    def processed_setup(self) -> Setup:
        """Setup for the processed grid: the capture's pixels, cropped and binned.

        The effective pixel pitch comes from the sensor width over the *source's* frame width
        (not the Setup page's frame width, which a preset could change after loading), times
        the binning factor.
        """
        s = self.project.setup
        b = max(1, self.project.process.binning)
        width = self.source.info.width if self.source is not None else s.image_width_px
        height = self.source.info.height if self.source is not None else s.image_height_px
        pitch = s.sensor_width_mm * 1e-3 / width
        cam = Camera(
            s.focal_length_mm * 1e-3,
            pitch * b * (width // b),
            width // b,
            height // b,
            s.camera_name,
        )
        return Setup(cam, s.z_a_m, s.z_d_m, s.f_number)

    @property
    def busy(self) -> bool:
        return bool(self._tasks)

    def ensure_idle(self, action: str) -> bool:
        """True if no background task runs; otherwise tell the user and return False.

        Saving, opening or reprocessing closes files that a running task may be using.
        """
        if self._tasks:
            self.status.emit(
                f"Wait for the current task to finish (or cancel it) before you {action}."
            )
            return False
        return True

    def mark_dirty(self) -> None:
        self.dirty = True

    def new_project(self) -> None:
        self._close_results()
        self._close_source()
        self.project = Project()
        self.source = None
        self.reconstructions.clear()
        self.dirty = False
        self.project_changed.emit()
        self.source_changed.emit()
        self.results_changed.emit()

    def load_project(self, path: str | Path) -> None:
        self._close_results()
        self._close_source()
        self.project = Project.load(path)
        self.source = None
        self.reconstructions.clear()
        if self.project.capture.paths:
            try:
                self.source = open_source(self.project.capture.paths, self.project.capture.channel)
            except Exception as exc:  # missing files must not prevent opening the project
                self.status.emit(f"Capture not found: {exc}")
        rp = self.project.results_path
        if rp is not None and rp.exists():
            self.results = ResultStore(rp)
        self.dirty = False
        self.project_changed.emit()
        self.source_changed.emit()
        self.results_changed.emit()

    def save_project(self, path: str | Path) -> None:
        self.project.save(path)
        rp = self.project.results_path
        if self.results is None and rp is not None and rp.exists():
            # Results from an earlier capture must not reappear when this project is opened.
            rp.unlink()
        if self.results is not None and rp is not None and self.results.path != rp:
            self.results.save_as(rp)
            old = self.results
            self.results = ResultStore(rp)
            old.close()
        self.dirty = False
        self.status.emit(f"Saved {Path(path).name}")

    # ------------------------------------------------------------------ capture
    def _close_source(self) -> None:
        if self.source is not None:
            self.source.close()
            self.source = None

    def reopen_capture(self) -> None:
        """Re-open the current capture (e.g. another colour channel), keeping all settings."""
        paths = self.project.capture.paths
        if not paths:
            return
        src = open_source(paths if len(paths) > 1 else paths[0], self.project.capture.channel)
        self._close_source()
        self.source = src
        self._close_results()
        self.reconstructions.clear()
        self.mark_dirty()
        self.source_changed.emit()
        self.results_changed.emit()

    def open_capture(self, paths: list[str]) -> None:
        src = open_source(paths if len(paths) > 1 else paths[0], self.project.capture.channel)
        self._close_source()
        self.source = src
        self.project.capture.paths = [str(p) for p in paths]
        self._close_results()
        self.reconstructions.clear()
        n = len(src)
        p = self.project.process
        p.reference_range = (0, min(8, max(1, n // 4))) if n > 1 else (0, 1)
        p.frame_range = None
        p.roi = None
        p.exclusions.clear()
        p.stable_regions.clear()
        self.current_frame = min(p.reference_range[1], n - 1)
        info = src.info
        s = self.project.setup
        if (info.width, info.height) != (s.image_width_px, s.image_height_px):
            # Keep the sensor width (field of view) and update the effective pixel grid.
            s.image_width_px, s.image_height_px = info.width, info.height
            self.setup_changed.emit()
        self.mark_dirty()
        if n < 2:
            self.status.emit(
                "This capture has a single frame: BOS needs a flow-free reference "
                "and at least one frame with flow. Add the reference image."
            )
        self.source_changed.emit()
        self.results_changed.emit()

    def frame(self, index: int) -> np.ndarray | None:
        if self.source is None:
            return None
        index = int(np.clip(index, 0, len(self.source) - 1))
        return self.source.frame(index)

    def set_frame(self, index: int) -> None:
        if self.source is None:
            return
        index = int(np.clip(index, 0, len(self.source) - 1))
        if index != self.current_frame:
            self.current_frame = index
            self.frame_changed.emit(index)

    # ------------------------------------------------------------------ results
    def new_results(self) -> ResultStore:
        self._close_results()
        self.reconstructions.clear()
        rp = self.project.results_path
        self.results = ResultStore(rp) if rp is not None else ResultStore()
        self.results.clear()
        self.results_changed.emit()
        return self.results

    def _close_results(self) -> None:
        if self.results is not None:
            self.results.close()
            self.results = None

    # ------------------------------------------------------------------ tasks
    def run_task(
        self,
        fn: Callable,
        on_done: Callable | None = None,
        on_progress: Callable | None = None,
        message: str = "Working",
    ) -> Task:
        task = Task(fn, self)
        self._tasks.append(task)
        self.busy_changed.emit(True, message)

        def finish(*_):
            if task in self._tasks:
                self._tasks.remove(task)
            self.busy_changed.emit(bool(self._tasks), "")

        if on_done:
            task.succeeded.connect(on_done)
        if on_progress:
            task.progressed.connect(on_progress)
        task.failed.connect(self.error.emit)
        task.finished.connect(finish)
        task.start()
        return task

    def cancel_tasks(self) -> None:
        for t in list(self._tasks):
            t.cancel()

    def shutdown(self) -> None:
        self.cancel_tasks()
        for t in list(self._tasks):
            t.wait(60000)
        self._close_results()
        self._close_source()
