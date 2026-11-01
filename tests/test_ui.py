"""End-to-end smoke test of the desktop UI, driven offscreen.

Loads a small synthetic capture, runs the capture check, a preview, a processing run and a
reconstruction through the real widgets, and exports data. Marked slow.
"""

import os
import time

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def demo_video(tmp_path_factory):
    from easy_bos.bench.demo import write_demo

    path = tmp_path_factory.mktemp("demo") / "demo.mp4"
    project = write_demo(path, width=320, height=192, n_ref=6, n_flow=6)
    return path, project


def _wait(app, state, timeout=120.0):
    end = time.time() + timeout
    while state._tasks and time.time() < end:
        app.processEvents()
        time.sleep(0.02)
    for _ in range(20):
        app.processEvents()
    assert not state._tasks, "background task did not finish"


def test_full_workflow(qapp, demo_video, tmp_path):
    from easy_bos.core.render import View
    from easy_bos.ui import theme
    from easy_bos.ui.main_window import MainWindow
    from easy_bos.ui.state import AppState

    path, project = demo_video
    theme.apply(qapp)
    state = AppState()
    errors = []
    state.error.connect(errors.append)
    win = MainWindow(state)
    win.show()
    state.project = project
    state.project_changed.emit()
    state.open_capture([str(path)])
    state.project.process.reference_range = (0, 6)
    state.source_changed.emit()
    assert len(state.source) == 12

    cap = win.pages[1]
    cap._run_checks()
    _wait(qapp, state)
    assert cap.rows_box.count() >= 7

    proc = win.pages[2]
    state.set_frame(10)
    proc.scrubber.set_frame(10)
    proc._frame_changed(10)
    proc._preview_frame()
    _wait(qapp, state)
    assert proc._preview is not None and np.isfinite(proc._preview).all()

    proc.first.setValue(8)
    proc.last.setValue(11)
    proc._run()
    _wait(qapp, state)
    assert state.results is not None and state.results.indices() == [8, 9, 10, 11]

    res = win.pages[3]
    win.go(3)
    qapp.processEvents()
    res._reconstruct()
    _wait(qapp, state)
    rec = state.reconstructions[res.index]
    assert np.nanmax(rec.temperature) > 293.15 + 10  # the demo plume is well above ambient
    for v in (View.KNIFE_EDGE, View.COLOUR_SCHLIEREN, View.TEMPERATURE):
        res.view.setCurrentIndex(res.view.findData(v.value))
        qapp.processEvents()

    out = tmp_path / "data.h5"
    import h5py

    state.results.save_as(out)
    with h5py.File(out) as f:
        assert len(f["frames"]) == 4
    assert not errors, errors
    state.shutdown()
