"""Regression tests for application-state and data-flow issues found in the UI review."""

import threading

import numpy as np
import pytest

from easy_bos.core import render
from easy_bos.core.export import write_video
from easy_bos.core.media import open_source
from easy_bos.core.pipeline import FrameResult
from easy_bos.core.project import Project
from easy_bos.ui.state import AppState


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "clip.mp4"
    frames = [np.full((48, 64, 3), 10 * k, np.uint8) for k in range(20)]
    write_video(path, frames, fps=25, crf=0)
    return path


def test_video_source_is_thread_safe(clip):
    src = open_source(clip, channel="red")
    errors = []

    def reader(order):
        try:
            for k in order:
                v = src.frame(k).mean() * 255
                assert abs(v - 10 * k) < 3, (k, v)
        except Exception as exc:  # collected and asserted below
            errors.append(exc)

    threads = [
        threading.Thread(target=reader, args=(list(range(20)) * 3,)),
        threading.Thread(target=reader, args=(list(range(19, -1, -1)) * 3,)),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors, errors


def test_saving_without_results_removes_stale_results(tmp_path, clip):
    state = AppState()
    state.open_capture([str(clip)])
    store = state.new_results()
    store.add(FrameResult(5, np.zeros((2, 4, 4), np.float32), None, None, 0.0, 0.0))
    path = tmp_path / "exp.ebos"
    state.save_project(path)
    assert state.project.results_path.exists()
    state.open_capture([str(clip)])  # new capture discards the results
    state.save_project(path)
    assert not state.project.results_path.exists()
    state.load_project(path)
    assert state.results is None
    state.shutdown()


def test_processed_setup_uses_source_width_and_binning(clip):
    state = AppState()
    state.open_capture([str(clip)])
    s = state.project.setup
    s.sensor_width_mm = 6.4
    s.image_width_px = 3840  # e.g. a preset chosen after loading a 64 px wide capture
    state.project.process.binning = 2
    st = state.processed_setup()
    assert st.p == pytest.approx(6.4e-3 / 64 * 2)
    full = state.project.setup.setup()
    assert st.pixels_per_radian < full.pixels_per_radian
    state.shutdown()


def test_busy_state_blocks_destructive_actions():
    state = AppState()
    messages = []
    state.status.connect(messages.append)
    state._tasks.append(object())  # simulate a running task
    assert not state.ensure_idle("save")
    assert messages
    state._tasks.clear()
    assert state.ensure_idle("save")


def test_divergence_colour_scale_matches_colorbar():
    d = np.random.default_rng(0).normal(0, 0.3, (2, 40, 50))
    s = 0.2
    img = render.render(render.View.DIVERGENCE, d, render.ViewSettings(scale=s))
    expected = render.colorize(render.divergence(d), "gray", -s, s)
    np.testing.assert_array_equal(img, expected)


def test_exclusions_round_outward_stable_regions_inward():
    p = Project()
    p.process.exclusions = [(0, 0, 3, 3)]
    p.process.stable_regions = [(1, 1, 4, 4)]
    p.process.binning = 2
    s = p.process.settings((8, 8))
    assert s.mask[1, 1] == 0  # source pixels 2-3 are partly excluded -> excluded
    assert s.stable_region[1, 1] and not s.stable_region[0, 0] and not s.stable_region[2, 2]
