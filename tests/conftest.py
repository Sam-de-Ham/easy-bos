import numpy as np
import pytest

from easy_bos.bench import fields, synth


@pytest.fixture(scope="session")
def bench_setup():
    return synth.bench_setup(synth.bench_camera(256, 192))


@pytest.fixture(scope="session")
def plume_pair(bench_setup):
    """A small realistic plume capture: (ref, cur, truth)."""
    scene = synth.Scene(
        fields.AxisymmetricPlume(delta_rho0=0.4, sigma0=0.004, y_source=0.012), bench_setup
    )
    cap = synth.Renderer(scene, synth.CameraModel(), seed=3).sequence(1, 1)
    return cap.frames[0], cap.frames[1], cap.truth[1]


def epe(d, truth, margin=12):
    e = np.sqrt(((d - truth) ** 2).sum(0))[margin:-margin, margin:-margin]
    return float(np.nanmean(e))


@pytest.fixture(scope="session")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
