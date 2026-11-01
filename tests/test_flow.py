"""Accuracy of every displacement estimator against the synthetic renderer's exact truth."""

from dataclasses import dataclass

import numpy as np
import pytest

from easy_bos.bench import fields, synth
from easy_bos.core import flow
from tests.conftest import epe

FAST = ["farneback", "dis", "correlation", "lucas_kanade", "variational", "wavelet"]

# Mean endpoint-error ceilings on a realistic plume (12-bit sensor noise, PSF, vignetting).
# Values sit ~1.5x above measured results so regressions are caught.
PLUME_EPE = {
    "farneback": 0.04,
    "dis": 0.09,
    "correlation": 0.03,
    "lucas_kanade": 0.04,
    "variational": 0.02,
    "wavelet": 0.03,
}


@dataclass
class Tilt(fields.Field):
    """Uniform deflection: a pure sub-pixel translation of the background."""

    ex: float = 0.0
    ey: float = 0.0
    geometry: str = "phase"

    def path(self, x, y, t):
        return self.ex * x + self.ey * y


@pytest.fixture(scope="module")
def shift_pair(bench_setup):
    k = bench_setup.pixels_per_radian
    scene = synth.Scene(Tilt(0.31 / k, -0.17 / k), bench_setup, apply_dof_blur=False)
    cap = synth.Renderer(scene, synth.CameraModel.ideal(), seed=5).sequence(1, 1)
    return cap.frames[0], cap.frames[1]


@pytest.mark.parametrize(
    "key", ["correlation", "lucas_kanade", "variational", "wavelet", "farneback"]
)
def test_uniform_subpixel_shift_is_unbiased(key, shift_pair):
    d = flow.create(key).estimate(*shift_pair).displacement[:, 16:-16, 16:-16]
    tol = 0.012 if key == "farneback" else 0.004
    assert np.median(d[0]) == pytest.approx(0.31, abs=tol)
    assert np.median(d[1]) == pytest.approx(-0.17, abs=tol)


@pytest.mark.parametrize("key", FAST)
def test_plume_accuracy(key, plume_pair):
    ref, cur, truth = plume_pair
    res = flow.create(key).estimate(ref, cur)
    assert res.displacement.shape == truth.shape
    assert np.isfinite(res.displacement).all()
    assert epe(res.displacement, truth) < PLUME_EPE[key]


def test_zero_flow_gives_zero(bench_setup):
    scene = synth.Scene(Tilt(), bench_setup)
    cap = synth.Renderer(scene, synth.CameraModel(), seed=9).sequence(2, 0)
    d = flow.create("variational").estimate(cap.frames[0], cap.frames[1]).displacement
    assert np.sqrt((d**2).mean()) < 0.01


def test_registry_and_param_metadata():
    for key, cls in flow.REGISTRY.items():
        assert cls.key == key
        for name, meta, _default in cls.Params.describe():
            assert meta.label
            assert hasattr(cls.Params(), name)


def test_mask_is_respected_by_variational(plume_pair):
    ref, cur, truth = plume_pair
    mask = np.ones_like(ref)
    mask[:, :40] = 0
    d = flow.create("variational").estimate(ref, cur, mask).displacement
    assert np.isfinite(d).all()
    assert epe(d[:, :, 50:], truth[:, :, 50:]) < PLUME_EPE["variational"]
