"""Tests for the performance-critical building blocks added for real footage."""

import numpy as np
import pytest
import torch
from scipy.fft import dct

from easy_bos.bench import fields, synth
from easy_bos.core import imops, preprocess, quality
from easy_bos.core.solvers import _dct, _idct, neumann_spectral_solver


@pytest.mark.parametrize("n", [7, 8, 33])
def test_dct_matches_scipy_and_inverts(n):
    x = np.random.default_rng(n).random((3, n))
    np.testing.assert_allclose(_dct(torch.tensor(x)).numpy(), dct(x, type=2), atol=1e-12)
    np.testing.assert_allclose(_idct(_dct(torch.tensor(x))).numpy(), x, atol=1e-12)


def test_spectral_solver_inverts_regulariser_operator():
    x = torch.randn(1, 2, 37, 50, dtype=torch.float64)
    solve = neumann_spectral_solver(37, 50, lambda lam: 1.0 + 3 * lam * lam + 2 * lam, "cpu")

    def op(v):
        return v + 3 * imops.laplacian(imops.laplacian(v)) - 2 * imops.laplacian(v)

    assert (op(solve(x.float()).double()) - x).abs().max() < 1e-4


def test_quick_shift_vector():
    rng = np.random.default_rng(0)
    img = rng.random((256, 320)).astype(np.float32)
    from scipy.ndimage import gaussian_filter, shift

    img = gaussian_filter(img, 1.5)
    moved = shift(img, (-1.2, 2.4), order=3, mode="nearest").astype(np.float32)
    dx, dy = preprocess.quick_shift_vector(img, moved, factor=1)
    assert dx == pytest.approx(2.4, abs=0.1) and dy == pytest.approx(-1.2, abs=0.1)


def test_auto_valid_rejects_blank_and_glowing_areas():
    rng = np.random.default_rng(1)
    ref = ((rng.random((200, 300)) > 0.5) * 0.6 + 0.2).astype(np.float32)
    ref[:, :40] = 0.05  # bezel: no texture
    cur = ref.copy()
    cur[80:120, 150:190] += 0.5  # a flame in front of the background
    valid = preprocess.auto_valid(ref, cur)
    assert not valid[:, :30].any()
    assert not valid[90:110, 160:180].any()
    assert valid[150:190, 80:140].all()


def test_suggest_reference_skips_initial_shake():
    st = synth.bench_setup(synth.bench_camera(320, 240))
    scene = synth.Scene(fields.WeakLens(focal_length=1e9), st)
    still = synth.Renderer(scene, synth.CameraModel(), seed=2).sequence(30, 0).frames
    shaky = (
        synth.Renderer(scene, synth.CameraModel(), synth.CaptureEffects(shake_px=0.8), seed=3)
        .sequence(12, 0)
        .frames
    )
    frames = np.concatenate([shaky, still])
    a, b = preprocess.suggest_reference(lambda i: frames[i], len(frames), 25.0, length=10)
    assert a >= 11 and b - a == 10


def test_moire_score_is_crop_independent_and_separates():
    st = synth.bench_setup(synth.bench_camera(1024, 1024), f_number=4)
    scene = synth.Scene(fields.WeakLens(focal_length=1e9), st)
    clean = synth.Renderer(scene, synth.CameraModel(psf_sigma_px=0.45), seed=1).sequence(1, 0)
    moire = synth.Renderer(
        scene,
        synth.CameraModel(psf_sigma_px=0.45),
        synth.CaptureEffects(display_pitch_px=1.7),
        seed=1,
    ).sequence(1, 0)
    c, m = clean.frames[0], moire.frames[0]
    assert quality.moire_score(c) < 20 and quality.moire_score(c[:512, :512]) < 20
    assert quality.moire_score(m) > 100 and quality.moire_score(m[:512, :512]) > 100
