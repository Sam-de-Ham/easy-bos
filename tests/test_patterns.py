import numpy as np
import pytest
from scipy import ndimage

from easy_bos.core.patterns import (
    PatternKind,
    PatternSpec,
    generate,
    linear_to_srgb,
    pattern_spectrum_score,
)


@pytest.mark.parametrize("kind", list(PatternKind))
def test_patterns_shape_range_and_determinism(kind):
    spec = PatternSpec(kind, element_px=3, seed=7)
    a = generate(spec, 97, 61)
    b = generate(spec, 97, 61)
    assert a.shape == (61, 97) and a.dtype == np.float32
    assert a.min() >= 0 and a.max() <= 1
    np.testing.assert_array_equal(a, b)


def test_speckle_is_half_filled_and_blocky():
    a = generate(PatternSpec(PatternKind.SPECKLE, element_px=4, seed=1), 400, 400)
    assert a.mean() == pytest.approx(0.5, abs=0.02)
    cells = a.reshape(100, 4, 100, 4)
    assert np.all(cells.min(axis=(1, 3)) == cells.max(axis=(1, 3)))


def test_random_dots_never_touch():
    a = generate(PatternSpec(PatternKind.RANDOM_DOTS, element_px=3, seed=2, dot_fill=1.0), 300, 300)
    labels, n = ndimage.label(a > 0.5)
    sizes = ndimage.sum(np.ones_like(a), labels, range(1, n + 1))
    assert n > 100
    assert np.all(sizes == 9)


def test_multiscale_noise_is_uniform_and_textured():
    a = generate(PatternSpec(PatternKind.MULTISCALE_NOISE, element_px=2, seed=3), 256, 256)
    hist, _ = np.histogram(a, bins=10, range=(0, 1))
    assert hist.min() > 0.08 * a.size
    assert pattern_spectrum_score(a) > 0.2


def test_srgb_encoding_endpoints():
    np.testing.assert_allclose(linear_to_srgb(np.array([0.0, 1.0])), [0.0, 1.0], atol=1e-12)
    assert linear_to_srgb(np.array([0.18]))[0] == pytest.approx(0.461, abs=2e-3)
