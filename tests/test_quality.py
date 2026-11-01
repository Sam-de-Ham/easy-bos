import numpy as np
import pytest

from easy_bos.bench import fields, synth
from easy_bos.core import quality
from easy_bos.core.patterns import PatternKind, PatternSpec


def _frames(effects=None, element=3.0, n=3, camera=None):
    st = synth.bench_setup(synth.bench_camera(384, 288))
    scene = synth.Scene(
        fields.WeakLens(focal_length=1e9), st, PatternSpec(PatternKind.SPECKLE, 3), element
    )
    cap = synth.Renderer(scene, camera or synth.CameraModel(), effects, seed=4).sequence(n, 0)
    return list(cap.frames), st


def _by_key(checks):
    return {c.key: c for c in checks}


def test_clean_capture_passes():
    frames, st = _frames()
    checks = _by_key(quality.run_checks(frames, st))
    assert quality.overall(list(checks.values())) is quality.Status.OK
    assert set(checks) >= {
        "exposure",
        "sampling",
        "focus",
        "noise",
        "stability",
        "moire",
        "compression",
        "banding",
    }


@pytest.mark.parametrize("element, expected", [(3.0, 3.8), (6.0, 6.5)])
def test_feature_size(element, expected):
    frames, _ = _frames(element=element, n=1)
    assert quality.feature_size_px(frames[0]) == pytest.approx(expected, abs=0.5)


@pytest.mark.parametrize(
    "effects, key",
    [
        (synth.CaptureEffects(banding_amplitude=0.03), "banding"),
        (synth.CaptureEffects(display_pitch_px=1.4), "moire"),
        (synth.CaptureEffects(codec="h264", crf=30), "compression"),
        (synth.CaptureEffects(shake_px=1.5), "stability"),
    ],
)
def test_problems_are_flagged(effects, key):
    frames, st = _frames(effects)
    checks = _by_key(quality.run_checks(frames, st))
    assert checks[key].status in (quality.Status.WARN, quality.Status.FAIL)


def test_shake_does_not_trigger_banding():
    frames, st = _frames(synth.CaptureEffects(shake_px=1.5))
    assert _by_key(quality.run_checks(frames, st))["banding"].status is quality.Status.OK


def test_clipping_detected():
    frames, _ = _frames(n=1, camera=synth.CameraModel(exposure=1.6))
    c = quality.check_exposure(frames[0])
    assert c.status is quality.Status.FAIL


def test_out_of_focus_region_detected():
    frames, _ = _frames(n=1)
    f = frames[0].copy()
    from scipy.ndimage import gaussian_filter

    f[:, : f.shape[1] // 2] = gaussian_filter(f[:, : f.shape[1] // 2], 6)
    assert quality.check_focus(f).status is not quality.Status.OK
    assert np.isfinite(quality.feature_size_px(f))
