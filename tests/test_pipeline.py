import numpy as np
import pytest

from easy_bos.bench import fields, synth
from easy_bos.core import preprocess, render
from easy_bos.core.media import ArraySource, open_source
from easy_bos.core.pipeline import ProcessingSettings, Processor, ReferenceMode
from tests.conftest import epe


def test_affine_fit_ignores_outlier_region():
    h, w = 120, 160
    true = preprocess.Motion(np.array([[1e-3, -2e-3, 0.4], [2e-3, 1e-3, -0.3]]), 0, 1)
    d = true.field(h, w)
    d[:, 40:80, 60:100] += 2.0  # "flow" region
    m = preprocess.fit_affine(d)
    np.testing.assert_allclose(m.matrix, true.matrix, atol=1e-4)
    assert m.inlier_fraction < 0.95


def test_banding_detection_and_correction():
    rng = np.random.default_rng(0)
    ref = rng.random((200, 150)).astype(np.float32) * 0.5 + 0.25
    gain = 1 + 0.05 * np.sin(np.arange(200) / 200 * 2 * np.pi * 7)
    cur = ref * gain[:, None].astype(np.float32)
    assert preprocess.banding_strength(cur, ref) > 0.02
    fixed = preprocess.correct_row_banding(cur, ref)
    assert preprocess.banding_strength(fixed, ref) < 0.004


def test_exposure_matching():
    rng = np.random.default_rng(1)
    ref = rng.random((50, 50)).astype(np.float32)
    cur = ref * 1.3
    np.testing.assert_allclose(preprocess.match_exposure(cur, ref), ref, atol=1e-5)


def test_exposure_matching_ignores_noise_level():
    rng = np.random.default_rng(3)
    clean = (rng.random((200, 200)) > 0.5).astype(np.float32) * 0.6 + 0.2
    noisy = clean + rng.normal(0, 0.05, clean.shape).astype(np.float32)
    fixed = preprocess.match_exposure(noisy, clean)
    assert np.mean(fixed / np.maximum(noisy, 1e-3)) == pytest.approx(1.0, abs=0.01)


def _capture(bench_setup, effects=None, n_ref=4, n_flow=2, seed=11, sigma0=0.004, spread=0.12):
    plume = fields.AxisymmetricPlume(delta_rho0=0.4, sigma0=sigma0, spread=spread, y_source=0.012)
    scene = synth.Scene(plume, bench_setup)
    return synth.Renderer(scene, synth.CameraModel(), effects, seed=seed).sequence(n_ref, n_flow)


def test_pipeline_with_shake_matches_truth(bench_setup):
    """Automatic stabilisation when the disturbance leaves most of the frame undisturbed."""
    cap = _capture(
        bench_setup,
        synth.CaptureEffects(shake_px=0.6, shake_rot_deg=0.05),
        sigma0=0.0025,
        spread=0.04,
    )
    settings = ProcessingSettings(method="variational", reference_range=(0, 1), stabilize=True)
    proc = Processor(ArraySource(cap.frames), settings)
    results = list(proc.run())
    assert [r.index for r in results] == [1, 2, 3, 4, 5]
    for r in results[-2:]:
        assert epe(r.displacement, cap.truth[r.index]) < 0.03
    # Without stabilisation the shake dominates the error.
    proc2 = Processor(ArraySource(cap.frames), settings.with_(stabilize=False))
    r = proc2.process_frame(5)
    assert epe(r.displacement, cap.truth[5]) > 0.1


def test_stable_region_handles_frame_filling_plume():
    """A wide plume across the frame: automatic and stable-region stabilisation both hold."""
    wide = synth.bench_setup(synth.bench_camera(448, 160))
    cap = _capture(wide, synth.CaptureEffects(shake_px=0.6, shake_rot_deg=0.05))
    stable = np.zeros(cap.frames.shape[1:], bool)
    stable[:, :40] = stable[:, -40:] = True
    assert np.abs(cap.truth[5][:, stable]).max() < 0.02  # the marked region really is flow-free
    auto = Processor(ArraySource(cap.frames), ProcessingSettings(reference_range=(0, 1)))
    assert epe(auto.process_frame(5).displacement, cap.truth[5]) < 0.03
    settings = ProcessingSettings(
        method="variational", reference_range=(0, 1), stable_region=stable
    )
    r = Processor(ArraySource(cap.frames), settings).process_frame(5)
    assert epe(r.displacement, cap.truth[5]) < 0.03


def test_pipeline_previous_frame_mode(bench_setup):
    cap = _capture(bench_setup, n_ref=1, n_flow=3)
    settings = ProcessingSettings(
        method="lucas_kanade", reference_mode=ReferenceMode.PREVIOUS, stabilize=False
    )
    results = list(Processor(ArraySource(cap.frames), settings).run())
    assert [r.index for r in results] == [1, 2, 3]
    # Frame 1 vs 0: full plume; frames 2 vs 1: static field -> near zero.
    assert np.abs(results[0].displacement).max() > 0.2
    assert np.abs(results[1].displacement).mean() < 0.02


def test_roi_and_binning(bench_setup):
    cap = _capture(bench_setup, n_ref=1, n_flow=1)
    s = ProcessingSettings(
        method="lucas_kanade",
        reference_range=(0, 1),
        roi=(10, 20, 200, 120),
        binning=2,
        stabilize=False,
    )
    r = Processor(ArraySource(cap.frames), s).process_frame(1)
    assert r.displacement.shape == (2, 60, 100)


def test_video_roundtrip_through_codec(tmp_path, bench_setup):
    import av

    cap = _capture(bench_setup, n_ref=2, n_flow=1)
    path = tmp_path / "clip.mp4"
    with av.open(str(path), "w") as out:
        st = out.add_stream("libx264", rate=30)
        st.width, st.height, st.pix_fmt = cap.frames.shape[2], cap.frames.shape[1], "yuv420p"
        st.options = {"crf": "10"}
        for f in cap.frames:
            frame = av.VideoFrame.from_ndarray((f * 255).round().astype(np.uint8), format="gray")
            for p in st.encode(frame.reformat(format="yuv420p")):
                out.mux(p)
        for p in st.encode():
            out.mux(p)
    src = open_source(path, channel="luma")
    assert len(src) == 3
    assert src.info.width == cap.frames.shape[2]
    # x264 at CRF 10 on an 8-bit BOS texture: small mean error, bounded worst case.
    assert np.abs(src.frame(1) - cap.frames[1]).mean() < 0.01


def test_image_sequence_16bit(tmp_path):
    import cv2

    for i in range(3):
        img = np.full((20, 30), 1000 * (i + 1), np.uint16)
        cv2.imwrite(str(tmp_path / f"f{i:02d}.tif"), img)
    src = open_source(tmp_path)
    assert len(src) == 3 and src.info.bit_depth == 16
    assert src.frame(2)[0, 0] == pytest.approx(3000 / 65535, rel=1e-6)


@pytest.mark.parametrize("view", list(render.View))
def test_every_view_renders(view):
    d = np.random.default_rng(0).normal(0, 0.2, (2, 30, 40)).astype(np.float32)
    d[:, 0, 0] = np.nan
    field = np.random.default_rng(1).random((30, 40))
    img = render.render(
        view, d, render.ViewSettings(scale=0.5), ref=field, cur=field * 1.01, field=field
    )
    assert img.shape == (30, 40, 3) and img.dtype == np.uint8


def test_exposure_matching_survives_misalignment():
    rng = np.random.default_rng(2)
    ref = (rng.random((120, 160)) > 0.5).astype(np.float32) * 0.6 + 0.2
    cur = np.roll(ref, 3, axis=1) * 1.15  # shifted by shake and brighter
    fixed = preprocess.match_exposure(cur, ref)
    np.testing.assert_allclose(fixed, np.roll(ref, 3, axis=1), atol=1e-5)


def test_auto_stabilisation_leaves_frame_filling_flow_alone():
    """Steady camera, flow everywhere: auto mode must not mistake the flow for motion."""
    st = synth.bench_setup(synth.bench_camera(256, 192), f_number=16, z_d=0.5)
    turb = fields.TurbulentPlume(amplitude=6e-7, extent=(0.04, 0.04))
    cap = synth.Renderer(synth.Scene(turb, st), synth.CameraModel(), seed=5).sequence(4, 1)
    base = ProcessingSettings(reference_range=(0, 4))
    auto = Processor(ArraySource(cap.frames), base)
    r = auto.process_frame(4)
    assert r.motion is None
    assert auto.reference_jitter_px is not None and auto.reference_jitter_px < 0.03
    off = Processor(ArraySource(cap.frames), base.with_(stabilize="off")).process_frame(4)
    np.testing.assert_allclose(r.displacement, off.displacement, atol=1e-5)
    forced = Processor(ArraySource(cap.frames), base.with_(stabilize="on")).process_frame(4)
    assert epe(forced.displacement, cap.truth[4]) > epe(r.displacement, cap.truth[4])
