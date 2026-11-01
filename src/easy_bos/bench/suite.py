"""Synthetic benchmark: every method on every scenario, through the production pipeline.

Each scenario renders reference frames and one distorted frame with exact ground truth
(``synth.py``), runs ``pipeline.Processor`` with the method's default parameters (exactly what
the app does), and scores the displacement. Scenarios with a physical density field are also
reconstructed (planar or axisymmetric) and scored in kelvin, alongside an "oracle" run on the
true displacement that separates optical limits (depth-of-field blur, discretisation) from
estimator error.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field

import numpy as np

from easy_bos.bench import fields, synth
from easy_bos.core.media import ArraySource
from easy_bos.core.patterns import PatternKind, PatternSpec
from easy_bos.core.pipeline import ProcessingSettings, Processor
from easy_bos.core.reconstruct.density import FlowGeometry, ReconstructionSettings, reconstruct

METHODS = ["dis", "farneback", "correlation", "lucas_kanade", "variational", "wavelet", "raft"]
MARGIN = 16


@dataclass
class Scenario:
    key: str
    title: str
    description: str
    make_field: Callable[[], fields.Field]
    camera: synth.CameraModel = field(default_factory=synth.CameraModel)
    effects: synth.CaptureEffects = field(default_factory=synth.CaptureEffects)
    pattern: PatternSpec = field(default_factory=lambda: PatternSpec(PatternKind.SPECKLE, 3))
    element_cam_px: float = 3.0
    f_number: float = 8.0
    z_a: float = 1.0
    z_d: float = 1.0
    n_ref: int = 4
    settings: dict = field(default_factory=dict)  # ProcessingSettings overrides
    reconstruction: ReconstructionSettings | None = None
    seed: int = 1


def _plume(**kw):
    def make():
        return fields.AxisymmetricPlume(
            **{"delta_rho0": 0.3, "sigma0": 0.005, "spread": 0.06, "y_source": 0.02, **kw}
        )

    return make


RAW = synth.CameraModel(full_well_e=20000, read_noise_e=3, bit_depth=14)
PHONE = synth.CameraModel(
    full_well_e=6000, read_noise_e=2.5, bit_depth=8, transfer="bt709", psf_sigma_px=0.8
)
CAM10 = synth.CameraModel(full_well_e=12000, read_noise_e=3, bit_depth=10, transfer="bt709")
AXI = ReconstructionSettings(FlowGeometry.AXISYMMETRIC, anchor_sides="lr")


def scenarios() -> list[Scenario]:
    return [
        Scenario(
            "plume-ideal",
            "Plume, ideal camera",
            "Axisymmetric plume (~+75 K), noise-free 16-bit sensor: algorithmic accuracy.",
            _plume(),
            camera=synth.CameraModel.ideal(),
            reconstruction=AXI,
        ),
        Scenario(
            "plume-raw",
            "Plume, RAW stills",
            "14-bit linear sensor with shot/read noise and PRNU, 4 averaged references.",
            _plume(),
            camera=RAW,
            reconstruction=AXI,
        ),
        Scenario(
            "plume-h264",
            "Plume, phone H.264",
            "8-bit BT.709 video, x264 CRF 20, phone-like noise and softer optics.",
            _plume(),
            camera=PHONE,
            effects=synth.CaptureEffects(codec="h264", crf=20),
            reconstruction=AXI,
        ),
        Scenario(
            "plume-hevc10",
            "Plume, 10-bit HEVC",
            "10-bit BT.709 video, x265 CRF 22 (mirrorless S-log/HLG-style capture).",
            _plume(),
            camera=CAM10,
            effects=synth.CaptureEffects(codec="hevc10", crf=22),
            reconstruction=AXI,
        ),
        Scenario(
            "breath",
            "Weak disturbance (breath), RAW",
            "~+8 K plume, peak displacement ~0.04 px, 14-bit stills, 16 averaged references: "
            "the noise floor.",
            _plume(delta_rho0=0.03),
            camera=RAW,
            n_ref=16,
        ),
        Scenario(
            "breath-h264",
            "Weak disturbance (breath), H.264",
            "The same ~+8 K plume through 8-bit H.264 at CRF 18: how much a consumer codec "
            "removes.",
            _plume(delta_rho0=0.03),
            camera=PHONE,
            effects=synth.CaptureEffects(codec="h264", crf=18),
            n_ref=16,
        ),
        Scenario(
            "turbulent",
            "Turbulent multi-scale field",
            "Random eddies 2-12 mm at f/16 and short Z_D (little blur): spatial resolution.",
            lambda: fields.TurbulentPlume(amplitude=6e-7),
            camera=RAW,
            f_number=16,
            z_d=0.5,
        ),
        Scenario(
            "lens",
            "Weak lens (calibration object)",
            "Thin lens F = 5 m placed 10 cm in front of the background: radial displacement "
            "linear in radius, up to ~2 px.",
            lambda: fields.WeakLens(focal_length=5.0, radius=0.025),
            camera=RAW,
            f_number=16,
            z_a=1.9,
            z_d=0.1,
        ),
        Scenario(
            "plate",
            "Heated vertical plate (planar)",
            "2-D natural-convection boundary layer, 15 cm deep: planar reconstruction.",
            lambda: fields.HeatedPlate(delta_rho=0.25, x_wall=-0.015, y_leading_edge=0.025),
            camera=RAW,
            f_number=16,
            z_d=0.5,
            reconstruction=ReconstructionSettings(
                FlowGeometry.PLANAR, depth_m=0.15, anchor_sides="r"
            ),
        ),
        Scenario(
            "shake",
            "Camera shake",
            "Plume with 0.8 px rms shake and 0.05 deg rotation per frame; stabilisation on.",
            _plume(),
            camera=RAW,
            effects=synth.CaptureEffects(shake_px=0.8, shake_rot_deg=0.05),
            reconstruction=AXI,
        ),
        Scenario(
            "rolling-shutter",
            "Rolling shutter, unsteady plume",
            "Plume swaying at 2 Hz (4 mm), sensor read out over 90 % of the frame interval at "
            "30 fps; truth is the field at mid-readout.",
            _plume(sway_amplitude=0.004, sway_period=0.5),
            camera=RAW,
            effects=synth.CaptureEffects(rolling_shutter=0.9),
        ),
        Scenario(
            "banding",
            "Display flicker banding",
            "3 % PWM row banding seen through a rolling shutter; banding correction on.",
            _plume(),
            camera=RAW,
            effects=synth.CaptureEffects(banding_amplitude=0.03),
            settings={"banding_correction": True},
        ),
        Scenario(
            "moire",
            "Display moiré",
            "Background on a display whose pixel grid images to 1.4 camera px at f/4.",
            _plume(),
            camera=synth.CameraModel(psf_sigma_px=0.45),
            effects=synth.CaptureEffects(display_pitch_px=1.4),
            element_cam_px=2.8,
            f_number=4,
        ),
        Scenario(
            "dots",
            "Random-dot background",
            "Same RAW plume with a sparse random-dot pattern instead of speckle.",
            _plume(),
            camera=RAW,
            pattern=PatternSpec(PatternKind.RANDOM_DOTS, 3),
        ),
        Scenario(
            "multiscale",
            "Multi-scale noise background",
            "Same RAW plume with a multi-scale noise pattern.",
            _plume(),
            camera=RAW,
            pattern=PatternSpec(PatternKind.MULTISCALE_NOISE, 2),
            element_cam_px=1.5,
        ),
    ]


@dataclass
class MethodScore:
    scenario: str
    method: str
    epe: float
    rmse: float
    p95: float
    relative: float  # mean EPE / mean |truth|
    peak_ratio: float  # 99th percentile |D| estimate / truth
    seconds: float
    t_rmse_k: float | None = None
    t_peak_err_k: float | None = None
    error: str | None = None


def _setup(sc: Scenario, size: tuple[int, int]):
    cam = synth.bench_camera(*size)
    return synth.bench_setup(cam, z_a=sc.z_a, z_d=sc.z_d, f_number=sc.f_number)


def render_scenario(sc: Scenario, size: tuple[int, int] = (512, 384)):
    setup = _setup(sc, size)
    scene = synth.Scene(sc.make_field(), setup, sc.pattern, sc.element_cam_px)
    cap = synth.Renderer(scene, sc.camera, sc.effects, seed=sc.seed).sequence(sc.n_ref, 1)
    return setup, scene, cap


def _true_temperature(fld, setup, shape):
    h, w = shape
    hp = setup.object_pixel_size_m
    ys, xs = np.mgrid[0:h, 0:w]
    xo = (xs - (w - 1) / 2) * hp
    yo = (ys - (h - 1) / 2) * hp
    if fld.geometry == "axisymmetric":
        return fld.temperature(np.abs(xo - fld.x_axis), yo)
    return fld.temperature(xo, yo)


def _temperature_error(d, sc, setup, scene, truth_t):
    rec = reconstruct(d, setup, sc.reconstruction)
    m = MARGIN
    e = (rec.temperature - truth_t)[m:-m, m:-m]
    rise = truth_t[m:-m, m:-m] - scene.field.t_ambient
    hot = rise > 0.2 * rise.max()
    t_rmse = float(np.sqrt(np.nanmean(e[hot] ** 2)))
    peak_est = np.nanpercentile(rec.temperature[m:-m, m:-m][hot], 99)
    peak_true = np.percentile(truth_t[m:-m, m:-m][hot], 99)
    return t_rmse, float(peak_est - peak_true)


def score(d: np.ndarray, truth: np.ndarray) -> dict:
    m = MARGIN
    dd, tt = d[:, m:-m, m:-m], truth[:, m:-m, m:-m]
    err = np.sqrt(((dd - tt) ** 2).sum(0))
    mag_t = np.sqrt((tt**2).sum(0))
    mag_e = np.sqrt((np.nan_to_num(dd) ** 2).sum(0))
    return {
        "epe": float(np.nanmean(err)),
        "rmse": float(np.sqrt(np.nanmean(err**2))),
        "p95": float(np.nanpercentile(err, 95)),
        "relative": float(np.nanmean(err) / max(mag_t.mean(), 1e-9)),
        "peak_ratio": float(np.percentile(mag_e, 99) / max(np.percentile(mag_t, 99), 1e-9)),
    }


def run(
    methods: list[str] | None = None,
    scenario_keys: list[str] | None = None,
    size: tuple[int, int] = (512, 384),
    params: dict[str, dict] | None = None,
    log: Callable[[str], None] = print,
) -> list[MethodScore]:
    methods = methods or METHODS
    params = params or {}
    out: list[MethodScore] = []
    for sc in scenarios():
        if scenario_keys and sc.key not in scenario_keys:
            continue
        setup, scene, cap = render_scenario(sc, size)
        idx = cap.flow_frames[0]
        truth = cap.truth[idx]
        truth_t = None
        if sc.reconstruction is not None:
            truth_t = _true_temperature(scene.field, setup, truth.shape[1:])
            o_rmse, o_peak = _temperature_error(truth, sc, setup, scene, truth_t)
            out.append(MethodScore(sc.key, "oracle", 0, 0, 0, 0, 1, 0, o_rmse, o_peak))
        for key in methods:
            settings = ProcessingSettings(
                method=key,
                method_params=params.get(key, {}),
                reference_range=(0, sc.n_ref),
                **sc.settings,
            )
            try:
                proc = Processor(ArraySource(cap.frames), settings)
                proc.reference  # noqa: B018 - load outside the timed section
                t0 = time.perf_counter()
                res = proc.process_frame(idx)
                secs = time.perf_counter() - t0
                s = score(res.displacement, truth)
                ms = MethodScore(sc.key, key, seconds=secs, **s)
                if truth_t is not None:
                    ms.t_rmse_k, ms.t_peak_err_k = _temperature_error(
                        res.displacement, sc, setup, scene, truth_t
                    )
            except Exception as exc:  # report and continue: one failure must not hide the rest
                ms = MethodScore(sc.key, key, *(np.nan,) * 6, error=repr(exc))
            out.append(ms)
            log(
                f"{sc.key:13s} {key:13s} EPE {ms.epe:.4f}  rel {ms.relative:.3f}  "
                f"peak {ms.peak_ratio:.2f}  {ms.seconds:.2f}s"
                + (f"  T {ms.t_rmse_k:.2f} K" if ms.t_rmse_k is not None else "")
            )
    return out


def save(results: list[MethodScore], path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in results], f, indent=1)
