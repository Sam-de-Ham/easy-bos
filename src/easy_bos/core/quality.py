"""Capture checks: is this footage good enough for BOS, and if not, what to change.

All checks run on two or more flow-free frames of the background as captured. Thresholds come
from the literature where it exists (dot size 2.5-5 px: Raffel 2015, Mercier & Lacassagne 2023;
noise floor targets: docs/research/physics-and-setup.md) and from the synthetic benchmark
otherwise. Every result carries a plain-language explanation and a concrete fix.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from easy_bos.core import preprocess
from easy_bos.core.geometry import Setup


class Status(Enum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    INFO = "info"


@dataclass
class Check:
    key: str
    title: str
    status: Status
    value: str
    detail: str


def _status(value: float, warn: float, fail: float, higher_is_worse: bool = True) -> Status:
    if higher_is_worse:
        return Status.FAIL if value >= fail else Status.WARN if value >= warn else Status.OK
    return Status.FAIL if value <= fail else Status.WARN if value <= warn else Status.OK


def check_exposure(frame: np.ndarray) -> Check:
    clipped_hi = float((frame >= 0.995).mean())
    clipped_lo = float((frame <= 0.005).mean())
    p1, p99 = np.percentile(frame, [1, 99])
    span = float(p99 - p1)
    clip = clipped_hi + clipped_lo
    st = _status(clip, 0.005, 0.03)
    if span < 0.2:
        st = Status.FAIL if span < 0.1 else max(st, Status.WARN, key=_rank)
    if st is Status.OK:
        detail = "Highlights and shadows are in range and the pattern uses enough tonal range."
    elif clip >= 0.005:
        detail = (
            "Parts of the background are clipped. Clipped pixels carry no displacement "
            "information. Lower the exposure, ISO or the screen brightness."
        )
    else:
        detail = (
            "The pattern uses little of the tonal range: low contrast raises noise. Increase "
            "exposure or screen brightness, or reduce glare on the background."
        )
    return Check(
        "exposure", "Exposure", st, f"{clip * 100:.2f}% clipped · range {span * 100:.0f}%", detail
    )


def feature_size_px(frame: np.ndarray) -> float:
    """Characteristic element size from the autocorrelation width.

    A random binary texture of element size e has a tent-shaped autocorrelation that falls to
    one half at a lag of e/2 along each axis, so e = 2 x (half-maximum lag). Illumination
    gradients are removed first with a broad high-pass; the lag is interpolated linearly.
    """
    from scipy.ndimage import gaussian_filter

    f = frame.astype(np.float64)
    f = f - gaussian_filter(f, 16)
    spec = np.abs(np.fft.rfft2(f)) ** 2
    ac = np.fft.irfft2(spec, s=f.shape)
    ac /= ac[0, 0]
    sizes = []
    for profile in (ac[0, :32], ac[:32, 0]):
        below = np.nonzero(profile < 0.5)[0]
        if below.size == 0:
            sizes.append(64.0)
            continue
        k = int(below[0])
        a0, a1 = profile[k - 1], profile[k]
        lag = (k - 1) + (a0 - 0.5) / max(a0 - a1, 1e-12)
        sizes.append(2.0 * lag)
    return float(np.mean(sizes))


def check_sampling(frame: np.ndarray) -> Check:
    size = feature_size_px(frame)
    if size < 1.8:
        st, detail = (
            Status.FAIL,
            (
                "Pattern features are smaller than two pixels: they alias and "
                "cause moiré and peak-locking. Use a coarser pattern, move "
                "closer to the background or use a longer focal length."
            ),
        )
    elif size < 2.5:
        st, detail = (
            Status.WARN,
            (
                "Features are slightly small. 3-4 px per element is the sweet "
                "spot; aliasing and compression hit fine textures first."
            ),
        )
    elif size <= 6.0:
        st, detail = Status.OK, "Pattern elements are 2.5-6 px on the sensor, ideal for BOS."
    elif size <= 10.0:
        st, detail = (
            Status.WARN,
            (
                "Features are coarse: fewer textured pixels means lower "
                "resolution and more noise. Use a finer pattern."
            ),
        )
    else:
        st, detail = (
            Status.FAIL,
            (
                "The texture is far too coarse (or out of focus). Use a "
                "finer pattern or focus on the background."
            ),
        )
    return Check("sampling", "Pattern scale", st, f"≈ {size:.1f} px per element", detail)


def texture_coverage(frame: np.ndarray, tile: int = 64) -> tuple[float, np.ndarray]:
    """Fraction of tiles with usable texture, and the per-tile contrast map."""
    h, w = frame.shape
    ny, nx = max(1, h // tile), max(1, w // tile)
    g = np.hypot(*np.gradient(frame.astype(np.float64)))
    tiles = g[: ny * tile, : nx * tile].reshape(ny, tile, nx, tile).mean(axis=(1, 3))
    ref = np.percentile(tiles, 90)
    rel = tiles / max(ref, 1e-12)
    return float((rel > 0.35).mean()), rel


def check_focus(frame: np.ndarray) -> Check:
    cov, _ = texture_coverage(frame)
    st = _status(cov, 0.85, 0.6, higher_is_worse=False)
    detail = (
        "The background is sharp and textured across the frame."
        if st is Status.OK
        else "Parts of the frame have little texture: the background is out of focus there, "
        "does not fill the frame, or is too dark. Focus manually on the background and "
        "make sure the pattern covers the area behind the flow."
    )
    return Check("focus", "Focus and coverage", st, f"{cov * 100:.0f}% of frame textured", detail)


def check_noise_floor(frames: list[np.ndarray], setup: Setup | None = None) -> tuple[Check, Check]:
    """Displacement noise between flow-free frames, and camera stability."""
    from easy_bos.core.flow import create

    ref = frames[0]
    lk = create("lucas_kanade", {"window_sigma": 2.0})
    noise, motions, nonrigid = [], [], []
    for cur in frames[1:4]:
        d = lk.estimate(ref, cur).displacement
        m = preprocess.fit_motion(d, model="rigid")
        resid = d - m.field(*d.shape[1:])
        inner = resid[:, 16:-16, 16:-16]
        noise.append(float(np.sqrt(np.mean(inner**2))))
        motions.append(np.hypot(*m.translation))
        aff = preprocess.fit_motion(d, model="affine")
        nonrigid.append(float(np.abs(aff.matrix[:, :2] - m.matrix[:, :2]).max() * max(d.shape[1:])))
    n = float(np.median(noise))
    st = _status(n, 0.02, 0.06)
    value = f"{n:.3f} px rms"
    if setup is not None:
        mrad = n / setup.pixels_per_radian * 1e3
        d_t = _min_detectable_dt(setup, n)
        value += f" · {mrad * 1e3:.1f} µrad · ≈{d_t:.1f} K plume"
    detail = {
        Status.OK: "The background is steady and the camera noise is low.",
        Status.WARN: (
            "Noticeable noise between still frames. Lower ISO, more light, a higher "
            "bitrate or RAW, and averaging more reference frames all help."
        ),
        Status.FAIL: (
            "High noise between still frames: weak flows will drown. Check for "
            "vibration, air currents, low light, heavy compression or a flickering "
            "display."
        ),
    }[st]
    noise_check = Check("noise", "Noise floor", st, value, detail)

    motion = float(np.max(motions))
    warp = float(np.max(nonrigid))
    if warp > 0.3:
        mst, mdetail = (
            Status.FAIL,
            (
                "Frames are warped, not just shifted: this is the signature of "
                "electronic/optical stabilisation or a moving lens. Turn off "
                "all image stabilisation (IBIS, OIS, EIS) and use a tripod."
            ),
        )
    elif motion > 0.5:
        mst, mdetail = (
            Status.WARN,
            (
                "The camera moves between frames. Stabilisation in "
                "processing will compensate, but a sturdier mount or remote "
                "trigger is better."
            ),
        )
    else:
        mst, mdetail = Status.OK, "The camera is steady."
    return noise_check, Check(
        "stability", "Camera stability", mst, f"{motion:.2f} px shift · {warp:.2f} px warp", mdetail
    )


def _min_detectable_dt(setup: Setup, noise_px: float) -> float:
    """Smallest Gaussian-plume temperature rise whose peak displacement is 3x the noise."""
    lo, hi = 0.0, 2000.0
    target = 3.0 * noise_px
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if setup.displacement_for_temperature_rise(mid) < target:
            lo = mid
        else:
            hi = mid
    return hi


def check_banding(frames: list[np.ndarray]) -> Check:
    ref = frames[0]
    strengths = []
    for f in frames[1:4]:
        aligned = preprocess.warp_reference(ref, preprocess.estimate_motion(ref, f))
        strengths.append(preprocess.banding_strength(f, aligned))
    b = max(strengths)
    st = _status(b, 0.003, 0.01)
    detail = (
        "No flicker bands."
        if st is Status.OK
        else "Horizontal bands change from frame to frame: a display or room light flickers "
        "against the rolling shutter. Set the display to full brightness, turn off "
        "flickering lights, and use a shutter time that is a multiple of the mains period "
        "(1/50 s or 1/100 s at 50 Hz; 1/60 s or 1/120 s at 60 Hz). Banding correction in "
        "processing removes what is left."
    )
    return Check("banding", "Flicker banding", st, f"{b * 100:.2f}% row modulation", detail)


def check_moire(frame: np.ndarray) -> Check:
    f = frame - frame.mean()
    h, w = f.shape
    win = np.outer(np.hanning(h), np.hanning(w))
    p = np.abs(np.fft.rfft2(f * win)) ** 2
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    fr = np.hypot(fy, fx)
    band = (fr > 0.02) & (fr < 0.5)
    vals = p[band]
    # Random textures have smooth spectra; periodic structure (display grid aliasing,
    # moiré) shows as isolated peaks far above the local spectral level.
    from scipy.ndimage import median_filter

    local = median_filter(p, size=9)
    ratio = (p / np.maximum(local, 1e-30))[band]
    peak = float(np.percentile(ratio, 99.99)) if vals.size else 0.0
    st = _status(peak, 40.0, 150.0)
    detail = (
        "No periodic interference."
        if st is Status.OK
        else "Strong periodic structure detected, typically moiré between the display's pixel "
        "grid and the camera sensor. Change the distance or focal length (see Setup), "
        "use larger pattern blocks, or defocus the background very slightly."
    )
    return Check("moire", "Moiré", st, f"peak {peak:.0f}× local spectrum", detail)


def check_compression(frame: np.ndarray) -> Check:
    """8x8 block-boundary energy ratio (H.264/HEVC/JPEG blocking)."""
    g = np.abs(np.diff(frame.astype(np.float64), axis=1))
    cols = np.arange(g.shape[1])
    on = g[:, (cols % 8) == 7].mean()
    off = g[:, (cols % 8) != 7].mean()
    ratio = float(on / max(off, 1e-12))
    st = _status(ratio, 1.08, 1.25)
    detail = (
        "No visible compression blocking."
        if st is Status.OK
        else "Compression artefacts align to 8-pixel blocks. Use the highest bitrate, an "
        "intra-frame or 10-bit codec, or RAW/stills; a slightly coarser pattern survives "
        "compression better."
    )
    return Check("compression", "Compression", st, f"block ratio {ratio:.2f}", detail)


def _rank(s: Status) -> int:
    return {Status.OK: 0, Status.INFO: 0, Status.WARN: 1, Status.FAIL: 2}[s]


def run_checks(frames: list[np.ndarray], setup: Setup | None = None) -> list[Check]:
    """All capture checks. ``frames`` are >= 2 flow-free frames, float32 in [0, 1]."""
    if not frames:
        return []
    f0 = frames[0]
    out = [
        check_exposure(f0),
        check_sampling(f0),
        check_focus(f0),
        check_moire(f0),
        check_compression(f0),
    ]
    if len(frames) >= 2:
        out[3:3] = list(check_noise_floor(frames, setup))
        out.append(check_banding(frames))
    else:
        out.append(
            Check(
                "noise",
                "Noise floor",
                Status.INFO,
                "needs ≥ 2 frames",
                "Load at least two frames without flow to measure noise and stability.",
            )
        )
    return out


def overall(checks: list[Check]) -> Status:
    worst = max((_rank(c.status) for c in checks), default=0)
    return {0: Status.OK, 1: Status.WARN, 2: Status.FAIL}[worst]
