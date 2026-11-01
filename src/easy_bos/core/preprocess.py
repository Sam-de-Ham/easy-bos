"""Corrections applied before displacement estimation.

* Reference: average of flow-free frames (noise falls as 1/sqrt(N)), or a temporal median when
  the flow is intermittent, or the previous frame for frame-to-frame (turbulence) views.
* Exposure: global gain/offset matching, then optional row-gain correction for PWM/mains flicker
  banding seen through a rolling shutter (horizontal bands that move between frames).
* Camera motion: a global affine model fitted robustly (Tukey IRLS) to a fast displacement
  estimate, restricted to a user-chosen stable region if given. The reference is pre-warped by
  the fitted motion, so the main estimator measures only the refractive displacement while D
  stays on the grid of the current frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from easy_bos.core import imops
from easy_bos.core.spline import SplineImage


def average_reference(frames: list[np.ndarray] | np.ndarray, robust: bool = False) -> np.ndarray:
    stack = np.stack(list(frames)).astype(np.float32)
    if robust:
        return np.median(stack, axis=0)
    return stack.mean(axis=0)


def match_exposure(cur: np.ndarray, ref: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Scale cur so its mean intensity matches ref's (auto-exposure or brightness drift).

    Exposure, ISO and display-brightness changes are multiplicative. The mean is the one
    statistic that is unaffected by both noise and pixel alignment: spreads or percentiles
    differ between an averaged (quiet) reference and a single noisy frame, and per-pixel
    regressions collapse as soon as the camera shakes. Local gain left over (vignetting
    changes, flicker bands) is handled elsewhere.
    """
    sel = (mask > 0) if mask is not None else slice(None)
    gain = float(
        np.mean(ref[sel], dtype=np.float64) / max(np.mean(cur[sel], dtype=np.float64), 1e-9)
    )
    return (cur * gain).astype(np.float32)


def row_gain_profile(
    cur: np.ndarray, ref: np.ndarray, smooth_rows: float = 1.5, border: int = 8
) -> np.ndarray:
    """Per-row multiplicative gain of ``cur`` relative to an *aligned* ``ref``.

    Rows are smoothed horizontally (averaging out the texture), divided, and the median taken
    along each row, so regions moved by the flow (a minority of each row) do not bias the gain.
    Rolling-shutter flicker modulates whole rows, which is exactly what this isolates. A
    ``border`` of rows and columns is excluded (a motion-compensated reference is undefined
    there) and the edge rows take the nearest interior gain.
    """
    from scipy.ndimage import uniform_filter1d

    b = max(0, min(border, cur.shape[0] // 4, cur.shape[1] // 4))
    c = uniform_filter1d(cur.astype(np.float64), 15, axis=1)
    r = uniform_filter1d(ref.astype(np.float64), 15, axis=1)
    ratio = c / np.maximum(r, 1e-4)
    if b:
        ratio = ratio[b:-b, b:-b]
    inner = np.median(ratio, axis=1)
    gain = np.pad(inner, (b, b), mode="edge") if b else inner
    gain /= np.median(gain)
    if smooth_rows > 0:
        k = np.exp(-0.5 * (np.arange(-6, 7) / smooth_rows) ** 2)
        k /= k.sum()
        gain = np.convolve(np.pad(gain, 6, mode="edge"), k, mode="valid")
    return gain


def correct_row_banding(cur: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Divide out the row gain profile (``ref`` must be aligned to ``cur``)."""
    return (cur / row_gain_profile(cur, ref)[:, None]).astype(np.float32)


def banding_strength(cur: np.ndarray, ref: np.ndarray) -> float:
    """RMS of the high-passed row gain profile (0 = no banding); ``ref`` aligned to ``cur``.

    The high-pass keeps periodic flicker bands and drops smooth illumination changes.
    """
    g = row_gain_profile(cur, ref, smooth_rows=0.0)
    k = np.ones(31) / 31
    hp = g - np.convolve(np.pad(g, 15, mode="edge"), k, mode="valid")
    return float(np.sqrt(np.mean(hp**2)))


@dataclass
class Motion:
    """Global displacement model d(x) = A [x - cx, y - cy, 1]^T (pixels).

    ``fit_motion`` can constrain A to a translation, a rigid motion (translation + small
    rotation) or a full affine map.
    """

    matrix: np.ndarray  # 2 x 3
    residual_px: float
    inlier_fraction: float

    @property
    def translation(self) -> tuple[float, float]:
        return float(self.matrix[0, 2]), float(self.matrix[1, 2])

    @property
    def rotation_deg(self) -> float:
        return float(np.degrees(0.5 * (self.matrix[1, 0] - self.matrix[0, 1])))

    def field(self, h: int, w: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx -= (w - 1) / 2
        yy -= (h - 1) / 2
        m = self.matrix
        return np.stack(
            [m[0, 0] * xx + m[0, 1] * yy + m[0, 2], m[1, 0] * xx + m[1, 1] * yy + m[1, 2]]
        ).astype(np.float32)


MOTION_MODELS = ("translation", "rigid", "affine")


def _mode_translation(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Dominant mode of the (u, v) distribution by annealed mean shift.

    Flow-free background pixels all share the camera motion, so they form the densest cluster;
    pixels inside a disturbance spread over a range of values. A least-squares (or even Tukey)
    fit blends the two when the disturbance covers much of the frame; the mode does not.
    """
    tu, tv = float(np.median(u[w > 0])), float(np.median(v[w > 0]))
    spread = 1.4826 * float(np.median(np.hypot(u - tu, v - tv)[w > 0])) + 1e-6
    bw = max(spread, 0.02)
    while True:
        for _ in range(10):
            k = w * np.exp(-0.5 * ((u - tu) ** 2 + (v - tv) ** 2) / bw**2)
            sk = k.sum()
            if sk <= 0:
                break
            nu, nv = float((k * u).sum() / sk), float((k * v).sum() / sk)
            done = abs(nu - tu) + abs(nv - tv) < 1e-5
            tu, tv = nu, nv
            if done:
                break
        if bw <= 0.006:
            return tu, tv
        bw *= 0.5


def fit_motion(disp: np.ndarray, weights: np.ndarray | None = None, model: str = "rigid") -> Motion:
    """Fit a global camera-motion model to a (2, H, W) displacement field.

    BOS displacement is curl-free, while a camera rotation has uniform curl 2*theta, so the
    rotation is the median curl / 2 and is not biased by the flow. The translation is the
    dominant mode of the remaining field (the flow-free background), or a weighted mean when a
    stable region is given. An affine model (scale/shear too) is fitted robustly on top; use it
    only with a stable-region mask, because a plume's near-linear core mimics scaling.
    """
    if model not in MOTION_MODELS:
        raise ValueError(f"unknown motion model {model}")
    _, h, w = disp.shape
    step = max(1, int(np.sqrt(h * w / 60000)))
    # A rigid/affine fit needs only a sparse sample of the smooth motion field.
    stride = max(1, int(np.sqrt(h * w / 250000)))
    disp = disp[:, ::stride, ::stride]
    if weights is not None:
        weights = weights[::stride, ::stride]
    _, h, w = disp.shape
    d = disp.astype(np.float64)
    finite = np.isfinite(d[0]) & np.isfinite(d[1])
    wt_full = (np.ones((h, w)) if weights is None else weights.astype(np.float64)) * finite
    d = np.nan_to_num(d)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    xx -= (w - 1) / 2
    yy -= (h - 1) / 2
    stable_given = weights is not None

    theta = 0.0
    if model in ("rigid", "affine"):
        curl = np.gradient(d[1], axis=1) - np.gradient(d[0], axis=0)
        sel = wt_full[1:-1, 1:-1] > 0
        theta = 0.5 * float(np.median(curl[1:-1, 1:-1][sel])) if np.any(sel) else 0.0
    u = (d[0] + theta * yy)[::step, ::step].ravel()
    v = (d[1] - theta * xx)[::step, ::step].ravel()
    wt = wt_full[::step, ::step].ravel()
    if stable_given:
        sw = wt.sum()
        tx, ty = float((wt * u).sum() / sw), float((wt * v).sum() / sw)
    else:
        tx, ty = _mode_translation(u, v, wt)
    m = np.array([[0.0, -theta, tx], [theta, 0.0, ty]])
    if model == "translation":
        u0, v0 = d[0][::step, ::step].ravel(), d[1][::step, ::step].ravel()
        if stable_given:
            tx, ty = float((wt * u0).sum() / wt.sum()), float((wt * v0).sum() / wt.sum())
        else:
            tx, ty = _mode_translation(u0, v0, wt)
        m = np.array([[0.0, 0.0, tx], [0.0, 0.0, ty]])
    elif model == "affine":
        m = _robust_affine(d, wt_full, step, m)
    res = d - Motion(m, 0.0, 1.0).field(h, w)
    r = np.hypot(res[0], res[1])[::step, ::step].ravel()
    inl = (r < 0.05) & (wt > 0)
    rms = float(np.sqrt(np.mean(r[inl] ** 2))) if np.any(inl) else float("nan")
    return Motion(m, rms, float(inl.sum() / max((wt > 0).sum(), 1)))


def _robust_affine(d, wt_full, step, start: np.ndarray, iterations: int = 8) -> np.ndarray:
    _, h, w = d.shape
    yy, xx = np.mgrid[0:h:step, 0:w:step].astype(np.float64)
    xx = (xx - (w - 1) / 2).ravel()
    yy = (yy - (h - 1) / 2).ravel()
    one, zero = np.ones_like(xx), np.zeros_like(xx)
    a = np.stack(
        [
            np.r_[xx, zero],
            np.r_[yy, zero],
            np.r_[one, zero],
            np.r_[zero, xx],
            np.r_[zero, yy],
            np.r_[zero, one],
        ],
        axis=1,
    )
    n = xx.size
    target = np.r_[d[0][::step, ::step].ravel(), d[1][::step, ::step].ravel()]
    wt = wt_full[::step, ::step].ravel()
    p = start.reshape(-1).copy()
    for _ in range(iterations):
        res = target - a @ p
        r = np.hypot(res[:n], res[n:])
        scale = 1.4826 * np.median(r[wt > 0]) + 1e-4
        c = 4.685 * scale
        rob = np.where(r < c, (1 - (r / c) ** 2) ** 2, 0.0) * wt
        sw = np.sqrt(np.r_[rob, rob])
        p = np.linalg.lstsq(a * sw[:, None], target * sw, rcond=None)[0]
    return p.reshape(2, 3)


def fit_affine(disp: np.ndarray, weights: np.ndarray | None = None) -> Motion:
    return fit_motion(disp, weights, "affine")


def align_to(frame: np.ndarray, target: np.ndarray, device=None) -> np.ndarray:
    """Warp ``frame`` onto ``target``'s pixel grid (rigid motion), for reference averaging."""
    m = estimate_motion(target, frame, device=device)
    # frame(x) = target(x + a(x))  =>  frame(x - a(x)) ~ target(x)
    inverse = Motion(-m.matrix, m.residual_px, m.inlier_fraction)
    return warp_reference(frame, inverse, device)


def estimate_motion(
    ref: np.ndarray,
    cur: np.ndarray,
    stable: np.ndarray | None = None,
    model: str = "rigid",
    device=None,
    passes: int = 3,
) -> Motion:
    """Global camera motion between ``ref`` and ``cur``.

    A smooth Lucas–Kanade field is fitted with ``fit_motion``; the reference is then warped by
    the estimate and the residual motion re-estimated, so multi-pixel shake converges to the
    same accuracy as sub-pixel motion.
    """
    from easy_bos.core.flow import create

    # Exposure is matched beforehand, so the cheaper brightness-constancy LK suffices here.
    lk = create(
        "lucas_kanade",
        {"window_sigma": 6.0, "levels": 4, "iterations": 3, "illumination": False},
        device,
    )
    weights = None if stable is None else stable.astype(np.float32)
    total = np.zeros((2, 3))
    m = Motion(total, float("nan"), 0.0)
    for _ in range(max(1, passes)):
        ref_w = ref if not total.any() else warp_reference(ref, Motion(total, 0.0, 1.0), device)
        d = lk.estimate(ref_w, cur).displacement
        m = fit_motion(d, weights, model)
        total = total + m.matrix
        if np.abs(m.matrix[:, 2]).max() < 0.002 and np.abs(m.matrix[:, :2]).max() < 1e-6:
            break
    return Motion(total, m.residual_px, m.inlier_fraction)


def quick_shift_vector(ref: np.ndarray, cur: np.ndarray, factor: int = 4) -> tuple[float, float]:
    """Global (dx, dy) shift between two frames from phase correlation on a ``factor``-times
    reduced copy (a few milliseconds even at 4K; ~0.05 px accurate)."""
    h, w = ref.shape
    size = (max(16, w // factor), max(16, h // factor))
    a = cv2.resize(ref, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    b = cv2.resize(cur, size, interpolation=cv2.INTER_AREA).astype(np.float32)
    win = cv2.createHanningWindow(size, cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(a, b, win)
    return float(dx * factor), float(dy * factor)


def quick_shift(ref: np.ndarray, cur: np.ndarray, factor: int = 4) -> float:
    """Magnitude of ``quick_shift_vector``: used to detect camera bumps cheaply."""
    return float(np.hypot(*quick_shift_vector(ref, cur, factor)))


def warp_reference(ref: np.ndarray, motion: Motion, device=None) -> np.ndarray:
    """ref(x + a(x)) for the fitted motion a, with B-spline interpolation."""
    from easy_bos.core.device import resolve

    dev = resolve(device)
    h, w = ref.shape
    t = imops.as_4d(imops.to_tensor(ref, dev))
    flow = imops.to_tensor(motion.field(h, w), dev)[None]
    with torch.no_grad():
        out = SplineImage(t).warp(flow)
    return out[0, 0].cpu().numpy()


def texture_mask(ref: np.ndarray, threshold: float = 0.08, sigma: float = 4.0) -> np.ndarray:
    """Pixels where the background has enough local contrast to measure displacement."""
    t = torch.from_numpy(np.ascontiguousarray(ref, dtype=np.float32))[None, None]
    gx, gy = imops.central_gradient(t)
    energy = imops.gaussian_blur(gx**2 + gy**2, sigma)[0, 0].numpy()
    norm = energy / max(float(np.percentile(energy, 90)), 1e-12)
    return norm > threshold


def auto_valid(
    ref: np.ndarray,
    cur: np.ndarray,
    texture_threshold: float = 0.08,
    change_ratio: float = 1.43,
    margin_px: int = 6,
) -> np.ndarray:
    """Pixels whose displacement can be trusted: textured background, not occluded or lit.

    * Texture: local gradient energy above ``texture_threshold`` of the frame's 90th
      percentile (rejects bezels, black borders, out-of-focus or blank areas).
    * Unchanged local brightness: a refractive disturbance moves the texture but barely changes
      the local mean, whereas a flame, a hand or an object in front of the background raises or
      lowers it. Areas whose smoothed brightness ratio leaves [1/r, r] are excluded. Strong
      refraction alone can shift a random texture's local mean by up to ~30 % (a 7 px
      boundary layer in the benchmark), so r = 1.43; flames and hands change it far more.

    The rejected area is grown by ``margin_px`` so edge pixels are excluded too.
    """
    full_shape = ref.shape
    # A mask needs no full resolution: work at half size (4x less work) and scale back.
    half = (max(8, ref.shape[1] // 2), max(8, ref.shape[0] // 2))
    r = cv2.resize(ref.astype(np.float32), half, interpolation=cv2.INTER_AREA)
    c = cv2.resize(cur.astype(np.float32), half, interpolation=cv2.INTER_AREA)
    margin_px = max(1, margin_px // 2)
    gx = cv2.Sobel(r, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(r, cv2.CV_32F, 0, 1, ksize=3)
    energy = cv2.GaussianBlur(gx * gx + gy * gy, (0, 0), 2)
    textured = energy > texture_threshold * max(float(np.percentile(energy[::4, ::4], 90)), 1e-12)
    mr = cv2.GaussianBlur(r, (0, 0), 5)
    mc = cv2.GaussianBlur(c, (0, 0), 5)
    ratio = (mc + 1e-3) / (mr + 1e-3)
    steady = (ratio < change_ratio) & (ratio > 1.0 / change_ratio)
    bad = (~(textured & steady)).astype(np.uint8)
    if margin_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin_px + 1, 2 * margin_px + 1))
        bad = cv2.dilate(bad, k)
    bad = cv2.resize(bad, (full_shape[1], full_shape[0]), interpolation=cv2.INTER_NEAREST)
    return bad == 0


def suggest_reference(
    frame, count: int, fps: float | None, length: int = 16, scan_seconds: float = 3.0
) -> tuple[int, int]:
    """Pick the reference frames: the first steady stretch at the start of a clip.

    Pressing the record button shakes a tripod for a second or more, as shifts and, through the
    rolling shutter, shears of 0.3 px or more; the flow is usually started a few seconds
    later. Frame-to-frame motion over the first ``scan_seconds`` is measured as the rms of a
    dense optical flow (DIS) between consecutive quarter-resolution frames, and the window of
    ``length`` frames with the least motion is returned as [start, stop). The shake decays
    over its whole tail, so "the first steady-enough window" lands in it; the quietest window
    does not, and needs no absolute threshold.

    (Frame-difference statistics cannot tell flow onset reliably in compressed video: they
    are dominated by the codec's GOP structure.) ``frame(i)`` returns frame i.
    """
    n_scan = min(count, max(length + 2, round(scan_seconds * (fps or 25.0))))
    if n_scan < 3:
        return 0, max(1, count)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    prev = None
    motion = [np.inf]
    for i in range(n_scan):
        f = frame(i)
        s = cv2.resize(
            f, (max(16, f.shape[1] // 4), max(16, f.shape[0] // 4)), interpolation=cv2.INTER_AREA
        )
        s8 = np.clip(s * 255.0, 0, 255).astype(np.uint8)
        if prev is not None:
            flow = dis.calc(prev, s8, None)[8:-8, 8:-8]
            motion.append(float(np.sqrt(np.mean(flow**2))))
        prev = s8
    m = np.array(motion)
    length = max(2, min(length, n_scan))
    # Window w covers frames [w, w + length): its steps are w+1 .. w+length-1.
    costs = np.array([m[w + 1 : w + length].mean() for w in range(n_scan - length + 1)])
    start = int(np.argmin(costs))
    return start, start + length
