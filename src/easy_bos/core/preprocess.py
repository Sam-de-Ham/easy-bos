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
    """Least-squares gain and offset so that cur matches ref (handles auto-exposure drift)."""
    sel = mask if mask is not None else slice(None)
    c = cur[sel].ravel().astype(np.float64)
    r = ref[sel].ravel().astype(np.float64)
    a = np.vstack([c, np.ones_like(c)]).T
    (gain, offset), *_ = np.linalg.lstsq(a, r, rcond=None)
    return (cur * gain + offset).astype(np.float32)


def correct_row_banding(cur: np.ndarray, ref: np.ndarray, smooth_rows: float = 2.0) -> np.ndarray:
    """Remove horizontal flicker bands: per-row gain from robust row statistics.

    BOS displacements barely change a row's mean intensity, while PWM banding scales whole rows,
    so the ratio of row means isolates the banding (medians are unstable on binary patterns).
    Lightly smoothed across rows.
    """
    rc = cur.mean(axis=1)
    rr = ref.mean(axis=1)
    gain = rr / np.maximum(rc, 1e-6)
    if smooth_rows > 0:
        k = np.exp(-0.5 * (np.arange(-8, 9) / smooth_rows) ** 2)
        k /= k.sum()
        gain = np.convolve(np.pad(gain, 8, mode="edge"), k, mode="valid")
    return (cur * gain[:, None]).astype(np.float32)


def banding_strength(cur: np.ndarray, ref: np.ndarray) -> float:
    """Relative rms of the row-gain profile (0 = no banding). Used by the setup checks."""
    rc = cur.mean(axis=1)
    rr = ref.mean(axis=1)
    g = rc / np.maximum(rr, 1e-6)
    g = g / np.median(g)
    # High-pass across rows: banding is periodic, illumination gradients are smooth.
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


def fit_motion(
    disp: np.ndarray, weights: np.ndarray | None = None, model: str = "rigid", iterations: int = 8
) -> Motion:
    """Robust (Tukey biweight IRLS) fit of a global motion model to a (2, H, W) field.

    The default rigid model (translation + rotation) cannot absorb BOS signal: a BOS field is
    curl-free and, for a disturbance inside the frame, has ~zero mean, which makes it
    orthogonal to both translations and rotations. An affine model also has scale and shear
    terms, which a plume's near-linear core mimics; use it only with a stable-region mask.
    """
    if model not in MOTION_MODELS:
        raise ValueError(f"unknown motion model {model}")
    _, h, w = disp.shape
    step = max(1, int(np.sqrt(h * w / 40000)))
    d = disp[:, ::step, ::step]
    yy, xx = np.mgrid[0:h:step, 0:w:step].astype(np.float64)
    xx = (xx - (w - 1) / 2).ravel()
    yy = (yy - (h - 1) / 2).ravel()
    one, zero = np.ones_like(xx), np.zeros_like(xx)
    # Stacked design for [u; v] = A p
    if model == "translation":
        cols = [np.r_[one, zero], np.r_[zero, one]]
    elif model == "rigid":
        cols = [np.r_[one, zero], np.r_[zero, one], np.r_[-yy, xx]]
    else:
        cols = [
            np.r_[xx, zero],
            np.r_[yy, zero],
            np.r_[one, zero],
            np.r_[zero, xx],
            np.r_[zero, yy],
            np.r_[zero, one],
        ]
    a = np.stack(cols, axis=1)
    target = np.r_[d[0].ravel(), d[1].ravel()].astype(np.float64)
    wt = np.ones(xx.size) if weights is None else weights[::step, ::step].ravel().astype(np.float64)
    finite = np.isfinite(target[: xx.size]) & np.isfinite(target[xx.size :])
    wt = np.where(finite, wt, 0.0)
    target = np.nan_to_num(target)
    # Start from the median translation (the flow-free background is the majority), so the
    # redescending Tukey weights lock onto the background rather than a least-squares blend.
    sel = wt > 0
    n = xx.size
    r = np.hypot(target[:n] - np.median(target[:n][sel]), target[n:] - np.median(target[n:][sel]))
    for _ in range(iterations):
        scale = 1.4826 * np.median(r[sel]) + 1e-4
        c = 4.685 * scale
        robust = np.where(r < c, (1 - (r / c) ** 2) ** 2, 0.0)
        ww = np.r_[wt * robust, wt * robust]
        sw = np.sqrt(ww)
        p = np.linalg.lstsq(a * sw[:, None], target * sw, rcond=None)[0]
        res = target - a @ p
        r = np.hypot(res[:n], res[n:])
    if model == "translation":
        m = np.array([[0, 0, p[0]], [0, 0, p[1]]])
    elif model == "rigid":
        m = np.array([[0, -p[2], p[0]], [p[2], 0, p[1]]])
    else:
        m = np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]]])
    inl = (robust > 0) & (wt > 0)
    res_px = float(np.sqrt(np.mean(r[inl] ** 2))) if np.any(inl) else float("nan")
    return Motion(m, res_px, float(inl.sum() / max((wt > 0).sum(), 1)))


def fit_affine(disp: np.ndarray, weights: np.ndarray | None = None) -> Motion:
    return fit_motion(disp, weights, "affine")


def estimate_motion(
    ref: np.ndarray,
    cur: np.ndarray,
    stable: np.ndarray | None = None,
    model: str = "rigid",
    device=None,
) -> Motion:
    """Global camera motion between ``ref`` and ``cur`` from a smooth Lucas–Kanade field."""
    from easy_bos.core.flow import create

    lk = create("lucas_kanade", {"window_sigma": 6.0, "levels": 4, "iterations": 4}, device)
    d = lk.estimate(ref, cur).displacement
    return fit_motion(d, None if stable is None else stable.astype(np.float32), model)


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
