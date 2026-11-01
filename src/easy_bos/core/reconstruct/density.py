"""From displacement to refractive index, density and temperature.

The deflection angle is eps = D / (pixels per radian) (``geometry.Setup``), and

    eps = (1/n0) grad_perp int (n - n_amb) dz = (K / n0) grad_perp int (rho - rho_amb) dz.

* Planar: the disturbance has known depth W and is uniform along the line of sight, so
  grad rho = n0 eps / (K W); integrate by weighted least squares with rho = rho_amb at anchors.
* Axisymmetric: integrate eps to the projection P = (1/n0) int (n - n_amb) dz (anchors at
  P = 0), locate the symmetry axis, fold, and Abel-invert each row with the three-point method
  (the "indirect" route, more robust to noise; Xiong et al. 2020), or invert eps directly.

Temperature follows from the ideal gas at constant pressure and composition:
T = T_amb rho_amb / rho. Gradients use the object-plane pixel size (``Setup.object_pixel_size_m``).

Anchor sides ("l", "r", "t", "b") always refer to the image as displayed, also for a horizontal
symmetry axis. Masked or NaN displacement is excluded from the data; the projection is filled
smoothly through such holes for the Abel inversion and the result is masked again afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np

from easy_bos.core import physics
from easy_bos.core.geometry import Setup
from easy_bos.core.reconstruct import abel
from easy_bos.core.reconstruct.integrate import border_anchors, integrate_gradient


class FlowGeometry(Enum):
    PLANAR = "planar"
    AXISYMMETRIC = "axisymmetric"

    @property
    def label(self) -> str:
        return {
            "planar": "Planar (known depth)",
            "axisymmetric": "Axisymmetric (round plume or jet)",
        }[self.value]


@dataclass
class Gas:
    t_ambient_k: float = 293.15
    p_ambient_pa: float = physics.P_STANDARD
    k_gd: float = physics.GLADSTONE_DALE_AIR

    @property
    def rho_ambient(self) -> float:
        return float(physics.air_density(self.p_ambient_pa, self.t_ambient_k))

    @property
    def n_ambient(self) -> float:
        return 1.0 + self.k_gd * self.rho_ambient


@dataclass
class ReconstructionSettings:
    geometry: FlowGeometry = FlowGeometry.AXISYMMETRIC
    depth_m: float = 0.10  # planar only
    anchor_sides: str = "lr"  # displayed-image borders where the field is ambient
    axis: str | float = "auto"  # axisymmetric: "auto" or the axis position in pixels
    axis_orientation: str = "vertical"  # "vertical" (plume) or "horizontal" (jet)
    abel_method: str = "indirect"  # "indirect" (integrate, then invert P) or "direct" (eps)
    downsample: int = 1
    gas: Gas = field(default_factory=Gas)


@dataclass
class Reconstruction:
    density: np.ndarray  # kg/m^3 at full resolution, NaN where not reconstructable
    temperature: np.ndarray  # K
    delta_n: np.ndarray  # n - n_ambient
    pixel_size_m: float  # object-plane size of one pixel of the arrays above
    settings: ReconstructionSettings
    residual_rms: float  # gradient misfit: kg/m^4 (planar) or rad (axisymmetric)
    axis_px: float | None = None  # column (vertical axis) or row (horizontal axis)
    radius_px: float | None = None
    radial_density: np.ndarray | None = None  # (rows or cols, r) at the processing resolution
    radial_spacing_m: float | None = None
    warnings: list[str] = field(default_factory=list)


_TRANSPOSE_SIDES = str.maketrans("lrtb", "tblr")


def reconstruct(
    displacement: np.ndarray,
    setup: Setup,
    settings: ReconstructionSettings,
    valid: np.ndarray | None = None,
    sigma_px: np.ndarray | None = None,
    device=None,
) -> Reconstruction:
    """Reconstruct density and temperature from a (2, H, W) displacement field in pixels."""
    s = settings
    d = np.asarray(displacement, dtype=np.float64)
    full_shape = d.shape[1:]
    ok = np.isfinite(d[0]) & np.isfinite(d[1])
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    sides = s.anchor_sides
    horizontal = s.axis_orientation == "horizontal" and s.geometry is FlowGeometry.AXISYMMETRIC
    if horizontal:
        # Work with a vertical axis: transpose the image and swap vector components.
        d = np.stack([d[1].T, d[0].T])
        ok = ok.T
        sigma_px = sigma_px.T if sigma_px is not None else None
        sides = sides.translate(_TRANSPOSE_SIDES)
    k = max(1, int(s.downsample))
    if k > 1:
        d = _pool(np.where(ok, d, np.nan), k)
        ok_p = np.isfinite(d[0]) & np.isfinite(d[1])
        if sigma_px is not None:
            sigma_px = np.sqrt(_pool(sigma_px[None] ** 2, k)[0]) / k
    else:
        ok_p = ok
    h_full = setup.object_pixel_size_m
    h_m = h_full * k
    eps = d / setup.pixels_per_radian
    sigma_eps = sigma_px / setup.pixels_per_radian if sigma_px is not None else None
    gas = s.gas
    n0 = gas.n_ambient
    anchors = border_anchors(eps.shape[1:], sides)
    msgs: list[str] = []
    _check_anchors(eps, ok_p, anchors, msgs)

    if s.geometry is FlowGeometry.PLANAR:
        scale = n0 / (gas.k_gd * s.depth_m)
        res = integrate_gradient(
            eps[0] * scale, eps[1] * scale, h_m, anchors, gas.rho_ambient,
            sigma=None if sigma_eps is None else sigma_eps * scale, valid=ok_p, device=device)
        msgs += res.warnings
        rho = _upsample_bilinear(res.filled, k, full_shape if not horizontal else full_shape[::-1])
        out = _finish(rho, gas, h_full, s, res.residual_rms)
    else:
        res = integrate_gradient(eps[0], eps[1], h_m, anchors, 0.0, sigma=sigma_eps, valid=ok_p,
                                 device=device)
        msgs += res.warnings
        proj = res.filled
        ex = np.where(ok_p, eps[0], np.gradient(proj, h_m, axis=1))
        rows_signal = np.abs(ex).mean(axis=1)
        strong = rows_signal > 0.25 * rows_signal.max() if rows_signal.max() > 0 else None
        if s.axis == "auto":
            axis = abel.find_axis(ex, strong)
        else:
            axis = (float(s.axis) - (k - 1) / 2) / k  # full-resolution pixel -> pooled pixel
        if s.abel_method == "direct":
            f = abel.invert_deflection(abel.fold(ex, axis, odd=True))
        else:
            folded = abel.fold(proj, axis, odd=False)
            folded = folded - folded[:, -1:]  # P = 0 at the outer radius
            f = abel.invert_projection(folded, h_m)
        radial_rho = gas.rho_ambient + n0 * f / gas.k_gd
        n_r = f.shape[1]
        _check_axis(ex, axis, n_r, msgs)
        axis_full = axis * k + (k - 1) / 2
        rho = _radial_to_image(radial_rho, axis_full, k, full_shape if not horizontal
                               else full_shape[::-1])
        out = _finish(rho, gas, h_full, s, res.residual_rms)
        out.axis_px = axis_full
        out.radius_px = (n_r - 1) * k
        out.radial_density = radial_rho
        out.radial_spacing_m = h_m

    if horizontal:
        out.density, out.temperature, out.delta_n = out.density.T, out.temperature.T, out.delta_n.T
    mask = np.isfinite(displacement[0]) & np.isfinite(displacement[1])
    if valid is not None:
        mask &= np.asarray(valid, dtype=bool)
    for name in ("density", "temperature", "delta_n"):
        arr = getattr(out, name)
        arr[~mask] = np.nan
    out.warnings = msgs
    return out


def _check_anchors(eps, ok, anchors, msgs) -> None:
    """Warn when the anchored border is not flow-free (the result would be offset)."""
    mag = np.hypot(eps[0], eps[1])
    sel = anchors & ok
    if not sel.any() or not np.any(ok):
        return
    peak = np.nanpercentile(mag[ok], 99)
    at_anchor = np.nanpercentile(mag[sel], 90)
    if peak > 0 and at_anchor > 0.15 * peak:
        msgs.append(
            "The ambient (anchor) border still shows deflection "
            f"({100 * at_anchor / peak:.0f}% of the peak). The flow may cross that edge; choose "
            "borders where the air is undisturbed, or widen the field of view."
        )


def _check_axis(ex, axis, n_r, msgs) -> None:
    """Warn when the folded radius is short compared with the signal width."""
    w = ex.shape[1]
    prof = np.abs(ex).mean(axis=0)
    if prof.max() <= 0:
        return
    cols = np.nonzero(prof > 0.1 * prof.max())[0]
    half_width = max(abs(cols.min() - axis), abs(cols.max() - axis)) if cols.size else 0
    if n_r - 1 < 1.2 * half_width:
        msgs.append(
            "The symmetry axis is close to an image edge: the disturbance extends beyond the "
            "usable radius, so densities near the outer radius are biased. Centre the flow."
        )
    if axis < 0.1 * w or axis > 0.9 * w:
        msgs.append("The detected symmetry axis is near the image edge; check it visually.")


def _finish(rho, gas: Gas, h_m, s, residual) -> Reconstruction:
    t = physics.temperature_from_density(rho, gas.rho_ambient, gas.t_ambient_k)
    dn = gas.k_gd * (rho - gas.rho_ambient)
    return Reconstruction(rho, np.asarray(t, dtype=np.float64), dn, h_m, s, residual)


def _pool(a: np.ndarray, k: int) -> np.ndarray:
    """Block mean ignoring NaN (a block is NaN only if all its pixels are)."""
    c, h, w = a.shape
    hh, ww = h // k, w // k
    blocks = a[:, : hh * k, : ww * k].reshape(c, hh, k, ww, k)
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(blocks).sum(axis=(2, 4))
        tot = np.nansum(blocks, axis=(2, 4))
        return np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan)


def _upsample_bilinear(a: np.ndarray, k: int, shape) -> np.ndarray:
    """Centre-aligned bilinear upsampling of a pooled field to the full-resolution shape."""
    if k == 1:
        return a[: shape[0], : shape[1]].copy()
    up = cv2.resize(a.astype(np.float64), (a.shape[1] * k, a.shape[0] * k),
                    interpolation=cv2.INTER_LINEAR)
    out = np.full(shape, np.nan)
    hh, ww = min(shape[0], up.shape[0]), min(shape[1], up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    if hh < shape[0]:
        out[hh:, :ww] = up[hh - 1 : hh, :ww]
    if ww < shape[1]:
        out[:, ww:] = out[:, ww - 1 : ww]
    return out


def _radial_to_image(radial: np.ndarray, axis_full: float, k: int, shape) -> np.ndarray:
    """Evaluate the radial profiles at every full-resolution pixel.

    Radius in pooled units is |col - axis| / k; rows are interpolated linearly between the
    pooled row centres (ik + (k-1)/2). Beyond the reconstructed radius the result is NaN.
    """
    h, w = shape
    n_rows, n_r = radial.shape
    cols = np.arange(w)
    r = np.abs(cols - axis_full) / k
    inside = r <= n_r - 1
    rgrid = np.arange(n_r)
    by_row = np.full((n_rows, w), np.nan)
    for i in range(n_rows):
        by_row[i, inside] = np.interp(r[inside], rgrid, radial[i])
    centres = np.arange(n_rows) * k + (k - 1) / 2
    rows = np.arange(h)
    pos = np.clip(np.interp(rows, centres, np.arange(n_rows)), 0, n_rows - 1)
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, n_rows - 1)
    t = (pos - i0)[:, None]
    return by_row[i0] * (1 - t) + by_row[i1] * t
