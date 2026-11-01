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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
        return {"planar": "Planar (known depth)", "axisymmetric": "Axisymmetric (round plume or jet)"}[self.value]


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
    anchor_sides: str = "lr"  # borders where the field is ambient: l, r, t, b
    axis: str | float = "auto"  # axisymmetric: "auto" or the axis position in pixels
    axis_orientation: str = "vertical"  # "vertical" (plume) or "horizontal" (jet)
    abel_method: str = "indirect"  # "indirect" (integrate, then invert P) or "direct" (eps)
    downsample: int = 1
    gas: Gas = field(default_factory=Gas)


@dataclass
class Reconstruction:
    density: np.ndarray  # kg/m^3, NaN where not reconstructable
    temperature: np.ndarray  # K
    delta_n: np.ndarray  # n - n_ambient
    pixel_size_m: float
    settings: ReconstructionSettings
    residual_rms: float
    axis_px: float | None = None
    radius_px: int | None = None
    radial_density: np.ndarray | None = None  # (rows, r) for axisymmetric


def reconstruct(displacement: np.ndarray, setup: Setup, settings: ReconstructionSettings,
                valid: np.ndarray | None = None, sigma_px: np.ndarray | None = None,
                device=None) -> Reconstruction:
    """Reconstruct density and temperature from a (2, H, W) displacement field in pixels."""
    s = settings
    d = np.asarray(displacement, dtype=np.float64)
    horizontal = s.axis_orientation == "horizontal" and s.geometry is FlowGeometry.AXISYMMETRIC
    if horizontal:
        # Rotate so the axis is vertical: swap image axes and vector components.
        d = np.stack([d[1].T, d[0].T])
        valid = valid.T if valid is not None else None
        sigma_px = sigma_px.T if sigma_px is not None else None
    k = max(1, int(s.downsample))
    if k > 1:
        d = _pool(d, k)
        valid = _pool(valid[None].astype(np.float64), k)[0] > 0.999 if valid is not None else None
        sigma_px = _pool(sigma_px[None], k)[0] / k if sigma_px is not None else None
    h_m = setup.object_pixel_size_m * k
    eps = d / setup.pixels_per_radian
    sigma_eps = sigma_px / setup.pixels_per_radian if sigma_px is not None else None
    gas = s.gas
    n0 = gas.n_ambient
    anchors = border_anchors(eps.shape[1:], s.anchor_sides)

    if s.geometry is FlowGeometry.PLANAR:
        scale = n0 / (gas.k_gd * s.depth_m)
        res = integrate_gradient(eps[0] * scale, eps[1] * scale, h_m, anchors, gas.rho_ambient,
                                 sigma=None if sigma_eps is None else sigma_eps * scale,
                                 valid=valid, device=device)
        rho = res.phi
        out = _finish(rho, gas, h_m, s, res.residual_rms)
    else:
        res = integrate_gradient(eps[0], eps[1], h_m, anchors, 0.0, sigma=sigma_eps, valid=valid,
                                 device=device)
        proj = np.nan_to_num(res.phi)
        rows_signal = np.abs(eps[0]).mean(axis=1)
        strong = rows_signal > 0.25 * rows_signal.max() if rows_signal.max() > 0 else None
        axis = abel.find_axis(eps[0], strong) if s.axis == "auto" else float(s.axis) / k
        if s.abel_method == "direct":
            folded = abel.fold(eps[0], axis, odd=True)
            f = abel.invert_deflection(folded)
        else:
            folded = abel.fold(proj, axis, odd=False)
            # The projection is zero at the anchors; remove any residual offset at the outer edge.
            folded = folded - folded[:, -1:]
            f = abel.invert_projection(folded, h_m)
        radial_rho = gas.rho_ambient + n0 * f / gas.k_gd
        n_r = f.shape[1]
        cols = np.arange(eps.shape[2])
        rr = np.abs(cols - axis)
        rho = np.full(eps.shape[1:], np.nan)
        inside = rr <= n_r - 1
        rgrid = np.arange(n_r)
        for i in range(rho.shape[0]):
            rho[i, inside] = np.interp(rr[inside], rgrid, radial_rho[i])
        out = _finish(rho, gas, h_m, s, res.residual_rms)
        out.axis_px = axis * k
        out.radius_px = (n_r - 1) * k
        out.radial_density = radial_rho

    if k > 1:
        out.density = _upsample(out.density, k, displacement.shape[1:] if not horizontal else displacement.shape[:0:-1])
        out.temperature = _upsample(out.temperature, k, out.density.shape)
        out.delta_n = _upsample(out.delta_n, k, out.density.shape)
    if horizontal:
        out.density, out.temperature, out.delta_n = out.density.T, out.temperature.T, out.delta_n.T
    return out


def _finish(rho, gas: Gas, h_m, s, residual) -> Reconstruction:
    t = physics.temperature_from_density(rho, gas.rho_ambient, gas.t_ambient_k)
    dn = gas.k_gd * (rho - gas.rho_ambient)
    return Reconstruction(rho, np.asarray(t), dn, h_m, s, residual)


def _pool(a: np.ndarray, k: int) -> np.ndarray:
    c, h, w = a.shape
    hh, ww = h // k, w // k
    return a[:, :hh * k, :ww * k].reshape(c, hh, k, ww, k).mean(axis=(2, 4))


def _upsample(a: np.ndarray, k: int, shape) -> np.ndarray:
    up = np.repeat(np.repeat(a, k, axis=0), k, axis=1)
    out = np.full(shape, np.nan)
    hh, ww = min(shape[0], up.shape[0]), min(shape[1], up.shape[1])
    out[:hh, :ww] = up[:hh, :ww]
    return out
