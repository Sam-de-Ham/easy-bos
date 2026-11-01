"""Analytic refractive-index fields with exact ground truth.

Each field defines its optical path perturbation

    P(x, y, t) = (1/n0) * integral (n - n_amb) dz      [m]

in object-plane coordinates (x right, y down, metres, origin at the image centre). The
deflection angle is its transverse gradient, eps = grad P (paraxial BOS relation, Raffel 2015
Eq. 2), computed exactly with autograd in float64. Fields that correspond to a physical gas
also expose the 3-D density so reconstructions can be checked against the truth.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import torch

from easy_bos.core import physics

N0 = 1.0 + physics.GLADSTONE_DALE_AIR * 1.204


class Field(ABC):
    """A refractive-index disturbance in front of the background."""

    name: str = "field"
    #: "axisymmetric", "planar", "phase" (no 3-D truth) or "lens"
    geometry: str = "phase"

    @abstractmethod
    def path(self, x: torch.Tensor, y: torch.Tensor, t: float) -> torch.Tensor:
        """Optical path perturbation P [m] at object-plane positions (float64 tensors)."""

    def deflection(
        self, x: np.ndarray, y: np.ndarray, t: float = 0.0
    ) -> tuple[np.ndarray, np.ndarray]:
        """Exact deflection angles (eps_x, eps_y) [rad] via autograd of ``path``."""
        xt = torch.as_tensor(np.asarray(x, dtype=np.float64)).requires_grad_(True)
        yt = torch.as_tensor(np.asarray(y, dtype=np.float64)).requires_grad_(True)
        p = self.path(xt, yt, t)
        gx, gy = torch.autograd.grad(p.sum(), (xt, yt), allow_unused=True)
        gx = torch.zeros_like(xt) if gx is None else gx
        gy = torch.zeros_like(yt) if gy is None else gy
        return gx.detach().numpy(), gy.detach().numpy()

    def path_numpy(self, x: np.ndarray, y: np.ndarray, t: float = 0.0) -> np.ndarray:
        with torch.no_grad():
            return self.path(
                torch.as_tensor(np.asarray(x, np.float64)),
                torch.as_tensor(np.asarray(y, np.float64)),
                t,
            ).numpy()


@dataclass
class AxisymmetricPlume(Field):
    """Rising hot plume: Gaussian density deficit in r that widens and weakens with height.

    rho(r, h) = rho_amb - drho(h) exp(-r^2 / 2 sigma(h)^2), h = height above the source.
    Axisymmetric about a vertical axis at x = x_axis (+ optional time-dependent sway, which
    breaks axisymmetry and is used only for video tests).
    """

    delta_rho0: float = 0.25  # kg/m^3 density deficit at the source (~ +100 K)
    sigma0: float = 0.008  # m
    spread: float = 0.12  # d sigma / d h
    decay_length: float = 0.15  # m, amplitude falls as sigma0/sigma(h) * exp(-h/L)
    y_source: float = 0.03  # m, below the image centre
    x_axis: float = 0.0
    sway_amplitude: float = 0.0  # m
    sway_period: float = 1.0  # s
    k_gd: float = physics.GLADSTONE_DALE_AIR
    rho_ambient: float = 1.204
    t_ambient: float = 293.15
    name: str = "Axisymmetric plume"
    geometry: str = "axisymmetric"

    def _height(self, y):
        # Smooth onset below the source so the field stays differentiable.
        h = self.y_source - y
        return torch.nn.functional.softplus(h / 0.004) * 0.004

    def _sigma(self, y):
        return self.sigma0 + self.spread * self._height(y)

    def _amplitude(self, y):
        h_raw = self.y_source - y
        onset = torch.sigmoid(h_raw / 0.004)
        return self.delta_rho0 * onset * torch.exp(-self._height(y) / self.decay_length)

    def _axis(self, y, t):
        if self.sway_amplitude == 0.0:
            return self.x_axis
        h = self._height(y)
        return self.x_axis + self.sway_amplitude * torch.sin(
            2 * math.pi * t / self.sway_period - 20.0 * h
        ) * (h / (h + 0.05))

    def path(self, x, y, t):
        s = self._sigma(y)
        r2 = (x - self._axis(y, t)) ** 2
        # Line-of-sight integral of a 2-D Gaussian: sqrt(2 pi) sigma exp(-x^2 / 2 sigma^2).
        return (
            -(self.k_gd / N0)
            * self._amplitude(y)
            * math.sqrt(2 * math.pi)
            * s
            * torch.exp(-r2 / (2 * s**2))
        )

    def density(self, r: np.ndarray, y: np.ndarray) -> np.ndarray:
        """True density [kg/m^3] at radius r and height coordinate y (static, no sway)."""
        yt = torch.as_tensor(np.asarray(y, np.float64))
        with torch.no_grad():
            s = self._sigma(yt).numpy()
            a = self._amplitude(yt).numpy()
        return self.rho_ambient - a * np.exp(-(np.asarray(r) ** 2) / (2 * s**2))

    def temperature(self, r, y):
        return physics.temperature_from_density(
            self.density(r, y), self.rho_ambient, self.t_ambient
        )


@dataclass
class PlanarBlob(Field):
    """2-D density disturbance of uniform depth W along the line of sight.

    rho(x, y) = rho_amb - drho * exp(-((x-xc)^2/2sx^2 + (y-yc)^2/2sy^2)). Models a slot jet or
    a heated plate boundary layer viewed end-on — the planar reconstruction case.
    """

    delta_rho: float = 0.15
    sigma_x: float = 0.01
    sigma_y: float = 0.02
    xc: float = 0.0
    yc: float = 0.0
    depth: float = 0.10  # W [m]
    k_gd: float = physics.GLADSTONE_DALE_AIR
    rho_ambient: float = 1.204
    t_ambient: float = 293.15
    name: str = "Planar blob"
    geometry: str = "planar"

    def _deficit(self, x, y):
        return self.delta_rho * torch.exp(
            -((x - self.xc) ** 2) / (2 * self.sigma_x**2)
            - (y - self.yc) ** 2 / (2 * self.sigma_y**2)
        )

    def path(self, x, y, t):
        return -(self.k_gd / N0) * self.depth * self._deficit(x, y)

    def density(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            d = self._deficit(
                torch.as_tensor(np.asarray(x, np.float64)),
                torch.as_tensor(np.asarray(y, np.float64)),
            ).numpy()
        return self.rho_ambient - d

    def temperature(self, x, y):
        return physics.temperature_from_density(
            self.density(x, y), self.rho_ambient, self.t_ambient
        )


@dataclass
class HeatedPlate(Field):
    """Natural-convection boundary layer on a vertical heated plate, viewed edge-on.

    Smooth Ostrach-like profile: deficit(x, h) = drho * exp(-x/delta(h)) * (x/delta) shape is
    replaced by a smooth one-sided Gaussian-error profile; thickness grows as h^(1/4).
    Uniform along the line of sight over the plate width W.
    """

    delta_rho: float = 0.20
    x_wall: float = -0.02
    delta0: float = 0.004  # boundary-layer thickness scale at h = h_ref
    h_ref: float = 0.05
    y_leading_edge: float = 0.05
    depth: float = 0.15
    k_gd: float = physics.GLADSTONE_DALE_AIR
    rho_ambient: float = 1.204
    t_ambient: float = 293.15
    name: str = "Heated vertical plate"
    geometry: str = "planar"

    def _deficit(self, x, y):
        h = torch.nn.functional.softplus((self.y_leading_edge - y) / 0.003) * 0.003 + 1e-4
        delta = self.delta0 * (h / self.h_ref) ** 0.25
        xi = torch.nn.functional.softplus((x - self.x_wall) / 0.0005) * 0.0005
        return (
            self.delta_rho
            * torch.exp(-((xi / delta) ** 2))
            * torch.sigmoid((self.y_leading_edge - y) / 0.003)
        )

    def path(self, x, y, t):
        return -(self.k_gd / N0) * self.depth * self._deficit(x, y)

    def density(self, x, y):
        with torch.no_grad():
            d = self._deficit(
                torch.as_tensor(np.asarray(x, np.float64)),
                torch.as_tensor(np.asarray(y, np.float64)),
            ).numpy()
        return self.rho_ambient - d

    def temperature(self, x, y):
        return physics.temperature_from_density(
            self.density(x, y), self.rho_ambient, self.t_ambient
        )


@dataclass
class WeakLens(Field):
    """Thin lens of focal length F centred at (xc, yc): P = -r^2 / 2F, eps = -r / F."""

    focal_length: float = 4.0
    radius: float = 0.03
    xc: float = 0.0
    yc: float = 0.0
    name: str = "Weak lens"
    geometry: str = "lens"

    def path(self, x, y, t):
        r2 = (x - self.xc) ** 2 + (y - self.yc) ** 2
        # Smoothly bounded aperture (C1 at the rim) so the phase screen stays differentiable.
        edge = torch.sigmoid((self.radius**2 - r2) / (0.05 * self.radius**2))
        inside = -r2 / (2 * self.focal_length)
        rim = -(self.radius**2) / (2 * self.focal_length)
        return rim + (inside - rim) * edge


@dataclass
class TurbulentPlume(Field):
    """Multi-scale random field: Gaussian eddies advected upward. No 3-D truth (phase only).

    Used to test spatial resolution and temporal behaviour of displacement estimators.
    """

    n_eddies: int = 60
    seed: int = 3
    amplitude: float = 4e-7  # m of optical path per eddy
    sigma_range: tuple[float, float] = (0.002, 0.012)
    extent: tuple[float, float] = (0.06, 0.06)
    rise_speed: float = 0.2  # m/s
    name: str = "Turbulent plume"
    geometry: str = "phase"

    def __post_init__(self):
        rng = np.random.default_rng(self.seed)
        ex, ey = self.extent
        self._x = rng.uniform(-ex / 2, ex / 2, self.n_eddies)
        self._y = rng.uniform(-ey / 2, ey / 2, self.n_eddies)
        self._s = np.exp(rng.uniform(*np.log(self.sigma_range), self.n_eddies))
        self._a = rng.choice([-1.0, 1.0], self.n_eddies) * rng.uniform(0.4, 1.0, self.n_eddies)

    def path(self, x, y, t):
        ey = self.extent[1]
        out = torch.zeros_like(x * y)
        for xi, yi, si, ai in zip(self._x, self._y, self._s, self._a, strict=True):
            yc = (yi - self.rise_speed * t + ey / 2) % ey - ey / 2
            out = out + ai * self.amplitude * torch.exp(
                -((x - xi) ** 2 + (y - yc) ** 2) / (2 * si**2)
            )
        # Envelope keeps the field inside the frame.
        env = torch.exp(-(x**2 / (2 * (0.4 * self.extent[0]) ** 2) + y**2 / (2 * (0.4 * ey) ** 2)))
        return out * env


@dataclass
class Superposition(Field):
    fields: tuple[Field, ...] = ()
    name: str = "Superposition"
    geometry: str = "phase"

    def path(self, x, y, t):
        return sum((f.path(x, y, t) for f in self.fields), torch.zeros_like(x * y))
