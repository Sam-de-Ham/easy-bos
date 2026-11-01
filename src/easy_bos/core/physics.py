"""Physical constants and gas relations for air.

All quantities are SI. References are collected in ``docs/research/physics-and-setup.md``.
"""

from __future__ import annotations

import numpy as np

R_UNIVERSAL = 8.31446261815324  # J/(mol K)
M_AIR = 0.028965  # kg/mol, dry air
P_STANDARD = 101325.0  # Pa
T_ZERO_C = 273.15  # K

#: Gladstone–Dale constant of air at visible wavelengths [m^3/kg].
#: Ciddor (1996) dispersion gives 2.290e-4 @ 450 nm, 2.268e-4 @ 550 nm, 2.257e-4 @ 633 nm.
GLADSTONE_DALE_AIR = 2.26e-4


def gladstone_dale_air(wavelength_m: float) -> float:
    """Gladstone–Dale constant of dry air from Ciddor's (1996) dispersion formula.

    Evaluated at the standard state (15 °C, 101325 Pa, 450 ppm CO2) where the formula is
    defined, divided by the density of that state. K is independent of state to first order.
    """
    sigma2 = (1.0 / (wavelength_m * 1e6)) ** 2  # (1/µm)^2
    n_minus_1 = (5792105.0 / (238.0185 - sigma2) + 167917.0 / (57.362 - sigma2)) * 1e-8
    rho_std = air_density(P_STANDARD, T_ZERO_C + 15.0)
    return n_minus_1 / rho_std


def air_density(pressure_pa: float, temperature_k: np.ndarray | float) -> np.ndarray | float:
    """Ideal-gas density of dry air [kg/m^3]."""
    return pressure_pa * M_AIR / (R_UNIVERSAL * np.asarray(temperature_k, dtype=np.float64))


def refractive_index(density: np.ndarray | float, k_gd: float = GLADSTONE_DALE_AIR):
    """Refractive index from density via Gladstone–Dale, n = 1 + K rho."""
    return 1.0 + k_gd * np.asarray(density, dtype=np.float64)


def density_from_index(n: np.ndarray | float, k_gd: float = GLADSTONE_DALE_AIR):
    return (np.asarray(n, dtype=np.float64) - 1.0) / k_gd


def temperature_from_density(density, rho_ambient: float, t_ambient_k: float):
    """Temperature at constant pressure and composition: T = T_amb * rho_amb / rho."""
    rho = np.asarray(density, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = t_ambient_k * rho_ambient / rho
    return np.where(rho > 0, t, np.nan)


def density_from_temperature(temperature_k, rho_ambient: float, t_ambient_k: float):
    return rho_ambient * t_ambient_k / np.asarray(temperature_k, dtype=np.float64)


def celsius(t_k):
    return np.asarray(t_k) - T_ZERO_C


def kelvin(t_c):
    return np.asarray(t_c) + T_ZERO_C
