"""Visualisations of displacement fields and reconstructions.

All functions take a (2, H, W) displacement in pixels (NaN = invalid) and return float data or
RGB uint8 images ready for display/export. Scales are explicit (``scale`` in pixels for full
saturation) so a sequence rendered with a fixed scale stays comparable frame to frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from matplotlib import colormaps


class View(Enum):
    KNIFE_EDGE = "knife_edge"
    MAGNITUDE = "magnitude"
    DIRECTION = "direction"
    COLOUR_SCHLIEREN = "colour_schlieren"
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    DIVERGENCE = "divergence"
    DIFFERENCE = "difference"
    DENSITY = "density"
    TEMPERATURE = "temperature"

    @property
    def label(self) -> str:
        return {
            "knife_edge": "Knife-edge schlieren",
            "magnitude": "Deflection magnitude",
            "direction": "Deflection direction",
            "colour_schlieren": "Colour schlieren",
            "horizontal": "Horizontal component",
            "vertical": "Vertical component",
            "divergence": "Shadowgraph (divergence)",
            "difference": "Quick look (difference)",
            "density": "Density",
            "temperature": "Temperature",
        }[self.value]

    @property
    def needs_displacement(self) -> bool:
        return self not in (View.DIFFERENCE,)

    @property
    def needs_reconstruction(self) -> bool:
        return self in (View.DENSITY, View.TEMPERATURE)


@dataclass
class ViewSettings:
    scale: float = 0.5  # px for full colour / contrast (auto if <= 0)
    knife_angle_deg: float = 0.0  # 0 = vertical knife edge (horizontal gradients)
    colormap: str = "auto"
    background: float = 0.5  # grey level of knife-edge view at zero deflection


def auto_scale(values: np.ndarray, percentile: float = 99.5) -> float:
    v = np.abs(values[np.isfinite(values)])
    if v.size == 0:
        return 1.0
    s = float(np.percentile(v, percentile))
    return s if s > 0 else 1.0


def knife_edge(
    d: np.ndarray, angle_deg: float, scale: float, background: float = 0.5
) -> np.ndarray:
    """Classic Z-type schlieren look: grey + displacement component normal to the knife edge."""
    a = np.radians(angle_deg)
    comp = d[0] * np.cos(a) + d[1] * np.sin(a)
    img = background + 0.5 * comp / scale
    return np.clip(np.nan_to_num(img, nan=background), 0.0, 1.0)


def magnitude(d: np.ndarray) -> np.ndarray:
    return np.hypot(d[0], d[1])


def direction_rgb(d: np.ndarray, scale: float, white_background: bool = False) -> np.ndarray:
    """Hue = direction, saturation (or value) = magnitude: the standard colour-wheel coding,
    equivalent to colour (rainbow) schlieren with a circular colour filter."""
    ang = (np.arctan2(-d[1], d[0]) / (2 * np.pi)) % 1.0  # 0 = right, counter-clockwise
    mag = np.clip(np.nan_to_num(magnitude(d)) / scale, 0.0, 1.0)
    rgb = colormaps["hsv"](np.nan_to_num(ang))[..., :3]
    if white_background:
        out = 1.0 - mag[..., None] * (1.0 - rgb)
    else:
        out = rgb * mag[..., None]
    out[~np.isfinite(d[0])] = 0.0
    return (np.clip(out, 0, 1) * 255).astype(np.uint8)


def divergence(d: np.ndarray) -> np.ndarray:
    """div D: proportional to the Laplacian of the projected index (shadowgraph-like)."""
    return np.gradient(d[0], axis=1) + np.gradient(d[1], axis=0)


def colorize(values: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
    norm = (values - vmin) / max(vmax - vmin, 1e-30)
    rgb = colormaps[cmap](np.clip(np.nan_to_num(norm, nan=0.0), 0, 1))[..., :3]
    rgb[~np.isfinite(values)] = 0.12  # neutral dark for invalid
    return (rgb * 255).astype(np.uint8)


DEFAULT_CMAPS = {
    View.KNIFE_EDGE: "gray",
    View.MAGNITUDE: "magma",
    View.HORIZONTAL: "RdBu_r",
    View.VERTICAL: "RdBu_r",
    View.DIVERGENCE: "gray",
    View.DIFFERENCE: "gray",
    View.DENSITY: "viridis",
    View.TEMPERATURE: "inferno",
}

COLORMAPS = (
    "gray",
    "magma",
    "inferno",
    "viridis",
    "cividis",
    "RdBu_r",
    "coolwarm",
    "PuOr",
    "turbo",
)


def render(
    view: View,
    d: np.ndarray | None,
    settings: ViewSettings,
    ref: np.ndarray | None = None,
    cur: np.ndarray | None = None,
    field: np.ndarray | None = None,
    field_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Render a view to an (H, W, 3) uint8 image. ``field`` holds density/temperature data."""
    s = settings
    cmap = DEFAULT_CMAPS.get(view, "gray") if s.colormap == "auto" else s.colormap
    if view is View.DIFFERENCE:
        diff = (cur - ref) if (cur is not None and ref is not None) else np.zeros((2, 2))
        sc = s.scale if s.scale > 0 else auto_scale(diff)
        return colorize(diff, cmap, -sc, sc)
    if view in (View.DENSITY, View.TEMPERATURE):
        lo, hi = field_range if field_range else (np.nanmin(field), np.nanmax(field))
        return colorize(field, cmap, lo, hi)
    assert d is not None
    if view in (View.DIRECTION, View.COLOUR_SCHLIEREN):
        sc = s.scale if s.scale > 0 else auto_scale(magnitude(d))
        return direction_rgb(d, sc, white_background=view is View.COLOUR_SCHLIEREN)
    if view is View.KNIFE_EDGE:
        a = np.radians(s.knife_angle_deg)
        sc = s.scale if s.scale > 0 else auto_scale(d[0] * np.cos(a) + d[1] * np.sin(a))
        g = knife_edge(d, s.knife_angle_deg, sc, s.background)
        return colorize(g, cmap, 0.0, 1.0)
    if view is View.MAGNITUDE:
        m = magnitude(d)
        sc = s.scale if s.scale > 0 else auto_scale(m)
        return colorize(m, cmap, 0.0, sc)
    if view is View.DIVERGENCE:
        dv = divergence(d)
        sc = s.scale if s.scale > 0 else auto_scale(dv)
        return colorize(dv, cmap, -sc, sc)
    comp = d[0] if view is View.HORIZONTAL else d[1]
    sc = s.scale if s.scale > 0 else auto_scale(comp)
    return colorize(comp, cmap, -sc, sc)


def scalar_field(view: View, d: np.ndarray) -> np.ndarray | None:
    """The scalar behind a view (for probes, profiles and colour bars)."""
    if view is View.MAGNITUDE:
        return magnitude(d)
    if view is View.HORIZONTAL or view is View.KNIFE_EDGE:
        return d[0]
    if view is View.VERTICAL:
        return d[1]
    if view is View.DIVERGENCE:
        return divergence(d)
    return None
