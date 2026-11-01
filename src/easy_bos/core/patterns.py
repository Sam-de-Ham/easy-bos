"""Background pattern generation.

Patterns are produced as linear-intensity float32 arrays in [0, 1]. Sizes are expressed in
*output pixels* (display pixels or printer dots); the finest feature is ``element_px`` output
pixels so that it images to roughly 3 camera pixels (see ``geometry.analyze_display``).

Every pattern is fully determined by its ``PatternSpec`` (including the seed), so a pattern
shown on screen can be regenerated exactly later.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

import numpy as np


class PatternKind(Enum):
    RANDOM_DOTS = "random_dots"
    SPECKLE = "speckle"
    MULTISCALE_NOISE = "multiscale_noise"
    CHECKER = "checker"

    @property
    def label(self) -> str:
        return {
            PatternKind.RANDOM_DOTS: "Random dots",
            PatternKind.SPECKLE: "Binary speckle",
            PatternKind.MULTISCALE_NOISE: "Multi-scale noise",
            PatternKind.CHECKER: "Focus and scale target",
        }[self]

    @property
    def description(self) -> str:
        return {
            PatternKind.RANDOM_DOTS: (
                "Square dots on a jittered grid. The classic BOS background; robust with "
                "cross-correlation and with heavy video compression."
            ),
            PatternKind.SPECKLE: (
                "A random black/white texture at a single scale with 50 % fill. Maximum "
                "information per pixel at the chosen scale. Best default for screens."
            ),
            PatternKind.MULTISCALE_NOISE: (
                "Band-limited noise across several octaves (wavelet-noise style). Gives "
                "optical flow texture at every scale; tolerant of focus and distance changes."
            ),
            PatternKind.CHECKER: (
                "Checkerboard with a scale bar. Use it to check focus, framing and display "
                "scaling. Not for measurements."
            ),
        }[self]


@dataclass(frozen=True)
class PatternSpec:
    kind: PatternKind = PatternKind.SPECKLE
    element_px: int = 3  # finest feature size in output pixels (integer blocks)
    seed: int = 1
    dot_fill: float = 0.8  # RANDOM_DOTS: fraction of 2e x 2e cells holding a dot
    octaves: int = 4  # MULTISCALE_NOISE
    binary: bool = False  # MULTISCALE_NOISE: threshold at the median
    invert: bool = False

    def with_(self, **kwargs) -> PatternSpec:
        return replace(self, **kwargs)


def generate(spec: PatternSpec, width: int, height: int) -> np.ndarray:
    """Render ``spec`` at ``width`` x ``height`` output pixels, linear intensity [0, 1]."""
    if width <= 0 or height <= 0:
        raise ValueError("pattern size must be positive")
    e = max(1, int(spec.element_px))
    rng = np.random.default_rng(spec.seed)
    match spec.kind:
        case PatternKind.RANDOM_DOTS:
            img = _random_dots(rng, width, height, e, spec.dot_fill)
        case PatternKind.SPECKLE:
            img = _block_upsample(rng.random((_cells(height, e), _cells(width, e))) < 0.5, e)
            img = img[:height, :width].astype(np.float32)
        case PatternKind.MULTISCALE_NOISE:
            img = _multiscale_noise(rng, width, height, e, spec.octaves, spec.binary)
        case PatternKind.CHECKER:
            img = _checker(width, height, e)
        case _:
            raise ValueError(f"unknown pattern kind {spec.kind}")
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return 1.0 - img if spec.invert else img


def _cells(n: int, e: int) -> int:
    return -(-n // e)


def _block_upsample(a: np.ndarray, e: int) -> np.ndarray:
    return np.repeat(np.repeat(a, e, axis=0), e, axis=1)


def _random_dots(rng: np.random.Generator, w: int, h: int, e: int, fill: float) -> np.ndarray:
    """Square dots of ``e`` pixels, one candidate per 2e x 2e cell with random jitter.

    The jittered grid keeps a minimum spacing (no touching dots) while staying random, which
    gives sharp, unambiguous correlation peaks.
    """
    cell = 2 * e
    ny, nx = _cells(h, cell), _cells(w, cell)
    img = np.zeros((ny * cell, nx * cell), dtype=np.float32)
    present = rng.random((ny, nx)) < np.clip(fill, 0.0, 1.0)
    # Jitter keeps every dot inside its own cell with at least one pixel to the next cell.
    jy = rng.integers(0, cell - e, size=(ny, nx))
    jx = rng.integers(0, cell - e, size=(ny, nx))
    ys, xs = np.nonzero(present)
    for dy in range(e):
        for dx in range(e):
            img[ys * cell + jy[ys, xs] + dy, xs * cell + jx[ys, xs] + dx] = 1.0
    return img[:h, :w]


def _multiscale_noise(
    rng: np.random.Generator, w: int, h: int, e: int, octaves: int, binary: bool
) -> np.ndarray:
    """Sum of octave band-pass noise, finest band peaking at a period of 2e pixels.

    Each octave is white noise band-limited in the Fourier domain to [f/2, f] (a smooth
    raised-cosine window in log-frequency), which is the spectral shape Cook & DeRose wavelet
    noise approximates. Octaves are weighted equally in amplitude so coarse texture remains
    after defocus. The result is rank-equalised to a uniform distribution to use the full
    dynamic range.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    fr = np.hypot(fy, fx)
    spectrum = np.zeros(fr.shape, dtype=np.complex128)
    white = np.fft.rfft2(rng.standard_normal((h, w)))
    f_top = 1.0 / (2.0 * e)  # cycles/pixel of the finest band
    for k in range(max(1, octaves)):
        fc = f_top / (2.0**k)
        with np.errstate(divide="ignore"):
            t = np.log2(np.where(fr > 0, fr, 1e-12) / fc)
        window = np.where(np.abs(t) < 1.0, 0.5 * (1.0 + np.cos(np.pi * t)), 0.0)
        band = white * window
        energy = np.sqrt(np.sum(np.abs(band) ** 2)) + 1e-12
        spectrum += band / energy
    noise = np.fft.irfft2(spectrum, s=(h, w))
    ranks = np.argsort(np.argsort(noise, axis=None), axis=None).reshape(h, w)
    uniform = (ranks + 0.5) / ranks.size
    if binary:
        return (uniform >= 0.5).astype(np.float32)
    return uniform.astype(np.float32)


def _checker(w: int, h: int, e: int) -> np.ndarray:
    size = max(8, 8 * e)
    yy, xx = np.mgrid[0:h, 0:w]
    img = (((yy // size) + (xx // size)) % 2).astype(np.float32)
    # Fine 1-element stripes in the centre to judge focus and scaling.
    cy, cx, r = h // 2, w // 2, min(h, w) // 8
    sl = (slice(max(0, cy - r), cy + r), slice(max(0, cx - r), cx + r))
    img[sl] = ((xx[sl] // e) % 2).astype(np.float32)
    return img


def linear_to_srgb(linear: np.ndarray) -> np.ndarray:
    """Encode linear intensity for display (IEC 61966-2-1)."""
    x = np.clip(linear, 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_display_uint8(linear: np.ndarray, black: float = 0.0, white: float = 1.0) -> np.ndarray:
    """Map a linear pattern to 8-bit sRGB, optionally reducing contrast to avoid clipping."""
    x = black + (white - black) * np.clip(linear, 0.0, 1.0)
    return np.round(linear_to_srgb(x) * 255.0).astype(np.uint8)


def pattern_spectrum_score(img: np.ndarray) -> float:
    """Fraction of image variance at spatial periods of 2-8 pixels (useful BOS texture)."""
    f = np.abs(np.fft.rfft2(img - img.mean())) ** 2
    fy = np.fft.fftfreq(img.shape[0])[:, None]
    fx = np.fft.rfftfreq(img.shape[1])[None, :]
    fr = np.hypot(fy, fx)
    band = (fr >= 1 / 8) & (fr <= 1 / 2)
    return float(f[band].sum() / max(f.sum(), 1e-12))
