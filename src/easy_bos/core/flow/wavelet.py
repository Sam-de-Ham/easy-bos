"""Wavelet-based optical flow (wOFV / wOFA) for BOS (GPU).

Follows Schmidt & Sutton (Exp Fluids 2019-2021), Schmidt & Woike (AIAA J 2021, BOS "wOFA") and
Nicolas et al. (arXiv:2310.03980); see docs/research/wavelet-optical-flow.md.

* The displacement components are represented by wavelet coefficients, U_k = Psi theta_k, where
  Psi is the separable 2-D inverse DWT with the CDF 17/11 (bior6.8) synthesis filters.
* Data term: the non-linearised displaced-frame difference sum m(x) [cur(x) - ref(x + U)]^2,
  with ref evaluated through a prefiltered cubic B-spline (one-sided form of Nicolas Eq. 7,
  which keeps U on the grid of the distorted image as everywhere else in easy-bos).
* Regulariser: lambda * sum |lap U_1|^2 + |lap U_2|^2 on the reconstructed field.
* Multiscale: coefficients are unlocked coarse-to-fine; at stage s the whole top-left block up
  to scale s is optimised (coarser coefficients stay free) with L-BFGS, warm-started from the
  previous stage. The finest detail band is not estimated (truncation regularisation).

Psi is built from explicit per-level synthesis matrices obtained from PyWavelets
(periodization mode), applied as U = S_y C S_x^T level by level. The image is padded to a
multiple of 2^L; the padding carries no data or regulariser weight, so periodic wrap-around
never couples measured pixels.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass

import numpy as np
import pywt
import torch
import torch.nn.functional as F

from easy_bos.core import imops
from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register
from easy_bos.core.spline import SplineImage

WAVELET = "bior6.8"


@dataclass
class WaveletParams(MethodParams):
    lam: float = param(30.0, "Smoothness", "Regularisation weight lambda. Err on the high side: "
                       "too little regularisation amplifies noise sharply.",
                       minimum=0.01, maximum=10000.0, log=True)
    estimate_finest: bool = param(False, "Estimate finest scale", "Also estimate pixel-scale "
                                  "detail coefficients (less implicit regularisation).",
                                  advanced=True)
    max_iterations: int = param(100, "Iterations per scale", minimum=10, maximum=500, advanced=True)
    presmooth: float = param(0.5, "Pre-smoothing", minimum=0.0, maximum=2.0, advanced=True)
    margin: int = param(16, "Padding margin", minimum=0, maximum=128, advanced=True)


@functools.cache
def synthesis_matrix(n: int, wavelet: str = WAVELET) -> np.ndarray:
    """One-level inverse DWT (periodization) as an n x n matrix acting on [approx; detail]."""
    half = n // 2
    s = np.zeros((n, n))
    for k in range(n):
        e = np.zeros(n)
        e[k] = 1.0
        s[:, k] = pywt.idwt(e[:half], e[half:], wavelet, mode="periodization")
    return s


class WaveletSynthesis:
    """Multi-level separable inverse DWT on an (H, W) grid, H and W divisible by 2^levels."""

    def __init__(self, h: int, w: int, levels: int, device):
        self.h, self.w, self.levels = h, w, levels
        self.sizes = [(h >> (levels - 1 - k), w >> (levels - 1 - k)) for k in range(levels)]
        self.sy = [torch.from_numpy(synthesis_matrix(hh)).float().to(device) for hh, _ in self.sizes]
        self.sx = [torch.from_numpy(synthesis_matrix(ww)).float().to(device) for _, ww in self.sizes]

    @property
    def coarsest(self) -> tuple[int, int]:
        return self.h >> self.levels, self.w >> self.levels

    def __call__(self, coeffs: torch.Tensor) -> torch.Tensor:
        """coeffs: (C, H, W) in Mallat layout -> field (C, H, W)."""
        c = coeffs
        for k, (hh, ww) in enumerate(self.sizes):
            block = self.sy[k] @ c[:, :hh, :ww] @ self.sx[k].T
            if hh == self.h and ww == self.w:
                c = block
            else:
                c = torch.cat([torch.cat([block, c[:, :hh, ww:]], dim=2), c[:, hh:, :]], dim=1)
        return c


@register
class WaveletFlow(FlowMethod):
    key = "wavelet"
    label = "Wavelet flow"
    summary = ("Wavelet-based optical flow (wOFA) with coarse-to-fine scale unlocking (GPU). "
               "Highest accuracy, slowest.")
    Params = WaveletParams
    speed = "slow"

    def _estimate(self, ref, cur, mask=None):
        p: WaveletParams = self.params
        dev = ref.device
        ref = imops.gaussian_blur(ref, p.presmooth)
        cur = imops.gaussian_blur(cur, p.presmooth)
        h, w = ref.shape[-2:]
        levels = max(1, int(math.floor(math.log2(max(4, min(h, w) + 2 * p.margin) / 8))))
        step = 2**levels
        hp = math.ceil((h + 2 * p.margin) / step) * step
        wp = math.ceil((w + 2 * p.margin) / step) * step
        oy, ox = (hp - h) // 2, (wp - w) // 2
        synth = WaveletSynthesis(hp, wp, levels, dev)

        spline = SplineImage(ref)
        xx, yy = imops.pixel_grid(h, w, dev)
        weight = torch.ones(1, 1, h, w, device=dev) if mask is None else mask
        n_pix = float(weight.sum().clamp_min(1.0))
        theta = torch.zeros(2, hp, wp, device=dev)

        def field(t: torch.Tensor) -> torch.Tensor:
            return synth(t)[:, oy:oy + h, ox:ox + w]

        n_stages = levels + (1 if p.estimate_finest else 0)
        bh, bw = synth.coarsest
        for _ in range(n_stages):
            block = theta[:, :bh, :bw].clone().requires_grad_(True)
            frozen = theta.clone()
            frozen[:, :bh, :bw] = 0.0

            def embed(b, frozen=frozen, bh=bh, bw=bw):
                return frozen + F.pad(b, (0, frozen.shape[2] - bw, 0, frozen.shape[1] - bh))

            opt = torch.optim.LBFGS([block], lr=1.0, max_iter=int(p.max_iterations),
                                    max_eval=int(p.max_iterations) * 3, history_size=8,
                                    tolerance_grad=1e-7, tolerance_change=1e-9,
                                    line_search_fn="strong_wolfe")

            def closure(block=block, embed=embed):
                opt.zero_grad()
                u = field(embed(block))
                warped = spline.sample(xx + u[0], yy + u[1])
                data = (weight * (cur - warped) ** 2).sum()
                lap = _laplacian(u[None])
                loss = (data + p.lam * (lap**2).sum()) / n_pix
                loss.backward()
                return loss

            with torch.enable_grad():
                opt.step(closure)
            theta = embed(block.detach()).detach()
            bh, bw = min(bh * 2, hp), min(bw * 2, wp)

        with torch.no_grad():
            u = field(theta)[None]
            warped, ix, iy = spline.warp(u, gradient=True)
            texture = imops.gaussian_blur(ix**2 + iy**2, 2.0)
            resid = imops.gaussian_blur((cur - warped) ** 2, 2.0)
            conf = (texture / (texture + resid * 4.0 + 1e-6)).clamp(0, 1) * weight
        return u, conf


def _laplacian(x: torch.Tensor) -> torch.Tensor:
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:] + xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1] - 4 * x
