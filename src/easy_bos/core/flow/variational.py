"""Variational optical flow for BOS (GPU), with an optional irrotationality prior.

Energy for the displacement D = (u, v), minimised coarse-to-fine with image warping
(Horn & Schunck 1981; Brox et al. 2004; evaluated for BOS by Atcheson et al. 2009):

    E(D) = sum_x psi( (cur - ref(x + D))^2 ) + alpha * R(D) + beta * |curl D|^2

* data term: Charbonnier psi(s^2) = sqrt(s^2 + eps^2), handled by iteratively reweighted
  least squares on the linearisation around the current warp; the warped reference and its
  gradient come from the same prefiltered B-spline, so linearisation and warp agree;
* regulariser R:
    - "first"  : |grad u|^2 + |grad v|^2  (Horn–Schunck, quadratic)
    - "second" : (lap u)^2 + (lap v)^2   (no penalty on uniform gradients; the
                                          "viscosity-like" choice of wavelet OF for BOS)
* curl penalty: single-view BOS displacement is the gradient of the projected refractive index
  and therefore irrotational (cf. the Poisson parameterisation of Shiba et al., T-PAMI 2024).
  A quadratic penalty on the discrete curl keeps the linear systems well conditioned, whereas
  solving for a scalar potential directly turns the problem into a badly conditioned
  higher-order one.

Each linearised problem is symmetric positive semi-definite and is solved matrix-free with
Jacobi-preconditioned conjugate gradients on the GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from easy_bos.core import imops
from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register
from easy_bos.core.spline import SplineImage


@dataclass
class VariationalParams(MethodParams):
    alpha: float = param(3.0, "Smoothness", "Regularisation weight. Higher = smoother, less noise.",
                         minimum=1e-3, maximum=1000.0, log=True)
    regularizer: str = param("first", "Regulariser", "First order (Horn–Schunck) or second "
                             "order (Laplacian).", choices=("first", "second"))
    curl_weight: float = param(0.0, "Irrotationality", "Penalty on the curl of the field; BOS "
                               "displacement from a single view is curl-free. 0 disables.",
                               minimum=0.0, maximum=1000.0)
    robust_eps: float = param(0.1, "Robustness", "Charbonnier epsilon (image std units). "
                              "Smaller ignores outliers more.", minimum=0.01, maximum=10.0,
                              log=True, advanced=True)
    levels: int = param(3, "Pyramid levels", minimum=1, maximum=6, advanced=True)
    warps: int = param(4, "Warps per level", minimum=1, maximum=12, advanced=True)
    cg_iterations: int = param(150, "Solver iterations", minimum=20, maximum=2000, advanced=True)
    presmooth: float = param(0.5, "Pre-smoothing", minimum=0.0, maximum=2.0, advanced=True)


@register
class Variational(FlowMethod):
    key = "variational"
    label = "Variational flow"
    summary = ("Global optical flow with a robust data term and smoothness prior (GPU). "
               "Dense, with optional curl-free prior.")
    Params = VariationalParams
    speed = "moderate"

    def _estimate(self, ref, cur, mask=None):
        p: VariationalParams = self.params
        ref = imops.gaussian_blur(ref, p.presmooth)
        cur = imops.gaussian_blur(cur, p.presmooth)
        pyr = [(ref, cur, mask)]
        for _ in range(int(p.levels) - 1):
            r, c, m = pyr[-1]
            if min(r.shape[-2:]) < 48:
                break
            m2 = F.interpolate(m, size=((m.shape[-2] + 1) // 2, (m.shape[-1] + 1) // 2),
                               mode="area") if m is not None else None
            pyr.append((imops.downsample2(r), imops.downsample2(c), m2))
        disp = None
        weight = None
        for r, c, m in reversed(pyr):
            h, w = r.shape[-2:]
            disp = torch.zeros(1, 2, h, w, device=r.device) if disp is None else imops.upsample_flow(disp, (h, w))
            spline = SplineImage(r)
            for _ in range(int(p.warps)):
                disp, weight = _solve_increment(spline, c, disp, m, p)
        return disp, weight


# --------------------------------------------------------------------------- operators
def laplacian(x: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian with Neumann (reflecting) borders; a symmetric operator."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:] + xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1] - 4 * x


def _laplacian_diag(shape, device) -> torch.Tensor:
    """Diagonal of the Neumann 5-point Laplacian (-4 inside, -3 on edges, -2 at corners)."""
    h, w = shape[-2:]
    d = torch.full((h, w), -4.0, device=device)
    d[0, :] += 1
    d[-1, :] += 1
    d[:, 0] += 1
    d[:, -1] += 1
    return d


def dx_fwd(f: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(f)
    out[..., :-1] = f[..., 1:] - f[..., :-1]
    return out


def dx_fwd_adjoint(g: torch.Tensor) -> torch.Tensor:
    g = g.clone()
    g[..., -1] = 0
    out = -g
    out[..., 1:] += g[..., :-1]
    return out


def dy_fwd(f: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(f)
    out[..., :-1, :] = f[..., 1:, :] - f[..., :-1, :]
    return out


def dy_fwd_adjoint(g: torch.Tensor) -> torch.Tensor:
    g = g.clone()
    g[..., -1, :] = 0
    out = -g
    out[..., 1:, :] += g[..., :-1, :]
    return out


def curl(d: torch.Tensor) -> torch.Tensor:
    """Discrete curl dv/dx - du/dy with forward differences."""
    return dx_fwd(d[:, 1:]) - dy_fwd(d[:, :1])


def curl_adjoint(c: torch.Tensor) -> torch.Tensor:
    return torch.cat([-dy_fwd_adjoint(c), dx_fwd_adjoint(c)], dim=1)


class _System:
    """Linearised normal equations around the current warp, applied matrix-free."""

    def __init__(self, ix, iy, wdata, alpha, regularizer, beta):
        self.ix, self.iy, self.w = ix, iy, wdata
        self.alpha = alpha
        self.beta = beta
        self.second = regularizer == "second"

    def reg(self, d: torch.Tensor) -> torch.Tensor:
        lap = laplacian(d)
        out = self.alpha * (laplacian(lap) if self.second else -lap)
        if self.beta > 0:
            out = out + self.beta * curl_adjoint(curl(d))
        return out

    def apply(self, d: torch.Tensor) -> torch.Tensor:
        """(J^T W J + R) d for a flow-shaped (1, 2, H, W) tensor."""
        r = self.w * (self.ix * d[:, :1] + self.iy * d[:, 1:])
        return torch.cat([self.ix * r, self.iy * r], dim=1) + self.reg(d)

    def diag(self) -> torch.Tensor:
        ld = _laplacian_diag(self.ix.shape, self.ix.device)
        # Diagonal of L^2 = sum of squared row entries of L: ld^2 + number of neighbours (-ld).
        reg_diag = self.alpha * ((ld**2 - ld) if self.second else -ld) + 2.0 * self.beta
        return torch.cat([self.w * self.ix**2, self.w * self.iy**2], dim=1) + reg_diag


def pcg(apply, b, x0, precond, iters: int, tol: float = 1e-4):
    """Preconditioned conjugate gradients for a symmetric positive (semi-)definite operator."""
    x = x0.clone()
    r = b - apply(x)
    z = precond(r)
    p = z.clone()
    rz = (r * z).sum()
    b_norm = b.norm().clamp_min(1e-20)
    for _ in range(iters):
        ap = apply(p)
        denom = (p * ap).sum()
        if denom.abs() < 1e-30:
            break
        a = rz / denom
        x = x + a * p
        r = r - a * ap
        if r.norm() / b_norm < tol:
            break
        z = precond(r)
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    return x


def _solve_increment(ref: SplineImage, cur, disp, mask, p: VariationalParams):
    ref_w, ix, iy = ref.warp(disp, gradient=True)
    e = cur - ref_w  # linear model: e ~ ix du + iy dv
    wdata = p.robust_eps / torch.sqrt(e**2 + p.robust_eps**2)
    if mask is not None:
        wdata = wdata * mask
    sys = _System(ix, iy, wdata, p.alpha, p.regularizer, p.curl_weight)
    rhs = torch.cat([ix * wdata * e, iy * wdata * e], dim=1) - sys.reg(disp)
    diag = sys.diag().clamp_min(1e-6)
    delta = pcg(sys.apply, rhs, torch.zeros_like(rhs), lambda r: r / diag, int(p.cg_iterations))
    texture = imops.gaussian_blur(ix**2 + iy**2, 2.0)
    conf = (wdata * texture / texture.mean().clamp_min(1e-12)).clamp(0, 1)
    return disp + delta, conf
