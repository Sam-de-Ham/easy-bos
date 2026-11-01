"""Weighted least-squares integration of a measured gradient field (GPU).

Solves for a scalar field phi on the pixel grid such that its forward differences match the
measured gradient (g_x, g_y) averaged to the cell edges, weighted by the inverse variance of
each edge, with soft Dirichlet anchors where phi is known:

    min_phi  sum_edges w_e (D phi - h <g>_e)^2  +  gamma sum_anchors (phi - phi_ref)^2

This is the uncertainty-weighted least-squares integration of Rajendran et al. (Exp Fluids
2020, arXiv:2004.01217); with uniform weights it reduces to the Poisson equation with Neumann
borders (Richard & Raffel 2001).

Invalid pixels (masked, or NaN in the input) carry no data. Their edges are kept with a tiny
weight and zero target, so phi is filled smoothly (harmonically) through holes instead of
being left undetermined; this matters for Abel inversion, which couples every radius inside a
hole. The filled values are returned separately from the masked result.

The normal equations are solved by conjugate gradients preconditioned with an exact FFT
Neumann-Poisson solve (float64 throughout).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import torch
from scipy import ndimage

from easy_bos.core.device import resolve
from easy_bos.core.imops import dx_fwd, dx_fwd_adjoint, dy_fwd, dy_fwd_adjoint
from easy_bos.core.solvers import SolveInfo, pcg, solve_neumann_poisson

FILL_WEIGHT = 1e-3  # relative weight of the smoothness edges inside invalid regions


@dataclass
class IntegrationResult:
    phi: np.ndarray  # NaN at invalid pixels
    filled: np.ndarray  # phi with invalid pixels filled harmonically
    residual_rms: float  # rms gradient misfit over valid edges, units of the input gradient
    solve: SolveInfo
    warnings: list[str]


def integrate_gradient(
    gx: np.ndarray,
    gy: np.ndarray,
    spacing: float,
    anchors: np.ndarray,
    anchor_value: float | np.ndarray = 0.0,
    sigma: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    device=None,
    iterations: int = 3000,
    tol: float = 1e-10,
) -> IntegrationResult:
    """Integrate the gradient (per unit length) on a grid of the given spacing.

    ``anchors`` is a boolean (H, W) mask where phi equals ``anchor_value``; every connected
    valid region needs at least one anchor. ``sigma`` is the per-pixel standard uncertainty of
    the gradient (same units as g); ``valid`` masks unusable pixels; NaNs are treated as invalid.
    """
    dev = resolve(device)
    gx = np.asarray(gx, dtype=np.float64)
    gy = np.asarray(gy, dtype=np.float64)
    h, w = gx.shape
    ok = np.isfinite(gx) & np.isfinite(gy)
    if valid is not None:
        ok &= np.asarray(valid, dtype=bool)
    if sigma is not None:
        sigma = np.asarray(sigma, dtype=np.float64)
        ok &= np.isfinite(sigma) & (sigma > 0)
    msgs: list[str] = []
    anc = np.asarray(anchors, dtype=bool) & ok
    if not anc.any():
        raise ValueError("integration needs at least one valid anchor pixel")
    labels, n_regions = ndimage.label(ok)
    anchored = set(np.unique(labels[anc]).tolist()) - {0}
    if len(anchored) < n_regions:
        msgs.append(
            f"{n_regions - len(anchored)} valid region(s) touch no anchor; their level is set "
            "by smooth continuation and may be offset."
        )

    def t(a):
        return torch.tensor(np.ascontiguousarray(a), dtype=torch.float64, device=dev)

    gx_t, gy_t = t(np.where(ok, gx, 0.0)), t(np.where(ok, gy, 0.0))
    var = t(np.where(ok, sigma**2, 1.0)) if sigma is not None else torch.ones_like(gx_t)
    ok_t = t(ok).bool()

    tx = torch.zeros_like(gx_t)
    ty = torch.zeros_like(gy_t)
    wx = torch.zeros_like(gx_t)
    wy = torch.zeros_like(gy_t)
    tx[:, :-1] = 0.5 * (gx_t[:, :-1] + gx_t[:, 1:]) * spacing
    ty[:-1, :] = 0.5 * (gy_t[:-1, :] + gy_t[1:, :]) * spacing
    wx[:, :-1] = 1.0 / (0.25 * (var[:, :-1] + var[:, 1:]))
    wy[:-1, :] = 1.0 / (0.25 * (var[:-1, :] + var[1:, :]))
    ex_ok = ok_t[:, :-1] & ok_t[:, 1:]
    ey_ok = ok_t[:-1, :] & ok_t[1:, :]
    scale = torch.cat([wx[:, :-1][ex_ok], wy[:-1, :][ey_ok]]).mean().clamp_min(1e-300)
    wx, wy = wx / scale, wy / scale
    # Edges touching invalid pixels: zero target, small weight (harmonic fill).
    wx[:, :-1] = torch.where(ex_ok, wx[:, :-1], torch.full_like(wx[:, :-1], FILL_WEIGHT))
    wy[:-1, :] = torch.where(ey_ok, wy[:-1, :], torch.full_like(wy[:-1, :], FILL_WEIGHT))
    tx[:, :-1] = torch.where(ex_ok, tx[:, :-1], torch.zeros_like(tx[:, :-1]))
    ty[:-1, :] = torch.where(ey_ok, ty[:-1, :], torch.zeros_like(ty[:-1, :]))

    gamma = 10.0
    ref = t(np.broadcast_to(anchor_value, (h, w)))
    a_w = t(anc) * gamma

    def apply(phi):
        return dx_fwd_adjoint(wx * dx_fwd(phi)) + dy_fwd_adjoint(wy * dy_fwd(phi)) + a_w * phi

    b = dx_fwd_adjoint(wx * tx) + dy_fwd_adjoint(wy * ty) + a_w * ref
    # Preconditioner: the Neumann Laplacian with a tiny shift (only to keep it invertible).
    # A shift comparable to the anchor strength mismatches the low modes and stalls CG.
    # (A diagonally rescaled variant S L^-1 S diverged in tests with 100:1 weight contrast.)
    shift = 1e-6

    def precond(r):
        return solve_neumann_poisson(r, shift=shift)

    info: list[SolveInfo] = []
    phi = pcg(apply, b, precond(b), precond, iterations, tol, info)
    if not info[0].converged:
        msg = (
            f"integration stopped after {info[0].iterations} iterations "
            f"(relative residual {info[0].relative_residual:.1e})"
        )
        msgs.append(msg)
        warnings.warn(msg, RuntimeWarning, stacklevel=2)
    res = torch.cat([(dx_fwd(phi) - tx)[:, :-1][ex_ok], (dy_fwd(phi) - ty)[:-1, :][ey_ok]])
    rms = float(res.pow(2).mean().sqrt() / spacing) if res.numel() else 0.0
    filled = phi.cpu().numpy()
    out = filled.copy()
    out[~ok] = np.nan
    return IntegrationResult(out, filled, rms, info[0], msgs)


def border_anchors(shape: tuple[int, int], sides: str = "lr", width: int = 3) -> np.ndarray:
    """Anchor mask on the chosen image borders: any of 'l', 'r', 't', 'b'."""
    h, w = shape
    m = np.zeros((h, w), dtype=bool)
    if "l" in sides:
        m[:, :width] = True
    if "r" in sides:
        m[:, -width:] = True
    if "t" in sides:
        m[:width, :] = True
    if "b" in sides:
        m[-width:, :] = True
    return m
