"""Weighted least-squares integration of a measured gradient field (GPU).

Solves for a scalar field phi on the pixel grid such that its forward differences match the
measured gradient (g_x, g_y) averaged to the cell edges, weighted by the inverse variance of
each edge, with soft Dirichlet anchors where phi is known:

    min_phi  sum_edges w_e (D phi - h <g>_e)^2  +  gamma sum_anchors (phi - phi_ref)^2

This is the uncertainty-weighted least-squares integration of Rajendran et al. (Exp Fluids
2020, arXiv:2004.01217); with uniform weights it reduces to the Poisson equation with Neumann
borders (Richard & Raffel 2001). Masked pixels (weight 0) drop out. The normal equations are
solved by conjugate gradients preconditioned with an exact FFT Neumann-Poisson solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from easy_bos.core.device import resolve
from easy_bos.core.imops import dx_fwd, dx_fwd_adjoint, dy_fwd, dy_fwd_adjoint
from easy_bos.core.solvers import pcg, solve_neumann_poisson


@dataclass
class IntegrationResult:
    phi: np.ndarray
    residual_rms: float  # rms misfit of the gradient, same units as the input gradient


def integrate_gradient(
    gx: np.ndarray,
    gy: np.ndarray,
    spacing: float,
    anchors: np.ndarray,
    anchor_value: float | np.ndarray = 0.0,
    sigma: np.ndarray | None = None,
    valid: np.ndarray | None = None,
    device=None,
    iterations: int = 4000,
    tol: float = 1e-11,
) -> IntegrationResult:
    """Integrate the gradient (per unit length) on a grid of the given spacing.

    ``anchors`` is a boolean (H, W) mask where phi equals ``anchor_value``; at least one anchor
    per connected valid region is required. ``sigma`` is the per-pixel standard uncertainty of
    the gradient (same units as g); ``valid`` masks unusable pixels.
    """
    dev = resolve(device)
    t = lambda a: torch.tensor(np.array(a), dtype=torch.float64, device=dev)  # noqa: E731
    gx_t, gy_t = t(gx), t(gy)
    h, w = gx_t.shape
    if not np.any(anchors):
        raise ValueError("integration needs at least one anchor pixel")
    var = t(sigma) ** 2 if sigma is not None else torch.ones(h, w, dtype=torch.float64, device=dev)
    var = var.clamp_min(1e-30)
    ok = t(valid).bool() if valid is not None else torch.ones(h, w, dtype=torch.bool, device=dev)

    # Edge targets and weights (edges between (i, j) and (i, j+1), and (i, j) and (i+1, j)).
    tx = torch.zeros_like(gx_t)
    ty = torch.zeros_like(gy_t)
    wx = torch.zeros_like(gx_t)
    wy = torch.zeros_like(gy_t)
    tx[:, :-1] = 0.5 * (gx_t[:, :-1] + gx_t[:, 1:]) * spacing
    ty[:-1, :] = 0.5 * (gy_t[:-1, :] + gy_t[1:, :]) * spacing
    wx[:, :-1] = 1.0 / (0.25 * (var[:, :-1] + var[:, 1:]))
    wy[:-1, :] = 1.0 / (0.25 * (var[:-1, :] + var[1:, :]))
    wx[:, :-1] *= (ok[:, :-1] & ok[:, 1:]).double()
    wy[:-1, :] *= (ok[:-1, :] & ok[1:, :]).double()
    w_scale = torch.cat([wx[:, :-1].flatten(), wy[:-1, :].flatten()]).mean().clamp_min(1e-30)
    wx, wy = wx / w_scale, wy / w_scale

    anc = t(anchors).bool() & ok
    gamma = 10.0
    ref = t(np.broadcast_to(anchor_value, (h, w)))
    a_w = anc.double() * gamma

    def apply(phi):
        return dx_fwd_adjoint(wx * dx_fwd(phi)) + dy_fwd_adjoint(wy * dy_fwd(phi)) + a_w * phi

    b = dx_fwd_adjoint(wx * tx) + dy_fwd_adjoint(wy * ty) + a_w * ref
    # Preconditioner: the unweighted Neumann Laplacian with a small shift standing in for the
    # anchors (keeps it invertible).
    shift = float(gamma * anc.double().mean().clamp_min(1e-6))

    def precond(r):
        return solve_neumann_poisson(r.float(), shift=shift).double()

    x0 = solve_neumann_poisson(b.float(), shift=shift).double()
    phi = pcg(apply, b, x0, precond, iterations, tol)
    res = torch.cat([(dx_fwd(phi) - tx)[:, :-1][(wx[:, :-1] > 0)],
                     (dy_fwd(phi) - ty)[:-1, :][(wy[:-1, :] > 0)]])
    rms = float(res.pow(2).mean().sqrt() / spacing) if res.numel() else 0.0
    out = phi.cpu().numpy()
    out[~ok.cpu().numpy()] = np.nan
    return IntegrationResult(out, rms)


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
