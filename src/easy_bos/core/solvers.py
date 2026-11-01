"""Linear solvers shared by the flow and reconstruction code (GPU, matrix-free)."""

from __future__ import annotations

import math

import torch


def pcg(apply, b: torch.Tensor, x0: torch.Tensor, precond, iters: int, tol: float = 1e-4) -> torch.Tensor:
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


def neumann_laplacian_eigenvalues(h: int, w: int, device) -> torch.Tensor:
    """Eigenvalues of -L for the 5-point Neumann (half-sample symmetric) Laplacian on the
    mirrored 2h x 2w periodic domain, in FFT layout."""
    ky = torch.arange(2 * h, device=device, dtype=torch.float64)
    kx = torch.arange(2 * w, device=device, dtype=torch.float64)
    ly = 2 - 2 * torch.cos(math.pi * ky / h)
    lx = 2 - 2 * torch.cos(math.pi * kx / w)
    return (ly[:, None] + lx[None, :]).float()


def solve_neumann_poisson(rhs: torch.Tensor, shift: float = 0.0) -> torch.Tensor:
    """Solve (-L + shift) x = rhs with Neumann borders by FFT on the mirrored domain.

    -L is the 5-point Laplacian with half-sample symmetric (Neumann) boundaries, which equals
    G^T G for forward differences. With ``shift == 0`` the constant mode is set to zero.
    """
    h, w = rhs.shape[-2:]
    ext = torch.cat([rhs, rhs.flip(-1)], dim=-1)
    ext = torch.cat([ext, ext.flip(-2)], dim=-2)
    spec = torch.fft.fft2(ext)
    lam = neumann_laplacian_eigenvalues(h, w, rhs.device) + shift
    if shift == 0.0:
        lam = lam.clone()
        lam[0, 0] = math.inf
    sol = torch.fft.ifft2(spec / lam).real
    return sol[..., :h, :w]
