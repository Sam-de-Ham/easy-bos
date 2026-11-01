"""Linear solvers shared by the flow and reconstruction code (GPU, matrix-free)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class SolveInfo:
    iterations: int
    relative_residual: float
    converged: bool


def pcg(
    apply,
    b: torch.Tensor,
    x0: torch.Tensor,
    precond,
    iters: int,
    tol: float = 1e-4,
    info: list | None = None,
) -> torch.Tensor:
    """Preconditioned conjugate gradients for a symmetric positive (semi-)definite operator.

    If ``info`` is a list, a ``SolveInfo`` is appended to it.
    """
    x = x0.clone()
    r = b - apply(x)
    z = precond(r)
    p = z.clone()
    rz = (r * z).sum()
    b_norm = b.norm().clamp_min(1e-30)
    rel = float(r.norm() / b_norm)
    k = 0
    for k in range(1, iters + 1):  # noqa: B007 - k is reported after the loop
        ap = apply(p)
        denom = (p * ap).sum()
        if denom.abs() < 1e-300:
            break
        a = rz / denom
        x = x + a * p
        r = r - a * ap
        rel = float(r.norm() / b_norm)
        if rel < tol:
            break
        z = precond(r)
        rz_new = (r * z).sum()
        p = z + (rz_new / rz) * p
        rz = rz_new
    if info is not None:
        info.append(SolveInfo(k, rel, rel < tol))
    return x


def neumann_laplacian_eigenvalues(h: int, w: int, device) -> torch.Tensor:
    """Eigenvalues of -L for the 5-point Neumann (half-sample symmetric) Laplacian on the
    mirrored 2h x 2w periodic domain, in FFT layout."""
    ky = torch.arange(2 * h, device=device, dtype=torch.float64)
    kx = torch.arange(2 * w, device=device, dtype=torch.float64)
    ly = 2 - 2 * torch.cos(math.pi * ky / h)
    lx = 2 - 2 * torch.cos(math.pi * kx / w)
    return ly[:, None] + lx[None, :]


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
    sol = torch.fft.ifft2(spec / lam.to(spec.real.dtype)).real
    return sol[..., :h, :w]


def _dct(x: torch.Tensor) -> torch.Tensor:
    """Unnormalised DCT-II along the last axis via one same-size FFT (Makhoul 1980):
    X_k = 2 sum_n x_n cos(pi k (2n + 1) / 2N)."""
    n = x.shape[-1]
    v = torch.cat([x[..., ::2], x[..., 1::2].flip(-1)], dim=-1)
    k = torch.arange(n, device=x.device, dtype=x.dtype)
    w = torch.exp(-1j * math.pi * k / (2 * n))
    return 2.0 * (torch.fft.fft(v) * w).real


def _idct(y: torch.Tensor) -> torch.Tensor:
    """Inverse of ``_dct`` along the last axis."""
    n = y.shape[-1]
    k = torch.arange(n, device=y.device, dtype=y.dtype)
    half = y / 2.0
    imag = torch.cat([torch.zeros_like(half[..., :1]), -half.flip(-1)[..., :-1]], dim=-1)
    w = torch.exp(1j * math.pi * k / (2 * n))
    v = torch.fft.ifft(torch.complex(half, imag) * w).real
    x = torch.empty_like(v)
    x[..., ::2] = v[..., : n - n // 2]
    x[..., 1::2] = v.flip(-1)[..., : n // 2]
    return x


def dct2(x: torch.Tensor) -> torch.Tensor:
    return _dct(_dct(x).transpose(-1, -2)).transpose(-1, -2)


def idct2(y: torch.Tensor) -> torch.Tensor:
    return _idct(_idct(y).transpose(-1, -2)).transpose(-1, -2)


def neumann_spectral_solver(h: int, w: int, denominator, device):
    """Return f(rhs) solving D(-L) x = rhs per channel, with D a function of the Neumann
    Laplacian eigenvalues (e.g. c + a*lam^2 for a biharmonic-plus-identity operator).

    Polynomials in the 5-point Neumann Laplacian are diagonalised by the DCT-II basis
    (eigenvalues 2 - 2cos(pi k / N) per axis). Used as a preconditioner whose cost does not
    grow with the conditioning of the operator.
    """
    ky = torch.arange(h, device=device, dtype=torch.float32)
    kx = torch.arange(w, device=device, dtype=torch.float32)
    lam = (2 - 2 * torch.cos(math.pi * ky / h))[:, None] + (2 - 2 * torch.cos(math.pi * kx / w))[
        None, :
    ]
    inv = 1.0 / denominator(lam).clamp_min(1e-12)

    def solve(rhs: torch.Tensor) -> torch.Tensor:
        return idct2(dct2(rhs) * inv)

    return solve
