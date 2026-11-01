"""Abel inversion for axisymmetric BOS (three-point method).

Operators follow Dasch (1992, Appl Opt 31:1146) as corrected by Martin (2002) and implemented in
PyAbel (``abel/dasch.py``); see docs/research/reconstruction.md. Samples are at y_j = j dr with
j = 0 on the axis; the field is assumed zero beyond the last sample.

* ``three_point_matrix``: f_i = (1/dr) sum_j D_ij P_j inverts a projection P.
* ``deflection_matrix``: f_i = sum_j K_ij eps_j inverts the deflection eps = dP/dy directly
  (same piecewise-quadratic quadrature applied to f(r) = -(1/pi) int eps / sqrt(y^2 - r^2) dy).
"""

from __future__ import annotations

import functools

import numpy as np


def _integrals(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Dasch's I0_ij and I1_ij for i, j = 0..n (one extra column for j + 1 terms)."""
    i = np.arange(n + 1, dtype=np.float64)[:, None]
    j = np.arange(n + 1, dtype=np.float64)[None, :]
    i_b = np.broadcast_to(i, (n + 1, n + 1))
    j_b = np.broadcast_to(j, (n + 1, n + 1))
    i0 = np.zeros((n + 1, n + 1))
    i1 = np.zeros((n + 1, n + 1))

    upper = np.sqrt(np.clip((2 * j_b + 1) ** 2 - 4 * i_b**2, 0, None))
    lower = np.sqrt(np.clip((2 * j_b - 1) ** 2 - 4 * i_b**2, 0, None))

    diag = (i_b == j_b) & (j_b > 0)
    i0[diag] = np.log((upper[diag] + 2 * j_b[diag] + 1) / (2 * j_b[diag])) / (2 * np.pi)
    above = j_b > i_b
    i0[above] = np.log((upper[above] + 2 * j_b[above] + 1) / (lower[above] + 2 * j_b[above] - 1)) / (2 * np.pi)

    eq = i_b == j_b
    i1[eq] = upper[eq] / (2 * np.pi) - 2 * j_b[eq] * i0[eq]
    i1[above] = (upper[above] - lower[above]) / (2 * np.pi) - 2 * j_b[above] * i0[above]
    return i0, i1


@functools.cache
def three_point_matrix(n: int) -> np.ndarray:
    """Dasch three-point deconvolution operator D (n x n) for projections."""
    i0, i1 = _integrals(n + 1)
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(max(0, i - 1), n):
            if j == i - 1:
                d[i, j] = i0[i, j + 1] - i1[i, j + 1]
            elif j == i:
                d[i, j] = i0[i, j + 1] - i1[i, j + 1] + 2 * i1[i, j]
            else:
                d[i, j] = i0[i, j + 1] - i1[i, j + 1] + 2 * i1[i, j] - i0[i, j - 1] - i1[i, j - 1]
    if n > 1:
        # Axis: P_{-1} = P_1 (P'(0) = 0) folds the j = -1 term into column 1.
        d[0, 1] = i0[0, 2] - i1[0, 2] + 2 * i1[0, 1] - 2 * i1[0, 0]
    return d


@functools.cache
def deflection_matrix(n: int) -> np.ndarray:
    """Operator K with f_i = sum_j K_ij eps_j (eps sampled at y_j = j dr, eps_0 = 0 implied).

    On each interval eps is modelled as eps_j + (eps_{j+1} - eps_{j-1})/2 (y - j), giving
    f_i = -sum_{j>=i} [2 I0_ij eps_j + 1/2 I1_ij (eps_{j+1} - eps_{j-1})], with eps_{-1} = -eps_1
    (odd symmetry) and eps_n = 0.
    """
    i0, i1 = _integrals(n + 1)
    i0 = i0[:n, :n].copy()
    i1 = i1[:n, :n].copy()
    k = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            k[i, j] -= 2 * i0[i, j]
            if j + 1 < n:
                k[i, j + 1] -= 0.5 * i1[i, j]
            if j - 1 >= 0:
                k[i, j - 1] += 0.5 * i1[i, j]
            else:  # eps_{-1} = -eps_1
                if n > 1:
                    k[i, 1] -= 0.5 * i1[i, j]
    k[:, 0] = 0.0  # eps_0 = 0 on the axis by symmetry
    return k


def invert_projection(p: np.ndarray, dr: float) -> np.ndarray:
    """Radial field f(r_i) from projections P(y_j) (last axis is j)."""
    n = p.shape[-1]
    return p @ three_point_matrix(n).T / dr


def invert_deflection(eps: np.ndarray) -> np.ndarray:
    """Radial field f(r_i) from deflection angles eps(y_j) = dP/dy (last axis is j)."""
    n = eps.shape[-1]
    return eps @ deflection_matrix(n).T


def forward_projection(f: np.ndarray, dr: float) -> np.ndarray:
    """Abel projection P(y) = 2 int_y^R f(r) r / sqrt(r^2 - y^2) dr for a piecewise-linear f.

    Used for tests and for consistency checks (re-projecting a reconstruction).
    """
    n = f.shape[-1]
    r = np.arange(n) * dr
    out = np.zeros_like(f, dtype=np.float64)
    for jj in range(n):
        y = r[jj]
        rk = r[jj:]
        fk = f[..., jj:]
        if rk.size < 2:
            continue
        # Exact integral of linear segments of f against r / sqrt(r^2 - y^2).
        a, b = rk[:-1], rk[1:]
        fa, fb = fk[..., :-1], fk[..., 1:]
        sa = np.sqrt(np.clip(a**2 - y**2, 0, None))
        sb = np.sqrt(np.clip(b**2 - y**2, 0, None))
        la = np.log(a + sa) if y > 0 else np.log(np.maximum(a, 1e-300))
        lb = np.log(b + sb)
        # int r/s dr = s ; int r^2/s dr = (r s + y^2 ln(r + s)) / 2
        i_r = sb - sa
        i_r2 = 0.5 * (b * sb - a * sa + y**2 * (lb - la))
        slope = (fb - fa) / (b - a)
        seg = (fa - slope * a) * i_r + slope * i_r2
        out[..., jj] = 2 * seg.sum(axis=-1)
    return out


def find_axis(eps_x: np.ndarray, rows: np.ndarray | None = None) -> float:
    """Locate a vertical symmetry axis from the horizontal deflection (odd about the axis).

    Minimises sum_rows sum_y [eps(c + y) + eps(c - y)]^2 over c with a coarse search plus
    parabolic refinement. Returns the axis column in pixels.
    """
    e = eps_x if rows is None else eps_x[rows]
    h, w = e.shape
    weights = np.abs(e).sum(axis=0)
    centre_guess = float((weights * np.arange(w)).sum() / max(weights.sum(), 1e-30))

    def cost(c: float) -> float:
        half = int(min(c, w - 1 - c))
        if half < 4:
            return np.inf
        y = np.arange(1, half + 1)
        xs = np.arange(w)
        right = np.stack([np.interp(c + y, xs, row) for row in e])
        left = np.stack([np.interp(c - y, xs, row) for row in e])
        return float(((right + left) ** 2).sum() / max((right**2 + left**2).sum(), 1e-30))

    lo, hi = max(4.0, centre_guess - w / 4), min(w - 5.0, centre_guess + w / 4)
    cs = np.linspace(lo, hi, 81)
    vals = np.array([cost(c) for c in cs])
    k = int(np.argmin(vals))
    step = cs[1] - cs[0]
    for _ in range(3):  # golden-free local refinement on a shrinking grid
        cs = np.linspace(cs[k] - step, cs[k] + step, 21)
        vals = np.array([cost(c) for c in cs])
        k = int(np.argmin(vals))
        step = cs[1] - cs[0]
    if 0 < k < len(cs) - 1:
        y0, y1, y2 = vals[k - 1], vals[k], vals[k + 1]
        den = y0 - 2 * y1 + y2
        if den > 0:
            return float(cs[k] + 0.5 * step * (y0 - y2) / den)
    return float(cs[k])


def fold(field: np.ndarray, axis_col: float, odd: bool) -> np.ndarray:
    """Sample rows at c + j (j = 0..N-1) and average with the mirrored side.

    ``odd`` fields (deflection) are antisymmetrised, even fields (projection) symmetrised.
    """
    h, w = field.shape
    n = int(np.floor(min(axis_col, w - 1 - axis_col))) + 1
    j = np.arange(n)
    xs = np.arange(w)
    right = np.stack([np.interp(axis_col + j, xs, row) for row in field])
    left = np.stack([np.interp(axis_col - j, xs, row) for row in field])
    return 0.5 * (right - left) if odd else 0.5 * (right + left)
