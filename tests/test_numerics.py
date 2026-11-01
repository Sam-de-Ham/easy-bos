import pytest
import torch

from easy_bos.core import imops
from easy_bos.core.solvers import pcg, solve_neumann_poisson
from easy_bos.core.spline import SplineImage


def _band_limited(h, w):
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij"
    )

    def f(x, y):
        return torch.sin(0.9 * x + 0.3) + torch.cos(0.7 * y) + torch.sin(0.4 * x + 0.5 * y)

    return f, xx, yy


@pytest.mark.parametrize("order", [3, 5])
def test_spline_interpolates_and_shifts_accurately(order):
    f, xx, yy = _band_limited(48, 64)
    img = f(xx, yy)[None, None]
    s = SplineImage(img, order)
    assert (s.sample(xx, yy) - img).abs().max() < 1e-5
    inner = (..., slice(6, -6), slice(6, -6))
    v = s.sample(xx + 0.37, yy - 0.21)
    assert (v - f(xx + 0.37, yy - 0.21))[inner].abs().max() < 5e-3
    # cubic convolution is an order of magnitude worse on the same data
    bicubic = imops.sample(img, (xx + 0.37)[None], (yy - 0.21)[None])
    assert (bicubic - f(xx + 0.37, yy - 0.21))[inner].abs().max() > 0.03


def test_spline_gradient_matches_finite_difference():
    f, xx, yy = _band_limited(32, 32)
    s = SplineImage(f(xx, yy)[None, None])
    _, gx, gy = s.sample(xx + 0.3, yy + 0.3, gradient=True)
    h = 1e-2
    fd_x = (s.sample(xx + 0.3 + h, yy + 0.3) - s.sample(xx + 0.3 - h, yy + 0.3)) / (2 * h)
    fd_y = (s.sample(xx + 0.3, yy + 0.3 + h) - s.sample(xx + 0.3, yy + 0.3 - h)) / (2 * h)
    assert (gx - fd_x)[..., 4:-4, 4:-4].abs().max() < 1e-2
    assert (gy - fd_y)[..., 4:-4, 4:-4].abs().max() < 1e-2


def test_adjoint_operators_are_exact():
    torch.manual_seed(0)
    a = torch.randn(1, 1, 9, 11, dtype=torch.float64)
    b = torch.randn(1, 1, 9, 11, dtype=torch.float64)
    for op, adj in [(imops.dx_fwd, imops.dx_fwd_adjoint), (imops.dy_fwd, imops.dy_fwd_adjoint)]:
        assert torch.allclose((op(a) * b).sum(), (a * adj(b)).sum())
    d = torch.randn(1, 2, 9, 11, dtype=torch.float64)
    c = torch.randn(1, 1, 9, 11, dtype=torch.float64)
    assert torch.allclose((imops.curl(d) * c).sum(), (d * imops.curl_adjoint(c)).sum())
    assert torch.allclose((imops.laplacian(a) * b).sum(), (a * imops.laplacian(b)).sum())


def test_neumann_poisson_solver_inverts_laplacian():
    x = torch.randn(1, 1, 40, 57, dtype=torch.float64)
    x -= x.mean()
    y = solve_neumann_poisson((-imops.laplacian(x)).float())
    assert (y.double() - x).abs().max() < 1e-4


def test_pcg_solves_spd_system():
    torch.manual_seed(1)
    m = torch.randn(20, 20, dtype=torch.float64)
    a = m @ m.T + 20 * torch.eye(20, dtype=torch.float64)
    b = torch.randn(20, dtype=torch.float64)
    x = pcg(lambda v: a @ v, b, torch.zeros(20, dtype=torch.float64), lambda r: r, 200, 1e-12)
    assert torch.allclose(a @ x, b, atol=1e-8)
