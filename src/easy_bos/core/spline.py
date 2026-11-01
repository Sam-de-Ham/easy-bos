"""Prefiltered B-spline image interpolation (GPU).

Cubic-convolution interpolation (``grid_sample(mode="bicubic")``) has a fractional-shift
dependent amplitude and phase error. With fine BOS textures (2-4 px features) this biases
sub-pixel displacement estimates by several percent towards integer values. Prefiltered
B-splines of order 3 or 5 have far smaller error (Unser 1999, IEEE Signal Proc Mag 16(6):22;
Schreier, Braasch & Sutton 2000, Opt Eng 39:2915 for DIC) and give exact analytic gradients of
the interpolant, which keeps warping and linearisation consistent.

Boundary condition: mirror-symmetric about the edge samples (whole-sample symmetry).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def bspline(x: torch.Tensor, n: int) -> torch.Tensor:
    """Centred B-spline of order n: 1/n! sum_j (-1)^j C(n+1, j) (x + (n+1)/2 - j)_+^n.

    Evaluated in float64: the alternating power form cancels badly in float32 for n = 5.
    """
    xd = x.double()
    out = torch.zeros_like(xd)
    for j in range(n + 2):
        out = out + (-1) ** j * math.comb(n + 1, j) * torch.clamp(xd + (n + 1) / 2 - j, min=0) ** n
    return (out / math.factorial(n)).to(x.dtype)


def bspline_derivative(x: torch.Tensor, n: int) -> torch.Tensor:
    """d/dx of the order-n B-spline (= beta_{n-1}(x + 1/2) - beta_{n-1}(x - 1/2))."""
    return bspline(x + 0.5, n - 1) - bspline(x - 0.5, n - 1)


def _prefilter_1d(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """Divide by the sampled B-spline spectrum using the mirror-symmetric periodic extension."""
    length = x.shape[dim]
    if length < 2:
        return x
    inner = x.narrow(dim, 1, length - 2).flip(dim)
    ext = torch.cat([x, inner], dim=dim)  # period 2N - 2, whole-sample symmetric
    period = ext.shape[dim]
    k = torch.arange(-(n // 2), n // 2 + 1, dtype=torch.float64)
    taps = bspline(k, n)
    omega = 2 * math.pi * torch.fft.rfftfreq(period, dtype=torch.float64)
    resp = sum(t * torch.cos(omega * kk) for t, kk in zip(taps, k, strict=True))
    shape = [1] * x.dim()
    shape[dim] = -1
    resp = resp.to(x.device, torch.float32).reshape(shape)
    spec = torch.fft.rfft(ext, dim=dim) / resp
    return torch.fft.irfft(spec, n=period, dim=dim).narrow(dim, 0, length)


class SplineImage:
    """B-spline representation of a (N, C, H, W) image for sampling and gradients."""

    def __init__(self, img: torch.Tensor, order: int = 3):
        if order not in (3, 5):
            raise ValueError("order must be 3 or 5")
        self.order = order
        self.shape = img.shape
        c = _prefilter_1d(_prefilter_1d(img.float(), -1, order), -2, order)
        self.pad = order // 2 + 2
        self.coeffs = F.pad(c, (self.pad,) * 4, mode="reflect")

    def sample(self, x: torch.Tensor, y: torch.Tensor, gradient: bool = False):
        """Sample at pixel coordinates x, y of shape (H', W') (or (N, H', W')).

        Coordinates are clamped to the image. Returns the values (N, C, H', W') and, if
        ``gradient``, also d/dx and d/dy of the interpolant.
        """
        n_, c_, h, w = self.shape
        x = x.clamp(0, w - 1)
        y = y.clamp(0, h - 1)
        if x.dim() == 2:
            x, y = x[None], y[None]
        if self.order == 3:
            return self._sample_cubic(x, y, gradient)
        xi = torch.floor(x)
        yi = torch.floor(y)
        tx, ty = x - xi, y - yi
        half = self.order // 2
        taps = range(-half, half + 2)  # order 3: -1..2, order 5: -2..3
        wx = [bspline(tx - k, self.order) for k in taps]
        wy = [bspline(ty - k, self.order) for k in taps]
        if gradient:
            dwx = [bspline_derivative(tx - k, self.order) for k in taps]
            dwy = [bspline_derivative(ty - k, self.order) for k in taps]
        cw = self.coeffs.shape[-1]
        flat = self.coeffs.reshape(n_, c_, -1)
        xi = xi.long() + self.pad
        yi = yi.long() + self.pad
        val = torch.zeros(n_, c_, *x.shape[-2:], device=x.device)
        gx = torch.zeros_like(val) if gradient else None
        gy = torch.zeros_like(val) if gradient else None
        for a, ky in enumerate(taps):
            rows = (yi + ky) * cw
            row_val = torch.zeros_like(val)
            row_dx = torch.zeros_like(val) if gradient else None
            for b, kx in enumerate(taps):
                idx = (rows + xi + kx).reshape(x.shape[0], 1, -1).expand(n_, c_, -1)
                g = torch.gather(flat, 2, idx).reshape(val.shape)
                row_val = row_val + wx[b][:, None] * g
                if gradient:
                    row_dx = row_dx + dwx[b][:, None] * g
            val = val + wy[a][:, None] * row_val
            if gradient:
                gx = gx + wy[a][:, None] * row_dx
                gy = gy + dwy[a][:, None] * row_val
        if gradient:
            return val, gx, gy
        return val

    def _lerp(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Bilinear lookup of the (padded) coefficient image at pixel coordinates x, y."""
        hp, wp = self.coeffs.shape[-2:]
        gx = (x + self.pad) * (2.0 / (wp - 1)) - 1.0
        gy = (y + self.pad) * (2.0 / (hp - 1)) - 1.0
        grid = torch.stack((gx, gy), dim=-1)
        return F.grid_sample(
            self.coeffs, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

    def _sample_cubic(self, x: torch.Tensor, y: torch.Tensor, gradient: bool):
        """Cubic B-spline evaluation as bilinear lookups (Sigg & Hadwiger 2005, GPU Gems 2).

        Per axis the four tap weights w0..w3 (taps -1..2) are positive, so the sum of the
        outer pairs can be read with one linear interpolation each at a shifted position:
        value = g0 * lerp(i + h0) + g1 * lerp(i + h1), g0 = w0 + w1, h0 = -1 + w1 / g0,
        g1 = w2 + w3, h1 = 1 + w3 / g1. The derivative weights have constant sign within each
        pair, so the same construction gives the exact interpolant gradient. In 2-D this is 4
        fused bilinear lookups per quantity instead of 16 gathers.
        """
        xi, yi = torch.floor(x), torch.floor(y)
        tx, ty = x - xi, y - yi

        def pairs(t, deriv: bool):
            s = 1.0 - t
            if deriv:
                w0, w1 = -0.5 * s * s, (1.5 * t - 2.0) * t
                w2, w3 = 0.5 + t - 1.5 * t * t, 0.5 * t * t
            else:
                w0, w1 = s * s * s / 6.0, (3.0 * t * t * t - 6.0 * t * t + 4.0) / 6.0
                w2, w3 = (-3.0 * t * t * t + 3.0 * t * t + 3.0 * t + 1.0) / 6.0, t * t * t / 6.0
            g0, g1 = w0 + w1, w2 + w3
            return g0, g1, -1.0 + w1 / g0, 1.0 + w3 / g1

        def evaluate(dx: bool, dy: bool):
            gx0, gx1, hx0, hx1 = pairs(tx, dx)
            gy0, gy1, hy0, hy1 = pairs(ty, dy)
            out = 0.0
            for gy_, hy_ in ((gy0, hy0), (gy1, hy1)):
                for gx_, hx_ in ((gx0, hx0), (gx1, hx1)):
                    out = out + (gy_ * gx_)[:, None] * self._lerp(xi + hx_, yi + hy_)
            return out

        val = evaluate(False, False)
        if gradient:
            return val, evaluate(True, False), evaluate(False, True)
        return val

    def inside(self, flow: torch.Tensor) -> torch.Tensor:
        """(1, 1, H, W) boolean: x + flow lies inside the image (samples are not clamped)."""
        h, w = self.shape[-2:]
        ys = torch.arange(h, device=flow.device, dtype=flow.dtype)[:, None]
        xs = torch.arange(w, device=flow.device, dtype=flow.dtype)[None, :]
        x = xs + flow[0, 0]
        y = ys + flow[0, 1]
        return ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1))[None, None]

    def warp(self, flow: torch.Tensor, gradient: bool = False):
        """img(x + flow(x)) for a (1, 2, H, W) flow on this image's grid."""
        h, w = self.shape[-2:]
        ys = torch.arange(h, device=flow.device, dtype=torch.float32)
        xs = torch.arange(w, device=flow.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return self.sample(xx + flow[0, 0], yy + flow[0, 1], gradient=gradient)
