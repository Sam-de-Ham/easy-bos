"""Shared image operations on torch tensors shaped (N, C, H, W) or (H, W)."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def as_4d(x: torch.Tensor) -> torch.Tensor:
    while x.dim() < 4:
        x = x.unsqueeze(0)
    return x


def to_tensor(
    a: np.ndarray | torch.Tensor, device: torch.device, dtype=torch.float32
) -> torch.Tensor:
    if isinstance(a, torch.Tensor):
        return a.to(device=device, dtype=dtype)
    return torch.from_numpy(np.ascontiguousarray(a)).to(device=device, dtype=dtype)


def gaussian_kernel1d(sigma: float, truncate: float = 4.0, device=None, dtype=torch.float32):
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float, mode: str = "replicate") -> torch.Tensor:
    """Separable Gaussian blur of a (N, C, H, W) tensor; ``sigma <= 0`` is a no-op."""
    if sigma <= 0:
        return x
    k = gaussian_kernel1d(sigma, device=x.device, dtype=x.dtype)
    r = (k.numel() - 1) // 2
    c = x.shape[1]
    kx = k.view(1, 1, 1, -1).expand(c, 1, 1, -1)
    ky = k.view(1, 1, -1, 1).expand(c, 1, -1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode=mode), kx, groups=c)
    return F.conv2d(F.pad(x, (0, 0, r, r), mode=mode), ky, groups=c)


def box_blur(x: torch.Tensor, size: int) -> torch.Tensor:
    if size <= 1:
        return x
    r = size // 2
    c = x.shape[1]
    k = torch.full((c, 1, size, size), 1.0 / (size * size), device=x.device, dtype=x.dtype)
    return F.conv2d(F.pad(x, (r, size - 1 - r, r, size - 1 - r), mode="replicate"), k, groups=c)


def disk_blur(x: torch.Tensor, diameter: float) -> torch.Tensor:
    """Convolve with a uniform disk (anti-aliased edge) of the given diameter in pixels."""
    if diameter <= 1.0:
        return x
    r = diameter / 2.0
    n = int(math.ceil(r)) + 1
    yy, xx = torch.meshgrid(
        torch.arange(-n, n + 1, device=x.device, dtype=x.dtype),
        torch.arange(-n, n + 1, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    k = torch.clamp(r + 0.5 - torch.sqrt(xx**2 + yy**2), 0.0, 1.0)
    k = k / k.sum()
    c = x.shape[1]
    return F.conv2d(F.pad(x, (n, n, n, n), mode="replicate"), k.expand(c, 1, -1, -1), groups=c)


def pixel_grid(h: int, w: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Pixel-centre coordinates (x, y) as (H, W) tensors."""
    ys = torch.arange(h, device=device, dtype=dtype)
    xs = torch.arange(w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx, yy


def sample(
    img: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    mode: str = "bicubic",
    padding_mode: str = "border",
) -> torch.Tensor:
    """Sample a (N, C, H, W) image at pixel coordinates x, y of shape (N, Hs, Ws)."""
    h, w = img.shape[-2:]
    gx = 2.0 * x / max(w - 1, 1) - 1.0
    gy = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(img, grid, mode=mode, padding_mode=padding_mode, align_corners=True)


def warp(img: torch.Tensor, flow: torch.Tensor, mode: str = "bicubic") -> torch.Tensor:
    """Return img(x + flow(x)). ``flow`` is (N, 2, H, W) in pixels (u right, v down)."""
    n, _, h, w = img.shape
    xx, yy = pixel_grid(h, w, img.device, img.dtype)
    return sample(img, xx + flow[:, 0], yy + flow[:, 1], mode=mode)


def central_gradient(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Fourth-order central differences [1, -8, 0, 8, -1]/12 with replicate borders."""
    c = img.shape[1]
    k = torch.tensor([1.0, -8.0, 0.0, 8.0, -1.0], device=img.device, dtype=img.dtype) / 12.0
    gx = F.conv2d(
        F.pad(img, (2, 2, 0, 0), mode="replicate"), k.view(1, 1, 1, 5).expand(c, 1, 1, 5), groups=c
    )
    gy = F.conv2d(
        F.pad(img, (0, 0, 2, 2), mode="replicate"), k.view(1, 1, 5, 1).expand(c, 1, 5, 1), groups=c
    )
    return gx, gy


def resize(x: torch.Tensor, size: tuple[int, int], mode: str = "area") -> torch.Tensor:
    if tuple(x.shape[-2:]) == tuple(size):
        return x
    if mode == "area":
        return F.interpolate(x, size=size, mode="area")
    return F.interpolate(x, size=size, mode=mode, align_corners=True)


def downsample2(x: torch.Tensor) -> torch.Tensor:
    """Anti-aliased factor-2 reduction (Gaussian sigma 1 then decimate)."""
    h, w = x.shape[-2:]
    return F.interpolate(
        gaussian_blur(x, 1.0),
        size=((h + 1) // 2, (w + 1) // 2),
        mode="bilinear",
        align_corners=True,
    )


def upsample_flow(flow: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Resize a flow field and rescale its vectors to the new pixel size (align-corners grid)."""
    h0, w0 = flow.shape[-2:]
    h1, w1 = size
    up = F.interpolate(flow, size=size, mode="bilinear", align_corners=True)
    sx = (w1 - 1) / max(w0 - 1, 1)
    sy = (h1 - 1) / max(h0 - 1, 1)
    return torch.stack((up[:, 0] * sx, up[:, 1] * sy), dim=1)


# --------------------------------------------------------------- finite differences
def laplacian(x: torch.Tensor) -> torch.Tensor:
    """5-point Laplacian with Neumann (reflecting) borders; a symmetric operator."""
    xp = F.pad(x, (1, 1, 1, 1), mode="replicate")
    return xp[..., 1:-1, :-2] + xp[..., 1:-1, 2:] + xp[..., :-2, 1:-1] + xp[..., 2:, 1:-1] - 4 * x


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
