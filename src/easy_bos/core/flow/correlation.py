"""GPU cross-correlation (PIV-style) with multi-pass window deformation.

Standard practice from PIV (Scarano 2002, Meas Sci Technol 13:R1; Raffel et al., *Particle
Image Velocimetry*, 3rd ed.):

* windows shrink over passes (4w -> 2w -> w, then w again);
* the reference is deformed by the current predictor (B-spline interpolation), so each pass
  measures only a small residual and D stays on the grid of the distorted image; the update is
  low-pass filtered at the window scale (Schrijer & Scarano 2008, Exp Fluids 45:927);
* FFT cross-correlation of mean-subtracted windows, peak by three-point Gaussian fit;
* normalised median test (Westerweel & Scarano 2005, Exp Fluids 39:1096) replaces outliers.

The vector grid is coarser than the image (window size / overlap); it is interpolated back to
pixels with bicubic interpolation so all methods return a dense field.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from easy_bos.core import imops
from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register
from easy_bos.core.spline import SplineImage


@dataclass
class CorrelationParams(MethodParams):
    window: int = param(
        16,
        "Final window",
        "Interrogation window of the last pass, pixels.",
        choices=("8", "12", "16", "24", "32", "48", "64"),
    )
    overlap: float = param(
        0.75, "Overlap", "Window overlap; 0.5-0.75 is typical.", minimum=0.0, maximum=0.875
    )
    passes: int = param(
        4, "Passes", "Total passes; the first uses a 4x larger window.", minimum=1, maximum=6
    )
    median_threshold: float = param(
        2.0, "Outlier threshold", "Normalised median test.", minimum=1.0, maximum=5.0, advanced=True
    )
    gaussian_window: bool = param(
        False,
        "Gaussian weighting",
        "Weight windows with a Gaussian (reduces edge effects).",
        advanced=True,
    )


@register
class CrossCorrelation(FlowMethod):
    key = "correlation"
    label = "Cross-correlation"
    summary = "Multi-pass PIV cross-correlation with window deformation (GPU). Robust baseline."
    Params = CorrelationParams
    speed = "fast"

    def _estimate(self, ref, cur, mask=None):
        p: CorrelationParams = self.params
        h, w = ref.shape[-2:]
        final = int(p.window)
        sizes = [final * 4, final * 2, final] + [final] * max(0, int(p.passes) - 3)
        sizes = sizes[-int(p.passes) :] if p.passes < 3 else sizes
        disp = torch.zeros(1, 2, h, w, device=ref.device)
        conf = torch.ones(1, 1, h, w, device=ref.device)
        ref_s = SplineImage(ref)
        for ws in sizes:
            ws = min(ws, h, w)
            step = max(1, round(ws * (1.0 - p.overlap)))
            # Deform the reference only: windows of cur stay on cur's grid, so D satisfies
            # cur(x) = ref(x + D(x)) exactly as the other estimators do.
            ref_w = ref_s.warp(disp) if disp.abs().max() > 0 else ref
            vec, peak, centers = _correlate(ref_w, cur, ws, step, p.gaussian_window)
            vec = _median_validate(vec, p.median_threshold)
            residual = _to_dense(vec, centers, (h, w))
            # Low-pass the update at the window scale (Schrijer & Scarano 2008): sub-window
            # components cannot be measured and would otherwise accumulate over passes.
            disp = disp + imops.gaussian_blur(residual, ws / 4.0)
            conf = _to_dense(peak, centers, (h, w)).clamp(0, 1)
        return disp, conf


def _correlate(ref: torch.Tensor, cur: torch.Tensor, ws: int, step: int, gaussian: bool):
    """Correlate all windows; returns vectors (1, 2, ny, nx), peak height and centre offsets."""
    h, w = ref.shape[-2:]
    ny = (h - ws) // step + 1
    nx = (w - ws) // step + 1
    # Centre the window grid in the image.
    oy = (h - ((ny - 1) * step + ws)) // 2
    ox = (w - ((nx - 1) * step + ws)) // 2
    r = ref[..., oy : oy + (ny - 1) * step + ws, ox : ox + (nx - 1) * step + ws]
    c = cur[..., oy : oy + (ny - 1) * step + ws, ox : ox + (nx - 1) * step + ws]
    rw = F.unfold(r, ws, stride=step).transpose(1, 2).reshape(-1, ws, ws)
    cw = F.unfold(c, ws, stride=step).transpose(1, 2).reshape(-1, ws, ws)
    rw = rw - rw.mean(dim=(1, 2), keepdim=True)
    cw = cw - cw.mean(dim=(1, 2), keepdim=True)
    if gaussian:
        t = (torch.arange(ws, device=ref.device, dtype=ref.dtype) - (ws - 1) / 2) / (ws / 4.0)
        g = torch.exp(-0.5 * t**2)
        g2 = g[:, None] * g[None, :]
        rw, cw = rw * g2, cw * g2
    norm = torch.sqrt((rw**2).sum(dim=(1, 2)) * (cw**2).sum(dim=(1, 2))).clamp_min(1e-12)
    # C(s) = sum_x cur(x) ref(x + s): the peak location is the displacement.
    spec = torch.conj(torch.fft.rfft2(cw)) * torch.fft.rfft2(rw)
    corr = torch.fft.irfft2(spec, s=(ws, ws)) / norm[:, None, None]
    corr = torch.fft.fftshift(corr, dim=(1, 2))
    flat = corr.reshape(corr.shape[0], -1).argmax(dim=1)
    py, px = flat // ws, flat % ws
    # Peaks on the border have no neighbours for the sub-pixel fit: invalid (NaN), so the
    # median test replaces them instead of keeping a clamped, wrong vector.
    on_border = (py < 1) | (py > ws - 2) | (px < 1) | (px > ws - 2)
    py = py.clamp(1, ws - 2)
    px = px.clamp(1, ws - 2)
    idx = torch.arange(corr.shape[0], device=ref.device)
    c0 = corr[idx, py, px]
    dy = _gauss3(corr[idx, py - 1, px], c0, corr[idx, py + 1, px])
    dx = _gauss3(corr[idx, py, px - 1], c0, corr[idx, py, px + 1])
    vy = py.float() - ws // 2 + dy
    vx = px.float() - ws // 2 + dx
    vx = torch.where(on_border, torch.nan, vx)
    vy = torch.where(on_border, torch.nan, vy)
    vec = torch.stack([vx, vy]).reshape(1, 2, ny, nx)
    peak = torch.where(on_border, 0.0, c0).reshape(1, 1, ny, nx)
    # Window centre of grid node (0, 0) and the spacing, in pixel coordinates.
    centers = (ox + (ws - 1) / 2.0, oy + (ws - 1) / 2.0, float(step))
    return vec, peak, centers


def _gauss3(cm, c0, cp):
    """Three-point Gaussian sub-pixel estimator; falls back to parabolic for non-positive values."""
    eps = 1e-6
    ok = (cm > eps) & (c0 > eps) & (cp > eps)
    lm, l0, lp = (
        torch.log(cm.clamp_min(eps)),
        torch.log(c0.clamp_min(eps)),
        torch.log(cp.clamp_min(eps)),
    )
    den_g = 2.0 * (lm - 2.0 * l0 + lp)
    den_p = 2.0 * (cm - 2.0 * c0 + cp)
    g = torch.where(den_g.abs() > 1e-12, (lm - lp) / den_g, torch.zeros_like(den_g))
    q = torch.where(den_p.abs() > 1e-12, (cm - cp) / den_p, torch.zeros_like(den_p))
    return torch.where(ok, g, q).clamp(-0.5, 0.5)


def _median_validate(vec: torch.Tensor, threshold: float, eps: float = 0.1) -> torch.Tensor:
    """Normalised median test (Westerweel & Scarano 2005) over the existing 3x3 neighbours.

    Outliers and invalid (NaN) vectors are replaced by the neighbourhood median; vectors with no
    valid neighbour fall back to the global median.
    """
    _, _, ny, nx = vec.shape
    pad = F.pad(vec, (1, 1, 1, 1), value=float("nan"))
    nb = F.unfold(pad, 3).reshape(1, 2, 9, ny, nx)
    nb = torch.cat([nb[:, :, :4], nb[:, :, 5:]], dim=2)  # exclude the centre
    med = torch.nanquantile(nb, 0.5, dim=2)
    resid = torch.nanquantile((nb - med[:, :, None]).abs(), 0.5, dim=2)
    r = (vec - med).abs() / (resid + eps)
    bad = (r > threshold).any(dim=1, keepdim=True) | torch.isnan(vec).any(dim=1, keepdim=True)
    out = torch.where(bad, med, vec)
    if torch.isnan(out).any():
        fallback = torch.nanmedian(vec.reshape(2, -1), dim=1).values.view(1, 2, 1, 1)
        out = torch.where(torch.isnan(out), torch.nan_to_num(fallback), out)
    return out


def _to_dense(grid: torch.Tensor, centers, size) -> torch.Tensor:
    """Bicubic interpolation of node values to every pixel (clamped outside the node hull)."""
    x0, y0, step = centers
    h, w = size
    xx, yy = imops.pixel_grid(h, w, grid.device)
    gx = (xx - x0) / step
    gy = (yy - y0) / step
    return imops.sample(grid, gx[None], gy[None], mode="bicubic")
