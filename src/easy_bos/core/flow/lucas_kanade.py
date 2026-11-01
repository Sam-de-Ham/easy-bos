"""Dense pyramidal Lucas–Kanade with an optional local illumination model (GPU).

At every pixel the displacement increment minimises, over a Gaussian window G,

    sum_G [ cur - (1 + a) ref_w - b - Ix du - Iy dv ]^2

where ref_w is the reference warped by the current estimate and (a, b) are a local gain and
offset (the iterative least-squares / DIC form recommended for BOS by Mercier & Lacassagne
2023). Iterated with re-warping on a coarse-to-fine pyramid.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from easy_bos.core import imops
from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register
from easy_bos.core.spline import SplineImage


@dataclass
class LucasKanadeParams(MethodParams):
    window_sigma: float = param(
        3.0,
        "Window sigma",
        "Gaussian window radius, pixels. Larger is smoother and less noisy.",
        minimum=0.75,
        maximum=8.0,
    )
    illumination: bool = param(True, "Illumination model", "Estimate a local gain and offset.")
    levels: int = param(3, "Pyramid levels", minimum=1, maximum=6, advanced=True)
    iterations: int = param(5, "Warps per level", minimum=1, maximum=20, advanced=True)
    presmooth: float = param(
        0.5,
        "Pre-smoothing",
        "Gaussian sigma applied to both images.",
        minimum=0.0,
        maximum=2.0,
        advanced=True,
    )


@register
class LucasKanade(FlowMethod):
    key = "lucas_kanade"
    label = "Lucas–Kanade"
    summary = "Dense local least squares with gain/offset (GPU). Fast, one main parameter."
    Params = LucasKanadeParams
    speed = "fast"

    def _estimate(self, ref, cur, mask=None):
        p: LucasKanadeParams = self.params
        ref = imops.gaussian_blur(ref, p.presmooth)
        cur = imops.gaussian_blur(cur, p.presmooth)
        pyr = [(ref, cur)]
        for _ in range(int(p.levels) - 1):
            r, c = pyr[-1]
            if min(r.shape[-2:]) < 32:
                break
            pyr.append((imops.downsample2(r), imops.downsample2(c)))
        disp = None
        cond = None
        for r, c in reversed(pyr):
            h, w = r.shape[-2:]
            disp = (
                torch.zeros(1, 2, h, w, device=r.device)
                if disp is None
                else imops.upsample_flow(disp, (h, w))
            )
            spline = SplineImage(r)
            for _ in range(int(p.iterations)):
                disp, cond = _lk_step(spline, c, disp, p.window_sigma, p.illumination)
        return disp, cond


def _lk_step(ref: SplineImage, cur, disp, sigma, illumination):
    """One Gauss-Newton step that solves for the *total* displacement in each window.

    Linearising ref(x + D) about the current estimate D_k gives
    cur - ref_w + grad(ref_w) . D_k  ~  grad(ref_w) . D  (+ a ref_w + b),
    so each step re-solves the window least-squares problem for D itself. Solving for an
    increment and accumulating it instead lets sub-window error components (which the window
    average cannot see) build up with every iteration.
    """
    ref_w, ix, iy = ref.warp(disp, gradient=True)
    valid = ref.inside(disp).float()
    e = (cur - ref_w + ix * disp[:, :1] + iy * disp[:, 1:]) * valid
    ix, iy = ix * valid, iy * valid
    blur = lambda t: imops.gaussian_blur(t, sigma)  # noqa: E731
    if not illumination:
        a11, a12, a22 = blur(ix * ix), blur(ix * iy), blur(iy * iy)
        b1, b2 = blur(ix * e), blur(iy * e)
        reg = 1e-4 * (a11 + a22).mean()
        a11, a22 = a11 + reg, a22 + reg
        det = a11 * a22 - a12 * a12
        du = (a22 * b1 - a12 * b2) / det
        dv = (a11 * b2 - a12 * b1) / det
        lam_min = 0.5 * (a11 + a22 - torch.sqrt((a11 - a22) ** 2 + 4 * a12**2))
        cond = (lam_min / (a11 + a22).mean()).clamp(0, 1)
        return torch.cat([du, dv], dim=1), cond
    feats = [ix, iy, ref_w * valid, valid]
    n = len(feats)
    m = torch.empty(*ix.shape[-2:], n, n, device=ix.device)
    rhs = torch.empty(*ix.shape[-2:], n, device=ix.device)
    for i in range(n):
        rhs[..., i] = blur(feats[i] * e)[0, 0]
        for j in range(i, n):
            v = blur(feats[i] * feats[j])[0, 0]
            m[..., i, j] = v
            m[..., j, i] = v
    scale = (m[..., 0, 0] + m[..., 1, 1]).mean()
    eye = torch.eye(n, device=ix.device)
    m = m + 1e-4 * scale * eye
    sol = torch.linalg.solve(m, rhs.unsqueeze(-1)).squeeze(-1)
    total = sol[..., :2].permute(2, 0, 1)[None]
    tr = m[..., 0, 0] + m[..., 1, 1]
    det = m[..., 0, 0] * m[..., 1, 1] - m[..., 0, 1] ** 2
    lam_min = 0.5 * (tr - torch.sqrt((tr**2 - 4 * det).clamp_min(0)))
    cond = (lam_min / tr.mean()).clamp(0, 1)[None, None]
    return total, cond
