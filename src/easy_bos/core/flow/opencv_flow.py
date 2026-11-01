"""OpenCV dense optical flow baselines: Farnebäck and DIS (CPU).

Farnebäck (polynomial expansion) is the method used by Pocket Schlieren and as pseudo ground
truth in event-based BOS; DIS is the fastest usable dense method. Both run on the CPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register


def _np(t: torch.Tensor) -> np.ndarray:
    return t[0, 0].detach().cpu().numpy().astype(np.float32)


@dataclass
class FarnebackParams(MethodParams):
    levels: int = param(3, "Pyramid levels", "Coarse-to-fine levels.", minimum=1, maximum=6)
    window: int = param(
        15, "Window", "Averaging window in pixels; larger is smoother.", minimum=5, maximum=61
    )
    iterations: int = param(5, "Iterations", minimum=1, maximum=20, advanced=True)
    poly_n: int = param(
        7, "Polynomial support", "5 or 7 pixels.", choices=("5", "7"), advanced=True
    )
    poly_sigma: float = param(1.5, "Polynomial sigma", minimum=0.8, maximum=2.5, advanced=True)
    gaussian_window: bool = param(True, "Gaussian window", advanced=True)


@register
class Farneback(FlowMethod):
    key = "farneback"
    label = "Farnebäck"
    summary = "Polynomial-expansion optical flow (OpenCV). Fast, dense, CPU."
    Params = FarnebackParams
    speed = "fast"
    uses_gpu = False

    def _estimate(self, ref, cur, mask=None):
        p: FarnebackParams = self.params
        flags = cv2.OPTFLOW_FARNEBACK_GAUSSIAN if p.gaussian_window else 0
        # Farnebäck's internal thresholds assume 8-bit-like intensity ranges, so map the
        # normalised images (zero mean, unit std) to roughly [0, 255] while staying float.
        to_range = lambda a: _np(a) * 40.0 + 128.0  # noqa: E731
        # OpenCV returns flow with prev(x) ~ next(x + flow): prev = cur, next = ref.
        f = cv2.calcOpticalFlowFarneback(
            to_range(cur),
            to_range(ref),
            None,
            0.5,
            int(p.levels),
            int(p.window),
            int(p.iterations),
            int(p.poly_n),
            float(p.poly_sigma),
            flags,
        )
        d = torch.from_numpy(f.transpose(2, 0, 1).copy()).to(ref.device)[None]
        return d, None


@dataclass
class DisParams(MethodParams):
    preset: str = param("medium", "Preset", choices=("fast", "medium", "ultrafast"))
    patch_size: int = param(8, "Patch size", minimum=4, maximum=32, advanced=True)
    patch_stride: int = param(3, "Patch stride", minimum=1, maximum=16, advanced=True)
    refinement_iterations: int = param(
        10, "Variational refinement iterations", minimum=0, maximum=50, advanced=True
    )


@register
class Dis(FlowMethod):
    key = "dis"
    label = "DIS"
    summary = "Dense inverse search (OpenCV). The fastest dense method; 8-bit internally."
    Params = DisParams
    speed = "instant"
    uses_gpu = False

    def _estimate(self, ref, cur, mask=None):
        p: DisParams = self.params
        preset = {
            "ultrafast": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
            "fast": cv2.DISOPTICAL_FLOW_PRESET_FAST,
            "medium": cv2.DISOPTICAL_FLOW_PRESET_MEDIUM,
        }[p.preset]
        dis = cv2.DISOpticalFlow_create(preset)
        dis.setPatchSize(int(p.patch_size))
        dis.setPatchStride(int(p.patch_stride))
        dis.setVariationalRefinementIterations(int(p.refinement_iterations))
        r, c = _np(ref), _np(cur)
        lo, hi = np.percentile(r, [0.1, 99.9])
        scale = 255.0 / max(hi - lo, 1e-6)
        to8 = lambda a: np.clip((a - lo) * scale, 0, 255).astype(np.uint8)  # noqa: E731
        f = dis.calc(to8(c), to8(r), None)
        d = torch.from_numpy(f.transpose(2, 0, 1).copy()).to(ref.device)[None]
        return d, None
