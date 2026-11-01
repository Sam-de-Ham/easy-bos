"""RAFT learned optical flow (torchvision), offered as an experimental comparison.

RAFT (Teed & Deng, ECCV 2020) was trained on large-motion synthetic datasets and computes its
correlation volume at 1/8 resolution with learned upsampling. There is no published evidence
that off-the-shelf learned flow resolves BOS-scale (0.01-2 px) displacements; event-based BOS
(Shiba et al. 2024) reports that FlowFormer and RIFE fail on real BOS. The synthetic benchmark
quantifies how it compares here.

Weights (raft_large C_T_SKHT_V2, ~20 MB) download on first use into the user data directory,
never into the repository. They were trained on research datasets (FlyingChairs/Things,
Sintel, KITTI, HD1K), whose terms may restrict commercial use.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from easy_bos.core.flow.base import FlowMethod, MethodParams, param, register
from easy_bos.core.paths import model_dir

_MODEL_CACHE: dict[str, torch.nn.Module] = {}


@dataclass
class RaftParams(MethodParams):
    iterations: int = param(20, "Refinement iterations", minimum=4, maximum=48)
    upscale: int = param(
        1,
        "Upscale input",
        "Enlarge images before inference so displacements are larger in RAFT's working resolution.",
        choices=("1", "2"),
    )


def weights_available() -> bool:
    from torchvision.models.optical_flow import Raft_Large_Weights

    url = Raft_Large_Weights.DEFAULT.url
    return (model_dir() / "hub" / "checkpoints" / os.path.basename(url)).exists()


def load_model(device: torch.device) -> torch.nn.Module:
    key = str(device)
    if key not in _MODEL_CACHE:
        os.environ["TORCH_HOME"] = str(model_dir())
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        _MODEL_CACHE[key] = model.eval().to(device)
    return _MODEL_CACHE[key]


@register
class Raft(FlowMethod):
    key = "raft"
    label = "RAFT (experimental)"
    summary = (
        "Learned optical flow trained on general video. Not designed for sub-pixel BOS; "
        "included for comparison. Downloads 20 MB of weights on first use."
    )
    Params = RaftParams
    speed = "moderate"

    def _estimate(self, ref, cur, mask=None):
        p: RaftParams = self.params
        model = load_model(self.device)
        h, w = ref.shape[-2:]
        up = int(p.upscale)

        def prep(img):
            x = (img / 3.0).clamp(-1, 1)
            if up > 1:
                x = F.interpolate(x, scale_factor=up, mode="bicubic", align_corners=False)
            hh, ww = x.shape[-2:]
            ph, pw = (-hh) % 8, (-ww) % 8
            x = F.pad(x, (0, pw, 0, ph), mode="replicate")
            return x.expand(-1, 3, -1, -1).contiguous()

        with torch.no_grad():
            # RAFT returns flow with img1(x) ~ img2(x + flow): img1 = cur, img2 = ref.
            flows = model(prep(cur), prep(ref), num_flow_updates=int(p.iterations))
            f = flows[-1][..., : h * up, : w * up]
            if up > 1:
                f = F.interpolate(f, size=(h, w), mode="area") / up
        return f, None
