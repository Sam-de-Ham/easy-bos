"""Common interface for displacement estimators.

Convention (shared with the synthetic renderer): for a reference image ``ref`` and a distorted
image ``cur``, the displacement field D satisfies

    cur(x) = ref(x + D(x))

with D = (u, v) in pixels, u to the right and v downward, sampled on the pixel grid of ``cur``.
D is proportional to the deflection angle (``geometry.Setup.pixels_per_radian``).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields, replace
from typing import Any, ClassVar

import numpy as np
import torch

from easy_bos.core import imops
from easy_bos.core.device import resolve


@dataclass
class Param:
    """UI metadata for a method parameter."""

    label: str
    help: str
    minimum: float | None = None
    maximum: float | None = None
    log: bool = False
    choices: tuple[str, ...] | None = None
    advanced: bool = False


def param(default, label: str, help: str = "", **kw):
    return field(default=default, metadata={"param": Param(label, help, **kw)})


@dataclass
class MethodParams:
    """Base for per-method parameter dataclasses."""

    def with_(self, **kw):
        return replace(self, **kw)

    def as_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def describe(cls) -> list[tuple[str, Param, Any]]:
        out = []
        for f in fields(cls):
            p = f.metadata.get("param")
            if p is not None:
                out.append((f.name, p, f.default))
        return out


@dataclass
class FlowResult:
    displacement: np.ndarray  # (2, H, W) float32, pixels
    method: str
    params: dict[str, Any]
    elapsed_s: float
    confidence: np.ndarray | None = None  # (H, W) in [0, 1], optional


class FlowMethod(ABC):
    key: ClassVar[str]
    label: ClassVar[str]
    summary: ClassVar[str]
    Params: ClassVar[type[MethodParams]]
    #: relative speed class for the UI: "instant", "fast", "moderate", "slow"
    speed: ClassVar[str] = "moderate"
    uses_gpu: ClassVar[bool] = True

    def __init__(self, params: MethodParams | None = None, device: str | torch.device | None = None):
        self.params = params or self.Params()
        self.device = resolve(device)

    @abstractmethod
    def _estimate(self, ref: torch.Tensor, cur: torch.Tensor, mask: torch.Tensor | None
                  ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """ref, cur: (1, 1, H, W) float32 on ``self.device``; mask: (1, 1, H, W) with 1 where
        the background is valid, or None. Returns ((1, 2, H, W), confidence or None)."""

    def estimate(self, ref: np.ndarray | torch.Tensor, cur: np.ndarray | torch.Tensor,
                 mask: np.ndarray | torch.Tensor | None = None) -> FlowResult:
        t0 = time.perf_counter()
        r = imops.as_4d(imops.to_tensor(ref, self.device))
        c = imops.as_4d(imops.to_tensor(cur, self.device))
        if r.shape != c.shape:
            raise ValueError(f"image shapes differ: {tuple(r.shape)} vs {tuple(c.shape)}")
        with torch.no_grad():
            r, c = normalize_pair(r, c)
        m = imops.as_4d(imops.to_tensor(mask, self.device)) if mask is not None else None
        d, conf = self._estimate(r, c, m)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        out = d[0].detach().float().cpu().numpy()
        conf_np = conf.detach().float().cpu().numpy().reshape(out.shape[1:]) if conf is not None else None
        return FlowResult(out, self.key, self.params.as_dict(), time.perf_counter() - t0, conf_np)


def normalize_pair(ref: torch.Tensor, cur: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-mean, unit-contrast scaling shared by both images (keeps brightness constancy)."""
    mean = ref.mean()
    std = ref.std().clamp_min(1e-6)
    return (ref - mean) / std, (cur - mean) / std


REGISTRY: dict[str, type[FlowMethod]] = {}


def register(cls: type[FlowMethod]) -> type[FlowMethod]:
    REGISTRY[cls.key] = cls
    return cls


def create(key: str, params: MethodParams | dict | None = None, device=None) -> FlowMethod:
    cls = REGISTRY[key]
    if isinstance(params, dict):
        params = cls.Params(**params)
    return cls(params, device)
