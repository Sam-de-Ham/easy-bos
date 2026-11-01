"""Processing pipeline: source frames -> corrected pairs -> displacement fields.

Usage::

    proc = Processor(source, ProcessingSettings(method="variational"))
    for result in proc.run(progress=print):
        ...

The pipeline is UI-agnostic; the GUI runs it in a worker thread and the benchmark drives it
directly, so benchmark numbers describe exactly what users get.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field, replace
from enum import Enum

import numpy as np

from easy_bos.core import preprocess
from easy_bos.core.flow import create
from easy_bos.core.media import Source


class ReferenceMode(Enum):
    AVERAGE = "average"
    MEDIAN = "median"
    PREVIOUS = "previous"

    @property
    def label(self) -> str:
        return {
            "average": "Average of reference frames",
            "median": "Median of reference frames",
            "previous": "Previous frame (frame to frame)",
        }[self.value]


@dataclass
class ProcessingSettings:
    method: str = "variational"
    method_params: dict = field(default_factory=dict)
    reference_mode: ReferenceMode = ReferenceMode.AVERAGE
    reference_range: tuple[int, int] = (0, 8)  # [start, stop) frames without flow
    frame_range: tuple[int, int, int] | None = None  # (start, stop, step) frames to process
    roi: tuple[int, int, int, int] | None = None  # x, y, w, h in source pixels
    mask: np.ndarray | None = None  # valid background (ROI shape after binning)
    stable_region: np.ndarray | None = None  # flow-free region for motion fit (same shape)
    stabilize: bool = True
    motion_model: str = "rigid"  # translation, rigid or affine
    exposure_correction: bool = True
    banding_correction: bool = False
    binning: int = 1
    temporal_average: int = 1
    device: str = "auto"

    def with_(self, **kw) -> ProcessingSettings:
        return replace(self, **kw)


@dataclass
class FrameResult:
    index: int
    displacement: np.ndarray  # (2, H, W) px in processed-image pixels
    confidence: np.ndarray | None
    motion: preprocess.Motion | None
    banding: float
    elapsed_s: float


class Processor:
    def __init__(self, source: Source, settings: ProcessingSettings):
        self.source = source
        self.settings = settings
        self.method = create(
            settings.method,
            settings.method_params,
            None if settings.device == "auto" else settings.device,
        )
        self._reference: np.ndarray | None = None

    # ------------------------------------------------------------------ frames
    def load(self, index: int) -> np.ndarray:
        """Frame ``index`` cropped to the ROI and binned."""
        img = self.source.frame(index)
        s = self.settings
        if s.roi is not None:
            x, y, w, h = s.roi
            img = img[y : y + h, x : x + w]
        b = max(1, int(s.binning))
        if b > 1:
            hh, ww = img.shape[0] // b, img.shape[1] // b
            img = img[: hh * b, : ww * b].reshape(hh, b, ww, b).mean(axis=(1, 3))
        return np.ascontiguousarray(img, dtype=np.float32)

    def frame_indices(self) -> list[int]:
        n = len(self.source)
        s = self.settings
        if s.frame_range is None:
            start = s.reference_range[1] if s.reference_mode is not ReferenceMode.PREVIOUS else 1
            return list(range(min(start, n - 1), n))
        a, b, c = s.frame_range
        return list(range(max(0, a), min(b, n), max(1, c)))

    @property
    def reference(self) -> np.ndarray:
        if self._reference is None:
            s = self.settings
            a, b = s.reference_range
            frames = [self.load(i) for i in range(max(0, a), min(max(b, a + 1), len(self.source)))]
            self._reference = preprocess.average_reference(
                frames, robust=s.reference_mode is ReferenceMode.MEDIAN
            )
        return self._reference

    # ------------------------------------------------------------------ processing
    def process_pair(self, ref: np.ndarray, cur: np.ndarray, index: int = -1) -> FrameResult:
        s = self.settings
        t0 = time.perf_counter()
        valid = s.mask
        if s.exposure_correction:
            cur = preprocess.match_exposure(cur, ref, valid)
        band = preprocess.banding_strength(cur, ref)
        if s.banding_correction:
            cur = preprocess.correct_row_banding(cur, ref)
        motion = None
        ref_used = ref
        if s.stabilize:
            dev = None if s.device == "auto" else s.device
            motion = preprocess.estimate_motion(ref, cur, s.stable_region, s.motion_model, dev)
            ref_used = preprocess.warp_reference(ref, motion, dev)
        res = self.method.estimate(ref_used, cur, valid)
        disp = res.displacement
        if valid is not None:
            disp = np.where(valid[None] > 0, disp, np.nan).astype(np.float32)
        return FrameResult(index, disp, res.confidence, motion, band, time.perf_counter() - t0)

    def process_frame(self, index: int, previous: np.ndarray | None = None) -> FrameResult:
        cur = self.load(index)
        if self.settings.reference_mode is ReferenceMode.PREVIOUS:
            ref = previous if previous is not None else self.load(max(0, index - 1))
        else:
            ref = self.reference
        return self.process_pair(ref, cur, index)

    def run(
        self,
        progress: Callable[[int, int, FrameResult], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[FrameResult]:
        """Process all selected frames, yielding results (temporally averaged if requested)."""
        indices = self.frame_indices()
        window: list[FrameResult] = []
        k = max(1, int(self.settings.temporal_average))
        prev = None
        for n, i in enumerate(indices):
            if cancelled is not None and cancelled():
                return
            res = self.process_frame(i, prev)
            if self.settings.reference_mode is ReferenceMode.PREVIOUS:
                prev = self.load(i)
            if k > 1:
                window.append(res)
                window = window[-k:]
                avg = np.nanmean(np.stack([w.displacement for w in window]), axis=0)
                res = replace(res, displacement=avg.astype(np.float32))
            if progress is not None:
                progress(n + 1, len(indices), res)
            yield res
