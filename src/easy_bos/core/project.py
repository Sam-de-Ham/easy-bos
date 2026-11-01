"""A project: setup geometry, background, capture, processing and reconstruction settings.

Serialised as human-readable JSON (``.ebos``). Results live next to it in an HDF5 file. Regions
(ROI, masks) are stored as rectangles in source-pixel coordinates and rasterised on demand, so
projects stay small and independent of binning.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from easy_bos.core.geometry import CAMERA_PRESETS, Camera, Display, Setup
from easy_bos.core.patterns import PatternKind, PatternSpec
from easy_bos.core.pipeline import QUALITY, ProcessingSettings, ReferenceMode, auto_binning
from easy_bos.core.reconstruct.density import FlowGeometry, Gas, ReconstructionSettings

VERSION = 1
Rect = tuple[int, int, int, int]  # x, y, w, h in source pixels


@dataclass
class SetupState:
    camera_name: str = "Sony A6400 · 4K video · 50 mm"
    focal_length_mm: float = 50.0
    sensor_width_mm: float = 23.5
    image_width_px: int = 3840
    image_height_px: int = 2160
    z_a_m: float = 1.0
    z_d_m: float = 1.0
    f_number: float = 8.0

    def camera(self) -> Camera:
        return Camera(
            self.focal_length_mm * 1e-3,
            self.sensor_width_mm * 1e-3,
            self.image_width_px,
            self.image_height_px,
            self.camera_name,
        )

    def setup(self) -> Setup:
        return Setup(self.camera(), self.z_a_m, self.z_d_m, self.f_number)

    def apply_preset(self, name: str) -> None:
        cam = CAMERA_PRESETS[name]
        self.camera_name = name
        self.focal_length_mm = cam.focal_length_m * 1e3
        self.sensor_width_mm = cam.sensor_width_m * 1e3
        self.image_width_px = cam.image_width_px
        self.image_height_px = cam.image_height_px


@dataclass
class BackgroundState:
    kind: str = "screen"  # "screen", "print", "natural"
    display_name: str = ""
    display_width_px: int = 2560
    display_height_px: int = 1440
    display_diagonal_in: float = 27.0
    pattern: str = PatternKind.SPECKLE.value
    element_px: int = 3
    auto_element: bool = True
    seed: int = 1
    invert: bool = False
    paper: str = "A3"
    print_dpi: int = 600
    print_element_mm: float = 0.6

    def display(self) -> Display:
        return Display(
            self.display_width_px,
            self.display_height_px,
            self.display_diagonal_in * 0.0254,
            self.display_name or "Display",
        )

    def spec(self, element_px: int | None = None) -> PatternSpec:
        return PatternSpec(
            PatternKind(self.pattern), element_px or self.element_px, self.seed, invert=self.invert
        )


@dataclass
class CaptureState:
    paths: list[str] = field(default_factory=list)
    channel: str = "green"


@dataclass
class ProcessState:
    quality: str = "best"  # fast, balanced, best, or custom (method + parameters below)
    method: str = "variational"
    method_params: dict = field(default_factory=dict)
    reference_mode: str = ReferenceMode.AVERAGE.value
    reference_range: tuple[int, int] = (0, 8)
    frame_range: tuple[int, int, int] | None = None
    roi: Rect | None = None
    exclusions: list[Rect] = field(default_factory=list)  # regions without valid background
    stable_regions: list[Rect] = field(default_factory=list)
    stabilize: str = "auto"  # auto, on, off
    motion_model: str = "rigid"
    exposure_correction: bool = True
    auto_mask: bool = True
    banding_correction: bool = False
    binning: int = 0  # 0 = automatic (see pipeline.auto_binning)
    temporal_average: int = 1

    def effective_binning(self, width: int) -> int:
        return self.binning if self.binning > 0 else auto_binning(width)

    def settings(self, image_shape: tuple[int, int]) -> ProcessingSettings:
        """Processing settings with masks rasterised for the ROI/binned image shape."""
        h, w = image_shape
        x0, y0, rw, rh = self.roi if self.roi else (0, 0, w, h)
        b = self.effective_binning(w)
        shape = (rh // b, rw // b)
        mask = (
            _raster(self.exclusions, (x0, y0), b, shape, inverse=True) if self.exclusions else None
        )
        stable = _raster(self.stable_regions, (x0, y0), b, shape) if self.stable_regions else None
        method, params = (
            (QUALITY[self.quality][0], {})
            if self.quality in QUALITY
            else (self.method, dict(self.method_params))
        )
        return ProcessingSettings(
            method=method,
            method_params=params,
            reference_mode=ReferenceMode(self.reference_mode),
            reference_range=tuple(self.reference_range),
            frame_range=tuple(self.frame_range) if self.frame_range else None,
            roi=self.roi,
            mask=None if mask is None else mask.astype(np.float32),
            stable_region=stable,
            stabilize=self.stabilize,
            motion_model=self.motion_model,
            exposure_correction=self.exposure_correction,
            auto_mask=self.auto_mask,
            banding_correction=self.banding_correction,
            binning=b,
            temporal_average=self.temporal_average,
        )


def _raster(rects: list[Rect], origin, binning, shape, inverse=False) -> np.ndarray:
    m = np.zeros(shape, dtype=bool)
    ox, oy = origin
    for x, y, w, h in rects:
        if inverse:  # exclusions: round outward, so partly excluded binned pixels are excluded
            xa, ya = max(0, (x - ox) // binning), max(0, (y - oy) // binning)
            xb = max(0, -(-(x - ox + w) // binning))
            yb = max(0, -(-(y - oy + h) // binning))
        else:  # stable regions: round inward, so every binned pixel is truly stable
            xa, ya = max(0, -(-(x - ox) // binning)), max(0, -(-(y - oy) // binning))
            xb, yb = max(0, (x - ox + w) // binning), max(0, (y - oy + h) // binning)
        m[ya:yb, xa:xb] = True
    return ~m if inverse else m


@dataclass
class ReconstructState:
    geometry: str = FlowGeometry.AXISYMMETRIC.value
    depth_mm: float = 100.0
    anchor_sides: str = "lr"
    axis_auto: bool = True
    axis_px: float = 0.0
    axis_orientation: str = "vertical"
    abel_method: str = "indirect"
    downsample: int = 2
    t_ambient_c: float = 20.0
    p_ambient_kpa: float = 101.325
    k_gd: float = 2.26e-4

    def settings(self) -> ReconstructionSettings:
        return ReconstructionSettings(
            FlowGeometry(self.geometry),
            self.depth_mm * 1e-3,
            self.anchor_sides,
            "auto" if self.axis_auto else float(self.axis_px),
            self.axis_orientation,
            self.abel_method,
            self.downsample,
            Gas(self.t_ambient_c + 273.15, self.p_ambient_kpa * 1e3, self.k_gd),
        )


@dataclass
class Project:
    setup: SetupState = field(default_factory=SetupState)
    background: BackgroundState = field(default_factory=BackgroundState)
    capture: CaptureState = field(default_factory=CaptureState)
    process: ProcessState = field(default_factory=ProcessState)
    reconstruct: ReconstructState = field(default_factory=ReconstructState)
    path: str | None = None

    def to_json(self) -> str:
        d = asdict(self)
        d.pop("path")
        d["version"] = VERSION
        return json.dumps(d, indent=2)

    @classmethod
    def from_json(cls, text: str) -> Project:
        d = json.loads(text)
        d.pop("version", None)
        p = cls()
        for name, sub in [
            ("setup", SetupState),
            ("background", BackgroundState),
            ("capture", CaptureState),
            ("process", ProcessState),
            ("reconstruct", ReconstructState),
        ]:
            if name in d:
                known = {k: v for k, v in d[name].items() if k in sub.__dataclass_fields__}
                setattr(p, name, sub(**known))
        if isinstance(p.process.stabilize, bool):  # projects saved before the auto mode
            p.process.stabilize = "on" if p.process.stabilize else "off"
        p.process.reference_range = tuple(p.process.reference_range)
        if p.process.frame_range is not None:
            p.process.frame_range = tuple(p.process.frame_range)
        p.process.roi = tuple(p.process.roi) if p.process.roi else None
        p.process.exclusions = [tuple(r) for r in p.process.exclusions]
        p.process.stable_regions = [tuple(r) for r in p.process.stable_regions]
        return p

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(self.to_json(), encoding="utf-8")
        self.path = str(path)

    @classmethod
    def load(cls, path: str | Path) -> Project:
        p = cls.from_json(Path(path).read_text(encoding="utf-8"))
        p.path = str(path)
        return p

    @property
    def results_path(self) -> Path | None:
        return Path(self.path).with_suffix(".results.h5") if self.path else None
