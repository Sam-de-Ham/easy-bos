"""Uniform access to captures: videos, image sequences, RAW stills and in-memory arrays.

Every source yields single-channel float32 frames in [0, 1] at full bit depth (10-bit HEVC and
16-bit TIFF/RAW keep their precision). Colour is reduced to the green channel by default:
it has twice the Bayer sampling density and the sharpest optics of the three.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".m2ts", ".webm", ".mxf"}
IMAGE_EXTS = {".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".webp"}
RAW_EXTS = {".arw", ".dng", ".cr2", ".cr3", ".nef", ".raf", ".orf", ".rw2", ".pef", ".srw"}


@dataclass
class SourceInfo:
    name: str
    width: int
    height: int
    frame_count: int
    fps: float | None
    bit_depth: int
    kind: str  # "video", "images", "raw", "array"
    details: dict = field(default_factory=dict)


class Source(ABC):
    channel: str = "green"  # "green", "luma", "red", "blue"

    @property
    @abstractmethod
    def info(self) -> SourceInfo: ...

    @abstractmethod
    def frame(self, index: int) -> np.ndarray:
        """Return frame ``index`` as float32 (H, W) in [0, 1]."""

    def __len__(self) -> int:
        return self.info.frame_count

    def close(self) -> None:  # noqa: B027 - optional hook
        pass


def to_gray(img: np.ndarray, channel: str = "green") -> np.ndarray:
    """Reduce an (H, W) or (H, W, 3) RGB image to one float32 channel."""
    if img.ndim == 2:
        return img.astype(np.float32)
    if channel == "luma":
        return (0.2126 * img[..., 0] + 0.7152 * img[..., 1] + 0.0722 * img[..., 2]).astype(
            np.float32
        )
    idx = {"red": 0, "green": 1, "blue": 2}[channel]
    return img[..., idx].astype(np.float32)


class ArraySource(Source):
    def __init__(self, frames: np.ndarray, fps: float | None = 30.0, name: str = "array"):
        self._frames = np.asarray(frames, dtype=np.float32)
        t, h, w = self._frames.shape
        self._info = SourceInfo(name, w, h, t, fps, 32, "array")

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame(self, index: int) -> np.ndarray:
        return self._frames[index]


class VideoSource(Source):
    """Video via PyAV (bundled FFmpeg), keeping 10/12-bit precision when present."""

    def __init__(self, path: str | os.PathLike, channel: str = "green"):
        import av

        self.path = Path(path)
        self.channel = channel
        self._container = av.open(str(self.path))
        stream = self._container.streams.video[0]
        stream.thread_type = "AUTO"
        self._stream = stream
        ctx = stream.codec_context
        fmt = ctx.pix_fmt or "yuv420p"
        self._bits = 10 if ("10" in fmt or "p010" in fmt) else 12 if "12" in fmt else 8
        fps = float(stream.average_rate) if stream.average_rate else None
        count = stream.frames or 0
        if count <= 0 and stream.duration and stream.time_base and fps:
            count = int(round(float(stream.duration * stream.time_base) * fps))
        self._index: list[int] | None = None  # pts per frame, built lazily for random access
        self._cache: dict[int, np.ndarray] = {}
        self._next = -1
        self._decoder = None
        self._info = SourceInfo(
            self.path.name,
            ctx.width,
            ctx.height,
            count,
            fps,
            self._bits,
            "video",
            {
                "codec": ctx.name,
                "pix_fmt": fmt,
                "bitrate_mbps": (self._container.bit_rate or 0) / 1e6,
            },
        )
        self._build_index()

    @property
    def info(self) -> SourceInfo:
        return self._info

    def _convert(self, frame) -> np.ndarray:
        if self.channel == "luma":
            if self._bits > 8:
                y = frame.to_ndarray(format="gray16le").astype(np.float32) / 65535.0
            else:
                y = frame.to_ndarray(format="gray").astype(np.float32) / 255.0
            return y
        if self._bits > 8:
            rgb = frame.to_ndarray(format="rgb48le").astype(np.float32) / 65535.0
        else:
            rgb = frame.to_ndarray(format="rgb24").astype(np.float32) / 255.0
        return to_gray(rgb, self.channel)

    def _build_index(self) -> list[int]:
        """Presentation timestamps of every frame, from packets (no decoding)."""
        if self._index is None:
            pts = [
                p.pts for p in self._container.demux(self._stream) if p.pts is not None and p.size
            ]
            self._index = sorted(pts)
            self._info.frame_count = len(self._index)
            self._decoder = None
        return self._index

    def frame(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        pts_list = self._build_index()
        if not 0 <= index < len(pts_list):
            raise IndexError(f"frame {index} outside {self.path.name} ({len(pts_list)} frames)")
        target = pts_list[index]
        # Sequential reads continue decoding; anything else seeks to the preceding keyframe.
        if self._decoder is None or index != self._next:
            self._container.seek(target, stream=self._stream, backward=True, any_frame=False)
            self._decoder = self._container.decode(self._stream)
        out = None
        for fr in self._decoder:
            if fr.pts is None or fr.pts < target:
                continue
            out = self._convert(fr)
            break
        if out is None:
            raise IndexError(f"could not decode frame {index} of {self.path.name}")
        self._next = index + 1
        if len(self._cache) > 32:
            self._cache.pop(next(iter(self._cache)))
        self._cache[index] = out
        return out

    def close(self) -> None:
        self._container.close()


class ImageSequence(Source):
    def __init__(
        self, paths: list[str | os.PathLike], channel: str = "green", fps: float | None = None
    ):
        if not paths:
            raise ValueError("no images")
        self.paths = [Path(p) for p in paths]
        self.channel = channel
        first = self._read(self.paths[0], raw_bits=True)
        img, bits = first
        h, w = img.shape[:2]
        self._info = SourceInfo(
            self.paths[0].parent.name or self.paths[0].name,
            w,
            h,
            len(self.paths),
            fps,
            bits,
            "images",
        )

    @staticmethod
    def _read(path: Path, raw_bits: bool = False):
        data = np.fromfile(str(path), dtype=np.uint8)  # handles non-ASCII paths on Windows
        img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if img is None:
            raise OSError(f"cannot read image {path}")
        if img.ndim == 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        if img.dtype == np.uint16:
            bits, scale = 16, 65535.0
        elif img.dtype == np.uint8:
            bits, scale = 8, 255.0
        else:
            bits, scale = 32, 1.0
        out = img.astype(np.float32) / scale
        return (out, bits) if raw_bits else out

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame(self, index: int) -> np.ndarray:
        img = self._read(self.paths[index])
        return to_gray(img, self.channel)


class RawSequence(Source):
    """Camera RAW stills via LibRaw: linear, no auto-brightness, no gamma, 16-bit.

    With ``mode="green"`` (default) the two green Bayer sites are averaged at half resolution,
    which avoids demosaicking artefacts entirely; ``mode="demosaic"`` returns full resolution.
    """

    def __init__(self, paths: list[str | os.PathLike], mode: str = "green"):
        import rawpy

        self._rawpy = rawpy
        self.paths = [Path(p) for p in paths]
        self.mode = mode
        f0 = self.frame(0)
        h, w = f0.shape
        self._info = SourceInfo(
            self.paths[0].parent.name or self.paths[0].name,
            w,
            h,
            len(self.paths),
            None,
            14,
            "raw",
            {"mode": mode},
        )

    @property
    def info(self) -> SourceInfo:
        return self._info

    def frame(self, index: int) -> np.ndarray:
        with self._rawpy.imread(str(self.paths[index])) as raw:
            if self.mode == "green":
                return _raw_green(raw)
            rgb = raw.postprocess(
                output_bps=16, gamma=(1, 1), no_auto_bright=True, use_camera_wb=True, user_flip=0
            )
            return to_gray(rgb.astype(np.float32) / 65535.0, "green")


def _raw_green(raw) -> np.ndarray:
    """Average of the two green sites of each 2x2 Bayer quad, black-subtracted, in [0, 1]."""
    img = raw.raw_image_visible.astype(np.float32)
    colors = raw.raw_colors_visible
    black = np.asarray(raw.black_level_per_channel, dtype=np.float32)
    white = float(raw.white_level)
    h, w = img.shape
    h2, w2 = h // 2 * 2, w // 2 * 2
    img, colors = img[:h2, :w2], colors[:h2, :w2]
    img = img - black[colors]
    green = np.isin(colors, (1, 3))
    quad = lambda a: a.reshape(h2 // 2, 2, w2 // 2, 2).sum(axis=(1, 3))  # noqa: E731
    g = quad(np.where(green, img, 0.0)) / np.maximum(quad(green.astype(np.float32)), 1)
    return np.clip(g / max(white - float(black.mean()), 1.0), 0, 1).astype(np.float32)


def open_source(
    paths: list[str | os.PathLike] | str | os.PathLike, channel: str = "green"
) -> Source:
    """Open a video file, a folder of images, or a list of image/RAW files."""
    if isinstance(paths, (str, os.PathLike)):
        p = Path(paths)
        if p.is_dir():
            files = sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS | RAW_EXTS)
            return open_source(files, channel)
        paths = [p]
    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("nothing to open")
    ext = paths[0].suffix.lower()
    if ext in VIDEO_EXTS:
        if len(paths) > 1:
            raise ValueError("open one video at a time")
        return VideoSource(paths[0], channel)
    if ext in RAW_EXTS:
        return RawSequence(sorted(paths))
    if ext in IMAGE_EXTS:
        return ImageSequence(sorted(paths), channel)
    raise ValueError(f"unsupported file type: {ext}")
