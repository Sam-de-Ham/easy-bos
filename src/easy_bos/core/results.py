"""Storage for processed results: per-frame displacement fields on disk (HDF5) with a cache.

A 4K displacement field is 66 MB in float32, so sequences are written to a chunked, compressed
HDF5 file as they are produced and read back lazily. The same file is the raw-data export:
it is self-describing (units and setup in attributes) and readable from Python, MATLAB and
ParaView.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections import OrderedDict
from pathlib import Path

import h5py
import numpy as np

from easy_bos.core.pipeline import FrameResult


class ResultStore:
    def __init__(self, path: str | Path | None = None, cache_frames: int = 8):
        if path is None:
            tmp = tempfile.NamedTemporaryFile(prefix="easybos-", suffix=".h5", delete=False)
            tmp.close()
            path = tmp.name
            self.temporary = True
        else:
            self.temporary = False
        self.path = Path(path)
        self._lock = threading.Lock()
        self._file = h5py.File(self.path, "a")
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._cache_frames = cache_frames
        self._file.require_group("frames")
        self._file.attrs.setdefault("format", "easy-bos results v1")
        self._file.attrs.setdefault(
            "displacement_convention", "cur(x) = ref(x + D(x)); D[0] right, D[1] down; pixels"
        )

    # ---------------------------------------------------------------- metadata
    def set_metadata(self, **kw) -> None:
        with self._lock:
            for k, v in kw.items():
                self._file.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v

    def metadata(self, key: str, default=None):
        v = self._file.attrs.get(key, default)
        if isinstance(v, str) and v[:1] in "[{":
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return v
        return v

    # ---------------------------------------------------------------- frames
    def clear(self) -> None:
        with self._lock:
            del self._file["frames"]
            self._file.create_group("frames")
            if "reconstruction" in self._file:
                del self._file["reconstruction"]
            self._cache.clear()
            self._file.flush()

    def add(self, r: FrameResult) -> None:
        with self._lock:
            g = self._file["frames"]
            name = f"{r.index:06d}"
            if name in g:
                del g[name]
            fg = g.create_group(name)
            chunks = (1, min(256, r.displacement.shape[1]), min(256, r.displacement.shape[2]))
            fg.create_dataset(
                "displacement",
                data=r.displacement.astype(np.float32),
                chunks=chunks,
                compression="lzf",
            )
            if r.confidence is not None:
                fg.create_dataset(
                    "confidence", data=r.confidence.astype(np.float16), compression="lzf"
                )
            if r.motion is not None:
                fg.attrs["motion_matrix"] = r.motion.matrix
                fg.attrs["motion_residual_px"] = r.motion.residual_px
            fg.attrs["banding"] = r.banding
            fg.attrs["elapsed_s"] = r.elapsed_s
            self._cache[r.index] = r.displacement
            self._trim()
            self._file.flush()

    def indices(self) -> list[int]:
        with self._lock:
            return sorted(int(k) for k in self._file["frames"].keys())

    def __len__(self) -> int:
        return len(self.indices())

    def displacement(self, index: int) -> np.ndarray:
        if index in self._cache:
            self._cache.move_to_end(index)
            return self._cache[index]
        with self._lock:
            d = self._file["frames"][f"{index:06d}"]["displacement"][...]
        self._cache[index] = d
        self._trim()
        return d

    def confidence(self, index: int) -> np.ndarray | None:
        with self._lock:
            g = self._file["frames"][f"{index:06d}"]
            return g["confidence"][...].astype(np.float32) if "confidence" in g else None

    def frame_attrs(self, index: int) -> dict:
        with self._lock:
            return dict(self._file["frames"][f"{index:06d}"].attrs)

    def mean_displacement(self, indices: list[int] | None = None) -> np.ndarray:
        idx = indices or self.indices()
        acc = None
        n = 0
        for i in idx:
            d = self.displacement(i).astype(np.float64)
            acc = d if acc is None else acc + d
            n += 1
        return (acc / max(n, 1)).astype(np.float32)

    # ---------------------------------------------------------------- reconstruction
    def add_reconstruction(
        self, index: int, density: np.ndarray, temperature: np.ndarray, attrs: dict
    ) -> None:
        with self._lock:
            g = self._file.require_group("reconstruction")
            name = f"{index:06d}"
            if name in g:
                del g[name]
            rg = g.create_group(name)
            rg.create_dataset("density", data=density.astype(np.float32), compression="lzf")
            rg.create_dataset("temperature", data=temperature.astype(np.float32), compression="lzf")
            for k, v in attrs.items():
                rg.attrs[k] = v
            self._file.flush()

    def reconstruction(self, index: int) -> tuple[np.ndarray, np.ndarray] | None:
        with self._lock:
            g = self._file.get("reconstruction")
            if g is None or f"{index:06d}" not in g:
                return None
            rg = g[f"{index:06d}"]
            return rg["density"][...], rg["temperature"][...]

    def _trim(self) -> None:
        while len(self._cache) > self._cache_frames:
            self._cache.popitem(last=False)

    def close(self) -> None:
        with self._lock:
            if self._file.id.valid:
                self._file.close()
        if self.temporary:
            try:
                self.path.unlink()
            except OSError:
                pass

    def save_as(self, path: str | Path) -> None:
        """Copy all results to ``path`` (used for 'export raw data')."""
        with self._lock:
            self._file.flush()
            with h5py.File(path, "w") as out:
                for k, v in self._file.attrs.items():
                    out.attrs[k] = v
                for k in self._file.keys():
                    self._file.copy(self._file[k], out, name=k)
