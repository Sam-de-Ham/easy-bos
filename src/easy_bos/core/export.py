"""Export: rendered images and videos, scientific data (HDF5, TIFF, CSV), figures."""

from __future__ import annotations

import csv
from collections.abc import Callable, Iterable
from pathlib import Path

import cv2
import h5py
import numpy as np

from easy_bos.core.geometry import Setup


def write_png(path: str | Path, rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(str(path))  # tofile handles non-ASCII paths on Windows


def write_tiff_float(path: str | Path, data: np.ndarray) -> None:
    """32-bit float TIFF (single channel), readable in ImageJ/Fiji and MATLAB."""
    ok, buf = cv2.imencode(".tiff", np.ascontiguousarray(data, dtype=np.float32))
    if not ok:
        raise OSError(f"could not encode {path}")
    buf.tofile(str(path))


VIDEO_FORMATS = {
    "mp4": ("H.264 MP4 (shareable)", "libx264", "yuv420p", {"crf": "15", "preset": "slow"}),
    "mov": ("ProRes 422 HQ MOV (editing)", "prores_ks", "yuv422p10le", {"profile": "3"}),
    "mkv": ("FFV1 MKV (lossless)", "ffv1", "yuv444p", {}),
}


def write_video(
    path: str | Path,
    frames: Iterable[np.ndarray],
    fps: float,
    progress: Callable[[int], None] | None = None,
    crf: int | None = None,
) -> None:
    """Encode RGB uint8 frames. Format is chosen from the file extension (mp4/mov/mkv).

    ``crf`` overrides the H.264 quality (lower is better; default 15).
    """
    import av

    ext = Path(path).suffix.lower().lstrip(".")
    if ext not in VIDEO_FORMATS:
        raise ValueError(f"unsupported video format .{ext}")
    _, codec, pix_fmt, opts = VIDEO_FORMATS[ext]
    with av.open(str(path), "w") as out:
        stream = None
        for n, rgb in enumerate(frames):
            h, w = rgb.shape[:2]
            if stream is None:
                stream = out.add_stream(codec, rate=max(1, round(fps)))
                stream.width, stream.height = w - w % 2, h - h % 2
                stream.pix_fmt = pix_fmt
                stream.options = dict(opts)
                if crf is not None and codec == "libx264":
                    stream.options["crf"] = str(crf)
            img = np.ascontiguousarray(rgb[: stream.height, : stream.width])
            frame = av.VideoFrame.from_ndarray(img, format="rgb24").reformat(format=pix_fmt)
            for p in stream.encode(frame):
                out.mux(p)
            if progress:
                progress(n + 1)
        if stream is not None:
            for p in stream.encode():
                out.mux(p)


def write_profile_csv(
    path: str | Path, distance: np.ndarray, columns: dict[str, np.ndarray]
) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance", *columns.keys()])
        for i in range(len(distance)):
            w.writerow([f"{distance[i]:.6g}", *(f"{c[i]:.6g}" for c in columns.values())])


def write_field_hdf5(
    path: str | Path,
    displacement: np.ndarray,
    setup: Setup | None,
    density: np.ndarray | None = None,
    temperature: np.ndarray | None = None,
    metadata: dict | None = None,
) -> None:
    """One frame of data with physical units: displacement, deflection, density, temperature."""
    with h5py.File(path, "w") as f:
        f.attrs["format"] = "easy-bos field v1"
        f.attrs["displacement_convention"] = "cur(x) = ref(x + D(x)); D[0] right, D[1] down"
        dset = f.create_dataset(
            "displacement_px", data=displacement.astype(np.float32), compression="gzip"
        )
        dset.attrs["units"] = "pixel"
        if setup is not None:
            eps = displacement / setup.pixels_per_radian
            e = f.create_dataset("deflection_rad", data=eps.astype(np.float32), compression="gzip")
            e.attrs["units"] = "rad"
            f.attrs["pixels_per_radian"] = setup.pixels_per_radian
            f.attrs["object_pixel_size_m"] = setup.object_pixel_size_m
            f.attrs["z_a_m"] = setup.z_a
            f.attrs["z_d_m"] = setup.z_d
            f.attrs["focal_length_m"] = setup.f
            f.attrs["pixel_pitch_m"] = setup.p
        if density is not None:
            f.create_dataset("density_kg_m3", data=density.astype(np.float32), compression="gzip")
        if temperature is not None:
            f.create_dataset(
                "temperature_K", data=temperature.astype(np.float32), compression="gzip"
            )
        for k, v in (metadata or {}).items():
            f.attrs[k] = v


def write_print_pattern_pdf(path: str | Path, image: np.ndarray, dpi: int, title: str) -> None:
    """Embed a printer-resolution pattern in a PDF at exact physical size, with a scale bar.

    Uses matplotlib's PDF backend (no extra dependency). The image is placed 1:1 at ``dpi``.
    """
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt

    h, w = image.shape
    margin_in = 0.4
    fig_w = w / dpi + 2 * margin_in
    fig_h = h / dpi + 2 * margin_in + 0.35
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes(
        [margin_in / fig_w, (margin_in + 0.35) / fig_h, (w / dpi) / fig_w, (h / dpi) / fig_h]
    )
    ax.imshow(image, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_axis_off()
    # 50 mm scale bar for checking the printer did not rescale the page.
    bar = fig.add_axes([margin_in / fig_w, 0.12 / fig_h, (50 / 25.4) / fig_w, 0.06 / fig_h])
    bar.set_facecolor("black")
    bar.set_xticks([])
    bar.set_yticks([])
    fig.text(
        (margin_in + 50 / 25.4 + 0.1) / fig_w,
        0.12 / fig_h,
        f"50 mm · print at 100 % · {title}",
        fontsize=7,
        va="bottom",
    )
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
