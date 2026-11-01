"""Render the README's footage images from a real clip.

    uv run python scripts/readme_media.py <clip> <first> <last> <still> <output folder>

Writes ``camera-vs-schlieren.jpg`` (the camera's frame ``still`` next to its knife-edge
schlieren), ``views.png`` (four views of the same frame) and ``plume.webp`` (the knife-edge
schlieren of frames first..last). Processing uses the app's defaults for the
"Best" quality: automatic reference frames, automatic mask, variational flow.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from easy_bos.core import media, preprocess, render
from easy_bos.core.pipeline import ProcessingSettings, Processor, auto_binning

TILE_HEIGHT = 540
ANIMATION_HEIGHT = 450


def colour_frame(path: Path, index: int) -> np.ndarray:
    """The camera's own frame ``index`` in colour."""
    with av.open(str(path)) as container:
        for i, frame in enumerate(container.decode(video=0)):
            if i == index:
                return frame.to_ndarray(format="rgb24")
    raise IndexError(index)


def crop_box(valid: np.ndarray, inset: int = 6) -> tuple[slice, slice]:
    """The rows and columns that are entirely valid background, minus an inset (drops bezels)."""
    rows = np.flatnonzero(valid.mean(axis=1) > 0.98)
    cols = np.flatnonzero(valid.mean(axis=0) > 0.98)
    return (
        slice(rows[0] + inset, rows[-1] + 1 - inset),
        slice(cols[0] + inset, cols[-1] + 1 - inset),
    )


def fit(img: np.ndarray, height: int) -> np.ndarray:
    w = round(img.shape[1] * height / img.shape[0])
    return cv2.resize(img, (w, height), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("clip", type=Path)
    ap.add_argument("first", type=int)
    ap.add_argument("last", type=int)
    ap.add_argument("still", type=int)
    ap.add_argument("out", type=Path)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.0, help="knife-edge range in pixels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    src = media.open_source(args.clip)
    reference = preprocess.suggest_reference(src.frame, len(src), src.info.fps)
    binning = auto_binning(src.info.width)
    proc = Processor(src, ProcessingSettings(reference_range=reference, binning=binning))
    print(f"reference frames {reference[0]}-{reference[1] - 1}, binning {binning}")

    still = proc.process_frame(args.still).displacement
    ys, xs = crop_box(np.isfinite(proc.process_frame(reference[1]).displacement[0]))

    settings = render.ViewSettings(scale=args.scale)
    tiles = [
        render.render(view, still, view_settings)[ys, xs]
        for view, view_settings in [
            (render.View.KNIFE_EDGE, settings),
            (render.View.KNIFE_EDGE, render.ViewSettings(args.scale, knife_angle_deg=90.0)),
            (render.View.COLOUR_SCHLIEREN, settings),
            (render.View.MAGNITUDE, settings),
        ]
    ]
    tiles = [fit(t, TILE_HEIGHT) for t in tiles]

    # What the camera recorded, next to what Easy BOS makes of it.
    rgb = colour_frame(args.clip, args.still)
    camera = rgb[ys.start * binning : ys.stop * binning, xs.start * binning : xs.stop * binning]
    pair = np.hstack([fit(camera, TILE_HEIGHT), tiles[0]])
    Image.fromarray(pair).save(args.out / "camera-vs-schlieren.jpg", quality=90)

    # Four views of the same frame: knife edge vertical and horizontal, colour, magnitude.
    grid = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    Image.fromarray(grid).save(args.out / "views.png", optimize=True)

    # The schlieren as an animation.
    frames = []
    for i in range(args.first, args.last + 1, args.step):
        d = proc.process_frame(i).displacement
        gray = render.render(render.View.KNIFE_EDGE, d, settings)[ys, xs]
        frames.append(fit(gray, ANIMATION_HEIGHT))
        print(f"frame {i}", flush=True)
    images = [Image.fromarray(f) for f in frames]
    images[0].save(
        args.out / "plume.webp",
        save_all=True,
        append_images=images[1:],
        duration=round(1000 * args.step / src.info.fps),
        loop=0,
        quality=75,
        method=6,
    )
    src.close()


if __name__ == "__main__":
    main()
