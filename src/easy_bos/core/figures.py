"""Publication-style figures (matplotlib, offscreen): field with scale bar and colour bar."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_field_figure(
    path: str | Path,
    rgb: np.ndarray,
    mm_per_px: float,
    title: str,
    cbar_label: str,
    vrange: tuple[float, float],
    cmap: str | None,
    dpi: int = 200,
) -> None:
    """Save a rendered view with a physical scale bar and (unless ``cmap`` is None) a colour bar."""
    import matplotlib

    matplotlib.use("Agg", force=False)
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors

    h, w = rgb.shape[:2]
    width_in = 7.0
    fig_h = width_in * h / w + 0.6
    fig, ax = plt.subplots(figsize=(width_in + (1.2 if cmap else 0.2), fig_h), dpi=dpi)
    ax.imshow(rgb, interpolation="nearest")
    ax.set_axis_off()
    ax.set_title(title, fontsize=10, loc="left")
    # Scale bar: a round length about a fifth of the width.
    target = w * mm_per_px / 5
    nice = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    length_mm = min(nice, key=lambda v: abs(v - target))
    px = length_mm / mm_per_px
    y = h * 0.94
    ax.plot([w * 0.04, w * 0.04 + px], [y, y], color="white", lw=3, solid_capstyle="butt")
    ax.text(
        w * 0.04 + px / 2,
        y - h * 0.025,
        f"{length_mm:g} mm",
        color="white",
        ha="center",
        va="bottom",
        fontsize=8,
    )
    if cmap:
        sm = plt.cm.ScalarMappable(norm=mcolors.Normalize(*vrange), cmap=cmap)
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(cbar_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(str(path), dpi=dpi)
    plt.close(fig)
