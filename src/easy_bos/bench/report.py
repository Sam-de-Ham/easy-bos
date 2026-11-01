"""Markdown report for benchmark results."""

from __future__ import annotations

import datetime as dt
from collections import defaultdict
from pathlib import Path

import numpy as np

from easy_bos.bench import suite
from easy_bos.core.device import describe
from easy_bos.core.flow import REGISTRY


def write(results: list[suite.MethodScore], path: Path, size: tuple[int, int]) -> None:
    by_sc: dict[str, list[suite.MethodScore]] = defaultdict(list)
    for r in results:
        by_sc[r.scenario].append(r)
    scs = {s.key: s for s in suite.scenarios()}
    methods = [m for m in suite.METHODS if any(r.method == m for r in results)]
    lines = [
        "# Synthetic benchmark",
        "",
        f"Generated {dt.date.today().isoformat()} by `uv run easy-bos-bench` on {describe()}, "
        f"{size[0]}×{size[1]} px frames. Every method runs through the production pipeline "
        "with its default parameters. Displacement errors are endpoint errors in pixels over "
        f"the frame minus a {suite.MARGIN}-px border; *rel* is mean EPE divided by the mean true "
        "magnitude; *peak* is the ratio of the 99th-percentile magnitudes (1 = no attenuation). "
        "Temperature error is the RMS over the region more than 20 % of the peak rise above "
        "ambient. *Oracle* reconstructs from the true (depth-of-field blurred) displacement: "
        "its error is the floor set by optics and reconstruction, not by the estimator.",
        "",
        "## Mean endpoint error (px), lower is better",
        "",
        "| Scenario | " + " | ".join(REGISTRY[m].label for m in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for key, rows in by_sc.items():
        vals = {r.method: r for r in rows}
        best = min(
            (vals[m].epe for m in methods if m in vals and np.isfinite(vals[m].epe)), default=np.nan
        )
        cells = []
        for m in methods:
            r = vals.get(m)
            if r is None or not np.isfinite(r.epe):
                cells.append("—")
            else:
                c = f"{r.epe:.4f}"
                cells.append(f"**{c}**" if r.epe == best else c)
        lines.append(f"| {scs[key].title} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Relative error and peak recovery",
        "",
        "| Scenario | " + " | ".join(REGISTRY[m].label for m in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for key, rows in by_sc.items():
        vals = {r.method: r for r in rows}
        cells = [
            f"{vals[m].relative:.3f} / {vals[m].peak_ratio:.2f}"
            if m in vals and np.isfinite(vals[m].epe)
            else "—"
            for m in methods
        ]
        lines.append(f"| {scs[key].title} | " + " | ".join(cells) + " |")

    t_rows = [k for k, rows in by_sc.items() if any(r.t_rmse_k is not None for r in rows)]
    if t_rows:
        lines += [
            "",
            "## Temperature RMS error (K) after reconstruction",
            "",
            "| Scenario | Oracle | " + " | ".join(REGISTRY[m].label for m in methods) + " |",
            "|---|---:|" + "---:|" * len(methods),
        ]
        for key in t_rows:
            vals = {r.method: r for r in by_sc[key]}
            o = vals.get("oracle")
            cells = [
                f"{vals[m].t_rmse_k:.2f}"
                if m in vals and vals[m].t_rmse_k is not None and np.isfinite(vals[m].t_rmse_k)
                else "—"
                for m in methods
            ]
            lines.append(f"| {scs[key].title} | {o.t_rmse_k:.2f} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## Runtime per frame (s)",
        "",
        "| Scenario | " + " | ".join(REGISTRY[m].label for m in methods) + " |",
        "|---|" + "---:|" * len(methods),
    ]
    for key, rows in by_sc.items():
        vals = {r.method: r for r in rows}
        cells = [
            f"{vals[m].seconds:.2f}" if m in vals and np.isfinite(vals[m].seconds) else "—"
            for m in methods
        ]
        lines.append(f"| {scs[key].title} | " + " | ".join(cells) + " |")

    lines += ["", "## Scenarios", ""]
    for key in by_sc:
        lines.append(f"- **{scs[key].title}** (`{key}`): {scs[key].description}")
    errors = [r for r in results if r.error]
    if errors:
        lines += ["", "## Failures", ""] + [
            f"- {r.scenario} / {r.method}: `{r.error}`" for r in errors
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
