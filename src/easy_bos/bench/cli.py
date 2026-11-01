"""Command-line entry point: ``uv run easy-bos-bench``."""

from __future__ import annotations

import argparse
from pathlib import Path

from easy_bos.bench import report, suite


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Run the Easy BOS synthetic benchmark.")
    ap.add_argument("--methods", nargs="*", default=None, help=f"subset of {suite.METHODS}")
    ap.add_argument("--scenarios", nargs="*", default=None, help="subset of scenario keys")
    ap.add_argument("--size", type=int, nargs=2, default=(512, 384), metavar=("W", "H"))
    ap.add_argument("--out", type=Path, default=Path("bench_out"))
    ap.add_argument(
        "--report",
        type=Path,
        default=Path("docs/benchmark.md"),
        help="Markdown report to write (only for a full run)",
    )
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    results = suite.run(args.methods, args.scenarios, tuple(args.size))
    suite.save(results, args.out / "results.json")
    full = args.methods is None and args.scenarios is None
    if full:
        report.write(results, args.report, tuple(args.size))
        readme = Path("README.md")
        if readme.exists():
            report.insert(readme, "ACCURACY-TABLE", report.summary(results))
        print(f"report written to {args.report}; README summary updated")
    print(f"results written to {args.out / 'results.json'}")


if __name__ == "__main__":
    main()
