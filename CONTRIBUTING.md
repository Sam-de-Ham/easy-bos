# Contributing

Bug reports, sample footage and ideas are welcome; please use the issue templates. For
anything about a capture, the Capture page's file details and check results usually make the
problem reproducible, and a few seconds of the clip help even more.

## Development setup

```bash
git clone https://github.com/Sam-de-Ham/easy-bos
cd easy-bos
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

Dependencies are managed with [uv](https://docs.astral.sh/uv/): add them with `uv add`.
`uv sync` installs PyTorch with CUDA 13.0 (about 3 GB). Set `EASY_BOS_DEVICE=cpu` to force
the CPU path.

## How the code is organised

See [docs/architecture.md](docs/architecture.md). In short: `core/` is all the processing and
has no Qt dependency, `core/flow/` holds the displacement estimators, `core/reconstruct/` the
density and temperature reconstruction, `bench/` the synthetic benchmark and `ui/` the
PySide6 application.

## Accuracy first

A change to a method or a default needs to hold up on the synthetic benchmark, which renders
captures through refractive-index fields with exact ground truth:

```bash
uv run easy-bos-bench          # full run, writes docs/benchmark.md and the README table
uv run easy-bos-bench --methods variational --scenarios plume-raw plume-h264
uv run python -m easy_bos.bench.tune   # parameter sweeps, writes docs/tuning.md
```

Check it on real footage too, and note anything surprising. The research notes behind the
methods are in [docs/research](docs/research).

## Tests

```bash
uv run pytest            # fast tests: numerics, estimators against synthetic truth, pipeline
uv run pytest -m slow    # end-to-end UI test, offscreen
```

## Screenshots

```bash
uv run python scripts/screenshots.py docs/images/app --clip <video> --frame <n>
uv run python scripts/screenshots.py docs/images/app --demo
uv run python scripts/readme_media.py <video> <first> <last> <still> docs/images
```

render the app's pages and the footage images used in the README.

## Pull requests

- Keep them focused, with tests, and run the checks above.
- Note user-visible changes in `CHANGELOG.md` under an `## Unreleased` heading.

## Releases

Set the version in `pyproject.toml` and `src/easy_bos/__init__.py`, move the changelog entries
under it, then tag and push (`git tag v1.2.3 && git push origin v1.2.3`). The release workflow
runs the tests, builds the installer package and publishes the GitHub release with the
changelog notes.
