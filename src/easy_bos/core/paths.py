"""Per-user directories (settings, model weights, logs). Nothing here lives in the repository."""

from __future__ import annotations

from pathlib import Path

from platformdirs import user_data_dir, user_log_dir

APP_NAME = "EasyBOS"


def data_dir() -> Path:
    d = Path(user_data_dir(APP_NAME, appauthor=False))
    d.mkdir(parents=True, exist_ok=True)
    return d


def model_dir() -> Path:
    d = data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_dir() -> Path:
    d = Path(user_log_dir(APP_NAME, appauthor=False))
    d.mkdir(parents=True, exist_ok=True)
    return d
