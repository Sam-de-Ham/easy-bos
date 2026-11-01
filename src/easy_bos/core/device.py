"""Compute device selection (CUDA when available, CPU otherwise)."""

from __future__ import annotations

import functools
import os

import torch


@functools.cache
def default_device() -> torch.device:
    """CUDA if usable, else CPU. ``EASY_BOS_DEVICE=cpu`` forces the CPU path."""
    forced = os.environ.get("EASY_BOS_DEVICE", "").strip().lower()
    if forced:
        return torch.device(forced)
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except RuntimeError:
            pass
    return torch.device("cpu")


def resolve(device: str | torch.device | None) -> torch.device:
    if device is None or device == "auto":
        return default_device()
    return torch.device(device)


def describe(device: torch.device | None = None) -> str:
    device = device or default_device()
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        return f"{props.name} · {props.total_memory / 2**30:.0f} GB · CUDA"
    return f"CPU · {torch.get_num_threads()} threads"


def free_memory_bytes(device: torch.device) -> int | None:
    if device.type != "cuda":
        return None
    free, _ = torch.cuda.mem_get_info(device)
    return int(free)
