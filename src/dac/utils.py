"""Small helpers shared across the package."""

from __future__ import annotations

import contextlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm


class EMA:
    """Exponential moving average, used to smooth logged training losses."""

    def __init__(self, weighting: float = 0.9) -> None:
        self.weighting = weighting  # weight given to the newest observation
        self.value: float | None = None

    def update(self, value: float) -> None:
        if self.value is None:
            self.value = value
        else:
            self.value = self.weighting * value + (1 - self.weighting) * self.value

    def __str__(self) -> str:
        return "nan" if self.value is None else f"{self.value:.2e}"


def progress_bar(**kwargs: Any) -> tqdm:
    """A tqdm progress bar that is silenced when ``DISABLE_TQDM`` is set."""
    kwargs.setdefault("mininterval", 0.5)
    kwargs.setdefault("disable", bool(os.environ.get("DISABLE_TQDM")))
    return tqdm(**kwargs)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def seed_dataloader_worker(_worker_id: int) -> None:
    """Derive numpy/random seeds from the per-worker torch seed."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def load_torch_file(path: str | Path, map_location: torch.device | str) -> Any:
    """``torch.load`` restricted to tensors and plain containers where supported."""
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:  # torch < 1.13 has no ``weights_only``
        return torch.load(path, map_location=map_location)


def make_grad_scaler(enabled: bool) -> Any:
    if hasattr(torch.amp, "GradScaler"):  # torch >= 2.3
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast(enabled: bool) -> contextlib.AbstractContextManager[Any]:
    """fp16 autocast on CUDA, or a no-op context."""
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()
