"""Shared fixtures: a tiny procedurally generated NeRF-Synthetic-style scene."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml
from PIL import Image

# Run on CPU with the pure-PyTorch fallback, without progress bars.
os.environ.setdefault("DAC_DISABLE_TCNN", "1")
os.environ.setdefault("DISABLE_TQDM", "1")

IMAGE_SIZE = 16


def look_at_origin(position: np.ndarray) -> list[list[float]]:
    """OpenGL camera-to-world matrix for a camera at ``position`` looking at the origin."""
    from dac.data.rays import _view_matrix

    c2w = np.eye(4)
    c2w[:3] = _view_matrix(position, np.array([0.0, 0.0, 1.0]), position)
    return c2w.tolist()


def ring_positions(n: int, radius: float = 4.0, height: float = 1.0, offset: float = 0.1) -> np.ndarray:
    angles = offset + np.arange(n) * 2 * np.pi / n
    return np.stack([radius * np.cos(angles), radius * np.sin(angles), np.full(n, height)], axis=1)


def write_split(root: Path, split: str, positions: np.ndarray, rng: np.random.Generator) -> None:
    (root / split).mkdir(parents=True, exist_ok=True)
    frames = []
    for i, position in enumerate(positions):
        image = rng.integers(0, 256, size=(IMAGE_SIZE, IMAGE_SIZE, 4), dtype=np.uint8)
        Image.fromarray(image).save(root / split / f"r_{i}.png")
        frames.append({"file_path": f"./{split}/r_{i}", "transform_matrix": look_at_origin(position)})
    meta = {"camera_angle_x": 0.69, "frames": frames}
    (root / f"transforms_{split}.json").write_text(json.dumps(meta))


@pytest.fixture
def scene(tmp_path: Path) -> Path:
    """A NeRF-Synthetic-layout scene with 8 training and 4 test views on a ring."""
    root = tmp_path / "data" / "nerf_synthetic" / "toy"
    rng = np.random.default_rng(0)
    write_split(root, "train", ring_positions(8), rng)
    write_split(root, "test", ring_positions(4, offset=0.3), rng)
    return root


TINY_MODEL: dict[str, Any] = {
    "grid": {"output_coordinate_dim": 4, "resolution": [8, 8, 8]},
    "multiscale_res": [1, 2],
    "linear_decoder_layers": 1,
    "num_samples": 8,
    "num_proposal_samples": [16, 8],
    "proposal_nets": [
        {"num_output_coords": 4, "resolution": [8, 8, 8]},
        {"num_output_coords": 4, "resolution": [8, 8, 8]},
    ],
}


def tiny_config(scene: Path, log_dir: Path, **sections: dict[str, Any]) -> dict[str, Any]:
    """A config small enough to train in a few seconds on CPU."""
    cfg: dict[str, Any] = {
        "experiment": {"name": "toy", "log_dir": str(log_dir), "device": "cpu"},
        "data": {"dataset": "nerf_synthetic", "root": str(scene), "num_workers": 0},
        "optim": {"num_steps": 3, "batch_size": 64, "fp16": False},
        "regularization": {"plane_tv": 1e-4, "plane_tv_proposal_net": 1e-4, "histogram": 1.0, "distortion": 1e-3},
        "logging": {"save_every": 1000, "valid_every": 1000, "log_every": 1, "eval_batch_size": 128},
        "model": copy.deepcopy(TINY_MODEL),
        "dac": {"num_experts": 2},
    }
    for section, values in sections.items():
        cfg.setdefault(section, {}).update(values)
    return cfg


@pytest.fixture
def config_file(scene: Path, tmp_path: Path) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(tiny_config(scene, tmp_path / "logs")))
    return path
