"""Parallel image loading for Blender-format (``transforms.json``) datasets."""

from __future__ import annotations

import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from PIL import Image

from dac.utils import progress_bar

_LANCZOS = getattr(Image, "Resampling", Image).LANCZOS
_MAX_THREADS = 10


def resolve_image_path(data_dir: str, file_path: str) -> str:
    path = os.path.join(data_dir, file_path)
    if "." not in os.path.basename(path):
        path += ".png"  # NeRF-Synthetic stores paths without an extension
    return path


def load_image(path: str, downsample: float) -> torch.Tensor:
    """Load an image as a float ``[H, W, C]`` tensor in ``[0, 1]``, resized by ``1/downsample``."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"image not found: {path}")
    with Image.open(path) as img:
        width, height = img.size
        size = (int(width / downsample), int(height / downsample))
        resized = img.resize(size, _LANCZOS) if size != img.size else img
        array = np.array(resized, dtype=np.uint8, copy=True)
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).to(torch.float32).div(255)


def load_frames(
    frames: Sequence[dict[str, Any]], data_dir: str, downsample: float, desc: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load all frames of a transforms file.

    Returns:
        ``(images, poses)`` with shapes ``[N, H, W, C]`` and ``[N, 4, 4]``.
    """
    if not frames:
        raise ValueError(f"no frames to load from {data_dir}")
    paths = [resolve_image_path(data_dir, frame["file_path"]) for frame in frames]
    with ThreadPoolExecutor(max_workers=min(_MAX_THREADS, len(paths))) as pool:
        images = list(
            progress_bar(
                iterable=pool.map(lambda p: load_image(p, downsample), paths),
                total=len(paths),
                desc=desc,
            )
        )
    shapes = {tuple(img.shape) for img in images}
    if len(shapes) != 1:
        raise ValueError(f"images in {data_dir} have different shapes: {sorted(shapes)}")
    poses = torch.tensor(np.array([frame["transform_matrix"] for frame in frames]), dtype=torch.float32)
    return torch.stack(images, dim=0), poses
