from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Union

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

PathLike = Union[str, Path]


def write_png(path: PathLike, image: np.ndarray) -> None:
    """Write an ``[H, W, C]`` ``uint8`` image."""
    Image.fromarray(image).save(path)


def write_video(path: PathLike, frames: Sequence[np.ndarray], fps: int = 30) -> None:
    """Write ``[H, W, 3]`` ``uint8`` frames to an mp4 file (requires ``imageio-ffmpeg``)."""
    import imageio.v2 as imageio

    log.info("Saving video (%d frames) to %s", len(frames), path)
    imageio.mimwrite(str(path), list(frames), fps=fps, macro_block_size=1)
