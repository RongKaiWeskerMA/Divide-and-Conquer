from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Intrinsics:
    """Pinhole camera intrinsics in pixels."""

    width: int
    height: int
    focal_x: float
    focal_y: float
    center_x: float
    center_y: float


def intrinsics_from_transforms(meta: dict, img_h: int, img_w: int, downsample: float) -> Intrinsics:
    """Read intrinsics from a Blender/instant-ngp style ``transforms.json``.

    ``img_h``/``img_w`` are the (already downsampled) image dimensions.
    """
    if "fl_x" in meta or "fl_y" in meta:
        fl_x = (meta["fl_x"] if "fl_x" in meta else meta["fl_y"]) / downsample
        fl_y = (meta["fl_y"] if "fl_y" in meta else meta["fl_x"]) / downsample
    elif "camera_angle_x" in meta or "camera_angle_y" in meta:
        # Field of view in radians; the focal length follows from the image size.
        fl_x = img_w / (2 * np.tan(meta["camera_angle_x"] / 2)) if "camera_angle_x" in meta else None
        fl_y = img_h / (2 * np.tan(meta["camera_angle_y"] / 2)) if "camera_angle_y" in meta else None
        fl_x = fl_x if fl_x is not None else fl_y
        fl_y = fl_y if fl_y is not None else fl_x
    else:
        raise ValueError("transforms file has no focal length (fl_x/fl_y or camera_angle_x/y)")

    cx = meta["cx"] / downsample if "cx" in meta else img_w / 2
    cy = meta["cy"] / downsample if "cy" in meta else img_h / 2
    return Intrinsics(width=img_w, height=img_h, focal_x=float(fl_x), focal_y=float(fl_y), center_x=cx, center_y=cy)
