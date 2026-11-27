"""Camera ray generation and render-path helpers."""

from __future__ import annotations

import numpy as np
import torch

from dac.data.intrinsics import Intrinsics


def get_ray_directions(intrinsics: Intrinsics, opengl_camera: bool = True) -> torch.Tensor:
    """Per-pixel ray directions in camera coordinates, sampled at pixel centres.

    Returns:
        ``[H, W, 3]`` directions (not normalised).
    """
    xs = torch.arange(intrinsics.width, dtype=torch.float32) + 0.5
    ys = torch.arange(intrinsics.height, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid([ys, xs], indexing="ij")
    sign = -1.0 if opengl_camera else 1.0
    return torch.stack(
        [
            (xx - intrinsics.center_x) / intrinsics.focal_x,
            (yy - intrinsics.center_y) / intrinsics.focal_y * sign,
            torch.full_like(xx, fill_value=sign),
        ],
        dim=-1,
    )


def get_rays(directions: torch.Tensor, c2w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """World-space ray origins and unit directions for every pixel of one camera.

    Args:
        directions: ``[H, W, 3]`` camera-space directions from :func:`get_ray_directions`.
        c2w: ``[3, 4]`` or ``[4, 4]`` camera-to-world matrix.

    Returns:
        ``(rays_o, rays_d)``, each ``[H * W, 3]``.
    """
    directions = directions.view(-1, 3)
    rays_d = (directions[:, None, :] * c2w[None, :3, :3]).sum(dim=-1)
    rays_o = torch.broadcast_to(c2w[None, :3, 3], directions.shape)
    rays_d = rays_d / torch.linalg.norm(rays_d, dim=-1, keepdim=True)
    return rays_o, rays_d


def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _view_matrix(z: np.ndarray, up: np.ndarray, pos: np.ndarray) -> np.ndarray:
    vec2 = _normalize(z)
    vec0 = _normalize(np.cross(up, vec2))
    vec1 = _normalize(np.cross(vec2, vec0))
    return np.stack([-vec0, vec1, vec2, pos], axis=1)


def generate_hemispherical_orbit(poses: torch.Tensor, n_frames: int = 120) -> torch.Tensor:
    """Camera path orbiting the z-axis at the mean height and radius of ``poses``.

    Based on RegNeRF: https://github.com/google-research/google-research/blob/342bfc1/regnerf/internal/datasets.py

    Returns:
        ``[n_frames, 3, 4]`` camera-to-world matrices.
    """
    origins = poses[:, :3, 3]
    radius = torch.sqrt(torch.mean(torch.sum(origins**2, dim=-1)))
    # Assume the z-axis points up towards the camera hemisphere.
    sin_phi = torch.mean(origins[:, 2], dim=0) / radius
    cos_phi = torch.sqrt(1 - sin_phi**2)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    render_poses = []
    for theta in np.linspace(0.0, 2.0 * np.pi, n_frames, endpoint=False):
        direction = [float(cos_phi) * np.cos(theta), float(cos_phi) * np.sin(theta), float(sin_phi)]
        origin = (float(radius) * np.asarray(direction)).astype(np.float32)
        render_poses.append(torch.from_numpy(_view_matrix(origin, up, origin)))
    return torch.stack(render_poses, dim=0).float()
