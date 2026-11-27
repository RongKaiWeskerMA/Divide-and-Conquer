from __future__ import annotations

import torch
import torch.nn.functional as F


def grid_sample_wrapper(grid: torch.Tensor, coords: torch.Tensor, align_corners: bool = True) -> torch.Tensor:
    """Bilinearly sample a feature grid at normalised coordinates in ``[-1, 1]``.

    Args:
        grid: ``[B, C, *reso]`` (or ``[C, *reso]``) feature grid with 2 or 3 spatial dims.
        coords: ``[B, n, D]`` (or ``[n, D]``) sample coordinates, ``D`` matching the grid.

    Returns:
        ``[n, C]`` (batch dimension squeezed when ``B == 1``) interpolated features.
    """
    grid_dim = coords.shape[-1]
    if grid_dim not in (2, 3):
        raise NotImplementedError(f"grid_sample supports 2D and 3D grids, got {grid_dim}D")
    if grid.dim() == grid_dim + 1:
        grid = grid.unsqueeze(0)
    if coords.dim() == 2:
        coords = coords.unsqueeze(0)

    coords = coords.view([coords.shape[0]] + [1] * (grid_dim - 1) + list(coords.shape[1:]))
    batch, feature_dim = grid.shape[:2]
    n = coords.shape[-2]
    interp = F.grid_sample(grid, coords, align_corners=align_corners, mode="bilinear", padding_mode="border")
    interp = interp.view(batch, feature_dim, n).transpose(-1, -2)  # [B, n, C]
    return interp.squeeze()
