from __future__ import annotations

import torch
from torch import nn


class SceneContraction(nn.Module):
    """MipNeRF-360 contraction of unbounded space.

    .. math::
        f(x) = x \\text{ if } ||x|| \\le 1 \\text{ else } (2 - 1 / ||x||) (x / ||x||)

    With the L-inf norm (``order=inf``) space is contracted into a cube of side 4, which
    suits grid-based encodings. Points are scaled and translated before contraction.
    """

    def __init__(
        self,
        order: float | None = None,
        global_translation: torch.Tensor | None = None,
        global_scale: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.order = order
        if global_translation is None:
            global_translation = torch.tensor([0.0, 0.0, 0.0])
        if global_scale is None:
            global_scale = torch.tensor([1.0, 1.0, 1.0])
        self.global_translation = nn.Parameter(global_translation, requires_grad=False)
        self.global_scale = nn.Parameter(global_scale, requires_grad=False)

    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        positions = positions * self.global_scale[None, None, :] + self.global_translation[None, None, :]
        mag = torch.linalg.norm(positions, ord=self.order, dim=-1)
        mask = mag >= 1
        contracted = positions.clone()
        contracted[mask] = (2 - (1 / mag[mask][..., None])) * (positions[mask] / mag[mask][..., None])
        return contracted

    def extra_repr(self) -> str:
        return f"order={self.order}"
