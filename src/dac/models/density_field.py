"""Low-resolution K-Planes density field used by the proposal sampler."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Callable

import torch
from torch import nn

from dac.models import tcnn_compat
from dac.models.kplane_field import ParamGroups, _split_params, init_grid_param, interpolate_ms_features, normalize_aabb
from dac.rendering.spatial_distortions import SceneContraction

log = logging.getLogger(__name__)


class KPlaneDensityField(nn.Module):
    def __init__(
        self,
        aabb: torch.Tensor,
        resolution: Sequence[int],
        num_input_coords: int,
        num_output_coords: int,
        density_activation: Callable[[torch.Tensor], torch.Tensor],
        spatial_distortion: SceneContraction | None = None,
        linear_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.feature_dim = num_output_coords
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder
        self.grids = init_grid_param(
            grid_nd=2, in_dim=num_input_coords, out_dim=num_output_coords, reso=resolution, a=0.1, b=0.15
        )
        self.sigma_net = tcnn_compat.network(
            n_input_dims=self.feature_dim,
            n_output_dims=1,
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "None" if self.linear_decoder else "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            },
        )
        log.debug("Initialized KPlaneDensityField with resolution %s", list(resolution))

    def get_density(self, pts: torch.Tensor) -> torch.Tensor:
        """Densities ``[n_rays, n_samples, 1]`` at world-space ``pts``."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts) / 2  # contracted [-2, 2] -> [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(pts, ms_grids=[self.grids], grid_dimensions=2, concat_features=False)
        return self.density_activation(self.sigma_net(features).to(pts)).view(n_rays, n_samples, 1)

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        return self.get_density(pts)

    def get_params(self) -> ParamGroups:
        return _split_params(self, ["grids"], ["sigma_net"])
