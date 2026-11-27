"""The K-Planes radiance field (https://arxiv.org/abs/2301.10241)."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import Callable

import torch
from torch import nn

from dac.config import GridConfig
from dac.models import tcnn_compat
from dac.ops.interpolation import grid_sample_wrapper
from dac.rendering.spatial_distortions import SceneContraction

log = logging.getLogger(__name__)

ParamGroups = dict[str, list[nn.Parameter]]

SH_DEGREE = 4


def normalize_aabb(pts: torch.Tensor, aabb: torch.Tensor) -> torch.Tensor:
    """Map points inside ``aabb`` (``[2, 3]``: min and max corner) to ``[-1, 1]^3``."""
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
    grid_nd: int, in_dim: int, out_dim: int, reso: Sequence[int], a: float = 0.1, b: float = 0.5
) -> nn.ParameterList:
    """One ``[1, out_dim, *reso]`` plane per ``grid_nd``-combination of the input axes."""
    if in_dim != len(reso):
        raise ValueError("resolution must have one entry per input dimension")
    if grid_nd > in_dim:
        raise ValueError("grid_nd cannot exceed in_dim")
    grid_coefs = nn.ParameterList()
    for coo_comb in itertools.combinations(range(in_dim), grid_nd):
        plane = nn.Parameter(torch.empty([1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]))
        nn.init.uniform_(plane, a=a, b=b)
        grid_coefs.append(plane)
    return grid_coefs


def interpolate_ms_features(
    pts: torch.Tensor,
    ms_grids: Sequence[nn.ParameterList],
    grid_dimensions: int,
    concat_features: bool,
) -> torch.Tensor:
    """Multi-scale plane features: a product over planes, then a sum or concat over scales.

    Args:
        pts: ``[n, in_dim]`` normalised points in ``[-1, 1]``.
    """
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))
    per_scale: list[torch.Tensor] = []
    for grid in ms_grids:
        interp_space: torch.Tensor | float = 1.0
        for ci, coo_comb in enumerate(coo_combs):
            feature_dim = grid[ci].shape[1]  # grid[ci]: [1, out_dim, *reso]
            interp_out_plane = grid_sample_wrapper(grid[ci], pts[..., coo_comb]).view(-1, feature_dim)
            interp_space = interp_space * interp_out_plane
        assert isinstance(interp_space, torch.Tensor)
        per_scale.append(interp_space)
    if concat_features:
        return torch.cat(per_scale, dim=-1)
    total = per_scale[0]
    for features in per_scale[1:]:
        total = total + features
    return total


def _split_params(module: nn.Module, field_prefixes: Sequence[str], nn_prefixes: Sequence[str]) -> ParamGroups:
    """Group parameters into plane features, network weights and everything else."""
    groups: ParamGroups = {"field": [], "nn": [], "other": []}
    named = dict(module.named_parameters())
    for prefixes, key in ((field_prefixes, "field"), (nn_prefixes, "nn")):
        for prefix in prefixes:
            submodule = module.get_submodule(prefix)
            for name, _ in submodule.named_parameters(prefix=prefix):
                groups[key].append(named.pop(name))
    groups["other"] = list(named.values())
    return groups


class KPlaneField(nn.Module):
    """Multi-scale K-Planes feature grid with a density head and a colour decoder.

    With ``linear_decoder`` (the "explicit" model) colours are a linear combination of
    the plane features with weights predicted from the view direction; otherwise (the
    "hybrid" model) a small MLP decodes features and encoded directions.
    """

    def __init__(
        self,
        aabb: torch.Tensor,
        grid_config: GridConfig,
        concat_features_across_scales: bool,
        multiscale_res: Sequence[int] | None,
        spatial_distortion: SceneContraction | None,
        density_activation: Callable[[torch.Tensor], torch.Tensor],
        linear_decoder: bool,
        linear_decoder_layers: int,
    ) -> None:
        super().__init__()
        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config
        self.multiscale_res_multipliers: list[int] = list(multiscale_res or [1])
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        # 1. Feature planes, one set per scale (multi-resolution on spatial axes only).
        self.grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            reso = [r * res for r in grid_config.resolution[:3]] + list(grid_config.resolution[3:])
            gp = init_grid_param(
                grid_nd=grid_config.grid_dimensions,
                in_dim=grid_config.input_coordinate_dim,
                out_dim=grid_config.output_coordinate_dim,
                reso=reso,
            )
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.grids.append(gp)
        log.debug("Initialized model grids: %s", self.grids)

        # 2. Direction encoding (used by the hybrid decoder).
        self.direction_encoder = tcnn_compat.encoding(
            n_input_dims=3, encoding_config={"otype": "SphericalHarmonics", "degree": SH_DEGREE}
        )

        # 3. Decoders.
        if self.linear_decoder:
            # A learned basis (instead of spherical harmonics) mapping view directions to
            # weights that combine the colour features into RGB, as in instant-NGP.
            self.color_basis = tcnn_compat.network(
                n_input_dims=3,
                n_output_dims=3 * self.feature_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": linear_decoder_layers,
                },
            )
            # A linear map from features to density.
            self.sigma_net = tcnn_compat.network(
                n_input_dims=self.feature_dim,
                n_output_dims=1,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "None",
                    "output_activation": "None",
                    "n_neurons": 128,
                    "n_hidden_layers": 0,
                },
            )
        else:
            self.geo_feat_dim = 15
            self.sigma_net = tcnn_compat.network(
                n_input_dims=self.feature_dim,
                n_output_dims=self.geo_feat_dim + 1,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                },
            )
            self.in_dim_color = SH_DEGREE**2 + self.geo_feat_dim
            self.color_net = tcnn_compat.network(
                n_input_dims=self.in_dim_color,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "Sigmoid",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                },
            )

    def get_density(self, pts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Densities ``[n_rays, n_samples, 1]`` and features for world-space ``pts``."""
        if self.spatial_distortion is not None:
            pts = self.spatial_distortion(pts) / 2  # contracted [-2, 2] -> [-1, 1]
        else:
            pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts,
            ms_grids=self.grids,  # type: ignore[arg-type]
            grid_dimensions=self.grid_config.grid_dimensions,
            concat_features=self.concat_features,
        )
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        if self.linear_decoder:
            density_before_activation = self.sigma_net(features)
        else:
            features = self.sigma_net(features)
            features, density_before_activation = torch.split(features, [self.geo_feat_dim, 1], dim=-1)
        density = self.density_activation(density_before_activation.to(pts)).view(n_rays, n_samples, 1)
        return density, features

    def forward(self, pts: torch.Tensor, directions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Colour and density at ``pts`` (``[n_rays, n_samples, 3]``) seen along ``directions``."""
        density, features = self.get_density(pts)
        n_rays, n_samples = pts.shape[:2]
        directions = directions.view(-1, 1, 3).expand(pts.shape).reshape(-1, 3)

        if self.linear_decoder:
            basis_values = self.color_basis(directions)  # [n, 3 * feature_dim]
            basis_values = basis_values.view(features.shape[0], 3, -1)
            rgb = torch.sum(features[:, None, :] * basis_values, dim=-1)
            rgb = torch.sigmoid(rgb.to(directions)).view(n_rays, n_samples, 3)
        else:
            encoded_directions = self.direction_encoder((directions + 1.0) / 2.0)  # SH expects [0, 1]
            color_features = torch.cat([encoded_directions, features.view(-1, self.geo_feat_dim)], dim=-1)
            rgb = self.color_net(color_features).to(directions).view(n_rays, n_samples, 3)
        return {"rgb": rgb, "density": density}

    def get_params(self) -> ParamGroups:
        decoder = "color_basis" if self.linear_decoder else "color_net"
        return _split_params(self, ["grids"], ["sigma_net", "direction_encoder", decoder])
