"""K-Planes NeRF: a :class:`KPlaneField` rendered with proposal-network sampling."""

from __future__ import annotations

import logging
import typing
from typing import Any

import numpy as np
import torch
from torch import nn

from dac.config import ModelConfig
from dac.models.density_field import KPlaneDensityField
from dac.models.kplane_field import KPlaneField
from dac.ops.activations import density_activation
from dac.rendering.ray_samplers import (
    ProposalNetworkSampler,
    RayBundle,
    RaySamples,
    SpacedSampler,
    UniformLinDispPiecewiseSampler,
    UniformSampler,
)
from dac.rendering.spatial_distortions import SceneContraction

log = logging.getLogger(__name__)


class KPlanesModel(nn.Module):
    """Renders rays through a K-Planes field.

    The module layout (and therefore the ``state_dict`` keys) matches the original
    K-Planes ``LowrankModel`` so existing checkpoints load unchanged.
    """

    def __init__(self, cfg: ModelConfig, aabb: torch.Tensor, is_contracted: bool) -> None:
        super().__init__()
        self.cfg = cfg
        self.is_contracted = is_contracted
        self.density_act = density_activation(cfg.density_activation)

        self.spatial_distortion: SceneContraction | None = None
        if is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float("inf"),
                global_scale=torch.as_tensor(cfg.global_scale, dtype=torch.float32),
                global_translation=torch.as_tensor(cfg.global_translation, dtype=torch.float32),
            )

        self.field = KPlaneField(
            aabb,
            grid_config=cfg.grid,
            concat_features_across_scales=cfg.concat_features_across_scales,
            multiscale_res=cfg.multiscale_res,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=cfg.linear_decoder,
            linear_decoder_layers=cfg.linear_decoder_layers,
        )

        self.num_proposal_iterations = cfg.num_proposal_iterations
        num_nets = 1 if cfg.use_same_proposal_network else cfg.num_proposal_iterations
        nets = []
        for i in range(num_nets):
            net_cfg = cfg.proposal_nets[min(i, len(cfg.proposal_nets) - 1)]
            nets.append(
                KPlaneDensityField(
                    aabb,
                    resolution=net_cfg.resolution,
                    num_input_coords=net_cfg.num_input_coords,
                    num_output_coords=net_cfg.num_output_coords,
                    density_activation=self.density_act,
                    spatial_distortion=self.spatial_distortion,
                    linear_decoder=cfg.linear_decoder,
                )
            )
        self.proposal_networks = nn.ModuleList(nets)
        self._proposal_nets = nets  # typed view of ``proposal_networks``
        if cfg.use_same_proposal_network:
            self.density_fns = [nets[0].get_density] * self.num_proposal_iterations
        else:
            self.density_fns = [net.get_density for net in nets]

        initial_sampler: SpacedSampler
        if is_contracted:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=cfg.single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=cfg.single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_proposal_samples_per_ray=cfg.num_proposal_samples,
            num_nerf_samples_per_ray=cfg.num_samples,
            num_proposal_network_iterations=cfg.num_proposal_iterations,
            single_jitter=cfg.single_jitter,
            update_sched=self._proposal_update_schedule,
            initial_sampler=initial_sampler,
        )

    def _proposal_update_schedule(self, step: int) -> float:
        """Number of steps between proposal-network gradient updates, ramped up during warmup."""
        every = self.cfg.proposal_update_every
        return float(np.clip(np.interp(step, [0, self.cfg.proposal_warmup], [0, every]), 1, every))

    def step_before_iter(self, step: int) -> None:
        if self.cfg.use_proposal_weight_anneal:
            # Anneal the proposal weights before PDF sampling (https://arxiv.org/abs/2111.12077, eq. 18).
            train_frac = np.clip(step / self.cfg.proposal_weights_anneal_max_num_iters, 0, 1)
            slope = self.cfg.proposal_weights_anneal_slope
            self.proposal_sampler.set_anneal((slope * train_frac) / ((slope - 1) * train_frac + 1))

    def step_after_iter(self, step: int) -> None:
        if self.cfg.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: torch.Tensor | None) -> torch.Tensor:
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        if bg_color is not None:
            comp_rgb = comp_rgb + (1.0 - torch.sum(weights, dim=-2)) * bg_color
        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor) -> torch.Tensor:
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        return torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]

    def forward(
        self,
        rays_o: torch.Tensor,
        rays_d: torch.Tensor,
        bg_color: torch.Tensor | None,
        near_far: torch.Tensor,
        return_samples: bool | None = None,
    ) -> dict[str, Any]:
        """Render a batch of rays.

        Args:
            rays_o: ``[n_rays, 3]`` origins.
            rays_d: ``[n_rays, 3]`` unit directions.
            bg_color: ``[1, 3]`` or ``[n_rays, 3]`` background colour (``None`` for black).
            near_far: ``[1, 2]`` or ``[n_rays, 2]`` sampling bounds.
            return_samples: Also return the per-level weights and ray samples, which the
                losses need. Defaults to ``self.training``.

        Returns:
            ``rgb``, ``depth``, ``accumulation`` and ``prop_depth_{i}`` (``[n_rays, ...]``),
            the per-sample colours ``rgb_pts`` of the final level, per-level ``density_list``
            and ``alpha_list`` and, if requested, ``weights_list`` and ``ray_samples_list``.
        """
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears, fars = ones * nears, ones * fars
        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)

        proposal = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)
        ray_samples = proposal.ray_samples
        field_out = self.field(ray_samples.get_positions(), ray_bundle.directions)
        rgb, density = field_out["rgb"], field_out["density"]
        weights, alphas = ray_samples.get_weights(density)
        weights_list = [*proposal.weights_list, weights]
        ray_samples_list = [*proposal.ray_samples_list, ray_samples]

        outputs: dict[str, Any] = {
            "rgb": self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color),
            "accumulation": torch.sum(weights, dim=-2),
            "depth": self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions),
            "rgb_pts": rgb,
            "density_list": [*proposal.density_list, density],
            "alpha_list": [*proposal.alpha_list, alphas],
        }
        if return_samples is None:
            return_samples = self.training
        if return_samples:  # memory-hungry, so not kept by default at eval time
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions
            )
        return outputs

    def feature_planes(self, proposal_networks: bool = False) -> list[nn.ParameterList]:
        """Feature planes of the main field (one set per scale) or of each proposal network."""
        if proposal_networks:
            return [net.grids for net in self._proposal_nets]
        return [typing.cast(nn.ParameterList, grids) for grids in self.field.grids]

    def param_groups(self, lr: float) -> tuple[list[dict[str, Any]], list[list[bool]]]:
        """Optimizer parameter groups, plus which entries of the original layout are kept.

        Groups are (plane features, networks, other), each concatenated over the main field
        and the proposal networks, as in K-Planes. Frozen and duplicated parameters (e.g.
        the shared scene-contraction constants) are dropped; the returned masks map the
        original K-Planes layout onto the kept parameters so that optimizer states saved
        by the original code can still be loaded (see :func:`remap_legacy_optimizer_state`).
        """
        per_module = [self.field.get_params()] + [pn.get_params() for pn in self._proposal_nets]
        groups, masks = [], []
        seen: set[int] = set()
        for key in ("field", "nn", "other"):
            legacy = [p for params in per_module for p in params[key]]
            mask = []
            for p in legacy:
                keep = p.requires_grad and id(p) not in seen
                seen.add(id(p))
                mask.append(keep)
            groups.append({"params": [p for p, keep in zip(legacy, mask) if keep], "lr": lr})
            masks.append(mask)
        return groups, masks


def remap_legacy_optimizer_state(state_dict: dict[str, Any], masks: list[list[bool]]) -> dict[str, Any]:
    """Convert an optimizer state saved with the original K-Planes parameter layout.

    Returns ``state_dict`` unchanged if it already matches the current layout.
    """
    groups = state_dict["param_groups"]
    if len(groups) != len(masks) or all(len(g["params"]) == sum(m) for g, m in zip(groups, masks)):
        return state_dict
    if any(len(g["params"]) != len(m) for g, m in zip(groups, masks)):
        raise ValueError("optimizer state does not match the model's parameter layout")
    new_state: dict[Any, Any] = {}
    new_groups = []
    next_index = 0
    for group, mask in zip(groups, masks):
        params = []
        for old_index, keep in zip(group["params"], mask):
            if not keep:
                continue
            if old_index in state_dict["state"]:
                new_state[next_index] = state_dict["state"][old_index]
            params.append(next_index)
            next_index += 1
        new_groups.append({**group, "params": params})
    return {"state": new_state, "param_groups": new_groups}


def build_model(cfg: ModelConfig, scene_bbox: torch.Tensor, is_contracted: bool) -> KPlanesModel:
    model = KPlanesModel(cfg, aabb=scene_bbox, is_contracted=is_contracted)
    num_params = sum(int(np.prod(p.shape)) for p in model.parameters())
    log.info(
        "Initialized K-Planes model with %s parameters (contracted=%s, linear decoder=%s)",
        f"{num_params:,}",
        is_contracted,
        cfg.linear_decoder,
    )
    return model
