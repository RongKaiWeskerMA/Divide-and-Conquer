"""Ray samplers, adapted from nerfstudio.

https://github.com/nerfstudio-project/nerfstudio/blob/628e4fe1a638e7fb3b7ad33d4d91a4b1d63a9b68/nerfstudio/model_components/ray_samplers.py

Copyright 2022 The Nerfstudio Team. All rights reserved.
Licensed under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

DensityFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class RaySamples:
    """Samples (frustums) along a batch of rays."""

    origins: torch.Tensor  # [n_rays, 1, 3]
    directions: torch.Tensor  # [n_rays, 1, 3]
    starts: torch.Tensor  # [n_rays, n_samples, 1], world-space distance along the ray
    ends: torch.Tensor  # [n_rays, n_samples, 1]
    deltas: torch.Tensor  # [n_rays, n_samples, 1], frustum lengths
    # Bin edges in the normalised [0, 1] "spacing" space, before warping to world space.
    spacing_starts: torch.Tensor | None = None  # [n_rays, n_samples, 1]
    spacing_ends: torch.Tensor | None = None  # [n_rays, n_samples, 1]
    spacing_to_euclidean_fn: Callable[[torch.Tensor], torch.Tensor] | None = None

    def get_positions(self) -> torch.Tensor:
        """World-space centre of every frustum, ``[n_rays, n_samples, 3]``."""
        return self.origins + self.directions * (self.starts + self.ends) / 2

    def get_weights(self, densities: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Volume-rendering weights and alphas for ``[n_rays, n_samples, 1]`` densities."""
        delta_mask = self.deltas > 0
        delta_density = torch.zeros_like(densities)
        delta_density[delta_mask] = self.deltas[delta_mask] * densities[delta_mask]
        alphas = 1 - torch.exp(-delta_density)

        transmittance = torch.cumsum(delta_density[..., :-1, :], dim=-2)
        transmittance = torch.cat(
            [torch.zeros((*transmittance.shape[:1], 1, 1), device=densities.device), transmittance],
            dim=-2,
        )
        transmittance = torch.exp(-transmittance)
        weights = alphas * transmittance
        return weights, alphas

    def sdist(self) -> torch.Tensor:
        """Bin edges in spacing space, ``[n_rays, n_samples + 1]``."""
        assert self.spacing_starts is not None
        assert self.spacing_ends is not None
        return torch.cat([self.spacing_starts[..., 0], self.spacing_ends[..., -1:, 0]], dim=-1)


@dataclass
class RayBundle:
    origins: torch.Tensor  # [n_rays, 3]
    directions: torch.Tensor  # [n_rays, 3], unit length
    nears: torch.Tensor  # [n_rays, 1]
    fars: torch.Tensor  # [n_rays, 1]

    def get_ray_samples(
        self,
        bin_starts: torch.Tensor,
        bin_ends: torch.Tensor,
        spacing_starts: torch.Tensor,
        spacing_ends: torch.Tensor,
        spacing_to_euclidean_fn: Callable[[torch.Tensor], torch.Tensor],
    ) -> RaySamples:
        return RaySamples(
            origins=self.origins[..., None, :],
            directions=self.directions[..., None, :],
            starts=bin_starts,
            ends=bin_ends,
            deltas=bin_ends - bin_starts,
            spacing_starts=spacing_starts,
            spacing_ends=spacing_ends,
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )


class SpacedSampler(nn.Module):
    """Stratified samples whose spacing along the ray is set by ``spacing_fn``.

    Args:
        spacing_fn: Maps world-space distance to the normalised spacing space.
        spacing_fn_inv: Inverse of ``spacing_fn``.
        train_stratified: Jitter the samples within their bins in training mode.
        single_jitter: Use one jitter value per ray instead of one per sample.
    """

    def __init__(
        self,
        spacing_fn: Callable[[torch.Tensor], torch.Tensor],
        spacing_fn_inv: Callable[[torch.Tensor], torch.Tensor],
        train_stratified: bool = True,
        single_jitter: bool = False,
    ) -> None:
        super().__init__()
        self.train_stratified = train_stratified
        self.single_jitter = single_jitter
        self.spacing_fn = spacing_fn
        self.spacing_fn_inv = spacing_fn_inv

    def forward(self, ray_bundle: RayBundle, num_samples: int) -> RaySamples:
        num_rays = ray_bundle.origins.shape[0]
        bins = torch.linspace(0.0, 1.0, num_samples + 1).to(ray_bundle.origins.device)[None, ...]

        if self.train_stratified and self.training:
            if self.single_jitter:
                t_rand = torch.rand((num_rays, 1), dtype=bins.dtype, device=bins.device)
            else:
                t_rand = torch.rand((num_rays, num_samples + 1), dtype=bins.dtype, device=bins.device)
            bin_centers = (bins[..., 1:] + bins[..., :-1]) / 2.0
            bin_upper = torch.cat([bin_centers, bins[..., -1:]], -1)
            bin_lower = torch.cat([bins[..., :1], bin_centers], -1)
            bins = bin_lower + (bin_upper - bin_lower) * t_rand
        else:
            bins = bins.repeat(num_rays, 1)

        s_near, s_far = (self.spacing_fn(x) for x in (ray_bundle.nears, ray_bundle.fars))

        def spacing_to_euclidean_fn(x: torch.Tensor) -> torch.Tensor:
            return self.spacing_fn_inv(x * s_far + (1 - x) * s_near)

        euclidean_bins = spacing_to_euclidean_fn(bins)  # [num_rays, num_samples + 1]
        return ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=spacing_to_euclidean_fn,
        )


class UniformSampler(SpacedSampler):
    """Samples uniformly in world space between near and far."""

    def __init__(self, train_stratified: bool = True, single_jitter: bool = False) -> None:
        super().__init__(
            spacing_fn=lambda x: x,
            spacing_fn_inv=lambda x: x,
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class UniformLinDispPiecewiseSampler(SpacedSampler):
    """Uniform samples up to distance 1, then linear in disparity (for unbounded scenes)."""

    def __init__(self, train_stratified: bool = True, single_jitter: bool = False) -> None:
        super().__init__(
            spacing_fn=lambda x: torch.where(x < 1, x / 2, 1 - 1 / (2 * x)),
            spacing_fn_inv=lambda x: torch.where(x < 0.5, 2 * x, 1 / (2 - 2 * x)),
            train_stratified=train_stratified,
            single_jitter=single_jitter,
        )


class PDFSampler(nn.Module):
    """Importance-samples bins in proportion to the weights of previous samples.

    Args:
        train_stratified: Randomise sample locations within bins in training mode.
        single_jitter: Use one jitter value per ray instead of one per sample.
        include_original: Keep the previous bin edges in addition to the new samples.
        histogram_padding: Added to the weights before building the PDF.
    """

    def __init__(
        self,
        train_stratified: bool = True,
        single_jitter: bool = False,
        include_original: bool = True,
        histogram_padding: float = 0.01,
    ) -> None:
        super().__init__()
        self.train_stratified = train_stratified
        self.include_original = include_original
        self.histogram_padding = histogram_padding
        self.single_jitter = single_jitter

    def forward(
        self,
        ray_bundle: RayBundle,
        ray_samples: RaySamples,
        weights: torch.Tensor,
        num_samples: int,
        eps: float = 1e-5,
    ) -> RaySamples:
        num_bins = num_samples + 1
        weights = weights[..., 0] + self.histogram_padding

        # Add a small offset to rays with zero weight to prevent NaNs.
        weights_sum = torch.sum(weights, dim=-1, keepdim=True)
        padding = torch.relu(eps - weights_sum)
        weights = weights + padding / weights.shape[-1]
        weights_sum += padding

        pdf = weights / weights_sum
        cdf = torch.min(torch.ones_like(pdf), torch.cumsum(pdf, dim=-1))
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)

        u = torch.linspace(0.0, 1.0 - (1.0 / num_bins), steps=num_bins, device=cdf.device)
        if self.train_stratified and self.training:
            u = u.expand((*cdf.shape[:-1], num_bins))
            if self.single_jitter:
                rand = torch.rand((*cdf.shape[:-1], 1), device=cdf.device) / num_bins
            else:
                rand = torch.rand((*cdf.shape[:-1], num_samples + 1), device=cdf.device) / num_bins
            u = u + rand
        else:
            u = u + 1.0 / (2 * num_bins)
            u = u.expand(size=(*cdf.shape[:-1], num_bins))
        u = u.contiguous()

        assert ray_samples.spacing_to_euclidean_fn is not None
        existing_bins = ray_samples.sdist()

        inds = torch.searchsorted(cdf, u, right=True)
        below = torch.clamp(inds - 1, 0, existing_bins.shape[-1] - 1)
        above = torch.clamp(inds, 0, existing_bins.shape[-1] - 1)
        cdf_g0 = torch.gather(cdf, -1, below)
        bins_g0 = torch.gather(existing_bins, -1, below)
        cdf_g1 = torch.gather(cdf, -1, above)
        bins_g1 = torch.gather(existing_bins, -1, above)

        t = torch.clip(torch.nan_to_num((u - cdf_g0) / (cdf_g1 - cdf_g0), 0), 0, 1)
        bins = bins_g0 + t * (bins_g1 - bins_g0)
        if self.include_original:
            bins, _ = torch.sort(torch.cat([existing_bins, bins], -1), -1)
        bins = bins.detach()

        euclidean_bins = ray_samples.spacing_to_euclidean_fn(bins)
        return ray_bundle.get_ray_samples(
            bin_starts=euclidean_bins[..., :-1, None],
            bin_ends=euclidean_bins[..., 1:, None],
            spacing_starts=bins[..., :-1, None],
            spacing_ends=bins[..., 1:, None],
            spacing_to_euclidean_fn=ray_samples.spacing_to_euclidean_fn,
        )


@dataclass
class ProposalSamples:
    """Output of :class:`ProposalNetworkSampler`.

    The ``*_list`` fields hold one entry per proposal level.
    """

    ray_samples: RaySamples  # samples for the main field
    weights_list: list[torch.Tensor]
    ray_samples_list: list[RaySamples]
    density_list: list[torch.Tensor]
    alpha_list: list[torch.Tensor]


class ProposalNetworkSampler(nn.Module):
    """Hierarchical sampler driven by proposal density networks (MipNeRF-360)."""

    def __init__(
        self,
        num_proposal_samples_per_ray: Sequence[int],
        num_nerf_samples_per_ray: int,
        num_proposal_network_iterations: int,
        single_jitter: bool = False,
        update_sched: Callable[[int], float] = lambda step: 1,
        initial_sampler: SpacedSampler | None = None,
    ) -> None:
        super().__init__()
        if num_proposal_network_iterations < 1:
            raise ValueError("num_proposal_network_iterations must be >= 1")
        self.num_proposal_samples_per_ray = list(num_proposal_samples_per_ray)
        self.num_nerf_samples_per_ray = num_nerf_samples_per_ray
        self.num_proposal_network_iterations = num_proposal_network_iterations
        self.update_sched = update_sched
        self.initial_sampler = initial_sampler or UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        self.pdf_sampler = PDFSampler(include_original=False, single_jitter=single_jitter)

        self._anneal = 1.0
        self._steps_since_update = 0
        self._step = 0

    def set_anneal(self, anneal: float) -> None:
        """Exponent applied to the proposal weights before PDF sampling."""
        self._anneal = anneal

    def step_cb(self, step: int) -> None:
        """Register that a training step has passed (drives the update schedule)."""
        self._step = step
        self._steps_since_update += 1

    def forward(self, ray_bundle: RayBundle, density_fns: Sequence[DensityFn]) -> ProposalSamples:
        n = self.num_proposal_network_iterations
        assert len(density_fns) == n
        weights_list: list[torch.Tensor] = []
        ray_samples_list: list[RaySamples] = []
        density_list: list[torch.Tensor] = []
        alpha_list: list[torch.Tensor] = []
        weights: torch.Tensor | None = None
        ray_samples: RaySamples | None = None
        # Always update on the first steps, otherwise the inf check in grad scaling crashes.
        updated = self._steps_since_update > self.update_sched(self._step) or self._step < 10
        for i_level in range(n + 1):
            is_prop = i_level < n
            num_samples = self.num_proposal_samples_per_ray[i_level] if is_prop else self.num_nerf_samples_per_ray
            if i_level == 0:
                ray_samples = self.initial_sampler(ray_bundle, num_samples=num_samples)
            else:
                assert weights is not None
                assert ray_samples is not None
                annealed_weights = torch.pow(weights, self._anneal)
                ray_samples = self.pdf_sampler(ray_bundle, ray_samples, annealed_weights, num_samples=num_samples)
            if is_prop:
                assert ray_samples is not None
                if updated:
                    density = density_fns[i_level](ray_samples.get_positions())
                else:
                    with torch.no_grad():
                        density = density_fns[i_level](ray_samples.get_positions())
                weights, alphas = ray_samples.get_weights(density)
                alpha_list.append(alphas)
                weights_list.append(weights)
                ray_samples_list.append(ray_samples)
                density_list.append(density)
        if updated:
            self._steps_since_update = 0

        assert ray_samples is not None
        return ProposalSamples(ray_samples, weights_list, ray_samples_list, density_list, alpha_list)

    def extra_repr(self) -> str:
        return (
            f"num_proposal_samples_per_ray={self.num_proposal_samples_per_ray}, "
            f"num_nerf_samples_per_ray={self.num_nerf_samples_per_ray}, "
            f"num_proposal_network_iterations={self.num_proposal_network_iterations}"
        )
