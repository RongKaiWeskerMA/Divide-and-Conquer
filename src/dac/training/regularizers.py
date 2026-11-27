"""Weighted K-Planes regularizers."""

from __future__ import annotations

import math
from typing import Any

import torch

from dac.config import RegularizationConfig
from dac.models.kplanes import KPlanesModel
from dac.ops.losses import distortion_loss, interlevel_loss, plane_tv


class Regularizer:
    """A loss term scaled by ``weight``; subclasses implement :meth:`compute`."""

    name = "regularizer"

    def __init__(self, weight: float) -> None:
        self.weight = float(weight)
        self.last_value: torch.Tensor | None = None

    def __call__(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        value = self.compute(model, outputs) * self.weight
        self.last_value = value.detach()
        return value

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{type(self).__name__}(weight={self.weight})"


class PlaneTV(Regularizer):
    """Total variation of the feature planes of the main field or the proposal networks."""

    def __init__(self, weight: float, proposal_networks: bool = False) -> None:
        super().__init__(weight)
        self.proposal_networks = proposal_networks
        self.name = "plane_tv_proposal" if proposal_networks else "plane_tv"

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        total: Any = 0
        for grids in model.feature_planes(self.proposal_networks):
            # As in K-Planes the three spatial planes are counted twice.
            for grid_id in (0, 1, 2):
                total += plane_tv(grids[grid_id])
            for grid in grids:
                total += plane_tv(grid)
        return total


class HistogramLoss(Regularizer):
    """MipNeRF-360 interlevel loss between proposal and final weight histograms."""

    name = "histogram"

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        return interlevel_loss(outputs["weights_list"], outputs["ray_samples_list"])


class L1ProposalNetwork(Regularizer):
    name = "l1_proposal"

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        total: Any = 0.0
        for grids in model.feature_planes(proposal_networks=True):
            for grid in grids:
                total += torch.abs(grid).mean()
        return torch.as_tensor(total)


class DepthTV(Regularizer):
    """Total variation of the rendered depth, treating the ray batch as a square patch."""

    name = "depth_tv"

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        depth = outputs["depth"]
        side = math.isqrt(depth.numel())
        if side * side != depth.numel():
            raise ValueError("depth TV needs a square number of rays per batch")
        return plane_tv(depth.reshape(side, side)[None, None])


class DistortionLoss(Regularizer):
    name = "distortion"

    def compute(self, model: KPlanesModel, outputs: dict[str, Any]) -> torch.Tensor:
        return distortion_loss(outputs["weights_list"][-1], outputs["ray_samples_list"][-1])


def build_regularizers(cfg: RegularizationConfig) -> list[Regularizer]:
    """All regularizers with a positive weight, in a fixed order."""
    candidates: list[Regularizer] = [
        PlaneTV(cfg.plane_tv),
        PlaneTV(cfg.plane_tv_proposal_net, proposal_networks=True),
        HistogramLoss(cfg.histogram),
        L1ProposalNetwork(cfg.l1_proposal_net),
        DepthTV(cfg.depth_tv),
        DistortionLoss(cfg.distortion),
    ]
    return [r for r in candidates if r.weight > 0]
