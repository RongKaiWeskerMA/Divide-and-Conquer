"""Divide-and-Conquer experts: assigning views to experts and querying them as teachers.

Each expert is a K-Planes model trained on one group of training views. During
distillation every ray is routed to the expert responsible for the view it came from
and the experts' outputs are merged into a single teacher prediction.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from dac.config import Config
from dac.models.kplanes import KPlanesModel, build_model
from dac.rendering.ray_samplers import RaySamples
from dac.utils import load_torch_file

log = logging.getLogger(__name__)

Router = Callable[[Mapping[str, torch.Tensor]], torch.Tensor]


def angular_centers(num_experts: int) -> np.ndarray:
    """Azimuths (radians) of the sector centres: ``(2k + 1) * pi / num_experts``.

    For 4 experts these are 45, 135, 225 and 315 degrees.
    """
    if num_experts < 1:
        raise ValueError("num_experts must be at least 1")
    return np.deg2rad([(2 * k + 1) * 180.0 / num_experts for k in range(num_experts)])


def angular_expert_index(camera_xy: torch.Tensor | np.ndarray, num_experts: int) -> torch.Tensor:
    """0-based expert index of every camera position, by azimuth around the z-axis.

    A position is assigned to the sector centre with the largest dot product, i.e. the
    nearest centre in angle. Computed in float64 so assignments are deterministic.
    """
    xy = torch.as_tensor(camera_xy).to(torch.float64)
    theta = angular_centers(num_experts)
    directions = torch.as_tensor(np.stack([np.cos(theta), np.sin(theta)], axis=1), device=xy.device)
    return torch.argmax(xy @ directions.T, dim=-1)


class AngularRouter:
    """Routes each ray by the azimuth of its origin (the camera centre)."""

    def __init__(self, num_experts: int) -> None:
        self.num_experts = num_experts

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return angular_expert_index(batch["rays_o"][:, :2], self.num_experts)


class LookupRouter:
    """Routes each ray by a precomputed frame -> expert table."""

    def __init__(self, frame_to_expert: torch.Tensor) -> None:
        self.frame_to_expert = frame_to_expert.long()

    def __call__(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        frame_idx = batch["frame_idx"]
        return self.frame_to_expert.to(frame_idx.device)[frame_idx.long()]

    @classmethod
    def from_file(cls, path: str | Path, frame_names: Sequence[str], num_experts: int) -> LookupRouter:
        assignment = load_assignment(path)
        missing = [name for name in frame_names if name not in assignment]
        if missing:
            raise ValueError(f"{path} has no expert for {len(missing)} training frames, e.g. {missing[:3]}")
        table = torch.tensor([int(assignment[name]) for name in frame_names])
        if table.numel() and (table.min() < 0 or table.max() >= num_experts):
            raise ValueError(f"{path} assigns experts outside [0, {num_experts - 1}]")
        return cls(table)


def load_assignment(path: str | Path) -> dict[str, int]:
    """Read a frame name -> 0-based expert index mapping (``.json``, or a legacy ``.pickle``)."""
    path = Path(path)
    if path.suffix in (".pickle", ".pkl"):
        # Legacy DaC files (pose2com.pickle). Only load pickles you created yourself.
        with open(path, "rb") as f:
            data = pickle.load(f)
    else:
        with open(path) as f:
            data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping from frame name to expert index")
    return {str(k): int(v) for k, v in data.items()}


def build_router(cfg: Config, frame_names: Sequence[str]) -> Router:
    if cfg.dac.router == "angular":
        return AngularRouter(cfg.dac.num_experts)
    path = os.path.join(cfg.data.root, cfg.dac.assignment_file)
    return LookupRouter.from_file(path, frame_names, cfg.dac.num_experts)


def expert_checkpoint_path(cfg: Config, index: int) -> Path:
    """Checkpoint of the 1-based expert ``index``."""
    return Path(cfg.dac.expert_checkpoint.format(log_dir=cfg.experiment.log_dir, scene=cfg.data.scene, index=index))


def load_experts(
    cfg: Config, scene_bbox: torch.Tensor, is_contracted: bool, device: torch.device
) -> list[KPlanesModel]:
    """Build every expert and load its trained weights; experts are frozen and in eval mode."""
    paths = [expert_checkpoint_path(cfg, i) for i in range(1, cfg.dac.num_experts + 1)]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing expert checkpoints (train them first, or point dac.expert_checkpoint at "
            "them):\n  " + "\n  ".join(missing)
        )
    experts = []
    for path in paths:
        expert = build_model(cfg.model, scene_bbox, is_contracted).to(device)
        expert.load_state_dict(load_torch_file(path, map_location=device)["model"])
        expert.eval().requires_grad_(False)
        experts.append(expert)
        log.info("Loaded expert from %s", path)
    return experts


@torch.no_grad()
def query_experts(
    experts: Sequence[KPlanesModel],
    assignment: torch.Tensor,
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    bg_color: torch.Tensor,
    near_far: torch.Tensor,
) -> dict[str, Any]:
    """Render every ray with its assigned expert and merge the outputs into one batch."""
    parts = []
    for index, expert in enumerate(experts):
        mask = assignment == index
        if mask.any():
            out = expert(rays_o[mask], rays_d[mask], bg_color=bg_color, near_far=near_far, return_samples=True)
            parts.append((mask, out))
    if not parts:
        raise ValueError("no rays were assigned to any expert")
    return {key: _merge([(mask, out[key]) for mask, out in parts], rays_o.shape[0]) for key in parts[0][1]}


def _merge(parts: list[tuple[torch.Tensor, Any]], num_rays: int) -> Any:
    """Scatter per-expert values (tensors, lists of them, or RaySamples) back into ray order."""
    first = parts[0][1]
    if isinstance(first, torch.Tensor):
        merged = first.new_zeros((num_rays, *first.shape[1:]))
        for mask, value in parts:
            merged[mask] = value
        return merged
    if isinstance(first, list):
        return [_merge([(mask, value[i]) for mask, value in parts], num_rays) for i in range(len(first))]
    if isinstance(first, RaySamples):
        fields: dict[str, Any] = {}
        for f in dataclasses.fields(RaySamples):
            if f.name == "spacing_to_euclidean_fn":
                continue
            values = [(mask, getattr(value, f.name)) for mask, value in parts]
            fields[f.name] = None if values[0][1] is None else _merge(values, num_rays)
        # The spacing function is specific to each expert's sub-batch, so it cannot be merged.
        return RaySamples(**fields, spacing_to_euclidean_fn=None)
    raise TypeError(f"cannot merge expert outputs of type {type(first).__name__}")
