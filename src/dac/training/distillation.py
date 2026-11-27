"""Distilling the experts into a single student model (the "conquer" step of DaC)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from dac.config import Config
from dac.data.dataset import NerfDataset
from dac.experts import build_router, load_experts, query_experts
from dac.ops.losses import alpha_interlevel_loss
from dac.training.trainer import Trainer


class DistillationTrainer(Trainer):
    """Trains a student to match the experts, each ray supervised by the expert that owns its view.

    The loss combines (weights in ``dac.loss_weights``):

    * ``alpha_histogram``: the teacher's alpha histogram must be bounded by each student
      proposal level (:func:`~dac.ops.losses.alpha_interlevel_loss`);
    * ``rgb_points``/``alpha_points``: student colour and alpha at the teacher's samples;
    * ``rgb``: rendered colours.
    """

    def __init__(self, cfg: Config, run_dir: Path, test_set: NerfDataset, train_set: NerfDataset) -> None:
        if cfg.dac.expert is not None:
            raise ValueError("distillation uses every view; unset dac.expert")
        # Experts are built before the student so that, for a given seed, the student's
        # initialisation matches the original implementation.
        self.experts = load_experts(
            cfg, train_set.scene_bbox, train_set.is_contracted, torch.device(cfg.experiment.device)
        )
        self.router = build_router(cfg, train_set.frame_names)
        super().__init__(cfg, run_dir, test_set, train_set)

    def compute_losses(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        teacher = query_experts(
            self.experts,
            self.router(batch),
            batch["rays_o"],
            batch["rays_d"],
            bg_color=batch["bg_color"],
            near_far=batch["near_fars"],
        )
        teacher_samples = teacher["ray_samples_list"][-1]
        teacher_alpha = teacher["alpha_list"][-1]

        # Query the student at the teacher's final sample positions. As in the paper's
        # implementation, alphas use the student's own sample spacing.
        student_at_teacher = self.model.field(teacher_samples.get_positions(), batch["rays_d"])
        _, student_alpha = outputs["ray_samples_list"][-1].get_weights(student_at_teacher["density"])

        w = self.cfg.dac.loss_weights
        alpha_hist = alpha_interlevel_loss(
            teacher_alpha, teacher_samples, outputs["alpha_list"], outputs["ray_samples_list"]
        )
        return {
            "alpha_histogram": w.alpha_histogram * alpha_hist,
            "rgb_points": w.rgb_points * F.mse_loss(student_at_teacher["rgb"], teacher["rgb_pts"]),
            "alpha_points": w.alpha_points * F.mse_loss(student_alpha, teacher_alpha),
            "rgb": w.rgb * F.mse_loss(outputs["rgb"], teacher["rgb"]),
        }
