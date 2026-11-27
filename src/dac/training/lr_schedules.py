from __future__ import annotations

import math
from collections.abc import Sequence

import torch

NUM_WARMUP_STEPS = 512


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    eta_min: float = 0.0,
    num_cycles: float = 0.999,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay.

    https://github.com/huggingface/transformers/blob/bd469c40659ce76c81f69c7726759d249b4aef49/src/transformers/optimization.py#L129
    """

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(eta_min, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0))))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def get_step_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, milestones: Sequence[int], gamma: float, num_warmup_steps: int
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a ``gamma`` decay at every milestone."""

    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        out = 1.0
        for m in milestones:
            if current_step < m:
                break
            out *= gamma
        return out

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(
    name: str | None, optimizer: torch.optim.Optimizer, num_steps: int
) -> torch.optim.lr_scheduler.LRScheduler | None:
    if name is None:
        return None
    milestones = [num_steps // 2, num_steps * 3 // 4, num_steps * 5 // 6, num_steps * 9 // 10]
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=0)
    if name == "warmup_cosine":
        return get_cosine_schedule_with_warmup(optimizer, NUM_WARMUP_STEPS, num_steps)
    if name == "step":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.33)
    if name == "warmup_step":
        return get_step_schedule_with_warmup(optimizer, milestones, gamma=0.33, num_warmup_steps=NUM_WARMUP_STEPS)
    raise ValueError(f"unknown scheduler {name!r}")
