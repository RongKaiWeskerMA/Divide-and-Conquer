"""Losses on sample-weight histograms along rays (MipNeRF-360) and on feature planes."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from dac.rendering.ray_samplers import RaySamples

EPS = 1.0e-7


def outer(
    t0_starts: torch.Tensor,
    t0_ends: torch.Tensor,
    t1_starts: torch.Tensor,
    t1_ends: torch.Tensor,
    y1: torch.Tensor,
) -> torch.Tensor:
    """Sum of the step function ``(t1, y1)`` over every interval of ``t0``.

    Faster version of
    https://github.com/kakaobrain/NeRF-Factory/blob/f61bb87/src/model/mipnerf360/helper.py#L117
    """
    cy1 = torch.cat([torch.zeros_like(y1[..., :1]), torch.cumsum(y1, dim=-1)], dim=-1)
    idx_lo = torch.searchsorted(t1_starts.contiguous(), t0_starts.contiguous(), right=True) - 1
    idx_lo = torch.clamp(idx_lo, min=0, max=y1.shape[-1] - 1)
    idx_hi = torch.searchsorted(t1_ends.contiguous(), t0_ends.contiguous(), right=True)
    idx_hi = torch.clamp(idx_hi, min=0, max=y1.shape[-1] - 1)
    cy1_lo = torch.take_along_dim(cy1[..., :-1], idx_lo, dim=-1)
    cy1_hi = torch.take_along_dim(cy1[..., 1:], idx_hi, dim=-1)
    return cy1_hi - cy1_lo


def lossfun_outer(t: torch.Tensor, w: torch.Tensor, t_env: torch.Tensor, w_env: torch.Tensor) -> torch.Tensor:
    """Penalty for histogram ``(t, w)`` exceeding the envelope ``(t_env, w_env)``.

    Args:
        t: ``[..., n + 1]`` interval edges.
        w: ``[..., n]`` weights.
        t_env: ``[..., m + 1]`` interval edges of the enveloping histogram.
        w_env: ``[..., m]`` weights that should upper-bound ``w``.
    """
    w_outer = outer(t[..., :-1], t[..., 1:], t_env[..., :-1], t_env[..., 1:], w_env)
    return torch.clip(w - w_outer, min=0) ** 2 / (w + EPS)


def interlevel_loss(weights_list: Sequence[torch.Tensor], ray_samples_list: Sequence[RaySamples]) -> torch.Tensor:
    """MipNeRF-360 proposal loss: every proposal histogram must bound the final one."""
    c = ray_samples_list[-1].sdist().detach()
    w = weights_list[-1][..., 0].detach()
    loss = torch.zeros((), device=c.device)
    for ray_samples, weights in zip(ray_samples_list[:-1], weights_list[:-1]):
        loss = loss + torch.mean(lossfun_outer(c, w, ray_samples.sdist(), weights[..., 0]))
    return loss


def alpha_interlevel_loss(
    target_alpha: torch.Tensor,
    target_samples: RaySamples,
    alpha_list: Sequence[torch.Tensor],
    ray_samples_list: Sequence[RaySamples],
) -> torch.Tensor:
    """Distillation variant of :func:`interlevel_loss` on alpha histograms.

    The teacher's final alpha histogram must be bounded by every proposal level of the
    student.
    """
    c = target_samples.sdist().detach()
    alpha = target_alpha[..., 0].detach()
    loss = torch.zeros((), device=c.device)
    for ray_samples, student_alpha in zip(ray_samples_list[:-1], alpha_list[:-1]):
        loss = loss + torch.mean(lossfun_outer(c, alpha, ray_samples.sdist(), student_alpha[..., 0]))
    return loss


def distortion_loss(weights: torch.Tensor, ray_samples: RaySamples) -> torch.Tensor:
    """Efficient O(N) MipNeRF-360 distortion loss.

    From https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py

    Note:
        As in K-Planes, ``weights`` is ``[n_rays, n_samples, 1]`` and the reductions run
        over the trailing singleton dimension, so only the uniform term contributes. This
        is kept as-is so results match the published models.
    """
    midpoints = (ray_samples.starts + ray_samples.ends) / 2
    interval = ray_samples.deltas
    loss_uni = (1 / 3) * (interval * weights.pow(2)).sum(dim=-1).mean()
    wm = weights * midpoints
    w_cumsum = weights.cumsum(dim=-1)
    wm_cumsum = wm.cumsum(dim=-1)
    loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
    loss_bi_1 = weights[..., 1:] * wm_cumsum[..., :-1]
    loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
    return loss_bi + loss_uni


def plane_tv(t: torch.Tensor) -> torch.Tensor:
    """Total variation of a ``[batch, channels, h, w]`` feature plane."""
    batch_size, c, h, w = t.shape
    count_h = batch_size * c * (h - 1) * w
    count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., : h - 1, :]).sum()
    w_tv = torch.square(t[..., :, 1:] - t[..., :, : w - 1]).sum()
    return 2 * (h_tv / count_h + w_tv / count_w)
