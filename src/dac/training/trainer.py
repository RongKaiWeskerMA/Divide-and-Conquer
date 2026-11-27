"""Training, validation and rendering of a single K-Planes model."""

from __future__ import annotations

import csv
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dac.config import Config, config_to_dict
from dac.data.dataset import NerfDataset
from dac.models.kplanes import build_model, remap_legacy_optimizer_state
from dac.ops.image_io import write_png, write_video
from dac.ops.metrics import image_metrics
from dac.training.lr_schedules import build_scheduler
from dac.training.regularizers import build_regularizers
from dac.utils import EMA, autocast, load_torch_file, make_grad_scaler, progress_bar, seed_dataloader_worker

log = logging.getLogger(__name__)

CHECKPOINT_NAME = "model.pth"
_BATCH_TENSORS = ("rays_o", "rays_d", "imgs", "near_fars", "bg_color", "frame_idx")


class Trainer:
    """Trains a K-Planes model on one scene and evaluates it.

    Used for the baseline, for every expert and for fine-tuning the distilled model;
    :class:`dac.training.distillation.DistillationTrainer` only changes the loss.

    Args:
        cfg: The experiment configuration.
        run_dir: Where checkpoints, metrics, images and TensorBoard logs are written.
        test_set: Images used for validation (or the render path for :meth:`render_video`).
        train_set: Training rays; may be omitted for evaluation-only use.
    """

    def __init__(self, cfg: Config, run_dir: Path, test_set: NerfDataset, train_set: NerfDataset | None = None) -> None:
        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(cfg.experiment.device)
        self.fp16 = cfg.optim.fp16 and self.device.type == "cuda"
        self.train_set = train_set
        self.test_set = test_set

        self.model = build_model(cfg.model, scene_bbox=test_set.scene_bbox, is_contracted=test_set.is_contracted)
        self.model.to(self.device)
        param_groups, self._param_masks = self.model.param_groups(cfg.optim.lr)
        self.optimizer = torch.optim.Adam(param_groups, eps=1e-15)
        self.scheduler = build_scheduler(cfg.optim.scheduler, self.optimizer, cfg.optim.num_steps)
        self.regularizers = build_regularizers(cfg.regularization)
        self.scaler = make_grad_scaler(self.fp16)

        self.global_step = 0
        self.losses: defaultdict[str, EMA] = defaultdict(lambda: EMA(0.9))
        self._writer: Any = None

    # ------------------------------------------------------------------ training

    def train(self) -> None:
        """Optimise until ``optim.num_steps``, validating and checkpointing periodically and at the end."""
        if self.train_set is None:
            raise RuntimeError("Trainer was created without a training set")
        num_steps = self.cfg.optim.num_steps
        log_cfg = self.cfg.logging
        loader = self._make_loader(self.train_set)
        log.info("Training %s from step %d to %d", self.run_dir, self.global_step + 1, num_steps)
        pbar = progress_bar(initial=self.global_step, total=num_steps)
        last_saved = last_validated = -1
        try:
            self.train_set.reset_iter()
            batches = iter(loader)
            while self.global_step < num_steps:
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                try:
                    batch = next(batches)
                except StopIteration:  # end of epoch: draw a new ray permutation
                    self.train_set.reset_iter()
                    batches = iter(loader)
                    batch = next(batches)

                if self.train_step(batch) and self.scheduler is not None:
                    self.scheduler.step()
                self.model.step_after_iter(self.global_step)

                if self.global_step % log_cfg.log_every == 0:
                    pbar.set_postfix_str(self._loss_summary(), refresh=False)
                    for name, ema in self.losses.items():
                        if ema.value is not None:
                            self.writer.add_scalar(f"train/loss/{name}", ema.value, self.global_step)
                pbar.update(1)
                if log_cfg.valid_every > 0 and self.global_step % log_cfg.valid_every == 0:
                    self.validate()
                    last_validated = self.global_step
                if log_cfg.save_every > 0 and self.global_step % log_cfg.save_every == 0:
                    self.save_checkpoint()
                    last_saved = self.global_step
            if last_validated != self.global_step and log_cfg.valid_every > 0:
                self.validate()
            if last_saved != self.global_step:
                self.save_checkpoint()
        finally:
            pbar.close()
            self.close()

    def train_step(self, batch: dict[str, Any]) -> bool:
        """One optimisation step. Returns False if the step was skipped by the gradient scaler."""
        self.model.train()
        batch = self._to_device(batch)
        with autocast(self.fp16):
            outputs = self.model(
                batch["rays_o"], batch["rays_d"], bg_color=batch["bg_color"], near_far=batch["near_fars"]
            )
            terms = self.compute_losses(batch, outputs)
            names = list(terms)
            loss = terms[names[0]]
            for name in names[1:]:
                loss = loss + terms[name]
            for reg in self.regularizers:
                loss = loss + reg(self.model, outputs)

        self.optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        scale = self.scaler.get_scale()
        self.scaler.update()

        if self.global_step % self.cfg.logging.log_every == 0:
            with torch.no_grad():
                self._record_losses(batch, outputs, loss, terms)
        return bool(scale <= self.scaler.get_scale())

    def compute_losses(self, batch: dict[str, Any], outputs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Loss terms to be summed (before regularizers): photometric MSE."""
        return {"mse": F.mse_loss(outputs["rgb"], batch["imgs"])}

    def _record_losses(
        self, batch: dict[str, Any], outputs: dict[str, Any], loss: torch.Tensor, terms: dict[str, torch.Tensor]
    ) -> None:
        mse = F.mse_loss(outputs["rgb"].float(), batch["imgs"].float()).item()
        self.losses["mse"].update(mse)
        self.losses["psnr"].update(-10 * math.log10(mse) if mse > 0 else float("inf"))
        if len(terms) > 1:
            self.losses["loss"].update(loss.item())
            for name, value in terms.items():
                self.losses[name].update(value.item())
        for reg in self.regularizers:
            if reg.last_value is not None:
                self.losses[reg.name].update(reg.last_value.item())

    def _loss_summary(self) -> str:
        parts = [f"{name}={ema}" for name, ema in self.losses.items()]
        parts.append(f"lr={self.optimizer.param_groups[0]['lr']:.2e}")
        return "  ".join(parts)

    def _make_loader(self, dataset: NerfDataset) -> DataLoader:
        workers = self.cfg.data.num_workers
        kwargs: dict[str, Any] = {"prefetch_factor": 4} if workers > 0 else {}
        return DataLoader(
            dataset,
            batch_size=None,  # the dataset already yields batches of rays
            num_workers=workers,
            pin_memory=self.device.type == "cuda",
            worker_init_fn=seed_dataloader_worker,
            **kwargs,
        )

    def _to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        return {
            k: v.to(self.device, non_blocking=True) if k in _BATCH_TENSORS and isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    # ------------------------------------------------------------------ evaluation

    @torch.no_grad()
    def render_image(self, data: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Render every ray of one image in chunks; returns ``rgb`` and depth maps on the CPU."""
        self.model.eval()
        rays_o, rays_d = data["rays_o"], data["rays_d"]
        near_far = data["near_fars"].to(self.device)
        bg_color = data["bg_color"].to(self.device)
        chunk = self.cfg.logging.eval_batch_size
        preds: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
        with autocast(self.fp16):
            for start in range(0, rays_o.shape[0], chunk):
                outputs = self.model(
                    rays_o[start : start + chunk].to(self.device),
                    rays_d[start : start + chunk].to(self.device),
                    bg_color=bg_color,
                    near_far=near_far,
                    return_samples=False,
                )
                for key, value in outputs.items():
                    if key == "rgb" or "depth" in key:
                        preds[key].append(value.cpu())
        return {key: torch.cat(values, dim=0) for key, values in preds.items()}

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Evaluate on every test view; writes metrics CSVs (and images) to the run directory."""
        dataset = self.test_set
        if dataset.imgs is None:
            raise RuntimeError("validation needs a dataset with ground-truth images")
        per_view: list[dict[str, float]] = []
        pbar = progress_bar(total=len(dataset), desc=f"Validating {dataset.name}")
        for index in range(len(dataset)):
            data = dataset[index]
            metrics = self._evaluate_view(data["imgs"], self.render_image(data), index)
            per_view.append(metrics)
            pbar.set_postfix_str(f"PSNR={metrics['psnr']:.2f}", refresh=False)
            pbar.update(1)
        pbar.close()

        names = list(per_view[0])
        summary = {name: float(np.mean([m[name] for m in per_view])) for name in names}
        log.info(
            "step %d/%d | %s",
            self.global_step,
            self.cfg.optim.num_steps,
            " | ".join(f"{k}: {v:.4f}" for k, v in summary.items()),
        )
        for name, value in summary.items():
            self.writer.add_scalar(f"test/{name}", value, self.global_step)
        stem = self.run_dir / f"test_metrics_step{self.global_step}"
        _write_csv(stem.with_suffix(".csv"), [{"step": self.global_step, **summary}])
        _write_csv(Path(f"{stem}_per_view.csv"), [{"view": i, **m} for i, m in enumerate(per_view)])
        return summary

    def _evaluate_view(self, gt: torch.Tensor, preds: dict[str, torch.Tensor], index: int) -> dict[str, float]:
        h, w = self.test_set.img_h, self.test_set.img_w
        rgb = preds["rgb"].reshape(h, w, 3).float().clamp(0, 1)
        if not torch.isfinite(rgb).all():
            log.warning("view %d: prediction has %d NaNs and %d infs", index, rgb.isnan().sum(), rgb.isinf().sum())
            rgb = torch.nan_to_num(rgb, nan=0.0)
        gt = gt.reshape(h, w, -1).cpu()
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
        metrics = image_metrics(rgb, gt)

        if self.cfg.logging.save_outputs:
            error = torch.abs(rgb - gt).mean(-1, keepdim=True).repeat(1, 1, 3)
            panel = torch.cat([rgb, gt, _normalize_01(error)], dim=0)
            name = f"step{self.global_step}-{index}"
            write_png(self.run_dir / f"{name}.png", _to_uint8(panel))
            depths = [preds["depth"]] + [v for k, v in sorted(preds.items()) if k.startswith("prop_depth")]
            depth = torch.cat([d.reshape(h, w, 1).float() for d in depths], dim=0)
            write_png(self.run_dir / f"{name}-depth.png", _to_uint8(_normalize_01(depth).repeat(1, 1, 3)))
        return metrics

    @torch.no_grad()
    def render_video(self, name: str = "rendering_path") -> Path:
        """Render every camera of ``test_set`` (normally the ``render`` split) to an mp4."""
        h, w = self.test_set.img_h, self.test_set.img_w
        frames = []
        for index in progress_bar(iterable=range(len(self.test_set)), desc="Rendering"):
            rgb = self.render_image(self.test_set[index])["rgb"].reshape(h, w, 3).float()
            frames.append(_to_uint8(rgb.clamp(0, 1)))
        path = self.run_dir / f"{name}.mp4"
        write_video(path, frames)
        return path

    # ------------------------------------------------------------------ checkpoints

    def save_checkpoint(self, path: Path | None = None) -> Path:
        path = path or self.run_dir / CHECKPOINT_NAME
        log.info("Saving checkpoint to %s", path)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
                "global_step": self.global_step,
                "config": config_to_dict(self.cfg),
            },
            path,
        )
        return path

    def load_checkpoint(self, path: str | Path, resume: bool = True) -> None:
        """Load model weights and the step; with ``resume`` also the optimizer and LR schedule.

        Checkpoints written by the original DaC/K-Planes code are supported.
        """
        checkpoint = load_torch_file(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        if resume:
            optimizer_state = remap_legacy_optimizer_state(checkpoint["optimizer"], self._param_masks)
            self.optimizer.load_state_dict(optimizer_state)
            if self.scheduler is not None and checkpoint.get("lr_scheduler") is not None:
                self.scheduler.load_state_dict(checkpoint["lr_scheduler"])
        self.global_step = int(checkpoint["global_step"])
        log.info("Loaded %s at step %d (resume=%s)", path, self.global_step, resume)

    @property
    def writer(self) -> Any:
        if self._writer is None:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(self.run_dir))
        return self._writer

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None


def _normalize_01(t: torch.Tensor) -> torch.Tensor:
    lo, hi = t.min(), t.max()
    return (t - lo) / (hi - lo).clamp_min(1e-12)


def _to_uint8(image: torch.Tensor) -> np.ndarray:
    return (image.cpu() * 255.0).byte().numpy()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
