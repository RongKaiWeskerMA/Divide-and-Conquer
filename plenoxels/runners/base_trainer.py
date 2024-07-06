import abc
import random
import logging as log
import math
import os
from copy import copy
from typing import Iterable, Optional, Union, Dict, Tuple, Sequence, MutableMapping

import numpy as np
import torch
import torch.utils.data
from torch.utils.tensorboard import SummaryWriter

from plenoxels.utils.timer import CudaTimer
from plenoxels.utils.ema import EMA
from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.ops.image import metrics
from plenoxels.ops.image.io import write_png
from plenoxels.runners.regularization import Regularizer
from plenoxels.ops.lr_scheduling import (
    get_cosine_schedule_with_warmup, get_step_schedule_with_warmup
)
from plenoxels.raymarching.ray_samplers import RaySamples
from plenoxels.ops.losses.histogram_loss import interlevel_loss_dns

class BaseTrainer(abc.ABC):
    def __init__(self,
                 train_data_loader: Iterable,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs):
        self.train_data_loader = train_data_loader
        self.num_steps = num_steps
        self.train_fp16 = train_fp16
        self.save_every = save_every
        self.valid_every = valid_every
        self.save_outputs = save_outputs
        self.device = device
        self.eval_batch_size = kwargs.get('eval_batch_size', 8129)
        self.extra_args = kwargs
        self.timer = CudaTimer(enabled=False)

        self.log_dir = os.path.join(logdir, expname)
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.global_step: Optional[int] = None
        self.loss_info: Optional[Dict[str, EMA]] = None

        self.model = self.init_model(**self.extra_args)
        self.optimizer = self.init_optim(**self.extra_args)
        self.scheduler = self.init_lr_scheduler(**self.extra_args)
        self.criterion = torch.nn.MSELoss(reduction='mean')
        self.regularizers = self.init_regularizers(**self.extra_args)
        self.gscaler = torch.cuda.amp.GradScaler(enabled=self.train_fp16)

        self.model.to(self.device)
        if kwargs['distil']:
            self.distil_data_loader = kwargs['distil_loader']
            # self.register_dns_backward_hooks(self.model)

    def register_dns_backward_hooks(self, model):
        for name, param in model.named_parameters():
            if ("sigma" in name) or ("proposal" in name) and (param.requires_grad == True):
                param.register_hook(lambda grad: torch.clamp(grad, min = 0))
                param.register_hook(hook_fn)
    @abc.abstractmethod
    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        self.model.eval()
        return None  # noqa

    def train_step(self, data, **kwargs) -> bool:
        self.model.train()
        data = self._move_data_to_device(data)
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            fwd_out = self.model(
                data['rays_o'], data['rays_d'], bg_color=data['bg_color'],
                near_far=data['near_fars'], timestamps=data['timestamps'])
            self.timer.check("model-forward")
            # Reconstruction loss
            recon_loss = self.criterion(fwd_out['rgb'], data['imgs'])
            # Regularization
            loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss
            self.timer.check("regularizaion-forward")
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.timer.check("backward")
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")
        self.writer.add_scalar("total loss", loss.item(), self.global_step)

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                self.loss_info[f"mse"].update(recon_loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def post_step(self, progress_bar):
        self.model.step_after_iter(self.global_step)
        if self.global_step % self.calc_metrics_every == 0:
            progress_bar.set_postfix_str(
                losses_to_postfix(self.loss_info, lr=self.lr), refresh=False)
            for loss_name, loss_val in self.loss_info.items():
                self.writer.add_scalar(f"train/loss/{loss_name}", loss_val.value, self.global_step)
                if self.timer.enabled:
                    tsum = 0.
                    tstr = "Timings: "
                    for tname, tval in self.timer.timings.items():
                        tstr += f"{tname}={tval:.1f}ms  "
                        tsum += tval
                    tstr += f"tot={tsum:.1f}ms"
                    log.info(tstr)
        progress_bar.update(1)
        if self.valid_every > -1 and self.global_step % self.valid_every == 0:
            print()
            self.validate()
        if self.save_every > -1 and self.global_step % self.save_every == 0:
            print()
            self.save_model()

    def pre_epoch(self):
        self.loss_info = self.init_epoch_info()

    def train(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        log.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.pre_epoch()
            batch_iter = iter(self.train_data_loader)
            while self.global_step < self.num_steps:
                self.timer.reset()
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                self.timer.check("step-before-iter")
                try:
                    data = next(batch_iter)
                    self.timer.check("dloader-next")
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    data = next(batch_iter)
                    log.info("Reset data-iterator")

                try:
                    step_successful = self.train_step(data)
                except StopIteration:
                    self.pre_epoch()
                    batch_iter = iter(self.train_data_loader)
                    log.info("Reset data-iterator")
                    step_successful = True

                if step_successful and self.scheduler is not None:
                    self.scheduler.step()
                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(progress_bar=pb)
                self.timer.check("after-step")
        finally:
            pb.close()
            self.writer.close()

    def distillation(self):
        """Override this if some very specific training procedure is needed."""
        if self.global_step is None:
            self.global_step = 0
        log.info(f"Starting training from step {self.global_step + 1}")
        pb = tqdm(initial=self.global_step, total=self.num_steps)
        try:
            self.pre_epoch_distil()
            batch_iter = iter(self.distil_data_loader)
            while self.global_step < self.num_steps:
                self.timer.reset()
                self.model.step_before_iter(self.global_step)
                self.global_step += 1
                self.timer.check("step-before-iter")
                try:
                    data = next(batch_iter)
                    self.timer.check("dloader-next")
                except StopIteration:
                    self.pre_epoch_distil()
                    batch_iter = iter(self.distil_data_loader)
                    data = next(batch_iter)
                    log.info("Reset data-iterator")

                try:
                    step_successful = self.distil_step(data)
                except StopIteration:
                    self.pre_epoch_distil()
                    batch_iter = iter(self.distil_data_loader)
                    log.info("Reset data-iterator")
                    step_successful = True

                if step_successful and self.scheduler is not None:
                    self.scheduler.step()
                for r in self.regularizers:
                    r.step(self.global_step)
                self.post_step(progress_bar=pb)
                self.timer.check("after-step")
        finally:
            pb.close()
            self.writer.close()


    def distil_step(self, data, **kwargs):
        self.model.train()
        data = self._move_data_to_device(data)
        if "timestamps" not in data:
            data["timestamps"] = None
        self.timer.check("move-to-device")

        with torch.cuda.amp.autocast(enabled=self.train_fp16):
            dataset = self.extra_args['data_dirs'][0].split('/')[1]
            if dataset == 'nerf_synthetic':
                fwd_out_exp = forward_expert(dataset, data, self.theta, self.experts[:self.num_splits])

            elif dataset == 'TanksAndTempleBG':
                fwd_out_exp = forward_expert(dataset, data, self.theta, self.experts[:self.num_splits], fname=self.fname, pose2com=self.pose2com)

            fwd_out = self.model(
                data['rays_o'], data['rays_d'], bg_color=data['bg_color'],
                near_far=data['near_fars'], timestamps=data['timestamps'])
            
            pts_exp = fwd_out_exp['ray_samples_list'][-1].get_positions()
            fwd_pts_rgb_dns = self.model.field(pts_exp, data['rays_d'], data['timestamps'])
            fwd_out['rgb_pts'] = fwd_pts_rgb_dns['rgb']
            dns_pts = fwd_pts_rgb_dns['density']
            _, alpha_pts = fwd_out['ray_samples_list'][-1].get_weights(dns_pts)
            fwd_out['alpha_pts'] = alpha_pts

            self.timer.check("model-forward")
            # Compute the loss for distillation process
            loss = self.compute_loss(fwd_out, fwd_out_exp)
            recon_loss = self.criterion(fwd_out['rgb'], data['imgs'])
            # Regularization
            # loss = recon_loss
            for r in self.regularizers:
                reg_loss = r.regularize(self.model, model_out=fwd_out)
                loss = loss + reg_loss
            self.timer.check("regularizaion-forward")
        # Update weights
        self.optimizer.zero_grad(set_to_none=True)
        self.gscaler.scale(loss).backward()
        self.timer.check("backward")
        self.gscaler.step(self.optimizer)
        scale = self.gscaler.get_scale()
        self.gscaler.update()
        self.timer.check("scaler-step")

        # Report on losses
        if self.global_step % self.calc_metrics_every == 0:
            with torch.no_grad():
                recon_loss_val = recon_loss.item()
                loss_val = loss.item()
                self.loss_info[f"mse"].update(loss_val)
                self.loss_info[f"psnr"].update(-10 * math.log10(recon_loss_val))
                for r in self.regularizers:
                    r.report(self.loss_info)

        return scale <= self.gscaler.get_scale()

    def compute_loss(self, model_outputs, exp_outputs):
        """
        This Function implement is composed of 2 main components
        1. Parse the output from experts and models (e.g., sigma, rgb, w, t)
        2. Compute the loss between each output pair
            a. We use histogram loss for sampled weights, and intervals (deltas)
            b. We use MSE loss for rgb and density pairs
        """
        
        # get the density from model and exp for both proposal networks and NeRF network
        density_prop_1, density_prop_2, density_nf = model_outputs['density_list'][0], \
                                                     model_outputs['density_list'][1], \
                                                     model_outputs['density_list'][2] 
        
        density_exp_prop_1, density_exp_prop_2, density_exp_nf = exp_outputs['density_list'][0], \
                                                                 exp_outputs['density_list'][1], \
                                                                 exp_outputs['density_list'][2]
        # get the sampled points from experts and distilled networks for both proposal and NeRF networks 
        ray_pts_prop_1, ray_pts_prop_2, ray_pts_nf = model_outputs['ray_samples_list'][0], \
                                                     model_outputs['ray_samples_list'][1], \
                                                     model_outputs['ray_samples_list'][2]
        ray_exp_pts_prop_1_exp, ray_exp_pts_prop_2, ray_exp_pts_nf = exp_outputs['ray_samples_list'][0], \
                                                                     exp_outputs['ray_samples_list'][1], \
                                                                     exp_outputs['ray_samples_list'][2]
        # get the sampled weights from experts and distilled for both proposal and NeRF networks
        w_prop_1, w_prop_2, w_nf = model_outputs['weights_list'][0],\
                                   model_outputs['weights_list'][0],\
                                   model_outputs['weights_list'][2]
        w_exp_prop_1, w_exp_prop_2, w_exp_nf = exp_outputs['weights_list'][0], \
                                               exp_outputs['weights_list'][1],\
                                               exp_outputs['weights_list'][2]
        # get the alpha values from experts and distilled for both proposal and NeRF networks
        a_prop_1, a_prop_2, a_nf = model_outputs['alpha_list'][0], \
                                   model_outputs['alpha_list'][1], \
                                   model_outputs['alpha_list'][2]
        a_exp_prop_1, a_exp_prop_2, a_exp_nf = exp_outputs['alpha_list'][0], \
                                               exp_outputs['alpha_list'][1], \
                                               exp_outputs['alpha_list'][2]

        # construct the density loss between expert NeRF network and model proposal networks
        # alpha_histo_loss = interlevel_loss_dns(w_exp_nf, ray_exp_pts_nf, model_outputs['weights_list'], model_outputs['ray_samples_list'])
        
        alpha_histo_loss = interlevel_loss_dns(a_exp_nf, ray_exp_pts_nf, model_outputs['alpha_list'], 
                                                   model_outputs['ray_samples_list'])

        # get the rgb for each sampled points along rays for both expert and model
        rgb_pts = model_outputs['rgb_pts']
        rgb_pts_exp = exp_outputs['rgb_pts']
        
        alpha_pts_loss = self.criterion(model_outputs['alpha_pts'], a_exp_nf)
        # construct the loss for each output pair
        # the easiest is the MLE loss for rgb, I suspect we need to use histogram loss for density pair
        # dns_pts_loss = self.criterion(model_outputs['dns_pts'], density_exp_nf) 
        rec_loss = self.criterion(rgb_pts, rgb_pts_exp)
        rec_loss_2 = self.criterion(model_outputs['rgb'], exp_outputs['rgb'])
        loss = 0.3 * alpha_histo_loss + 0.2 * rec_loss + 0.2 * alpha_pts_loss + 0.3 * rec_loss_2
        # log loss value into tensorboard
        self.writer.add_scalar("alpha_histo_loss", alpha_histo_loss, self.global_step)
        self.writer.add_scalar("alpha_pts_loss", alpha_pts_loss, self.global_step)
        self.writer.add_scalar("rgb_pts_loss", rec_loss, self.global_step)
        
        return loss
            

    def _move_data_to_device(self, data):
        data["rays_o"] = data["rays_o"].to(self.device)
        data["rays_d"] = data["rays_d"].to(self.device)
        data["imgs"] = data["imgs"].to(self.device)
        data["near_fars"] = data["near_fars"].to(self.device)
        if "timestamps" in data:
            data["timestamps"] = data["timestamps"].to(self.device)
        bg_color = data["bg_color"]
        if isinstance(bg_color, torch.Tensor):
            bg_color = bg_color.to(self.device)
        data["bg_color"] = bg_color
        return data

    def _normalize_err(self, preds: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        err = torch.abs(preds - gt)
        err = err.mean(-1, keepdim=True)  # mean over channels
        # normalize between 0, 1 where 1 corresponds to the 90th percentile
        # err = err.clamp_max(torch.quantile(err, 0.9))
        err = self._normalize_01(err)
        return err.repeat(1, 1, 3)

    @staticmethod
    def _normalize_01(t: torch.Tensor) -> torch.Tensor:
        return (t - t.min()) / t.max()

    def _normalize_depth(self, depth: torch.Tensor, img_h: int, img_w: int) -> torch.Tensor:
        return (
            self._normalize_01(depth)
        ).cpu().reshape(img_h, img_w)[..., None]

    def calc_metrics(self, preds: torch.Tensor, gt: torch.Tensor):
        if gt.shape[-1] == 4:
            gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])

        err = (gt - preds) ** 2
        return {
            "mse": torch.mean(err),
            "psnr": metrics.psnr(preds, gt),
            "ssim": metrics.ssim(preds, gt),
            "ms-ssim": metrics.msssim(preds, gt),
            #"alex_lpips": metrics.rgb_lpips(preds, gt, net_name='alex', device=err.device),
            #"vgg_lpips": metrics.rgb_lpips(preds, gt, net_name='vgg', device=err.device)
        }

    def evaluate_metrics(self,
                         gt: Optional[torch.Tensor],
                         preds: MutableMapping[str, torch.Tensor],
                         dset,
                         img_idx: int,
                         name: Optional[str] = None,
                         save_outputs: bool = True) -> Tuple[dict, np.ndarray, Optional[np.ndarray]]:
        if isinstance(dset.img_h, int):
            img_h, img_w = dset.img_h, dset.img_w
        else:
            img_h, img_w = dset.img_h[img_idx], dset.img_w[img_idx]
        preds_rgb = (
            preds["rgb"]
            .reshape(img_h, img_w, 3)
            .cpu()
            .clamp(0, 1)
        )
        if not torch.isfinite(preds_rgb).all():
            log.warning(f"Predictions have {torch.isnan(preds_rgb).sum()} NaNs, "
                        f"{torch.isinf(preds_rgb).sum()} infs.")
            preds_rgb = torch.nan_to_num(preds_rgb, nan=0.0)
        out_img = preds_rgb
        summary = dict()

        out_depth = None
        if "depth" in preds:
            out_depth = preds["depth"].cpu().reshape(img_h, img_w)[..., None]
        preds.pop("depth")

        # This is used for proposal-depth keys
        for k in preds.keys():
            if "depth" in k:
                prop_depth = preds[k].cpu().reshape(img_h, img_w)[..., None]
                out_depth = torch.cat((out_depth, prop_depth)) if out_depth is not None else prop_depth

        if gt is not None:
            gt = gt.reshape(img_h, img_w, -1).cpu()
            if gt.shape[-1] == 4:
                gt = gt[..., :3] * gt[..., 3:] + (1.0 - gt[..., 3:])
            summary.update(self.calc_metrics(preds_rgb, gt))
            out_img = torch.cat((out_img, gt), dim=0)
            out_img = torch.cat((out_img, self._normalize_err(preds_rgb, gt)), dim=0)

        out_img_np: np.ndarray = (out_img * 255.0).byte().numpy()
        out_depth_np: Optional[np.ndarray] = None
        if out_depth is not None:
            out_depth = self._normalize_01(out_depth)
            out_depth_np = (out_depth * 255.0).repeat(1, 1, 3).byte().numpy()

        if save_outputs:
            out_name = f"step{self.global_step}-{img_idx}"
            if name is not None and name != "":
                out_name += "-" + name
            write_png(os.path.join(self.log_dir, out_name + ".png"), out_img_np)
            if out_depth is not None:
                depth_name = out_name + "-depth"
                write_png(os.path.join(self.log_dir, depth_name + ".png"), out_depth_np)

        return summary, out_img_np, out_depth_np

    @abc.abstractmethod
    def validate(self):
        pass

    def report_test_metrics(self, scene_metrics: Dict[str, Sequence[float]], extra_name: Optional[str]):
        log_text = f"step {self.global_step}/{self.num_steps}"
        if extra_name is not None:
            log_text += f" | {extra_name}"
        scene_metrics_agg: Dict[str, float] = {}
        for k in scene_metrics:
            ak = f"{k}_{extra_name}"
            scene_metrics_agg[ak] = np.mean(np.asarray(scene_metrics[k])).item()
            log_text += f" | {k}: {scene_metrics_agg[ak]:.4f}"
            self.writer.add_scalar(f"test/{ak}", scene_metrics_agg[ak], self.global_step)

        log.info(log_text)
        return scene_metrics_agg

    def get_save_dict(self):
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "global_step": self.global_step
        }

    def save_model(self):
        if self.extra_args['expert_idx']:
            model_fname = os.path.join(self.log_dir,f"{self.extra_args['expert_idx']}.pth")
        
        else:
            model_fname = os.path.join(self.log_dir, f'model.pth')
        log.info(f'Saving model checkpoint to: {model_fname}')
        torch.save(self.get_save_dict(), model_fname)

    def load_model(self, checkpoint_data, training_needed: bool = True):
        self.model.load_state_dict(checkpoint_data["model"], strict=False)
        log.info("=> Loaded model state from checkpoint")

        if training_needed:
            self.optimizer.load_state_dict(checkpoint_data["optimizer"])
            log.info("=> Loaded optimizer state from checkpoint")

        if training_needed and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint_data['lr_scheduler'])
            log.info("=> Loaded scheduler state from checkpoint")

        self.global_step = checkpoint_data["global_step"]
        log.info(f"=> Loaded step {self.global_step} from checkpoints")

    @abc.abstractmethod
    def init_epoch_info(self) -> Dict[str, EMA]:
        pass

    # noinspection PyUnresolvedReferences,PyProtectedMember
    def init_lr_scheduler(self, **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        eta_min = 0
        lr_sched = None
        max_steps = self.num_steps
        scheduler_type = kwargs['scheduler_type']
        log.info(f"Initializing LR Scheduler of type {scheduler_type} with "
                 f"{max_steps} maximum steps.")
        if scheduler_type == "cosine":
            lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max_steps,
                eta_min=eta_min)
        elif scheduler_type == "warmup_cosine":
            lr_sched = get_cosine_schedule_with_warmup(
                self.optimizer, num_warmup_steps=512, num_training_steps=max_steps)
        elif scheduler_type == "step":
            lr_sched = torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33)
        elif scheduler_type == "warmup_step":
            lr_sched = get_step_schedule_with_warmup(
                self.optimizer, milestones=[
                    max_steps // 2,
                    max_steps * 3 // 4,
                    max_steps * 5 // 6,
                    max_steps * 9 // 10,
                ],
                gamma=0.33,
                num_warmup_steps=512)
        return lr_sched

    def init_optim(self, **kwargs) -> torch.optim.Optimizer:
        optim_type = kwargs['optim_type']
        if optim_type == 'adam':
            optim = torch.optim.Adam(params=self.model.get_params(kwargs['lr']), eps=1e-15)
        else:
            raise NotImplementedError()
        return optim

    @abc.abstractmethod
    def init_model(self, **kwargs) -> torch.nn.Module:
        pass

    def get_regularizers(self, **kwargs) -> Sequence[Regularizer]:
        return ()

    def init_regularizers(self, **kwargs):
        # Keep only the regularizers with a positive weight
        regularizers = [r for r in self.get_regularizers(**kwargs) if r.weight > 0]
        return regularizers

    @property
    def lr(self):
        return self.optimizer.param_groups[0]['lr']

    @property
    def calc_metrics_every(self):
        return 1


def losses_to_postfix(loss_dict: Dict[str, EMA], lr: Optional[float]) -> str:
    pfix = [f"{lname}={lval}" for lname, lval in loss_dict.items()]
    if lr is not None:
        pfix.append(f"lr={lr:.2e}")
    return "  ".join(pfix)


def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    np.random.seed(seed)
    random.seed(seed)


def initialize_model(
        runner: Union['StaticTrainer', 'PhototourismTrainer', 'VideoTrainer'],
        **kwargs) -> LowrankModel:
    """Initialize a `LowrankModel` according to the **kwargs parameters.

    Args:
        runner: The runner object which will hold the model.
                Needed here to fetch dataset parameters.
        **kwargs: Extra parameters to pass to the model

    Returns:
        Initialized LowrankModel.
    """
    from .phototourism_trainer import PhototourismTrainer
    extra_args = copy(kwargs)
    # extra_args.pop('global_scale', None)
    # extra_args.pop('global_translation', None)    
    if kwargs['data_dirs'][0].split('/')[-2] == "TanksAndTempleBG":
        global_scale =  torch.as_tensor(extra_args.pop('global_scale', None)).float()
        global_translation = torch.as_tensor(extra_args.pop('global_translation', None))
    else:
         global_translation = None
         global_scale = None
         
    if extra_args['distil']:
        dset = runner.distil_dataset
    else:
        dset = runner.test_dataset

    num_images = None
    if runner.train_dataset is not None:
        try:
            num_images = runner.train_dataset.num_images
        except AttributeError:
            num_images = None
    else:
        try:
            num_images = runner.test_dataset.num_images
        except AttributeError:
            num_images = None
    
    model = LowrankModel(
        grid_config=extra_args.pop("grid_config"),
        aabb=dset.scene_bbox,
        is_ndc=dset.is_ndc,
        is_contracted=dset.is_contracted,
        global_scale=global_scale,
        global_translation=global_translation,
        use_appearance_embedding=isinstance(runner, PhototourismTrainer),
        num_images=num_images,
        **extra_args)
    log.info(f"Initialized {model.__class__} model with "
             f"{sum(np.prod(p.shape) for p in model.parameters()):,} parameters, "
             f"using ndc {model.is_ndc} and contraction {model.is_contracted}. "
             f"Linear decoder: {model.linear_decoder}.")
    return model


def forward_expert(dataset, data, theta, experts, fname=None, pose2com=None):
    '''
    This function performs the following steps:
    1. Determines the expert index for each ray based on pre-defined angles or pose information.
    2. Aggregates rays based on their expert indices.
    3. Uses the corresponding expert model to get predictions for each group of rays.
    
    Parameters:
    dataset (str): The name of the dataset being used ('nerf_synthetic' or 'TanksAndTempleBG').
    data (dict): A dictionary containing ray and scene information.
    theta (ndarray): Array of angles used to determine expert indices for 'nerf_synthetic' dataset.
    experts (list): A list of expert models to make predictions.
    fname (list, optional): List of filenames for frames in 'TanksAndTempleBG' dataset.
    pose2com (dict, optional): A dictionary mapping poses to expert indices for 'TanksAndTempleBG' dataset.
    
    Returns:
    dict: A dictionary containing the aggregated outputs from all experts.
    '''
    
    # Extract rays origins, directions, background color, near-far range, and timestamps from the data
    rays_o = data["rays_o"]
    rays_d = data['rays_d']
    bg_color = data['bg_color']
    near_far = data['near_fars']
    timestamps = data['timestamps']
    
    # Determine expert index for each ray based on the dataset
    if dataset == 'nerf_synthetic':
        # For 'nerf_synthetic', compute expert indices based on 2D projection of camera centers
        camera_center_2d = rays_o[:, :-1].detach().cpu().numpy()    
        expert_idx = camera_center_2d.dot(theta).argmax(axis=1)
    
    elif dataset == 'TanksAndTempleBG':
        # For 'TanksAndTempleBG', compute expert indices based on frame indices and pose2com mapping
        pose_idx = data['frame_idx']
        keys = [fname[idx] for idx in pose_idx]
        expert_idx = torch.tensor([pose2com[k] for k in keys], device=rays_o.device)
    
    # Create masks to separate rays by expert index
    exp_1_mask = expert_idx == 0
    exp_2_mask = expert_idx == 1
    exp_3_mask = expert_idx == 2
    exp_4_mask = expert_idx == 3
    exp_5_mask = expert_idx == 4
    exp_6_mask = expert_idx == 5
    
    # List to store the outputs of each expert
    expert_outputs_list = []
    
    # Disable gradient computation for efficiency
    with torch.no_grad():
        # Get predictions from each expert and store them along with their masks
        if exp_1_mask.sum() > 0:
            outputs_exp1 = experts[0](rays_o[exp_1_mask], rays_d[exp_1_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp1, exp_1_mask])

        if exp_2_mask.sum() > 0:
            outputs_exp2 = experts[1](rays_o[exp_2_mask], rays_d[exp_2_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp2, exp_2_mask])

        if exp_3_mask.sum() > 0:
            outputs_exp3 = experts[2](rays_o[exp_3_mask], rays_d[exp_3_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp3, exp_3_mask])

        if exp_4_mask.sum() > 0:
            outputs_exp4 = experts[3](rays_o[exp_4_mask], rays_d[exp_4_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp4, exp_4_mask])

        if exp_5_mask.sum() > 0:
            outputs_exp5 = experts[4](rays_o[exp_5_mask], rays_d[exp_5_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp5, exp_5_mask])

        if exp_6_mask.sum() > 0:
            outputs_exp6 = experts[5](rays_o[exp_6_mask], rays_d[exp_6_mask], bg_color=bg_color, near_far=near_far, timestamps=timestamps)
            expert_outputs_list.append([outputs_exp6, exp_6_mask])

        batch_size = rays_o.shape[0]

        # Initialize a dictionary to store the combined outputs from all experts
        outputs_experts = {}
        
        # Aggregate outputs from different experts based on the keys in the output dictionary
        for k, v in outputs_exp1.items():  # Use the keys from the first expert's output as reference
            if k in ["rgb", "accumulation", "depth", "rgb_pts", "prop_depth_0", "prop_depth_1"]:
                # Initialize a tensor to store combined outputs for the current key
                value = torch.zeros(batch_size, *v.shape[1:]).to(v.device)
                for item in expert_outputs_list:
                    exp_output, mask = item
                    value[mask] = exp_output[k]
                outputs_experts[k] = value
        
            elif k in ["density_list", "weights_list", "alpha_list"]:
                # Initialize a list to store combined outputs for the current key
                combined_list = []
                for i, val in enumerate(v):
                    value = torch.zeros(batch_size, *val.shape[1:]).to(val.device)
                    for item in expert_outputs_list:
                        exp_output, mask = item
                        value[mask] = exp_output[k][i]
                    combined_list.append(value)
                outputs_experts[k] = combined_list
        
            elif k == "ray_samples_list":
                # Initialize a list to store combined RaySamples for the current key
                combined_list = []
                for i, val in enumerate(v):
                    origins = torch.zeros(batch_size, *v[i].origins.shape[1:]).to(v[i].origins.device)
                    directions = torch.zeros(batch_size, *v[i].directions.shape[1:]).to(v[i].directions.device)
                    starts = torch.zeros(batch_size, *v[i].starts.shape[1:]).to(v[i].starts.device)
                    ends = torch.zeros(batch_size, *v[i].ends.shape[1:]).to(v[i].ends.device)
                    deltas = torch.zeros(batch_size, *v[i].deltas.shape[1:]).to(v[i].deltas.device)
                    spacing_starts = torch.zeros(batch_size, *v[i].spacing_starts.shape[1:]).to(v[i].spacing_starts.device)
                    spacing_ends = torch.zeros(batch_size, *v[i].spacing_ends.shape[1:]).to(v[i].spacing_ends.device)
                    spacing_to_euclidean_fn = v[i].spacing_to_euclidean_fn
                    for item in expert_outputs_list:
                        exp_output, mask = item
                        origins[mask] = exp_output[k][i].origins
                        directions[mask] = exp_output[k][i].directions
                        starts[mask] = exp_output[k][i].starts
                        ends[mask] = exp_output[k][i].ends
                        deltas[mask] = exp_output[k][i].deltas
                        spacing_starts[mask] = exp_output[k][i].spacing_starts
                        spacing_ends[mask] = exp_output[k][i].spacing_ends
                    combined_list.append(RaySamples(
                        origins=origins, directions=directions,
                        starts=starts, ends=ends,
                        deltas=deltas, spacing_starts=spacing_starts,
                        spacing_ends=spacing_ends, spacing_to_euclidean_fn=spacing_to_euclidean_fn
                    ))

                outputs_experts[k] = combined_list

    return outputs_experts


def hook_fn(grad):
    print("max_grad", grad.max())
    print("min_grad", grad.min())
    pass