import math
import os
from collections import defaultdict
from typing import Dict, MutableMapping, Union, Sequence, Any
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import pickle
from plenoxels.datasets import SyntheticNerfDataset, LLFFDataset, SyntheticNerfDataset_Distil
from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.utils.ema import EMA
from plenoxels.utils.my_tqdm import tqdm
from plenoxels.utils.parse_args import parse_optint
from .base_trainer import BaseTrainer, init_dloader_random, initialize_model
from .regularization import (
    PlaneTV, HistogramLoss, L1ProposalNetwork,
    DepthTV, DistortionLoss, DnsHistogramLoss,
)

from plenoxels.models.lowrank_model import LowrankModel

class StaticTrainer_distil(BaseTrainer):
    def __init__(self,
                 tr_loader: torch.utils.data.DataLoader,
                 ts_dset: torch.utils.data.TensorDataset,
                 tr_dset: torch.utils.data.TensorDataset,
                 num_steps: int,
                 logdir: str,
                 expname: str,
                 train_fp16: bool,
                 save_every: int,
                 valid_every: int,
                 save_outputs: bool,
                 device: Union[str, torch.device],
                 **kwargs
                 ):
        self.test_dataset = ts_dset
        self.train_dataset = tr_dset
        self.distil_dataset = kwargs['distil_dset']
        self.is_ndc = self.distil_dataset.is_ndc
        self.is_contracted = self.distil_dataset.is_contracted
        self.device = device
        self.logdir = logdir
        self.num_splits = kwargs['num_splits']
        self.experts = self.load_experts(**kwargs)
        if kwargs['overlap_split']:
            theta = [1.0471975511965979, 3.141592653589793, -1.0471975511965979,
                    0.0,                2.0943951023931953, -2.0943951023931953]
        else:
            if kwargs['num_splits'] == 2:
                theta = [np.deg2rad(deg) for deg in [90, 270]]
            elif kwargs['num_splits'] == 3:
                theta = [1.04719755, 3.14159265, 5.23598776]
            elif kwargs['num_splits'] == 4:
                theta = [np.deg2rad(deg) for deg in [45, 135, 225, 315]]
            elif kwargs['num_splits'] == 5:
                theta = [np.deg2rad(deg) for deg in [36, 108, 180, 252, 324]]

            
        self.theta = np.array([[np.cos(x), np.sin(x)] for x in theta]).T
        
        if kwargs['data_dirs'][0].split('/')[-2] == 'TanksAndTempleBG':
            with open(os.path.join(kwargs['data_dirs'][0], "fnames.pickle"), 'rb') as f:
                self.fname = pickle.load(f)

            with open(os.path.join(kwargs['data_dirs'][0], "pose2com.pickle"), 'rb') as f:
                self.pose2com = pickle.load(f)
            print()

        super().__init__(
            train_data_loader=tr_loader,
            num_steps=num_steps,
            logdir=logdir,
            expname=expname,
            train_fp16=train_fp16,
            save_every=save_every,
            valid_every=valid_every,
            save_outputs=save_outputs,
            device=device,
            **kwargs
        )

    def eval_step(self, data, **kwargs) -> MutableMapping[str, torch.Tensor]:
        """
        Note that here `data` contains a whole image. we need to split it up before tracing
        for memory constraints.
        """
        super().eval_step(data, **kwargs)
        batch_size = self.eval_batch_size
        channels = {"rgb", "depth", "proposal_depth"}
        with torch.cuda.amp.autocast(enabled=self.train_fp16), torch.no_grad():
            rays_o = data["rays_o"]
            rays_d = data["rays_d"]
            # near_far and bg_color are constant over mini-batches
            near_far = data["near_fars"].to(self.device)
            bg_color = data["bg_color"]
            if isinstance(bg_color, torch.Tensor):
                bg_color = bg_color.to(self.device)
            preds = defaultdict(list)
            for b in range(math.ceil(rays_o.shape[0] / batch_size)):
                rays_o_b = rays_o[b * batch_size: (b + 1) * batch_size].to(self.device)
                rays_d_b = rays_d[b * batch_size: (b + 1) * batch_size].to(self.device)
                outputs = self.model(rays_o_b, rays_d_b, near_far=near_far,
                                     bg_color=bg_color)
                for k, v in outputs.items():
                    if k in channels or "depth" in k:
                        preds[k].append(v.cpu())
        return {k: torch.cat(v, 0) for k, v in preds.items()}


    @torch.no_grad()
    def validate(self):
        dataset = self.test_dataset
        per_scene_metrics = defaultdict(list)
        pb = tqdm(total=len(dataset), desc=f"Test scene {dataset.name}")
        for img_idx, data in enumerate(dataset):
            ts_render = self.eval_step(data)
            out_metrics, _, _ = self.evaluate_metrics(
                data["imgs"], ts_render, dset=dataset, img_idx=img_idx,
                name=None, save_outputs=self.save_outputs)
            for k, v in out_metrics.items():
                per_scene_metrics[k].append(v)
            pb.set_postfix_str(f"PSNR={out_metrics['psnr']:.2f}", refresh=False)
            pb.update(1)
        pb.close()
        val_metrics = [
            self.report_test_metrics(per_scene_metrics, extra_name="")
        ]
        df = pd.DataFrame.from_records(val_metrics)
        df.to_csv(os.path.join(self.log_dir, f"test_metrics_step{self.global_step}.csv"))
    
    @staticmethod
    def determin_expert(rays_o, theta):
        x, y, z = rays_o
        x = x.detach().cpu().numpy(); y = y.detach().cpu().numpy();
        exp_idx = np.array((x,y)).dot(theta).argmax() + 1
        return exp_idx

    def train_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        return super().train_step(data, **kwargs)

    def post_step(self, progress_bar):
        super().post_step(progress_bar)

    def pre_epoch(self):
        super().pre_epoch()
        self.train_dataset.reset_iter()

    def pre_epoch_distil(self):
        super().pre_epoch()
        self.distil_dataset.reset_iter()

    def distil_step(self, data: Dict[str, Union[int, torch.Tensor]], **kwargs):
        return super().distil_step(data, **kwargs)
   
    def load_experts(self, **kwargs):
        """
        This function loads the expert models based on the number of splits (num_splits).
        It initializes each expert model, loads their pre-trained weights from checkpoint files,
        and sets them to evaluation mode.

        Parameters:
        kwargs (dict): Additional arguments needed for model initialization and checkpoint loading.

        Returns:
        list: A list of expert models.
        """
    # Check the number of splits and load the corresponding expert models
        if self.num_splits == 2:
            # Initialize two expert models and move them to the specified device
            expert1 = self.init_model(**kwargs).to(self.device)
            expert2 = self.init_model(**kwargs).to(self.device)
            
            # Extract dataset name and construct checkpoint directory path
            dataset = kwargs['data_dirs'][0].split('/')[-1]
            ckpt_dir = os.path.join(self.logdir, dataset)
            
            # Load state dictionaries for each expert model from checkpoint files
            expert1.load_state_dict(torch.load(ckpt_dir + f'_N1_{self.num_splits}_splits/N1.pth')["model"])
            expert2.load_state_dict(torch.load(ckpt_dir + f'_N2_{self.num_splits}_splits/N2.pth')["model"])
            
            print(f"load expert 1 - {self.num_splits}")

            # Set the expert models to evaluation mode
            expert1.eval()
            expert2.eval()
            experts = [expert1, expert2]
        
        elif self.num_splits == 3:
            # Initialize three expert models and move them to the specified device
            expert1 = self.init_model(**kwargs).to(self.device)
            expert2 = self.init_model(**kwargs).to(self.device)
            expert3 = self.init_model(**kwargs).to(self.device)
            
            # Extract dataset name and construct checkpoint directory path
            dataset = kwargs['data_dirs'][0].split('/')[-1]
            ckpt_dir = os.path.join(self.logdir, dataset)
            
            # Load state dictionaries for each expert model from checkpoint files
            expert1.load_state_dict(torch.load(ckpt_dir + f'_N1/N1.pth')["model"])
            expert2.load_state_dict(torch.load(ckpt_dir + f'_N2/N2.pth')["model"])
            expert3.load_state_dict(torch.load(ckpt_dir + f'_N3/N3.pth')["model"])
            
            print(f"load expert 1 - {self.num_splits}")

            # Set the expert models to evaluation mode
            expert1.eval()
            expert2.eval()
            expert3.eval()
            experts = [expert1, expert2, expert3]
        
        elif self.num_splits == 4:
            # Initialize four expert models and move them to the specified device
            expert1 = self.init_model(**kwargs).to(self.device)
            expert2 = self.init_model(**kwargs).to(self.device)
            expert3 = self.init_model(**kwargs).to(self.device)
            expert4 = self.init_model(**kwargs).to(self.device)
            
            # Extract dataset name and construct checkpoint directory path
            dataset = kwargs['data_dirs'][0].split('/')[-1]
            ckpt_dir = os.path.join(self.logdir, dataset)
            
            # Load state dictionaries for each expert model from checkpoint files
            expert1.load_state_dict(torch.load(ckpt_dir + f'_N1_spc/N1.pth')["model"])
            expert2.load_state_dict(torch.load(ckpt_dir + f'_N2_spc/N2.pth')["model"])
            expert3.load_state_dict(torch.load(ckpt_dir + f'_N3_spc/N3.pth')["model"])
            expert4.load_state_dict(torch.load(ckpt_dir + f'_N4_spc/N4.pth')["model"])
            
            print(f"load expert 1 - {self.num_splits}")

            # Set the expert models to evaluation mode
            expert1.eval()
            expert2.eval()
            expert3.eval()
            expert4.eval()
            experts = [expert1, expert2, expert3, expert4]
        
        elif self.num_splits == 5:
            # Initialize five expert models and move them to the specified device
            expert1 = self.init_model(**kwargs).to(self.device)
            expert2 = self.init_model(**kwargs).to(self.device)
            expert3 = self.init_model(**kwargs).to(self.device)
            expert4 = self.init_model(**kwargs).to(self.device)
            expert5 = self.init_model(**kwargs).to(self.device)
            
            # Extract dataset name and construct checkpoint directory path
            dataset = kwargs['data_dirs'][0].split('/')[-1]
            ckpt_dir = os.path.join(self.logdir, dataset)
            
            # Load state dictionaries for each expert model from checkpoint files
            expert1.load_state_dict(torch.load(ckpt_dir + f'_N1_{self.num_splits}_splits/N1.pth')["model"])
            expert2.load_state_dict(torch.load(ckpt_dir + f'_N2_{self.num_splits}_splits/N2.pth')["model"])
            expert3.load_state_dict(torch.load(ckpt_dir + f'_N3_{self.num_splits}_splits/N3.pth')["model"])
            expert4.load_state_dict(torch.load(ckpt_dir + f'_N4_{self.num_splits}_splits/N4.pth')["model"])
            expert5.load_state_dict(torch.load(ckpt_dir + f'_N5_{self.num_splits}_splits/N5.pth')["model"])
            
            print(f"load expert 1 - {self.num_splits}")

            # Set the expert models to evaluation mode
            expert1.eval()
            expert2.eval()
            expert3.eval()
            expert4.eval()
            expert5.eval()
            experts = [expert1, expert2, expert3, expert4, expert5]
        
        return experts



    def get_save_dict(self):
        base_save_dict = super().get_save_dict()
        return base_save_dict

    def load_model(self, checkpoint_data, training_needed: bool = True):
        super().load_model(checkpoint_data, training_needed)

    def init_epoch_info(self):
        ema_weight = 0.9  # higher places higher weight to new observations
        loss_info = defaultdict(lambda: EMA(ema_weight))
        return loss_info

    def init_model(self, **kwargs) -> LowrankModel:
        return initialize_model(self, **kwargs)

    def get_regularizers(self, **kwargs):
        return [
            PlaneTV(kwargs.get('plane_tv_weight', 0.0), what='field'),
            PlaneTV(kwargs.get('plane_tv_weight_proposal_net', 0.0), what='proposal_network'),
            HistogramLoss(kwargs.get('histogram_loss_weight', 0.0)),
            L1ProposalNetwork(kwargs.get('l1_proposal_net_weight', 0.0)),
            DepthTV(kwargs.get('depth_tv_weight', 0.0)),
            DistortionLoss(kwargs.get('distortion_loss_weight', 0.0)),
        ]

    @property
    def calc_metrics_every(self):
        return 5

