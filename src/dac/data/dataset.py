"""Blender-format (``transforms_{split}.json``) datasets for NeRF-Synthetic and Tanks and Temples."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from dac.config import DataConfig
from dac.data.intrinsics import Intrinsics, intrinsics_from_transforms
from dac.data.loading import load_frames
from dac.data.rays import generate_hemispherical_orbit, get_ray_directions, get_rays

log = logging.getLogger(__name__)

SPLITS = ("train", "test", "render")


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset scene properties."""

    near_far: tuple[float, float]
    scene_radius: float
    # Unbounded scenes are contracted into a cube of side 4 before grid lookups.
    contracted: bool
    # Composite training pixels over a random colour (True) or over white (False).
    random_train_background: bool
    scene_radius_overrides: tuple[tuple[str, float], ...] = ()

    def radius_for(self, scene: str) -> float:
        return dict(self.scene_radius_overrides).get(scene, self.scene_radius)


DATASET_SPECS: dict[str, DatasetSpec] = {
    "nerf_synthetic": DatasetSpec(
        near_far=(2.0, 6.0),
        scene_radius=1.3,
        contracted=False,
        random_train_background=True,
        scene_radius_overrides=(("ship", 1.5),),
    ),
    "tanks_temples": DatasetSpec(
        near_far=(0.0, 6.0),
        scene_radius=2.0,
        contracted=True,
        random_train_background=False,
    ),
}


def expert_name(index: int) -> str:
    """Suffix of the per-expert transforms files, e.g. ``transforms_train_N1.json``."""
    return f"N{index}"


def transforms_path(root: str, split: str, frames_dir: str = ".", expert: int | None = None) -> str:
    suffix = f"_{expert_name(expert)}" if expert is not None else ""
    return os.path.normpath(os.path.join(root, frames_dir.format(split=split), f"transforms_{split}{suffix}.json"))


def read_transforms(path: str, max_frames: int | None = None) -> dict[str, Any]:
    """Load a transforms file, keeping roughly ``max_frames`` evenly spaced frames if given."""
    with open(path) as f:
        meta = json.load(f)
    total = len(meta["frames"])
    if max_frames is not None and max_frames < total:
        step = round(total / max_frames)
        log.info("Subsampling %s to 1 in every %d frames", path, step)
        meta["frames"] = meta["frames"][::step]
    return meta


class NerfDataset(Dataset):
    """Rays and colours of one scene split.

    * ``train``: rays of all frames are merged and each item is a random batch of rays.
    * ``test``: each item holds every ray of one image.
    * ``render``: each item holds every ray of one camera on an orbit around the scene
      (no ground truth).

    Every item is a dict with ``rays_o``, ``rays_d``, ``imgs`` (or ``None``), ``bg_color``
    and ``near_fars``; training items also carry the ``frame_idx`` of every ray.
    """

    def __init__(
        self,
        dataset: str,
        root: str,
        split: str,
        *,
        batch_size: int | None = None,
        downsample: float = 1.0,
        max_frames: int | None = None,
        frames_dir: str = ".",
        expert: int | None = None,
        expert_frames_dir: str = ".",
    ) -> None:
        if dataset not in DATASET_SPECS:
            raise ValueError(f"unknown dataset {dataset!r}; expected one of {list(DATASET_SPECS)}")
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {list(SPLITS)}")
        if split == "train" and batch_size is None:
            raise ValueError("batch_size is required for the train split")
        self.spec = DATASET_SPECS[dataset]
        self.root = root
        self.name = os.path.basename(os.path.normpath(root))
        self.split = split
        self.batch_size = batch_size
        self.expert = expert

        source_split = "test" if split == "render" else split
        if expert is not None and split != "render":
            path = transforms_path(root, source_split, expert_frames_dir, expert)
        else:
            path = transforms_path(root, source_split, frames_dir)
        meta = read_transforms(path, max_frames)
        frames = meta["frames"]
        images, poses = load_frames(frames, os.path.dirname(path), downsample, desc=f"Loading {split} images")
        self.intrinsics: Intrinsics = intrinsics_from_transforms(
            meta, img_h=images.shape[1], img_w=images.shape[2], downsample=downsample
        )
        self.frame_names = [os.path.basename(frame["file_path"]) for frame in frames]
        pixels: torch.Tensor | None = images
        if split == "render":
            poses = generate_hemispherical_orbit(poses, n_frames=120)
            pixels = None
            self.frame_names = []
        self.num_frames = poses.shape[0]

        self.rays_o, self.rays_d, self.imgs = _create_rays(
            pixels, poses, self.intrinsics, merge_frames=split == "train"
        )
        radius = self.spec.radius_for(self.name)
        self.scene_bbox = torch.tensor([[-radius] * 3, [radius] * 3])
        self.is_contracted = self.spec.contracted
        self.near_far = torch.tensor([self.spec.near_far], dtype=torch.float32)

        self.num_samples = self.rays_o.shape[0]
        self.use_permutation = self.num_samples < 100_000_000
        self.perm: torch.Tensor | None = None
        if split == "train":
            self.reset_iter()
        log.info(
            "Loaded %s split of %s from %s: %d frames of %dx%d",
            split,
            self.name,
            path,
            self.num_frames,
            self.img_h,
            self.img_w,
        )

    @property
    def img_h(self) -> int:
        return self.intrinsics.height

    @property
    def img_w(self) -> int:
        return self.intrinsics.width

    def reset_iter(self) -> None:
        """Draw a fresh ray permutation for the next epoch."""
        self.perm = torch.randperm(self.num_samples) if self.use_permutation else None

    def __len__(self) -> int:
        if self.split == "train":
            assert self.batch_size is not None
            return (self.num_samples + self.batch_size - 1) // self.batch_size
        return self.num_samples

    def _ray_indices(self, index: int) -> torch.Tensor:
        assert self.batch_size is not None
        if self.perm is not None:
            return self.perm[index * self.batch_size : (index + 1) * self.batch_size]
        return torch.randint(0, self.num_samples, size=(self.batch_size,))

    def __getitem__(self, index: int) -> dict[str, Any]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        out: dict[str, Any] = {}
        ids: Any = index
        if self.split == "train":
            ids = self._ray_indices(index)
            out["frame_idx"] = ids // (self.img_h * self.img_w)
        out["rays_o"] = self.rays_o[ids]
        out["rays_d"] = self.rays_d[ids]
        pixels = self.imgs[ids] if self.imgs is not None else None

        if self.split == "train" and self.spec.random_train_background:
            assert pixels is not None
            bg_color = torch.rand((1, 3), dtype=pixels.dtype, device=pixels.device)
        else:
            bg_color = torch.ones((1, 3), dtype=torch.float32)
        if pixels is not None and pixels.shape[-1] == 4:  # alpha compositing
            pixels = pixels[:, :3] * pixels[:, 3:] + bg_color * (1.0 - pixels[:, 3:])
        out["imgs"] = pixels
        out["bg_color"] = bg_color
        out["near_fars"] = self.near_far.clone()
        return out


def _create_rays(
    images: torch.Tensor | None, poses: torch.Tensor, intrinsics: Intrinsics, merge_frames: bool
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Rays (and pixels) as ``[N * H * W, C]`` if ``merge_frames`` else ``[N, H * W, C]``."""
    directions = get_ray_directions(intrinsics, opengl_camera=True)
    rays = [get_rays(directions, pose) for pose in poses]
    rays_o = torch.cat([r[0] for r in rays], dim=0).to(torch.float32)
    rays_d = torch.cat([r[1] for r in rays], dim=0).to(torch.float32)
    pixels = images.reshape(-1, images.shape[-1]).to(torch.float32) if images is not None else None
    if not merge_frames:
        n, hw = poses.shape[0], intrinsics.height * intrinsics.width
        rays_o, rays_d = rays_o.view(n, hw, 3), rays_d.view(n, hw, 3)
        pixels = pixels.view(n, hw, -1) if pixels is not None else None
    return rays_o, rays_d, pixels


def make_dataset(
    cfg: DataConfig, split: str, *, batch_size: int | None = None, expert: int | None = None
) -> NerfDataset:
    """Build the dataset for ``split`` from a :class:`~dac.config.DataConfig`."""
    max_frames = cfg.max_train_frames if split == "train" else cfg.max_test_frames
    return NerfDataset(
        cfg.dataset,
        cfg.root,
        split,
        batch_size=batch_size,
        downsample=cfg.downsample,
        max_frames=max_frames,
        frames_dir=cfg.frames_dir,
        expert=expert,
        expert_frames_dir=cfg.expert_frames_dir,
    )


def frame_camera_positions(frames: list[dict[str, Any]]) -> np.ndarray:
    """``[N, 3]`` camera centres of transforms-file frames."""
    return np.array([np.asarray(frame["transform_matrix"], dtype=np.float64)[:3, 3] for frame in frames])
