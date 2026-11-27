from __future__ import annotations

from pathlib import Path

import pytest
import torch

from dac.data.dataset import NerfDataset
from dac.splitting import split_views
from tests.conftest import IMAGE_SIZE, ring_positions


def test_train_split_yields_random_ray_batches(scene: Path) -> None:
    dataset = NerfDataset("nerf_synthetic", str(scene), "train", batch_size=100)
    num_rays = 8 * IMAGE_SIZE * IMAGE_SIZE
    assert dataset.num_samples == num_rays
    assert len(dataset) == -(-num_rays // 100)
    batch = dataset[0]
    assert batch["rays_o"].shape == (100, 3)
    assert batch["imgs"].shape == (100, 3)
    assert batch["near_fars"].tolist() == [[2.0, 6.0]]
    assert batch["imgs"].min() >= 0
    assert batch["imgs"].max() <= 1
    torch.testing.assert_close(batch["rays_d"].norm(dim=-1), torch.ones(100))
    # Every ray starts at the camera centre of the frame it came from.
    centres = torch.tensor(ring_positions(8), dtype=torch.float32)
    torch.testing.assert_close(batch["rays_o"], centres[batch["frame_idx"]])
    # The last batch is smaller and every ray is visited exactly once per epoch.
    seen = torch.cat([dataset._ray_indices(i) for i in range(len(dataset))])
    assert sorted(seen.tolist()) == list(range(num_rays))


def test_test_split_yields_whole_images_over_white(scene: Path) -> None:
    dataset = NerfDataset("nerf_synthetic", str(scene), "test")
    assert len(dataset) == 4
    item = dataset[1]
    assert item["rays_o"].shape == (IMAGE_SIZE * IMAGE_SIZE, 3)
    assert item["imgs"].shape == (IMAGE_SIZE * IMAGE_SIZE, 3)
    assert item["bg_color"].tolist() == [[1.0, 1.0, 1.0]]
    assert "frame_idx" not in item
    with pytest.raises(IndexError):
        dataset[4]


def test_render_split_orbits_without_images(scene: Path) -> None:
    dataset = NerfDataset("nerf_synthetic", str(scene), "render")
    assert len(dataset) == 120
    assert dataset.imgs is None
    assert dataset[0]["imgs"] is None


def test_expert_subsets_and_frame_subsampling(scene: Path) -> None:
    split_views(scene, num_experts=2)
    expert = NerfDataset("nerf_synthetic", str(scene), "train", batch_size=10, expert=2)
    assert expert.frame_names == ["r_4", "r_5", "r_6", "r_7"]
    subsampled = NerfDataset("nerf_synthetic", str(scene), "test", max_frames=2)
    assert subsampled.frame_names == ["r_0", "r_2"]


def test_dataset_specs(scene: Path) -> None:
    synthetic = NerfDataset("nerf_synthetic", str(scene), "test")
    assert not synthetic.is_contracted
    assert synthetic.scene_bbox[1].tolist() == pytest.approx([1.3] * 3)
    tnt = NerfDataset("tanks_temples", str(scene), "train", batch_size=10)
    assert tnt.is_contracted
    assert tnt.scene_bbox[1].tolist() == [2.0] * 3
    assert tnt[0]["bg_color"].tolist() == [[1.0, 1.0, 1.0]]  # no random background
    assert tnt[0]["near_fars"].tolist() == [[0.0, 6.0]]


def test_missing_images_are_an_error(scene: Path) -> None:
    (scene / "test" / "r_2.png").unlink()
    with pytest.raises(FileNotFoundError, match=r"r_2\.png"):
        NerfDataset("nerf_synthetic", str(scene), "test")


def test_invalid_arguments(scene: Path) -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        NerfDataset("llff", str(scene), "test")
    with pytest.raises(ValueError, match="batch_size"):
        NerfDataset("nerf_synthetic", str(scene), "train")
