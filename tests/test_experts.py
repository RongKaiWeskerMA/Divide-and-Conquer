from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from dac.config import Config
from dac.experts import (
    AngularRouter,
    LookupRouter,
    angular_centers,
    angular_expert_index,
    expert_checkpoint_path,
    load_experts,
    query_experts,
)
from dac.rendering.ray_samplers import RaySamples

# Sector centres hard-coded in the original implementation.
ORIGINAL_THETAS = {
    2: np.deg2rad([90, 270]),
    3: [1.04719755, 3.14159265, 5.23598776],
    4: np.deg2rad([45, 135, 225, 315]),
    5: np.deg2rad([36, 108, 180, 252, 324]),
}


@pytest.mark.parametrize("num_experts", sorted(ORIGINAL_THETAS))
def test_angular_centers_match_original(num_experts: int) -> None:
    np.testing.assert_allclose(angular_centers(num_experts), ORIGINAL_THETAS[num_experts], atol=1e-8)


@pytest.mark.parametrize("num_experts", sorted(ORIGINAL_THETAS))
def test_angular_assignment_matches_original_formula(num_experts: int) -> None:
    xy = torch.randn(1000, 2, generator=torch.Generator().manual_seed(num_experts))
    theta = np.array([[np.cos(t), np.sin(t)] for t in ORIGINAL_THETAS[num_experts]]).T
    expected = xy.numpy().dot(theta).argmax(axis=1)
    np.testing.assert_array_equal(angular_expert_index(xy, num_experts).numpy(), expected)


def test_angular_router_uses_ray_origins() -> None:
    centers = angular_centers(4)
    rays_o = torch.tensor([[np.cos(t), np.sin(t), 0.5] for t in centers], dtype=torch.float32)
    assert AngularRouter(4)({"rays_o": rays_o}).tolist() == [0, 1, 2, 3]


@pytest.mark.parametrize("suffix", [".json", ".pickle"])
def test_lookup_router_from_file(tmp_path: Path, suffix: str) -> None:
    mapping = {"a.png": 1, "b.png": 0, "c.png": 1}
    path = tmp_path / f"assignment{suffix}"
    if suffix == ".json":
        path.write_text(json.dumps(mapping))
    else:
        path.write_bytes(pickle.dumps(mapping))
    router = LookupRouter.from_file(path, ["c.png", "a.png", "b.png"], num_experts=2)
    assert router({"frame_idx": torch.tensor([2, 0, 1, 0])}).tolist() == [0, 1, 1, 1]


def test_lookup_router_validates_the_table(tmp_path: Path) -> None:
    path = tmp_path / "assignment.json"
    path.write_text(json.dumps({"a.png": 0, "b.png": 3}))
    with pytest.raises(ValueError, match="no expert for 1 training frames"):
        LookupRouter.from_file(path, ["a.png", "z.png"], num_experts=4)
    with pytest.raises(ValueError, match="outside"):
        LookupRouter.from_file(path, ["a.png", "b.png"], num_experts=2)


class FakeExpert:
    """Deterministic stand-in for a model: outputs depend on the rays and the expert id."""

    def __init__(self, offset: float) -> None:
        self.offset = offset
        self.calls = 0

    def __call__(self, rays_o: torch.Tensor, rays_d: torch.Tensor, **_: Any) -> dict[str, Any]:
        self.calls += 1
        n = rays_o.shape[0]
        value = rays_o[:, :1] + self.offset
        samples = RaySamples(
            origins=rays_o[:, None, :],
            directions=rays_d[:, None, :],
            starts=value[:, None].expand(n, 3, 1),
            ends=value[:, None].expand(n, 3, 1) + 1,
            deltas=torch.ones(n, 3, 1),
            spacing_starts=torch.zeros(n, 3, 1),
            spacing_ends=torch.ones(n, 3, 1),
            spacing_to_euclidean_fn=lambda x: x,
        )
        return {"rgb": value.expand(n, 3), "alpha_list": [value, 2 * value], "ray_samples_list": [samples]}


def test_query_experts_merges_outputs_in_ray_order() -> None:
    experts = [FakeExpert(0.0), FakeExpert(10.0), FakeExpert(20.0)]
    rays_o = torch.arange(5, dtype=torch.float32)[:, None].repeat(1, 3)
    rays_d = torch.ones(5, 3)
    # Expert 0 gets no rays: this used to crash the original implementation.
    assignment = torch.tensor([1, 2, 1, 2, 2])
    out = query_experts(experts, assignment, rays_o, rays_d, bg_color=torch.ones(1, 3), near_far=torch.ones(1, 2))

    expected = rays_o[:, :1] + torch.tensor([10.0, 20.0, 10.0, 20.0, 20.0])[:, None]
    assert experts[0].calls == 0
    torch.testing.assert_close(out["rgb"], expected.expand(5, 3))
    torch.testing.assert_close(out["alpha_list"][1], 2 * expected)
    samples = out["ray_samples_list"][0]
    assert isinstance(samples, RaySamples)
    torch.testing.assert_close(samples.starts[:, 0], expected)
    torch.testing.assert_close(samples.origins[:, 0], rays_o)
    assert samples.spacing_to_euclidean_fn is None


def test_query_experts_requires_assigned_rays() -> None:
    with pytest.raises(ValueError, match="no rays"):
        query_experts([FakeExpert(0.0)], torch.tensor([3]), torch.zeros(1, 3), torch.zeros(1, 3), None, None)


def test_load_experts_reports_every_missing_checkpoint(tmp_path: Path) -> None:
    cfg = Config()
    cfg.experiment.log_dir = str(tmp_path)
    cfg.data.root = "data/nerf_synthetic/lego"
    cfg.dac.num_experts = 2
    assert expert_checkpoint_path(cfg, 2) == tmp_path / "lego_N2" / "model.pth"
    with pytest.raises(FileNotFoundError) as excinfo:
        load_experts(cfg, torch.ones(2, 3), False, torch.device("cpu"))
    assert "lego_N1" in str(excinfo.value)
    assert "lego_N2" in str(excinfo.value)


def test_expert_checkpoint_template_supports_legacy_layout() -> None:
    cfg = Config()
    cfg.experiment.log_dir = "logs/TanksandTemple"
    cfg.data.root = "data/TanksAndTempleBG/Playground"
    cfg.dac.expert_checkpoint = "{log_dir}/{scene}_N{index}_spc/N{index}.pth"
    assert expert_checkpoint_path(cfg, 3) == Path("logs/TanksandTemple/Playground_N3_spc/N3.pth")
