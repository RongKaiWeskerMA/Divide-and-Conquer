from __future__ import annotations

import copy
import warnings

import pytest
import torch

from dac.config import ModelConfig, from_dict
from dac.models import tcnn_compat
from dac.models.kplanes import KPlanesModel, build_model, remap_legacy_optimizer_state
from tests.conftest import TINY_MODEL

NETWORK_PREFIXES = ("sigma_net", "color_basis", "color_net", "direction_encoder")


def tiny_model_config(**overrides: object) -> ModelConfig:
    raw = {**copy.deepcopy(TINY_MODEL), **overrides}
    return from_dict(ModelConfig, raw)


def make_model(contracted: bool = False, **overrides: object) -> KPlanesModel:
    bbox = torch.tensor([[-1.5] * 3, [1.5] * 3])
    return build_model(tiny_model_config(**overrides), bbox, is_contracted=contracted)


def rays(n: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(0)
    origins = torch.randn(n, 3, generator=gen) * 0.1 + torch.tensor([0.0, 0.0, 4.0])
    directions = torch.nn.functional.normalize(-origins + 0.1 * torch.randn(n, 3, generator=gen), dim=-1)
    return origins, directions


@pytest.mark.parametrize("linear_decoder", [True, False])
@pytest.mark.parametrize("contracted", [False, True])
def test_forward_shapes(linear_decoder: bool, contracted: bool) -> None:
    model = make_model(contracted, linear_decoder=linear_decoder)
    rays_o, rays_d = rays()
    near_far = torch.tensor([[2.0, 6.0]])
    out = model(rays_o, rays_d, bg_color=torch.ones(1, 3), near_far=near_far)
    assert out["rgb"].shape == (32, 3)
    assert out["depth"].shape == (32, 1)
    assert out["rgb_pts"].shape == (32, 8, 3)
    assert [a.shape[1] for a in out["alpha_list"]] == [16, 8, 8]
    assert len(out["ray_samples_list"]) == 3  # training mode keeps the samples
    assert {"prop_depth_0", "prop_depth_1"} <= out.keys()
    assert ((out["rgb"] >= 0) & (out["rgb"] <= 1 + 1e-6)).all()

    model.eval()
    with torch.no_grad():
        out = model(rays_o, rays_d, bg_color=None, near_far=near_far)
        assert "ray_samples_list" not in out
        assert "ray_samples_list" in model(rays_o, rays_d, None, near_far, return_samples=True)


def test_state_dict_layout_matches_original_kplanes() -> None:
    """Checkpoints of the original code must load, so the non-network keys are pinned."""
    state = make_model(contracted=True).state_dict()
    keys = {k for k in state if not set(k.split(".")) & set(NETWORK_PREFIXES)}
    contraction = {"global_translation", "global_scale"}
    expected = {f"spatial_distortion.{p}" for p in contraction}
    expected |= {"field.aabb"} | {f"field.spatial_distortion.{p}" for p in contraction}
    expected |= {f"field.grids.{scale}.{plane}" for scale in range(2) for plane in range(3)}
    for i in range(2):
        expected |= {f"proposal_networks.{i}.aabb"} | {f"proposal_networks.{i}.grids.{p}" for p in range(3)}
        expected |= {f"proposal_networks.{i}.spatial_distortion.{p}" for p in contraction}
    assert keys == expected


def test_shared_proposal_network() -> None:
    model = make_model(use_same_proposal_network=True, proposal_nets=[{"resolution": [8, 8, 8]}])
    assert len(model.proposal_networks) == 1
    assert len(model.density_fns) == 2


def test_param_groups_drop_frozen_and_duplicate_parameters() -> None:
    model = make_model(contracted=True)
    groups, masks = model.param_groups(lr=0.01)
    params = [p for g in groups for p in g["params"]]
    assert all(p.requires_grad for p in params)
    assert len({id(p) for p in params}) == len(params)
    assert len(groups) == len(masks) == 3
    # The original layout has 3 aabbs + 3 copies of the 2 shared contraction constants.
    assert len(masks[2]) == 9
    assert sum(masks[2]) == 0
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == sum(p.numel() for p in params)


def _legacy_groups(model: KPlanesModel) -> list[dict]:
    per_module = [model.field.get_params()] + [pn.get_params() for pn in model._proposal_nets]
    return [{"params": [p for m in per_module for p in m[key]], "lr": 0.01} for key in ("field", "nn", "other")]


def test_legacy_optimizer_state_is_remapped() -> None:
    model = make_model(contracted=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the legacy layout contains duplicate parameters
        legacy = torch.optim.Adam(_legacy_groups(model), eps=1e-15)
    rays_o, rays_d = rays()
    model(rays_o, rays_d, None, torch.tensor([[2.0, 6.0]]))["rgb"].sum().backward()
    legacy.step()

    groups, masks = model.param_groups(lr=0.01)
    optimizer = torch.optim.Adam(groups, eps=1e-15)
    optimizer.load_state_dict(remap_legacy_optimizer_state(legacy.state_dict(), masks))
    assert legacy.state
    for p in optimizer.param_groups[0]["params"] + optimizer.param_groups[1]["params"]:
        assert (p in optimizer.state) == (p in legacy.state)
        if p in legacy.state:
            torch.testing.assert_close(optimizer.state[p]["exp_avg"], legacy.state[p]["exp_avg"])
    # Current-layout states pass through unchanged.
    assert remap_legacy_optimizer_state(optimizer.state_dict(), masks) == optimizer.state_dict()


def test_fallback_spherical_harmonics_values() -> None:
    sh = tcnn_compat.TorchSphericalHarmonics(3, {"otype": "SphericalHarmonics", "degree": 4})
    up = sh(torch.tensor([[0.5, 0.5, 1.0]]))[0]  # direction (0, 0, 1) mapped to [0, 1]
    expected = torch.zeros(16)
    expected[0], expected[2], expected[6], expected[12] = 0.28209479, 0.48860251, 0.63078313, 0.74635267
    torch.testing.assert_close(up, expected)


def test_fallback_mlp_architecture() -> None:
    mlp = tcnn_compat.TorchMLP(
        3, 2, {"n_neurons": 16, "n_hidden_layers": 2, "activation": "ReLU", "output_activation": "Sigmoid"}
    )
    linears = [m for m in mlp.modules() if isinstance(m, torch.nn.Linear)]
    assert [(m.in_features, m.out_features) for m in linears] == [(3, 16), (16, 16), (16, 2)]
    assert all(m.bias is None for m in linears)
    out = mlp(torch.randn(5, 3))
    assert out.shape == (5, 2)
    assert ((out > 0) & (out < 1)).all()
