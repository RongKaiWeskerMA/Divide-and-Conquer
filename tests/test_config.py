from __future__ import annotations

import re
from pathlib import Path

import pytest

from dac.config import Config, ConfigError, load_config, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("path", sorted(CONFIG_DIR.glob("*.yaml")), ids=lambda p: p.name)
def test_shipped_configs_load(path: Path) -> None:
    cfg = load_config(path)
    assert cfg.optim.num_steps > 0
    assert cfg.model.num_proposal_iterations == len(cfg.model.num_proposal_samples)


def test_shipped_configs_match_the_paper_settings() -> None:
    synthetic = load_config(CONFIG_DIR / "nerf_synthetic.yaml")
    assert synthetic.dac.router == "angular"
    assert synthetic.model.linear_decoder_layers == 4
    assert synthetic.model.multiscale_res == [1, 2, 4]
    tnt = load_config(CONFIG_DIR / "tanks_temples.yaml")
    assert tnt.dac.router == "lookup"
    assert tnt.model.multiscale_res == [1, 2, 4, 8]
    assert tnt.regularization.plane_tv == pytest.approx(2e-4)


def test_overrides_are_parsed_as_yaml_and_coerced() -> None:
    cfg = load_config(
        CONFIG_DIR / "nerf_synthetic.yaml",
        [
            "optim.lr=1e-3",  # PyYAML reads this as a string
            "optim.num_steps=100",
            "model.multiscale_res=[1, 2]",
            "dac.expert=2",
            "experiment.name=123",
            "data.max_train_frames=null",
            "model.proposal_nets=[{resolution: [32, 32, 32]}]",
            "model.use_same_proposal_network=true",
        ],
    )
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert cfg.optim.num_steps == 100
    assert cfg.model.multiscale_res == [1, 2]
    assert cfg.dac.expert == 2
    assert cfg.experiment.name == "123"
    assert cfg.data.max_train_frames is None
    assert cfg.model.proposal_nets[0].resolution == [32, 32, 32]
    assert cfg.model.proposal_nets[0].num_output_coords == 8  # default kept


def test_override_can_add_fields_missing_from_the_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("")
    cfg = load_config(path, ["dac.loss_weights.rgb=0.5"])
    assert cfg.dac.loss_weights.rgb == 0.5
    assert cfg == Config(dac=cfg.dac)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ("optim.lrr=0.1", "did you mean 'optim.lr'"),
        ("optim.num_steps=abc", "optim.num_steps: expected int"),
        ("optim.fp16=1", "optim.fp16: expected bool"),
        ("model.multiscale_res=4", "expected a list"),
        ("optim", "not of the form"),
        ("optim.lr.value=1", "is not a section"),
        ("dac.expert=5", "dac.expert must be in [1, 4]"),
        ("dac.router=nearest", "dac.router must be one of"),
        ("model.num_proposal_samples=[64]", "one entry per proposal iteration"),
    ],
)
def test_invalid_overrides_are_rejected(override: str, message: str) -> None:
    with pytest.raises(ConfigError, match=re.escape(message)):
        load_config(CONFIG_DIR / "nerf_synthetic.yaml", [override])


def test_save_and_reload_roundtrip(tmp_path: Path) -> None:
    cfg = load_config(CONFIG_DIR / "tanks_temples.yaml", ["dac.expert=3"])
    save_config(cfg, tmp_path / "config.yaml")
    assert load_config(tmp_path / "config.yaml") == cfg


def test_derived_paths() -> None:
    cfg = load_config(CONFIG_DIR / "nerf_synthetic.yaml", ["data.root=data/nerf_synthetic/ship/"])
    assert cfg.data.scene == "ship"
    assert cfg.experiment.run_dir == Path("logs/nerf_synthetic/lego")
