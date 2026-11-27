"""End-to-end runs of the CLI on a tiny scene (CPU, PyTorch fallback)."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from dac.cli import main


def run(*args: str) -> None:
    assert main(list(args)) == 0


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path) as f:
        return list(csv.DictReader(f))


def test_full_dac_pipeline(scene: Path, config_file: Path, tmp_path: Path) -> None:
    logs = tmp_path / "logs"
    cfg = str(config_file)

    run("split", "--data-root", str(scene), "--num-experts", "2")
    for i in (1, 2):
        run("train", "-c", cfg, f"experiment.name=toy_N{i}", f"dac.expert={i}")
    expert_ckpt = torch.load(logs / "toy_N1" / "model.pth", weights_only=False)
    assert expert_ckpt["global_step"] == 3
    assert expert_ckpt["config"]["dac"]["expert"] == 1
    assert len(read_csv(logs / "toy_N1" / "test_metrics_step3_per_view.csv")) == 2  # expert test views only

    run("distill", "-c", cfg, "experiment.name=toy_distill")
    distilled = logs / "toy_distill" / "model.pth"
    assert distilled.is_file()

    run("train", "-c", cfg, "experiment.name=toy_finetune", "--checkpoint", str(distilled), "optim.num_steps=5")
    finetuned = logs / "toy_finetune"
    assert torch.load(finetuned / "model.pth", weights_only=False)["global_step"] == 5
    (row,) = read_csv(finetuned / "test_metrics_step5.csv")
    assert float(row["psnr"]) > 0
    assert (finetuned / "step5-0.png").is_file()
    assert (finetuned / "step5-0-depth.png").is_file()
    saved = yaml.safe_load((finetuned / "config.yaml").read_text())
    assert saved["experiment"]["name"] == "toy_finetune"

    # eval/render default to the config saved next to the checkpoint.
    run("eval", "--checkpoint", str(finetuned / "model.pth"), "--output-dir", str(tmp_path / "eval"))
    assert len(read_csv(tmp_path / "eval" / "test_metrics_step5_per_view.csv")) == 4
    run("render", "--checkpoint", str(finetuned / "model.pth"), "--output-dir", str(tmp_path / "render"))
    assert (tmp_path / "render" / "rendering_path.mp4").stat().st_size > 0


def test_periodic_validation_and_checkpoints(config_file: Path, tmp_path: Path) -> None:
    run("train", "-c", str(config_file), "optim.num_steps=4", "logging.valid_every=2", "logging.save_every=2")
    run_dir = tmp_path / "logs" / "toy"
    assert sorted(p.name for p in run_dir.glob("test_metrics_step?.csv")) == [
        "test_metrics_step2.csv",
        "test_metrics_step4.csv",
    ]


def test_distill_without_experts_fails_cleanly(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["distill", "-c", str(config_file)]) == 2
    assert "missing expert checkpoints" in capsys.readouterr().out


def test_lookup_router_pipeline(scene: Path, config_file: Path) -> None:
    cfg = str(config_file)
    run("split", "--data-root", str(scene), "--num-experts", "2")
    for i in (1, 2):
        run("train", "-c", cfg, f"experiment.name=toy_N{i}", f"dac.expert={i}", "optim.num_steps=1")
    run("distill", "-c", cfg, "experiment.name=toy_distill", "dac.router=lookup", "optim.num_steps=2")


def test_bad_override_exit_code(config_file: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["train", "-c", str(config_file), "optim.lrr=1"]) == 2
    assert "did you mean 'optim.lr'" in capsys.readouterr().out


def test_unknown_arguments_are_rejected(config_file: Path) -> None:
    with pytest.raises(SystemExit):
        main(["train", "-c", str(config_file), "--checkpont", "x.pth"])
    with pytest.raises(SystemExit):
        main(["split", "--data-root", "x", "--num-experts", "2", "a=1"])
