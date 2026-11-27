from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dac.data.dataset import frame_camera_positions
from dac.experts import angular_expert_index
from dac.splitting import ASSIGNMENT_FILE, split_views


def _frames(path: Path) -> list[dict]:
    return json.loads(path.read_text())["frames"]


def test_split_views_partitions_every_split(scene: Path) -> None:
    counts = split_views(scene, num_experts=2)
    assert counts == {"train": [4, 4], "test": [2, 2]}
    for split in ("train", "test"):
        full = _frames(scene / f"transforms_{split}.json")
        parts = [_frames(scene / f"transforms_{split}_N{i}.json") for i in (1, 2)]
        assert sorted(f["file_path"] for part in parts for f in part) == sorted(f["file_path"] for f in full)
        for index, part in enumerate(parts):
            xy = frame_camera_positions(part)[:, :2]
            assert (angular_expert_index(xy, 2).numpy() == index).all()
        # Metadata such as the field of view is kept.
        assert json.loads((scene / f"transforms_{split}_N1.json").read_text())["camera_angle_x"] == 0.69


def test_split_views_writes_the_lookup_table(scene: Path) -> None:
    split_views(scene, num_experts=2)
    assignment = json.loads((scene / ASSIGNMENT_FILE).read_text())
    assert assignment == {f"r_{i}": int(i >= 4) for i in range(8)}


def test_split_views_refuses_to_overwrite(scene: Path) -> None:
    split_views(scene, num_experts=2)
    with pytest.raises(FileExistsError, match="--force"):
        split_views(scene, num_experts=2)
    split_views(scene, num_experts=2, force=True)


def test_split_views_rejects_empty_experts(scene: Path) -> None:
    with pytest.raises(ValueError, match="no views"):
        split_views(scene, num_experts=16)
    assert not list(scene.glob("transforms_*_N*.json"))


def test_split_views_supports_a_frames_subdirectory(scene: Path) -> None:
    legacy = scene / "train_json_4"
    legacy.mkdir()
    frames = _frames(scene / "transforms_train.json")
    for frame in frames:
        frame["file_path"] = "../" + frame["file_path"]
    (legacy / "transforms_train.json").write_text(json.dumps({"camera_angle_x": 0.69, "frames": frames}))
    split_views(scene, num_experts=2, frames_dir="{split}_json_4", splits=("train",))
    assert np.array([len(_frames(legacy / f"transforms_train_N{i}.json")) for i in (1, 2)]).sum() == 8
