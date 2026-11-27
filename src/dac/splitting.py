"""Split a scene's views into expert groups (the "divide" step of DaC).

Views are grouped by the azimuth of their camera centre around the z-axis, the same
rule :class:`dac.experts.AngularRouter` uses to route rays during distillation. For each
split this writes ``transforms_{split}_N{i}.json`` next to ``transforms_{split}.json``,
and records the training-view assignment in ``expert_assignment.json`` (used by
:class:`dac.experts.LookupRouter`).
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from dac.data.dataset import expert_name, frame_camera_positions, transforms_path
from dac.experts import angular_expert_index

log = logging.getLogger(__name__)

ASSIGNMENT_FILE = "expert_assignment.json"


def split_views(
    data_root: str | Path,
    num_experts: int,
    frames_dir: str = ".",
    splits: Sequence[str] = ("train", "test"),
    force: bool = False,
) -> dict[str, list[int]]:
    """Write per-expert transforms files for ``splits`` of the scene at ``data_root``.

    Returns:
        The number of views given to each expert, per split.
    """
    root = str(data_root)
    outputs: list[tuple[str, dict]] = []
    counts: dict[str, list[int]] = {}
    assignment: dict[str, int] = {}
    for split in splits:
        src = transforms_path(root, split, frames_dir)
        with open(src) as f:
            meta = json.load(f)
        frames = meta["frames"]
        if not frames:
            raise ValueError(f"{src} has no frames")
        experts = angular_expert_index(frame_camera_positions(frames)[:, :2], num_experts).tolist()
        counts[split] = [experts.count(i) for i in range(num_experts)]
        empty = [i + 1 for i, n in enumerate(counts[split]) if n == 0]
        if empty:
            raise ValueError(f"{split}: experts {empty} would get no views; use fewer experts")
        for i in range(num_experts):
            subset = {**meta, "frames": [frame for frame, e in zip(frames, experts) if e == i]}
            outputs.append((transforms_path(root, split, frames_dir, expert=i + 1), subset))
        if split == "train":
            assignment = {os.path.basename(frame["file_path"]): e for frame, e in zip(frames, experts)}
            if len(assignment) != len(frames):
                raise ValueError(f"{src}: training frame file names are not unique")

    targets = [path for path, _ in outputs]
    if assignment:
        targets.append(os.path.join(root, ASSIGNMENT_FILE))
    existing = [p for p in targets if os.path.exists(p)]
    if existing and not force:
        raise FileExistsError("refusing to overwrite (pass --force):\n  " + "\n  ".join(existing))

    for path, subset in outputs:
        with open(path, "w") as f:
            json.dump(subset, f, indent=2)
    if assignment:
        with open(os.path.join(root, ASSIGNMENT_FILE), "w") as f:
            json.dump(assignment, f, indent=2)
    for split, per_expert in counts.items():
        summary = ", ".join(f"{expert_name(i + 1)}={n}" for i, n in enumerate(per_expert))
        log.info("%s views per expert: %s", split, summary)
    return counts
