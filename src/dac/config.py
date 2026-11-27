"""Typed experiment configuration.

A config is a YAML file whose sections mirror the dataclasses below. Any field can be
overridden from the command line with dotted ``section.field=value`` pairs, where
``value`` is parsed as YAML, e.g. ``optim.lr=0.001``, ``model.multiscale_res=[1,2]`` or
``dac.expert=null``.
"""

import dataclasses
import difflib
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, TypeVar, Union

import yaml

T = TypeVar("T")

DATASETS = ("nerf_synthetic", "tanks_temples")
ROUTERS = ("angular", "lookup")
SCHEDULERS = ("cosine", "warmup_cosine", "step", "warmup_step")
DENSITY_ACTIVATIONS = ("trunc_exp", "relu")


class ConfigError(ValueError):
    """Raised for malformed config files or command-line overrides."""


@dataclass
class ExperimentConfig:
    name: str = "default"
    log_dir: str = "logs"
    device: str = "cuda:0"
    seed: int = 0

    @property
    def run_dir(self) -> Path:
        return Path(self.log_dir) / self.name


@dataclass
class DataConfig:
    dataset: str = "nerf_synthetic"
    root: str = "data/nerf_synthetic/lego"
    downsample: float = 1.0
    max_train_frames: Optional[int] = None
    max_test_frames: Optional[int] = None
    # Directory (relative to ``root``) holding ``transforms_{split}.json``. ``{split}`` is
    # substituted, so the original DaC layout is ``"{split}_json_4"``.
    frames_dir: str = "."
    # Directory (relative to ``root``) holding the per-expert ``transforms_{split}_N{i}.json``.
    expert_frames_dir: str = "."
    num_workers: int = 4

    @property
    def scene(self) -> str:
        return Path(self.root).name


@dataclass
class GridConfig:
    grid_dimensions: int = 2
    input_coordinate_dim: int = 3
    output_coordinate_dim: int = 32
    resolution: list[int] = field(default_factory=lambda: [128, 128, 128])


@dataclass
class ProposalNetConfig:
    num_input_coords: int = 3
    num_output_coords: int = 8
    resolution: list[int] = field(default_factory=lambda: [64, 64, 64])


@dataclass
class ModelConfig:
    grid: GridConfig = field(default_factory=GridConfig)
    multiscale_res: list[int] = field(default_factory=lambda: [1, 2, 4])
    density_activation: str = "trunc_exp"
    concat_features_across_scales: bool = True
    # True: K-Planes "explicit" model (learned colour basis); False: "hybrid" (MLP decoder).
    linear_decoder: bool = True
    linear_decoder_layers: int = 1
    # Scene contraction (only used for unbounded scenes).
    global_translation: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    global_scale: list[float] = field(default_factory=lambda: [1.0, 1.0, 1.0])
    # Ray marching and proposal sampling.
    num_samples: int = 48
    num_proposal_samples: list[int] = field(default_factory=lambda: [256, 128])
    num_proposal_iterations: int = 2
    use_same_proposal_network: bool = False
    proposal_nets: list[ProposalNetConfig] = field(
        default_factory=lambda: [
            ProposalNetConfig(resolution=[64, 64, 64]),
            ProposalNetConfig(resolution=[128, 128, 128]),
        ]
    )
    single_jitter: bool = False
    proposal_warmup: int = 5000
    proposal_update_every: int = 5
    use_proposal_weight_anneal: bool = True
    proposal_weights_anneal_max_num_iters: int = 1000
    proposal_weights_anneal_slope: float = 10.0


@dataclass
class OptimConfig:
    num_steps: int = 30000
    batch_size: int = 4096
    lr: float = 0.01
    scheduler: Optional[str] = "warmup_cosine"
    fp16: bool = True


@dataclass
class RegularizationConfig:
    """Weights of the K-Planes regularizers; a weight of 0 disables the term."""

    plane_tv: float = 0.0
    plane_tv_proposal_net: float = 0.0
    histogram: float = 0.0
    distortion: float = 0.0
    l1_proposal_net: float = 0.0
    depth_tv: float = 0.0


@dataclass
class LoggingConfig:
    # Periodic checkpointing / validation interval in steps (-1 disables). A checkpoint
    # is always written and the model always validated after the final step.
    save_every: int = 30000
    valid_every: int = 30000
    save_outputs: bool = True
    log_every: int = 5
    eval_batch_size: int = 8192


@dataclass
class DistillationLossWeights:
    alpha_histogram: float = 0.3
    rgb_points: float = 0.2
    alpha_points: float = 0.2
    rgb: float = 0.3


@dataclass
class DaCConfig:
    """Divide-and-Conquer settings: how views are split into experts and distilled."""

    num_experts: int = 4
    # 1-based expert index when training an expert, ``None`` for every other run.
    expert: Optional[int] = None
    # ``angular`` routes a ray by the azimuth of its origin (NeRF-Synthetic); ``lookup``
    # reads a frame -> expert table from ``assignment_file`` (Tanks and Temples).
    router: str = "angular"
    # Relative to ``data.root``. JSON mapping frame file name -> 0-based expert index
    # (legacy ``.pickle`` files are also accepted).
    assignment_file: str = "expert_assignment.json"
    # Where trained experts are loaded from during distillation. Placeholders:
    # ``{log_dir}`` (experiment.log_dir), ``{scene}`` (data.root basename), ``{index}``.
    expert_checkpoint: str = "{log_dir}/{scene}_N{index}/model.pth"
    loss_weights: DistillationLossWeights = field(default_factory=DistillationLossWeights)


@dataclass
class Config:
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dac: DaCConfig = field(default_factory=DaCConfig)

    def validate(self) -> None:
        """Check cross-field invariants that the type system cannot express."""
        _check_choice("data.dataset", self.data.dataset, DATASETS)
        _check_choice("dac.router", self.dac.router, ROUTERS)
        _check_choice("model.density_activation", self.model.density_activation, DENSITY_ACTIVATIONS)
        if self.optim.scheduler is not None:
            _check_choice("optim.scheduler", self.optim.scheduler, SCHEDULERS)
        if self.dac.num_experts < 1:
            raise ConfigError("dac.num_experts must be at least 1")
        if self.dac.expert is not None and not 1 <= self.dac.expert <= self.dac.num_experts:
            raise ConfigError(f"dac.expert must be in [1, {self.dac.num_experts}], got {self.dac.expert}")
        grid = self.model.grid
        if grid.input_coordinate_dim != 3 or len(grid.resolution) != 3:
            raise ConfigError("only static 3D scenes are supported: model.grid needs 3 coordinates")
        if len(self.model.num_proposal_samples) != self.model.num_proposal_iterations:
            raise ConfigError(
                "model.num_proposal_samples must have one entry per proposal iteration "
                f"({self.model.num_proposal_iterations})"
            )
        if not self.model.proposal_nets:
            raise ConfigError("model.proposal_nets must not be empty")
        if self.model.use_same_proposal_network and len(self.model.proposal_nets) != 1:
            raise ConfigError("model.use_same_proposal_network requires exactly one proposal net")
        for name in ("global_translation", "global_scale"):
            if len(getattr(self.model, name)) != 3:
                raise ConfigError(f"model.{name} must have 3 entries")
        if self.optim.num_steps < 1 or self.optim.batch_size < 1:
            raise ConfigError("optim.num_steps and optim.batch_size must be positive")


def _check_choice(name: str, value: str, choices: tuple[str, ...]) -> None:
    if value not in choices:
        raise ConfigError(f"{name} must be one of {list(choices)}, got {value!r}")


def load_config(path: Union[str, Path, None] = None, overrides: typing.Sequence[str] = ()) -> Config:
    """Load a YAML config, apply ``key=value`` overrides and validate the result."""
    raw: dict[str, Any] = {}
    if path is not None:
        with open(path) as f:
            loaded = yaml.safe_load(f)
        if loaded is not None and not isinstance(loaded, dict):
            raise ConfigError(f"{path}: expected a mapping at the top level")
        raw = loaded or {}
    for override in overrides:
        _apply_override(raw, override)
    cfg = from_dict(Config, raw)
    cfg.validate()
    return cfg


def config_to_dict(cfg: Config) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)


def _apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if not sep or not key:
        raise ConfigError(f"override {override!r} is not of the form section.field=value")
    *parents, leaf = key.strip().split(".")
    node = raw
    for part in parents:
        child = node.setdefault(part, {})
        if not isinstance(child, dict):
            raise ConfigError(f"override {override!r}: {part!r} is not a section")
        node = child
    node[leaf] = yaml.safe_load(value)


def from_dict(cls: type[T], data: Any, path: str = "") -> T:
    """Recursively build dataclass ``cls`` from plain data, validating keys and types."""
    if not isinstance(data, dict):
        raise ConfigError(f"{path or 'config'}: expected a mapping, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = [f.name for f in dataclasses.fields(cls)]  # type: ignore[arg-type]
    unknown = sorted(set(data) - set(names))
    if unknown:
        key = unknown[0]
        hint = difflib.get_close_matches(key, names, n=1)
        suggestion = f" (did you mean {_join(path, hint[0])!r}?)" if hint else ""
        raise ConfigError(f"unknown config key {_join(path, key)!r}{suggestion}")
    kwargs = {name: _coerce(data[name], hints[name], _join(path, name)) for name in names if name in data}
    return cls(**kwargs)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _coerce(value: Any, tp: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    if origin is typing.Union:
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(value, args[0], path)
    if origin is list:
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        (item_tp,) = typing.get_args(tp)
        return [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
    if dataclasses.is_dataclass(tp):
        return from_dict(typing.cast(type, tp), value, path)
    if tp is bool:
        if isinstance(value, bool):
            return value
    elif tp is int:
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    elif tp is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        if isinstance(value, str):  # PyYAML reads e.g. "1e-3" as a string
            try:
                return float(value)
            except ValueError:
                pass
    elif tp is str:
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            return str(value)
    else:  # pragma: no cover - guards against unsupported annotations
        raise TypeError(f"unsupported config type {tp!r} at {path}")
    raise ConfigError(f"{path}: expected {tp.__name__}, got {value!r}")
