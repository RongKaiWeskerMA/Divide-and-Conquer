"""Command-line interface: ``dac {split,train,distill,eval,render}``."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

import torch
import yaml

from dac import __version__
from dac.config import Config, ConfigError, config_to_dict, load_config, save_config
from dac.data.dataset import make_dataset
from dac.splitting import split_views
from dac.utils import seed_everything

log = logging.getLogger("dac")

PIPELINE_HELP = """\
The DaC pipeline for one scene:
  1. dac split   --data-root data/nerf_synthetic/lego --num-experts 4
  2. dac train   -c configs/nerf_synthetic.yaml experiment.name=lego_N1 dac.expert=1   (for each expert)
  3. dac distill -c configs/nerf_synthetic.yaml experiment.name=lego_distill
  4. dac train   -c configs/nerf_synthetic.yaml experiment.name=lego_finetune \\
                 --checkpoint logs/nerf_synthetic/lego_distill/model.pth optim.num_steps=60000
Overrides are dotted section.field=value pairs, e.g. data.root=data/nerf_synthetic/ship.
"""


def _setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s|%(levelname)8s| %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )


def _check_device(cfg: Config) -> None:
    device = torch.device(cfg.experiment.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigError(f"experiment.device={cfg.experiment.device} but CUDA is not available")


def _prepare(cfg: Config, run_dir: Path) -> None:
    _check_device(cfg)
    seed_everything(cfg.experiment.seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    log.info("Config:\n%s", yaml.safe_dump(config_to_dict(cfg), sort_keys=False).rstrip())


def _config_for_checkpoint(args: argparse.Namespace) -> Config:
    """``--config`` if given, else the ``config.yaml`` saved next to the checkpoint."""
    path = args.config
    if path is None:
        path = Path(args.checkpoint).parent / "config.yaml"
        if not path.is_file():
            raise ConfigError(f"no --config given and {path} does not exist")
    return load_config(path, args.overrides)


def cmd_split(args: argparse.Namespace) -> int:
    split_views(args.data_root, args.num_experts, frames_dir=args.frames_dir, force=args.force)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    from dac.training.trainer import Trainer

    cfg = load_config(args.config, args.overrides)
    run_dir = cfg.experiment.run_dir
    _prepare(cfg, run_dir)
    save_config(cfg, run_dir / "config.yaml")
    # The training set is built first so that the random stream matches the original code.
    train_set = make_dataset(cfg.data, "train", batch_size=cfg.optim.batch_size, expert=cfg.dac.expert)
    test_set = make_dataset(cfg.data, "test", expert=cfg.dac.expert)
    trainer = Trainer(cfg, run_dir, test_set, train_set)
    if args.checkpoint is not None:
        trainer.load_checkpoint(args.checkpoint, resume=True)
    trainer.train()
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from dac.training.distillation import DistillationTrainer

    cfg = load_config(args.config, args.overrides)
    if cfg.dac.expert is not None:
        raise ConfigError("distillation uses every view; unset dac.expert")
    run_dir = cfg.experiment.run_dir
    _prepare(cfg, run_dir)
    save_config(cfg, run_dir / "config.yaml")
    train_set = make_dataset(cfg.data, "train", batch_size=cfg.optim.batch_size)
    test_set = make_dataset(cfg.data, "test")
    trainer = DistillationTrainer(cfg, run_dir, test_set, train_set)
    trainer.train()
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from dac.training.trainer import Trainer

    cfg = _config_for_checkpoint(args)
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    _prepare(cfg, run_dir)
    trainer = Trainer(cfg, run_dir, make_dataset(cfg.data, "test", expert=cfg.dac.expert))
    trainer.load_checkpoint(args.checkpoint, resume=False)
    trainer.validate()
    trainer.close()
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    from dac.training.trainer import Trainer

    cfg = _config_for_checkpoint(args)
    run_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    _prepare(cfg, run_dir)
    trainer = Trainer(cfg, run_dir, make_dataset(cfg.data, "render"))
    trainer.load_checkpoint(args.checkpoint, resume=False)
    log.info("Saved %s", trainer.render_video())
    trainer.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dac",
        description="Divide and Conquer: Rethinking the Training Paradigm of Neural Radiance Fields.",
        epilog=PIPELINE_HELP,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    split = sub.add_parser("split", help="split a scene's views into expert groups")
    split.add_argument("--data-root", required=True, help="scene directory, e.g. data/nerf_synthetic/lego")
    split.add_argument("--num-experts", type=int, required=True)
    split.add_argument("--frames-dir", default=".", help="directory under --data-root holding transforms_{split}.json")
    split.add_argument("--force", action="store_true", help="overwrite existing split files")
    split.set_defaults(func=cmd_split)

    def add_run_args(p: argparse.ArgumentParser, config_required: bool) -> None:
        p.add_argument(
            "-c",
            "--config",
            required=config_required,
            help="YAML config" + ("" if config_required else " (default: config.yaml next to the checkpoint)"),
        )
        p.add_argument("overrides", nargs="*", metavar="KEY=VALUE", help="config overrides")

    train = sub.add_parser("train", help="train a baseline, an expert, or fine-tune from a checkpoint")
    add_run_args(train, config_required=True)
    train.add_argument(
        "--checkpoint", help="resume model, optimizer, LR schedule and step from this checkpoint (fine-tuning)"
    )
    train.set_defaults(func=cmd_train)

    distill = sub.add_parser("distill", help="distill trained experts into one model")
    add_run_args(distill, config_required=True)
    distill.set_defaults(func=cmd_distill)

    for name, func, help_text in (
        ("eval", cmd_eval, "compute test metrics of a checkpoint"),
        ("render", cmd_render, "render a video orbiting the scene"),
    ):
        p = sub.add_parser(name, help=help_text)
        add_run_args(p, config_required=False)
        p.add_argument("--checkpoint", required=True)
        p.add_argument("--output-dir", help="where to write outputs (default: the checkpoint's directory)")
        p.set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    # Overrides may appear anywhere after the command, also after options such as
    # --checkpoint, which argparse cannot express with a positional argument alone.
    args, extra = parser.parse_known_args(argv)
    invalid = [a for a in extra if a.startswith("-") or "=" not in a]
    if invalid or (extra and not hasattr(args, "overrides")):
        parser.error(f"unrecognized arguments: {' '.join(invalid or extra)}")
    if extra:
        args.overrides = [*args.overrides, *extra]
    _setup_logging()
    try:
        return int(args.func(args))
    except (ConfigError, FileNotFoundError, FileExistsError) as e:
        log.error("%s", e)
        return 2
