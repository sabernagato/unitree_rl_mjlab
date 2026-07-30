"""Launcher for short B2W wheel-control reward ablations."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

import tyro

from train import TrainConfig, launch_training


TASK_IDS = {
  "base": "Unitree-B2W-Deployable-Flat-Base-Ablation",
  "rolling": "Unitree-B2W-Deployable-Flat-Rolling-Ablation",
  "stable": "Unitree-B2W-Deployable-Flat-Stable-Ablation",
  "precision": "Unitree-B2W-Deployable-Flat-Precision-Ablation",
}


@dataclass(frozen=True)
class B2WAblationTrainConfig:
  variant: Literal["base", "rolling", "stable", "precision"] = "base"
  num_envs: int = 256
  max_iterations: int = 500
  seed: int = 42
  run_name: str = ""
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  enable_nan_guard: bool = False


def main() -> None:
  args = tyro.cli(B2WAblationTrainConfig)
  if args.num_envs <= 0:
    raise ValueError("num_envs must be positive")
  if args.max_iterations <= 0:
    raise ValueError("max_iterations must be positive")

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id = TASK_IDS[args.variant]
  cfg = TrainConfig.from_task(task_id)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.max_iterations
  cfg.agent.seed = args.seed
  cfg.agent.run_name = args.run_name or f"{args.variant}-seed{args.seed}"
  cfg.agent.resume = args.resume
  cfg.agent.load_run = args.load_run
  cfg.agent.load_checkpoint = args.load_checkpoint
  cfg = replace(
    cfg,
    gpu_ids=args.gpu_ids,
    enable_nan_guard=args.enable_nan_guard,
  )
  launch_training(task_id, cfg)


if __name__ == "__main__":
  main()
