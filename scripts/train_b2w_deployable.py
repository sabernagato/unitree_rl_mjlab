"""Dedicated launcher for B2W real-deployment-oriented asymmetric training."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

import tyro

from train import TrainConfig, launch_training


TASK_ID = "Unitree-B2W-Deployable"


@dataclass(frozen=True)
class B2WDeployableTrainConfig:
  """High-level options for the B2W deployable-actor training run."""

  num_envs: int = 1280
  max_iterations: int = 20000
  seed: int = 42
  run_name: str = "deployable"
  payload_kg: float = 0.0
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  resume: bool = False
  load_run: str = ".*"
  load_checkpoint: str = "model_.*.pt"
  video: bool = False
  video_length: int = 400
  video_interval: int = 2000
  enable_nan_guard: bool = False


def main() -> None:
  args = tyro.cli(B2WDeployableTrainConfig)
  if args.num_envs <= 0:
    raise ValueError("num_envs must be positive")
  if args.max_iterations <= 0:
    raise ValueError("max_iterations must be positive")
  if args.payload_kg < 0.0:
    raise ValueError("payload_kg must be non-negative")

  os.environ["B2W_PAYLOAD_KG"] = str(args.payload_kg)

  # Populate the task registry before loading the dedicated configuration.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  cfg = TrainConfig.from_task(TASK_ID)
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.max_iterations
  cfg.agent.seed = args.seed
  cfg.agent.run_name = args.run_name
  cfg.agent.resume = args.resume
  cfg.agent.load_run = args.load_run
  cfg.agent.load_checkpoint = args.load_checkpoint

  cfg = replace(
    cfg,
    gpu_ids=args.gpu_ids,
    video=args.video,
    video_length=args.video_length,
    video_interval=args.video_interval,
    enable_nan_guard=args.enable_nan_guard,
  )
  launch_training(TASK_ID, cfg)


if __name__ == "__main__":
  main()
