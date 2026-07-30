"""Launcher for B2W terrain and physical-disturbance fine-tuning."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from typing import Literal

import tyro
import warp as wp

# MjLab 1.2 still queries the pre-1.13 public module path when deciding whether
# CUDA graphs are supported. Keep the existing path untouched on Warp 1.12 and
# expose the moved module only for the isolated Warp 1.13 validation runtime.
if not hasattr(wp, "context"):
  from warp._src import context as warp_context

  wp.context = warp_context  # type: ignore[attr-defined]

from train import TrainConfig, launch_training


TASK_ID = "Unitree-B2W-Deployable-Stage2"


@dataclass(frozen=True)
class B2WStage2TrainConfig:
  """Options for Stage 2, bootstrapped from the validated flat policy."""

  # Single-process heightfield contact is validated through 512 worlds. Use
  # multi-GPU 2x512 rather than raising this per-process value.
  num_envs: int = 512
  max_iterations: int = 1200
  seed: int = 42
  run_name: str = "terrain-force-stage2"
  resume: bool = True
  load_run: str = "bootstrap_precision1397"
  load_checkpoint: str = "model_1397.pt"
  start_iteration: int = 1397
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  enable_nan_guard: bool = False
  disable_domain_randomization: bool = False


def main() -> None:
  args = tyro.cli(B2WStage2TrainConfig)
  if args.num_envs <= 0:
    raise ValueError("num_envs must be positive")
  if args.max_iterations <= 0:
    raise ValueError("max_iterations must be positive")
  if args.start_iteration < 0:
    raise ValueError("start_iteration must be non-negative")

  os.environ["B2W_STAGE2_START_ITERATION"] = str(args.start_iteration)
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
  if args.disable_domain_randomization:
    for event_name in (
      "foot_friction",
      "encoder_bias",
      "base_com",
      "inertial_properties",
      "leg_pd_gains",
      "actuator_delay",
    ):
      cfg.env.events.pop(event_name, None)
  cfg = replace(
    cfg,
    gpu_ids=args.gpu_ids,
    enable_nan_guard=args.enable_nan_guard,
  )
  launch_training(TASK_ID, cfg)


if __name__ == "__main__":
  main()
