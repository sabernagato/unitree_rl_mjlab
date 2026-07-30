"""Deterministic B2W wheel-direction and differential-drive smoke test."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends


TASK_ID = "Unitree-B2W-Flat"
EXPECTED_WHEEL_ORDER = (
  "FL_wheel_joint",
  "FR_wheel_joint",
  "RL_wheel_joint",
  "RR_wheel_joint",
)


@dataclass(frozen=True)
class WheelDirectionCheckCfg:
  device: str | None = None
  wheel_speed: float = 5.0
  settle_steps: int = 100
  drive_steps: int = 200
  min_forward_speed: float = 0.10
  min_yaw_rate: float = 0.10


@dataclass(frozen=True)
class TestResult:
  name: str
  wheel_targets: dict[str, float]
  mean_base_velocity_b: list[float]
  mean_base_angular_velocity_b: list[float]
  mean_wheel_velocity: list[float]
  displacement_w: list[float]
  mean_contact_fraction: float


def _configure_env() -> object:
  # Use the real B2W model and controller, but remove all randomization so this
  # test isolates wheel joint order, sign, and ground interaction.
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.scene.num_envs = 1
  cfg.episode_length_s = 1e9
  cfg.terminations = {}
  cfg.curriculum = {}

  reset_base = cfg.events["reset_base"]
  reset_base.params["pose_range"] = {}
  reset_base.params["velocity_range"] = {}
  cfg.events = {
    "reset_base": reset_base,
    "reset_robot_joints": cfg.events["reset_robot_joints"],
  }

  cfg.commands["twist"].debug_vis = False
  cfg.actions["wheel_vel"].scale = 1.0
  return cfg


def _wheel_action_slice(env: ManagerBasedRlEnv) -> tuple[int, int]:
  start = 0
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    if name == "wheel_vel":
      return start, start + term.action_dim
    start += term.action_dim
  raise RuntimeError("wheel_vel action term is not active")


def _run_pattern(
  env: ManagerBasedRlEnv,
  wheel_order: tuple[str, ...],
  wheel_slice: tuple[int, int],
  name: str,
  targets: dict[str, float],
  settle_steps: int,
  drive_steps: int,
) -> TestResult:
  env.reset(seed=42)
  action = torch.zeros(
    (1, env.action_manager.total_action_dim),
    device=env.device,
    dtype=torch.float32,
  )
  for _ in range(settle_steps):
    env.step(action)

  robot = env.scene["robot"]
  initial_pos = robot.data.root_link_pos_w[0].clone()
  target_values = torch.tensor(
    [[targets.get(joint_name, 0.0) for joint_name in wheel_order]],
    device=env.device,
  )
  action[:, wheel_slice[0] : wheel_slice[1]] = target_values

  base_vel_samples = []
  base_ang_vel_samples = []
  wheel_vel_samples = []
  contact_samples = []
  wheel_term = env.action_manager.get_term("wheel_vel")
  for _ in range(drive_steps):
    env.step(action)
    base_vel_samples.append(robot.data.root_link_lin_vel_b[0].clone())
    base_ang_vel_samples.append(robot.data.root_link_ang_vel_b[0].clone())
    wheel_vel_samples.append(
      robot.data.joint_vel[0, wheel_term.target_ids].clone()
    )
    contact = env.scene["feet_ground_contact"].data.found
    if contact is not None:
      contact_samples.append((contact[0] > 0).float().mean())

  sample_start = drive_steps // 2
  mean_base_vel = torch.stack(base_vel_samples[sample_start:]).mean(dim=0)
  mean_base_ang_vel = torch.stack(base_ang_vel_samples[sample_start:]).mean(dim=0)
  mean_wheel_vel = torch.stack(wheel_vel_samples[sample_start:]).mean(dim=0)
  displacement = robot.data.root_link_pos_w[0] - initial_pos
  mean_contact = (
    torch.stack(contact_samples[sample_start:]).mean()
    if contact_samples
    else torch.tensor(float("nan"), device=env.device)
  )

  return TestResult(
    name=name,
    wheel_targets=targets,
    mean_base_velocity_b=mean_base_vel.cpu().tolist(),
    mean_base_angular_velocity_b=mean_base_ang_vel.cpu().tolist(),
    mean_wheel_velocity=mean_wheel_vel.cpu().tolist(),
    displacement_w=displacement.cpu().tolist(),
    mean_contact_fraction=float(mean_contact.cpu()),
  )


def main() -> None:
  args = tyro.cli(WheelDirectionCheckCfg)
  if args.wheel_speed <= 0.0:
    raise ValueError("wheel_speed must be positive")
  if args.settle_steps <= 0 or args.drive_steps <= 1:
    raise ValueError("settle_steps and drive_steps must be positive")

  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  # Populate the task registry only after CLI parsing.
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env = ManagerBasedRlEnv(cfg=_configure_env(), device=device)
  try:
    wheel_term = env.action_manager.get_term("wheel_vel")
    wheel_order = tuple(wheel_term.target_names)
    print(f"[CHECK] wheel action order: {wheel_order}")
    if wheel_order != EXPECTED_WHEEL_ORDER:
      raise RuntimeError(
        f"Unexpected wheel order: {wheel_order}, expected {EXPECTED_WHEEL_ORDER}"
      )

    wheel_slice = _wheel_action_slice(env)
    speed = args.wheel_speed
    patterns = {
      "forward": {joint: speed for joint in wheel_order},
      "reverse": {joint: -speed for joint in wheel_order},
      "positive_yaw": {
        joint: (-speed if joint.startswith(("FL_", "RL_")) else speed)
        for joint in wheel_order
      },
      **{
        f"single_{joint[:2]}": {joint: speed}
        for joint in wheel_order
      },
    }
    results = [
      _run_pattern(
        env,
        wheel_order,
        wheel_slice,
        name,
        targets,
        args.settle_steps,
        args.drive_steps,
      )
      for name, targets in patterns.items()
    ]

    print(json.dumps([asdict(result) for result in results], indent=2))
    by_name = {result.name: result for result in results}
    forward_vx = by_name["forward"].mean_base_velocity_b[0]
    reverse_vx = by_name["reverse"].mean_base_velocity_b[0]
    positive_yaw = by_name["positive_yaw"].mean_base_angular_velocity_b[2]
    if forward_vx < args.min_forward_speed:
      raise RuntimeError(
        f"Positive wheel speed did not drive +X: mean vx={forward_vx:.3f}"
      )
    if reverse_vx > -args.min_forward_speed:
      raise RuntimeError(
        f"Negative wheel speed did not drive -X: mean vx={reverse_vx:.3f}"
      )
    if positive_yaw < args.min_yaw_rate:
      raise RuntimeError(
        "Left-negative/right-positive differential command did not produce "
        f"+yaw: mean wz={positive_yaw:.3f}"
      )
    print("[PASS] B2W wheel order, forward sign, reverse sign, and yaw sign agree.")
  finally:
    env.close()


if __name__ == "__main__":
  main()
