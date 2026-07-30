"""Deterministic fixed-command evaluation for deployable B2W policies."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


TASK_IDS = {
  "base": "Unitree-B2W-Deployable-Flat-Base-Ablation",
  "rolling": "Unitree-B2W-Deployable-Flat-Rolling-Ablation",
  "stable": "Unitree-B2W-Deployable-Flat-Stable-Ablation",
  "precision": "Unitree-B2W-Deployable-Flat-Precision-Ablation",
}

SCENARIOS = {
  "stop": (0.0, 0.0, 0.0),
  "forward": (0.3, 0.0, 0.0),
  "reverse": (-0.25, 0.0, 0.0),
  "yaw_left": (0.0, 0.0, 0.15),
  "yaw_right": (0.0, 0.0, -0.15),
  "forward_yaw": (0.3, 0.0, 0.15),
}


@dataclass(frozen=True)
class EvaluationCfg:
  variant: str
  checkpoint_file: str
  num_envs: int = 64
  warmup_steps: int = 100
  measure_steps: int = 300
  seed: int = 42
  device: str | None = None
  domain_randomization: bool = False


class FixedVelocityCommand:
  """Replace command resampling with one fixed body-frame command."""

  def __init__(self, command_term: object):
    required = ("compute", "vel_command_b")
    if not all(hasattr(command_term, attr) for attr in required):
      raise TypeError("The twist command does not expose velocity controls")
    self._term = command_term
    self._original_compute = command_term.compute
    self._command = (0.0, 0.0, 0.0)

    def compute_with_override(dt: float) -> None:
      self._original_compute(dt)
      self.apply()

    command_term.compute = compute_with_override
    self.apply()

  def set(self, command: tuple[float, float, float]) -> None:
    self._command = command
    self.apply()

  def apply(self) -> None:
    self._term.vel_command_b[:, :] = torch.tensor(
      self._command,
      device=self._term.vel_command_b.device,
      dtype=self._term.vel_command_b.dtype,
    )
    if hasattr(self._term, "is_heading_env"):
      self._term.is_heading_env.fill_(False)
    if hasattr(self._term, "is_standing_env"):
      self._term.is_standing_env.fill_(False)


def _action_slice(env: ManagerBasedRlEnv, term_name: str) -> slice:
  start = 0
  for name in env.action_manager.active_terms:
    term = env.action_manager.get_term(name)
    if name == term_name:
      return slice(start, start + term.action_dim)
    start += term.action_dim
  raise KeyError(f"Action term is not active: {term_name}")


def _mean(samples: list[torch.Tensor]) -> float:
  return float(torch.cat(samples).mean().cpu())


def main() -> None:
  args = tyro.cli(EvaluationCfg)
  if args.variant not in TASK_IDS:
    raise ValueError(f"variant must be one of {tuple(TASK_IDS)}, got {args.variant!r}")
  if args.num_envs <= 0:
    raise ValueError("num_envs must be positive")
  if args.warmup_steps < 0 or args.measure_steps <= 0:
    raise ValueError("warmup_steps must be non-negative and measure_steps positive")

  checkpoint = Path(args.checkpoint_file).resolve()
  if not checkpoint.is_file():
    raise FileNotFoundError(checkpoint)

  os.environ.setdefault("MUJOCO_GL", "egl")
  configure_torch_backends()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id = TASK_IDS[args.variant]
  device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.seed = args.seed
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.commands["twist"].debug_vis = False
  if not args.domain_randomization:
    for event_name in (
      "foot_friction",
      "encoder_bias",
      "base_com",
      "inertial_properties",
      "leg_pd_gains",
      "actuator_delay",
      "directional_body_impulse",
    ):
      env_cfg.events.pop(event_name, None)

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  fixed_command = FixedVelocityCommand(
    raw_env.command_manager.get_term("twist")
  )
  wheel_slice = _action_slice(raw_env, "wheel_vel")
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  robot = raw_env.scene["robot"]

  results: dict[str, object] = {
    "variant": args.variant,
    "checkpoint": str(checkpoint),
    "num_envs": args.num_envs,
    "warmup_steps": args.warmup_steps,
    "measure_steps": args.measure_steps,
    "domain_randomization": args.domain_randomization,
    "scenarios": {},
  }

  try:
    for scenario_name, command in SCENARIOS.items():
      fixed_command.set(command)
      observations, _ = env.reset()
      fixed_command.apply()
      observations = env.get_observations()

      samples: dict[str, list[torch.Tensor]] = {
        "planar_error": [],
        "yaw_error": [],
        "vx": [],
        "vy_abs": [],
        "yaw_rate": [],
        "vertical_speed_abs": [],
        "tilt_sine": [],
        "base_height": [],
        "wheel_action_abs": [],
        "wheel_speed_abs": [],
      }
      fall_events = 0
      fell_once = torch.zeros(args.num_envs, dtype=torch.bool, device=device)

      total_steps = args.warmup_steps + args.measure_steps
      for step in range(total_steps):
        with torch.inference_mode():
          actions = policy(observations)
        observations, _, dones, _ = env.step(actions)

        if step < args.warmup_steps:
          continue

        lin_vel = robot.data.root_link_lin_vel_b
        ang_vel = robot.data.root_link_ang_vel_b
        command_tensor = torch.tensor(command, device=device)
        samples["planar_error"].append(
          torch.linalg.vector_norm(
            lin_vel[:, :2] - command_tensor[None, :2], dim=1
          )
        )
        samples["yaw_error"].append(
          torch.abs(ang_vel[:, 2] - command_tensor[2])
        )
        samples["vx"].append(lin_vel[:, 0])
        samples["vy_abs"].append(torch.abs(lin_vel[:, 1]))
        samples["yaw_rate"].append(ang_vel[:, 2])
        samples["vertical_speed_abs"].append(torch.abs(lin_vel[:, 2]))
        samples["tilt_sine"].append(
          torch.linalg.vector_norm(robot.data.projected_gravity_b[:, :2], dim=1)
        )
        samples["base_height"].append(robot.data.root_link_pos_w[:, 2])
        samples["wheel_action_abs"].append(
          torch.abs(actions[:, wheel_slice]).mean(dim=1)
        )
        samples["wheel_speed_abs"].append(
          torch.abs(
            robot.data.joint_vel[
              :, raw_env.action_manager.get_term("wheel_vel").target_ids
            ]
          ).mean(dim=1)
        )
        fall_events += int(dones.sum().item())
        fell_once |= dones.bool()

      scenario_result = {
        "command": command,
        "planar_error_mean": _mean(samples["planar_error"]),
        "yaw_error_mean": _mean(samples["yaw_error"]),
        "base_vx_mean": _mean(samples["vx"]),
        "base_vy_abs_mean": _mean(samples["vy_abs"]),
        "yaw_rate_mean": _mean(samples["yaw_rate"]),
        "vertical_speed_abs_mean": _mean(samples["vertical_speed_abs"]),
        "tilt_sine_mean": _mean(samples["tilt_sine"]),
        "base_height_mean": _mean(samples["base_height"]),
        "wheel_action_abs_mean": _mean(samples["wheel_action_abs"]),
        "wheel_speed_abs_mean": _mean(samples["wheel_speed_abs"]),
        "fall_events": fall_events,
        "fell_env_fraction": float(fell_once.float().mean().cpu()),
      }
      results["scenarios"][scenario_name] = scenario_result
      print(
        f"[EVAL] {scenario_name}: "
        f"planar_error={scenario_result['planar_error_mean']:.4f}, "
        f"yaw_error={scenario_result['yaw_error_mean']:.4f}, "
        f"fell={scenario_result['fell_env_fraction']:.3f}"
      )
  finally:
    env.close()

  print("[RESULT_JSON]")
  print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
