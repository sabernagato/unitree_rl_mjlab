"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Literal

import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer
from mjlab.viewer.native.keys import KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_S, KEY_UP


class KeyboardVelocityControl:
  """Thread-safe manual override for a velocity command term."""

  def __init__(self, command_term: Any):
    required_attrs = ("compute", "vel_command_b", "cfg")
    if not all(hasattr(command_term, attr) for attr in required_attrs):
      raise TypeError("The twist command does not expose velocity command controls")

    self._term = command_term
    self._lock = Lock()
    self._command = (0.0, 0.0, 0.0)
    self._original_compute = command_term.compute

    # CommandTerm.compute() runs on the simulation thread. Keep tensor writes
    # there; native viewer key callbacks only update the Python tuple above.
    def compute_with_manual_override(dt: float) -> None:
      self._original_compute(dt)
      self._apply()

    command_term.compute = compute_with_manual_override
    self._apply()

  def adjust(
    self, delta_b: tuple[float, float, float]
  ) -> tuple[float, float, float]:
    with self._lock:
      candidate = tuple(
        value + delta
        for value, delta in zip(self._command, delta_b, strict=True)
      )
      self._command = self._clamp(candidate)
      return self._command

  def zero(self) -> tuple[float, float, float]:
    with self._lock:
      self._command = (0.0, 0.0, 0.0)
      return self._command

  def _clamp(
    self, command_b: tuple[float, float, float]
  ) -> tuple[float, float, float]:
    ranges = self._term.cfg.ranges
    limits = (ranges.lin_vel_x, ranges.lin_vel_y, ranges.ang_vel_z)
    return tuple(
      min(max(float(value), axis_limits[0]), axis_limits[1])
      for value, axis_limits in zip(command_b, limits, strict=True)
    )

  def _apply(self) -> None:
    with self._lock:
      command = self._command
    self._term.vel_command_b[:, 0] = command[0]
    self._term.vel_command_b[:, 1] = command[1]
    self._term.vel_command_b[:, 2] = command[2]
    if hasattr(self._term, "is_heading_env"):
      self._term.is_heading_env.fill_(False)
    if hasattr(self._term, "is_standing_env"):
      self._term.is_standing_env.fill_(False)


@dataclass(frozen=True)
class PlayConfig:
  agent: Literal["zero", "random", "trained"] = "trained"
  checkpoint_file: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  keyboard_control: bool = False
  keyboard_linear_step: float = 0.25
  keyboard_yaw_step: float = 0.25
  no_terminations: bool = False
  """Disable all termination conditions (useful for viewing motions with dummy agents)."""

  # Internal flag used by demo script.
  _demo_mode: tyro.conf.Suppress[bool] = False


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task_id, play=True)
  agent_cfg = load_rl_cfg(task_id)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Disable terminations if requested (useful for viewing motions).
  if cfg.no_terminations:
    env_cfg.terminations = {}
    print("[INFO]: Terminations disabled")

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)

    # Check for local motion file first (works for both dummy and trained modes).
    if cfg.motion_file is not None and Path(cfg.motion_file).exists():
      print(f"[INFO]: Using local motion file: {cfg.motion_file}")
      motion_cmd.motion_file = cfg.motion_file
    elif DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require either:\n"
          "  --motion-file /path/to/motion.npz (local file)\n"
          "  --registry-name your-org/motions/motion-name (download from WandB)"
        )
  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
      str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  key_callback = None
  if cfg.keyboard_control:
    if resolved_viewer != "native":
      raise ValueError("--keyboard-control requires --viewer native")
    if "twist" not in env.unwrapped.command_manager.active_terms:
      raise ValueError("--keyboard-control requires a velocity task with a twist command")

    command_term = env.unwrapped.command_manager.get_term("twist")
    keyboard = KeyboardVelocityControl(command_term)

    def print_command(command: tuple[float, float, float]) -> None:
      print(
        "\r[KEYBOARD] "
        f"forward={command[0]:+.2f} m/s  yaw={command[2]:+.2f} rad/s  "
        "(arrows adjust, S stops)   ",
        end="",
        flush=True,
      )

    def handle_key(key: int) -> None:
      if key == KEY_UP:
        command = keyboard.adjust((cfg.keyboard_linear_step, 0.0, 0.0))
      elif key == KEY_DOWN:
        command = keyboard.adjust((-cfg.keyboard_linear_step, 0.0, 0.0))
      elif key == KEY_LEFT:
        command = keyboard.adjust((0.0, 0.0, cfg.keyboard_yaw_step))
      elif key == KEY_RIGHT:
        command = keyboard.adjust((0.0, 0.0, -cfg.keyboard_yaw_step))
      elif key == KEY_S:
        command = keyboard.zero()
      else:
        return
      print_command(command)

    key_callback = handle_key
    print("[INFO]: Keyboard control enabled: arrows adjust velocity, S stops")
    print_command((0.0, 0.0, 0.0))

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy, key_callback=key_callback).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401
  import src.tasks

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  # Parse the rest of the arguments + allow overriding env_cfg and agent_cfg.
  agent_cfg = load_rl_cfg(chosen_task)

  args = tyro.cli(
    PlayConfig,
    args=remaining_args,
    default=PlayConfig(),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )
  del remaining_args, agent_cfg

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
