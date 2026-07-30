"""Task-specific disturbance events for velocity training."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, TypedDict

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class BodyImpulseForceStage(TypedDict):
  """One stage of the body impulse force curriculum."""

  step: int
  magnitude_range: tuple[float, float]


class apply_directional_body_impulse:
  """Apply intermittent, direction-biased force impulses to selected bodies.

  Directions are sampled in the robot body frame from a mixture of lateral
  cones, a uniform sphere, and vertical cones. The force is rotated into the
  world frame at impact and held briefly. Applying it above the body CoM also
  generates a realistic attitude disturbance.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._body_ids = asset_cfg.body_ids
    self._num_envs = env.num_envs
    self._device = env.device
    self._step_dt = env.step_dt
    self._cooldown_s: tuple[float, float] = cfg.params["cooldown_s"]

    self._num_bodies = (
      len(self._body_ids)
      if isinstance(self._body_ids, list)
      else self._asset.num_bodies
    )
    self._time_remaining = torch.zeros(self._num_envs, device=self._device)
    self._cooldown_remaining = torch.zeros(self._num_envs, device=self._device)
    self._active = torch.zeros(
      self._num_envs, device=self._device, dtype=torch.bool
    )
    self._force_magnitude = torch.zeros(self._num_envs, device=self._device)
    self._sample_cooldown(torch.arange(self._num_envs, device=self._device))

  def _sample_cooldown(self, env_ids: torch.Tensor) -> None:
    low, high = self._cooldown_s
    self._cooldown_remaining[env_ids] = (
      torch.rand(len(env_ids), device=self._device) * (high - low) + low
    )

  def _clear_force(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    zeros = torch.zeros(
      (len(env_ids), self._num_bodies, 3),
      device=self._device,
    )
    self._asset.write_external_wrench_to_sim(
      zeros,
      zeros,
      env_ids=env_ids,
      body_ids=self._body_ids,
    )
    self._force_magnitude[env_ids] = 0.0

  @staticmethod
  def _active_force_range(
    common_step_counter: int,
    force_stages: list[BodyImpulseForceStage],
  ) -> tuple[float, float]:
    if not force_stages:
      raise ValueError("force_stages must contain at least one stage")
    active_range = force_stages[0]["magnitude_range"]
    for stage in force_stages:
      if common_step_counter >= stage["step"]:
        active_range = stage["magnitude_range"]
    return active_range

  def _sample_directions(
    self,
    count: int,
    lateral_probability: float,
    vertical_probability: float,
    lateral_cone_deg: float,
    vertical_cone_deg: float,
  ) -> torch.Tensor:
    """Sample body-frame unit vectors with extra lateral and vertical coverage."""
    if lateral_probability < 0.0 or vertical_probability < 0.0:
      raise ValueError("direction probabilities must be non-negative")
    if lateral_probability + vertical_probability > 1.0:
      raise ValueError("direction probabilities must sum to at most one")

    # The remaining probability is uniform over the full sphere.
    direction = torch.randn((count, 3), device=self._device)
    direction /= torch.clamp(
      torch.linalg.norm(direction, dim=1, keepdim=True),
      min=1e-6,
    )
    mode = torch.rand(count, device=self._device)
    lateral = mode < lateral_probability
    vertical = (
      (mode >= lateral_probability)
      & (mode < lateral_probability + vertical_probability)
    )

    lateral_count = int(torch.sum(lateral).item())
    if lateral_count > 0:
      cos_limit = math.cos(math.radians(lateral_cone_deg))
      cos_angle = (
        torch.rand(lateral_count, device=self._device) * (1.0 - cos_limit)
        + cos_limit
      )
      sin_angle = torch.sqrt(torch.clamp(1.0 - cos_angle**2, min=0.0))
      azimuth = (
        torch.rand(lateral_count, device=self._device) * (2.0 * math.pi)
      )
      lateral_direction = torch.stack(
        (
          sin_angle * torch.cos(azimuth),
          cos_angle,
          sin_angle * torch.sin(azimuth),
        ),
        dim=1,
      )
      lateral_direction[:, 1] = torch.where(
        torch.rand(lateral_count, device=self._device) < 0.5,
        -lateral_direction[:, 1],
        lateral_direction[:, 1],
      )
      direction[lateral] = lateral_direction

    vertical_count = int(torch.sum(vertical).item())
    if vertical_count > 0:
      cos_limit = math.cos(math.radians(vertical_cone_deg))
      cos_angle = (
        torch.rand(vertical_count, device=self._device) * (1.0 - cos_limit)
        + cos_limit
      )
      sin_angle = torch.sqrt(torch.clamp(1.0 - cos_angle**2, min=0.0))
      azimuth = (
        torch.rand(vertical_count, device=self._device) * (2.0 * math.pi)
      )
      vertical_direction = torch.stack(
        (
          sin_angle * torch.cos(azimuth),
          sin_angle * torch.sin(azimuth),
          cos_angle,
        ),
        dim=1,
      )
      vertical_direction[:, 2] = torch.where(
        torch.rand(vertical_count, device=self._device) < 0.5,
        -vertical_direction[:, 2],
        vertical_direction[:, 2],
      )
      direction[vertical] = vertical_direction

    return direction

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | None,
    force_stages: list[BodyImpulseForceStage],
    duration_s: tuple[float, float],
    cooldown_s: tuple[float, float],
    lateral_probability: float,
    vertical_probability: float,
    lateral_cone_deg: float,
    vertical_cone_deg: float,
    body_point_offset: tuple[float, float, float],
    asset_cfg: SceneEntityCfg,
    step_offset: int = 0,
  ) -> None:
    """Advance independent impulse timers and apply newly triggered pushes."""
    del env_ids, asset_cfg  # Step events operate on every environment.
    self._cooldown_s = cooldown_s

    self._time_remaining[self._active] -= self._step_dt
    expired = self._active & (self._time_remaining <= 0.0)
    if expired.any():
      expired_ids = expired.nonzero(as_tuple=False).squeeze(-1)
      self._clear_force(expired_ids)
      self._active[expired_ids] = False
      self._time_remaining[expired_ids] = 0.0
      self._sample_cooldown(expired_ids)

    inactive = ~self._active
    self._cooldown_remaining[inactive] -= self._step_dt
    eligible = inactive & (self._cooldown_remaining <= 0.0)
    if eligible.any():
      trigger_ids = eligible.nonzero(as_tuple=False).squeeze(-1)
      count = len(trigger_ids)
      low, high = self._active_force_range(
        max(0, env.common_step_counter - step_offset),
        force_stages,
      )
      magnitude = (
        torch.rand(count, device=self._device) * (high - low) + low
      )
      direction_b = self._sample_directions(
        count=count,
        lateral_probability=lateral_probability,
        vertical_probability=vertical_probability,
        lateral_cone_deg=lateral_cone_deg,
        vertical_cone_deg=vertical_cone_deg,
      )

      force_b = (
        direction_b * magnitude.unsqueeze(1)
      ).unsqueeze(1).expand(
        -1,
        self._num_bodies,
        -1,
      )
      body_quat_w = self._asset.data.body_com_quat_w[trigger_ids][
        :, self._body_ids
      ]
      force_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        force_b.reshape(-1, 3),
      ).reshape(count, self._num_bodies, 3)

      offset_b = torch.tensor(
        body_point_offset,
        device=self._device,
        dtype=force_w.dtype,
      ).expand(count * self._num_bodies, 3)
      offset_w = quat_apply(
        body_quat_w.reshape(-1, 4),
        offset_b,
      ).reshape(count, self._num_bodies, 3)
      torque_w = torch.cross(offset_w, force_w, dim=-1)
      self._asset.write_external_wrench_to_sim(
        force_w,
        torque_w,
        env_ids=trigger_ids,
        body_ids=self._body_ids,
      )

      duration_low, duration_high = duration_s
      self._time_remaining[trigger_ids] = (
        torch.rand(count, device=self._device)
        * (duration_high - duration_low)
        + duration_low
      )
      self._active[trigger_ids] = True
      self._force_magnitude[trigger_ids] = magnitude

    env.extras["log"]["Metrics/body_impulse_active"] = torch.mean(
      self._active.float()
    )
    env.extras["log"]["Metrics/body_impulse_force"] = torch.mean(
      self._force_magnitude
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    """Clear active wrenches and restart push cooldowns for reset environments."""
    if env_ids is None:
      reset_ids = torch.arange(self._num_envs, device=self._device)
    elif isinstance(env_ids, slice):
      reset_ids = torch.arange(self._num_envs, device=self._device)[env_ids]
    else:
      reset_ids = env_ids

    self._clear_force(reset_ids)
    self._time_remaining[reset_ids] = 0.0
    self._active[reset_ids] = False
    self._sample_cooldown(reset_ids)
