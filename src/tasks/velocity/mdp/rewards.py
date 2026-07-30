from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded base linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  z_error = torch.square(actual[:, 2])
  lin_vel_error = xy_error + (2 * z_error)
  return torch.exp(-lin_vel_error / std**2)


def track_planar_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Track commanded planar velocity without mixing in vertical motion.

  Wheel torque can compress B2W's serial legs during acceleration. Keeping the
  planar objective separate from vertical stabilization prevents that transient
  from erasing the forward-velocity learning signal.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_lin_vel_b
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)

  env.extras["log"]["Metrics/linear_velocity_error_instant"] = torch.mean(
    torch.sqrt(xy_error)
  )
  env.extras["log"]["Metrics/base_speed_xy"] = torch.mean(
    torch.linalg.vector_norm(actual[:, :2], dim=1)
  )
  env.extras["log"]["Metrics/command_speed_xy"] = torch.mean(
    torch.linalg.vector_norm(command[:, :2], dim=1)
  )
  return torch.exp(-xy_error / std**2)


def base_vertical_velocity_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical base motion independently from planar tracking."""
  asset: Entity = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_height_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from a flat-ground base-height target."""
  asset: Entity = env.scene[asset_cfg.name]
  height_error = asset.data.root_link_pos_w[:, 2] - target_height
  env.extras["log"]["Metrics/base_height"] = torch.mean(
    asset.data.root_link_pos_w[:, 2]
  )
  return torch.square(height_error)


def base_height_above_terrain_l2(
  env: ManagerBasedRlEnv,
  target_height: float,
  terrain_sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize base height relative to the median terrain-scan elevation.

  A world-frame height target is valid on flat ground but becomes contradictory
  on stairs. The median is deliberately used instead of the minimum/maximum so
  a step entering one edge of the 187-ray footprint does not abruptly move the
  target before most of the robot has reached it.
  """
  asset: Entity = env.scene[asset_cfg.name]
  sensor: RayCastSensor = env.scene[terrain_sensor_name]
  hit_height = sensor.data.hit_pos_w[..., 2]
  valid = sensor.data.distances >= 0.0
  masked_height = torch.where(
    valid,
    hit_height,
    torch.full_like(hit_height, torch.nan),
  )
  ground_height = torch.nanmedian(masked_height, dim=1).values

  # Ray misses should neither introduce NaNs nor a large artificial penalty.
  fallback_ground = asset.data.root_link_pos_w[:, 2] - target_height
  ground_height = torch.where(
    torch.isfinite(ground_height),
    ground_height,
    fallback_ground,
  )
  relative_height = asset.data.root_link_pos_w[:, 2] - ground_height
  env.extras["log"]["Metrics/base_height_above_terrain"] = torch.mean(
    relative_height
  )
  return torch.square(relative_height - target_height)


def track_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."
  actual = asset.data.root_link_ang_vel_b
  z_error = torch.square(command[:, 2] - actual[:, 2])
  xy_error = torch.sum(torch.square(actual[:, :2]), dim=1)
  ang_vel_error = z_error + (0.05 * xy_error)
  return torch.exp(-ang_vel_error / std**2)


def body_orientation_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward flat base orientation (robot being upright).

  If asset_cfg has body_ids specified, computes the projected gravity
  for that specific body. Otherwise, uses the root link projected gravity.
  """
  asset: Entity = env.scene[asset_cfg.name]

  # If body_ids are specified, compute projected gravity for that body.
  if asset_cfg.body_ids:
    body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]  # [B, N, 4]
    body_quat_w = body_quat_w.squeeze(1)  # [B, 4]
    gravity_w = asset.data.gravity_vec_w  # [3]
    projected_gravity_b = quat_apply_inverse(body_quat_w, gravity_w)  # [B, 3]
    xy_squared = torch.sum(torch.square(projected_gravity_b[:, :2]), dim=1)
  else:
    # Use root link projected gravity.
    xy_squared = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
  return xy_squared


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def body_angular_velocity_penalty(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize excessive body angular velocities."""
  asset: Entity = env.scene[asset_cfg.name]
  ang_vel = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  ang_vel = ang_vel.squeeze(1)
  ang_vel_xy = ang_vel[:, :2]  # Don't penalize z-angular velocity.
  return torch.sum(torch.square(ang_vel_xy), dim=1)


def angular_momentum_penalty(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize whole-body angular momentum to encourage natural arm swing."""
  angmom_sensor: BuiltinSensor = env.scene[sensor_name]
  angmom = angmom_sensor.data
  angmom_magnitude_sq = torch.sum(torch.square(angmom), dim=-1)
  angmom_magnitude = torch.sqrt(angmom_magnitude_sq)
  env.extras["log"]["Metrics/angular_momentum_mean"] = torch.mean(angmom_magnitude)
  return angmom_magnitude_sq


def feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold: float = 0.4,
  command_name: str | None = None,
  command_threshold: float = 0.1,
) -> torch.Tensor:
  """Reward feet air time."""
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  air_time = sensor_data.current_air_time
  contact_time = sensor_data.current_contact_time
  in_contact = contact_time > 0.0
  in_mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.mean(in_contact.float(), dim=1) == 0.5
  mode_time = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
  error = torch.abs(mode_time - threshold)
  reward = torch.clamp(threshold - error, min=0.0)
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      scale = (total_command > command_threshold).float()
      reward *= scale
  return reward


def feet_clearance(
  env: ManagerBasedRlEnv,
  target_height: float,
  command_name: str | None = None,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from target clearance height, weighted by foot velocity."""
  asset: Entity = env.scene[asset_cfg.name]
  foot_z = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  delta = torch.abs(foot_z - target_height)  # [B, N]
  cost = torch.sum(delta * vel_norm, dim=1)  # [B]
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


def feet_gait(
        env: ManagerBasedRlEnv,
        period: float,
        offset: list[float],
        threshold: float,
        command_threshold: float,
        command_name: str,
        sensor_name: str,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    is_contact = sensor.data.current_contact_time > 0
    global_phase = ((env.episode_length_buf * env.step_dt) / period).unsqueeze(1)
    offsets = torch.as_tensor(offset, device=env.device, dtype=global_phase.dtype).view(1, -1)
    leg_phase = (global_phase + offsets) % 1.0
    is_stance = (leg_phase < threshold)
    reward = (is_stance == is_contact).float().mean(dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command > command_threshold).float()
            reward *= scale
    return reward


class feet_swing_height:
  """Penalize deviation from target swing height, evaluated at landing."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.sensor_name = cfg.params["sensor_name"]
    self.site_names = cfg.params["asset_cfg"].site_names
    self.peak_heights = torch.zeros(
      (env.num_envs, len(self.site_names)), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    target_height: float,
    command_name: str,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    foot_heights = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air,
      torch.maximum(self.peak_heights, foot_heights),
      self.peak_heights,
    )
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    total_command = linear_norm + angular_norm
    active = (total_command > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active
    num_landings = torch.sum(first_contact.float())
    peak_heights_at_landing = self.peak_heights * first_contact.float()
    mean_peak_height = torch.sum(peak_heights_at_landing) / torch.clamp(
      num_landings, min=1
    )
    env.extras["log"]["Metrics/peak_height_mean"] = mean_peak_height
    self.peak_heights = torch.where(
      first_contact,
      torch.zeros_like(self.peak_heights),
      self.peak_heights,
    )
    return cost


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost


class variable_posture:
  """Penalize deviation from default pose with speed-dependent tolerance.

  Uses per-joint standard deviations to control how much each joint can deviate
  from default pose. Smaller std = stricter (less deviation allowed), larger
  std = more forgiving. The reward is: exp(-mean(error² / std²))

  Three speed regimes (based on linear + angular command velocity):
    - std_standing (speed < walking_threshold): Tight tolerance for holding pose.
    - std_walking (walking_threshold <= speed < running_threshold): Moderate.
    - std_running (speed >= running_threshold): Loose tolerance for large motion.

  Tune std values per joint based on how much motion that joint needs at each
  speed. Map joint name patterns to std values, e.g. {".*knee.*": 0.35}.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset: Entity = env.scene[cfg.params["asset_cfg"].name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    self.default_joint_pos = default_joint_pos

    _, joint_names = asset.find_joints(cfg.params["asset_cfg"].joint_names)

    _, _, std_standing = resolve_matching_names_values(
      data=cfg.params["std_standing"],
      list_of_strings=joint_names,
    )
    self.std_standing = torch.tensor(
      std_standing, device=env.device, dtype=torch.float32
    )

    _, _, std_walking = resolve_matching_names_values(
      data=cfg.params["std_walking"],
      list_of_strings=joint_names,
    )
    self.std_walking = torch.tensor(std_walking, device=env.device, dtype=torch.float32)

    _, _, std_running = resolve_matching_names_values(
      data=cfg.params["std_running"],
      list_of_strings=joint_names,
    )
    self.std_running = torch.tensor(std_running, device=env.device, dtype=torch.float32)

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    std_standing,
    std_walking,
    std_running,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    walking_threshold: float = 0.5,
    running_threshold: float = 1.5,
  ) -> torch.Tensor:
    del std_standing, std_walking, std_running  # Unused.

    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    linear_speed = torch.norm(command[:, :2], dim=1)
    angular_speed = torch.abs(command[:, 2])
    total_speed = linear_speed + angular_speed

    standing_mask = (total_speed < walking_threshold).float()
    walking_mask = (
      (total_speed >= walking_threshold) & (total_speed < running_threshold)
    ).float()
    running_mask = (total_speed >= running_threshold).float()

    std = (
      self.std_standing * standing_mask.unsqueeze(1)
      + self.std_walking * walking_mask.unsqueeze(1)
      + self.std_running * running_mask.unsqueeze(1)
    )

    current_joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    desired_joint_pos = self.default_joint_pos[:, asset_cfg.joint_ids]
    error_squared = torch.square(current_joint_pos - desired_joint_pos)

    return torch.exp(-torch.mean(error_squared / (std**2), dim=1))


def stand_still(
        env: ManagerBasedRlEnv,
        command_name: str,
        command_threshold: float = 0.1,
        asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    diff_angle = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    reward = torch.sum(torch.square(diff_angle), dim=1)
    if command_name is not None:
        command = env.command_manager.get_command(command_name)
        if command is not None:
            linear_norm = torch.norm(command[:, :2], dim=1)
            angular_norm = torch.abs(command[:, 2])
            total_command = linear_norm + angular_norm
            scale = (total_command <= command_threshold).float()
            reward *= scale
    return reward


def wheel_stand_still(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize wheel rotation only while the commanded base velocity is zero."""
  asset: Entity = env.scene[asset_cfg.name]
  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  cost = torch.sum(torch.square(wheel_vel), dim=1)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return cost * (command_magnitude <= command_threshold).float()


def _terrain_relief(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Return max-minus-min terrain height across a raycast footprint."""
  sensor: RayCastSensor = env.scene[sensor_name]
  hit_height = sensor.data.hit_pos_w[..., 2]
  valid = sensor.data.distances >= 0.0
  high = torch.where(valid, hit_height, torch.full_like(hit_height, -torch.inf))
  low = torch.where(valid, hit_height, torch.full_like(hit_height, torch.inf))
  relief = torch.amax(high, dim=1) - torch.amin(low, dim=1)
  return torch.where(torch.isfinite(relief), relief, torch.zeros_like(relief))


def wheel_command_tracking(
  env: ManagerBasedRlEnv,
  command_name: str,
  wheel_radius: float,
  std: float,
  asset_cfg: SceneEntityCfg,
  terrain_sensor_name: str | None = None,
  rough_terrain_scale: float = 0.25,
  roughness_threshold: float = 0.05,
) -> torch.Tensor:
  """Track differential-drive wheel speeds, relaxing the target on obstacles."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None

  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  site_pos_delta_w = (
    asset.data.site_pos_w[:, asset_cfg.site_ids]
    - asset.data.root_link_pos_w.unsqueeze(1)
  )
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, site_pos_delta_w.shape[1], -1
  )
  site_pos_b = quat_apply_inverse(root_quat, site_pos_delta_w)
  target_wheel_vel = (
    command[:, 0].unsqueeze(1) - command[:, 2].unsqueeze(1) * site_pos_b[..., 1]
  ) / wheel_radius
  error = torch.mean(torch.square(wheel_vel - target_wheel_vel), dim=1)
  reward = torch.exp(-error / std**2)

  if terrain_sensor_name is not None:
    relief = _terrain_relief(env, terrain_sensor_name)
    scale = torch.where(
      relief > roughness_threshold,
      torch.full_like(reward, rough_terrain_scale),
      torch.ones_like(reward),
    )
    reward *= scale

  env.extras["log"]["Metrics/wheel_speed_error"] = torch.mean(torch.sqrt(error))
  return reward


def wheel_rolling_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  wheel_radius: float,
  lateral_weight: float,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Penalize rolling-constraint and lateral slip errors at contacting wheels."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  wheel_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  wheel_center_vel_w = asset.data.site_lin_vel_w[:, asset_cfg.site_ids]
  root_quat = asset.data.root_link_quat_w.unsqueeze(1).expand(
    -1, wheel_center_vel_w.shape[1], -1
  )
  wheel_center_vel_b = quat_apply_inverse(root_quat, wheel_center_vel_w)
  rolling_error = wheel_center_vel_b[..., 0] - wheel_radius * wheel_vel
  lateral_velocity = wheel_center_vel_b[..., 1]

  in_contact = (contact_sensor.data.found > 0).float()
  per_wheel_cost = (
    torch.square(rolling_error)
    + lateral_weight * torch.square(lateral_velocity)
  )
  contact_count = torch.clamp(torch.sum(in_contact, dim=1), min=1.0)
  cost = torch.sum(per_wheel_cost * in_contact, dim=1) / contact_count

  env.extras["log"]["Metrics/wheel_rolling_slip"] = torch.mean(torch.sqrt(cost))
  return cost


class wheel_obstacle_clearance:
  """Reward wheel clearance above the last support height when terrain is uneven."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    asset_cfg = cfg.params["asset_cfg"]
    self.num_wheels = len(asset_cfg.site_names)
    self.ground_height = torch.zeros(
      (env.num_envs, self.num_wheels), device=env.device, dtype=torch.float32
    )
    self.peak_height = torch.zeros_like(self.ground_height)
    self.initialized = torch.zeros(
      env.num_envs, device=env.device, dtype=torch.bool
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self.ground_height[env_ids] = 0.0
    self.peak_height[env_ids] = 0.0
    self.initialized[env_ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    terrain_sensor_name: str,
    command_name: str,
    obstacle_threshold: float,
    clearance_margin: float,
    min_clearance: float,
    max_clearance: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
  ) -> torch.Tensor:
    asset: Entity = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None

    wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
    in_contact = contact_sensor.data.found > 0
    first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)

    init_mask = (~self.initialized).unsqueeze(1)
    self.ground_height = torch.where(
      init_mask, wheel_height, self.ground_height
    )
    self.peak_height = torch.where(init_mask, wheel_height, self.peak_height)
    self.initialized[:] = True

    self.peak_height = torch.where(
      in_contact,
      self.peak_height,
      torch.maximum(self.peak_height, wheel_height),
    )
    clearance = torch.clamp(self.peak_height - self.ground_height, min=0.0)

    relief = _terrain_relief(env, terrain_sensor_name)
    target = torch.clamp(
      relief + clearance_margin,
      min=min_clearance,
      max=max_clearance,
    ).unsqueeze(1)
    cleared_fraction = torch.clamp(clearance / target, min=0.0, max=1.0)

    command = env.command_manager.get_command(command_name)
    assert command is not None
    active = (
      (relief > obstacle_threshold)
      & (torch.norm(command[:, :2], dim=1) > command_threshold)
    )
    landing_reward = torch.sum(
      cleared_fraction * first_contact.float(), dim=1
    ) / torch.clamp(torch.sum(first_contact.float(), dim=1), min=1.0)

    self.ground_height = torch.where(in_contact, wheel_height, self.ground_height)
    self.peak_height = torch.where(in_contact, wheel_height, self.peak_height)

    env.extras["log"]["Metrics/terrain_relief"] = torch.mean(relief)
    env.extras["log"]["Metrics/wheel_clearance"] = torch.mean(clearance)
    return landing_reward * active.float()


def turning_diagonal_gait(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  period: float,
  target_height: float,
  linear_threshold: float,
  angular_threshold: float,
  group_a: tuple[int, int],
  group_b: tuple[int, int],
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Reward alternating diagonal support and wheel lift for near-pure yaw."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  assert contact_sensor.data.found is not None

  contact = contact_sensor.data.found > 0
  wheel_height = asset.data.site_pos_w[:, asset_cfg.site_ids, 2]
  phase = (env.episode_length_buf * env.step_dt) % period / period
  support_a = phase < 0.5

  group_a_ids = list(group_a)
  group_b_ids = list(group_b)
  a_contact = torch.mean(contact[:, group_a_ids].float(), dim=1)
  b_contact = torch.mean(contact[:, group_b_ids].float(), dim=1)
  a_height = torch.mean(wheel_height[:, group_a_ids], dim=1)
  b_height = torch.mean(wheel_height[:, group_b_ids], dim=1)

  contact_score_a = 0.5 * (a_contact + (1.0 - b_contact))
  contact_score_b = 0.5 * (b_contact + (1.0 - a_contact))
  lift_score_a = torch.clamp(
    (b_height - a_height) / target_height, min=0.0, max=1.0
  )
  lift_score_b = torch.clamp(
    (a_height - b_height) / target_height, min=0.0, max=1.0
  )
  score_a = 0.5 * (contact_score_a + lift_score_a)
  score_b = 0.5 * (contact_score_b + lift_score_b)
  reward = torch.where(support_a, score_a, score_b)

  command = env.command_manager.get_command(command_name)
  assert command is not None
  active = (
    (torch.norm(command[:, :2], dim=1) < linear_threshold)
    & (torch.abs(command[:, 2]) > angular_threshold)
  )
  return reward * active.float()
