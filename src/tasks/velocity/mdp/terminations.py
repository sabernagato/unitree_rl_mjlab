from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def illegal_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    return (force_mag > force_threshold).any(dim=-1).any(dim=-1)  # [B]
  assert data.found is not None
  return torch.any(data.found, dim=-1)


def terrain_patch_boundary(
  env: ManagerBasedRlEnv,
  boundary: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Terminate before the robot reaches a generated terrain-tile seam.

  Generated velocity terrains place different 8 m tiles next to each other.
  Crossing a seam can create a large instantaneous contact discontinuity,
  especially for rolling wheels. The boundary is measured independently along
  each horizontal axis relative to the active terrain origin.
  """
  if boundary <= 0.0:
    raise ValueError("boundary must be positive")
  asset: Entity = env.scene[asset_cfg.name]
  relative_xy = (
    asset.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
  )
  return torch.amax(torch.abs(relative_xy), dim=1) >= boundary
