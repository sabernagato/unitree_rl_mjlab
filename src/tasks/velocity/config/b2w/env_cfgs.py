"""Unitree B2W velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointVelocityActionCfg,
)
from mjlab.managers import RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.assets.robots import get_b2w_robot_cfg
import src.tasks.velocity.mdp as unitree_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_LEG_JOINTS = (r".*_(hip|thigh|calf)_joint",)
_WHEEL_JOINTS = (r".*_wheel_joint",)
_WHEEL_SITES = ("FR", "FL", "RR", "RL")
_WHEEL_GEOMS = tuple(f"{name}_wheel_collision" for name in _WHEEL_SITES)


def _leg_asset_cfg() -> SceneEntityCfg:
  """Return an unresolved leg-only entity config for one manager term."""
  return SceneEntityCfg("robot", joint_names=_LEG_JOINTS)


def _wheel_asset_cfg() -> SceneEntityCfg:
  """Return an unresolved wheel-only entity config for one manager term."""
  return SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS)


def unitree_b2w_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the B2W rough-terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.nconmax = 256
  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 1000
  cfg.scene.entities = {"robot": get_b2w_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  wheel_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=_WHEEL_GEOMS,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  nonwheel_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_touch",
    primary=ContactMatch(
      mode="geom",
      entity="robot",
      pattern=r".*_collision\d*$",
      exclude=_WHEEL_GEOMS,
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    wheel_ground_cfg,
    nonwheel_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  # B2W uses position targets for its 12 leg joints and velocity targets for
  # its four continuous wheel joints.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_LEG_JOINTS,
      scale=0.25,
      use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_WHEEL_JOINTS,
      scale=20.0,
      use_default_offset=True,
    ),
  }

  for group_name in ("actor", "critic"):
    cfg.observations[group_name].terms["joint_pos"].params[
      "asset_cfg"
    ] = _leg_asset_cfg()

  cfg.viewer.body_name = "base_link"
  cfg.viewer.distance = 2.5
  cfg.viewer.elevation = -10.0

  cfg.observations["critic"].terms["foot_height"].params[
    "asset_cfg"
  ].site_names = _WHEEL_SITES

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = _WHEEL_GEOMS
  cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

  cfg.rewards["pose"].params["asset_cfg"] = _leg_asset_cfg()
  cfg.rewards["pose"].params["std_standing"] = {
    r".*_hip_joint": 0.05,
    r".*_thigh_joint": 0.10,
    r".*_calf_joint": 0.15,
  }
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_joint": 0.10,
    r".*_thigh_joint": 0.20,
    r".*_calf_joint": 0.25,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_joint": 0.15,
    r".*_thigh_joint": 0.30,
    r".*_calf_joint": 0.40,
  }

  cfg.rewards["body_orientation_l2"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)
  cfg.rewards["stand_still"].params["asset_cfg"] = _leg_asset_cfg()
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = _leg_asset_cfg()
  cfg.rewards["wheel_stand_still"] = RewardTermCfg(
    func=unitree_mdp.wheel_stand_still,
    weight=-0.01,
    params={
      "command_name": "twist",
      "command_threshold": 0.1,
      "asset_cfg": _wheel_asset_cfg(),
    },
  )

  # Gait and foot-slip terms assume point feet. A rolling wheel has non-zero
  # surface velocity at contact, so those costs would penalize correct motion.
  for reward_name in ("foot_gait", "foot_clearance", "foot_slip", "soft_landing"):
    cfg.rewards.pop(reward_name)

  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={
      "sensor_name": nonwheel_ground_cfg.name,
      "force_threshold": 20.0,
    },
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.ranges.lin_vel_x = (-1.0, 2.0)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  command_curriculum = cfg.curriculum["command_vel"]
  command_curriculum.params["velocity_stages"] = [
    {
      "step": 0,
      "lin_vel_x": (-0.5, 1.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-0.5, 0.5),
    },
    {
      "step": 5000 * 24,
      "lin_vel_x": (-1.0, 2.0),
      "lin_vel_y": (0.0, 0.0),
      "ang_vel_z": (-1.0, 1.0),
    },
  ]

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    if cfg.scene.terrain is not None:
      if cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = False
        cfg.scene.terrain.terrain_generator.num_cols = 5
        cfg.scene.terrain.terrain_generator.num_rows = 5
        cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_b2w_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the B2W flat-terrain velocity configuration."""
  cfg = unitree_b2w_rough_env_cfg(play=play)

  cfg.sim.njmax = 500
  cfg.sim.nconmax = 128
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    sensor
    for sensor in (cfg.scene.sensors or ())
    if sensor.name != "terrain_scan"
  )
  del cfg.observations["actor"].terms["height_scan"]
  del cfg.observations["critic"].terms["height_scan"]
  cfg.curriculum.pop("terrain_levels", None)

  return cfg
