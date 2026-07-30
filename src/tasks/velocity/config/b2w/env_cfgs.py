"""Unitree B2W velocity environment configurations."""

from copy import deepcopy
import math
import os

import mjlab.terrains as terrain_gen
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  JointVelocityActionCfg,
)
from mjlab.managers import ObservationTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

from src.assets.robots import get_b2w_robot_cfg
import src.tasks.velocity.mdp as unitree_mdp
from src.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg

_LEG_JOINTS = (r".*_(hip|thigh|calf)_joint",)
_WHEEL_JOINTS = (r".*_wheel_joint",)
_WHEEL_SITES = ("FR", "FL", "RR", "RL")
_WHEEL_GEOMS = tuple(f"{name}_wheel_collision" for name in _WHEEL_SITES)
_ORDERED_WHEEL_SITES = ("FL", "FR", "RL", "RR")
_WHEEL_JOINT_NAMES = tuple(
  f"{name}_wheel_joint" for name in _ORDERED_WHEEL_SITES
)

# B2W-specific curriculum with a strong focus on stair traversal.
#
# Stair heights are fixed, discrete specifications rather than difficulty
# ranges: 15 cm stairs, 20 cm stairs, and a 40 cm single wide step. The
# large-step geometry uses the 8 m patch, 1 m border, 4 m center platform,
# and 1 m tread to produce exactly one step in every travel direction.
_B2W_STAIRS_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.10),
    "stairs_down_15cm": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.15, 0.15),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_up_15cm": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.10,
      step_height_range=(0.15, 0.15),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_down_20cm": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.20, 0.20),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_up_20cm": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.15,
      step_height_range=(0.20, 0.20),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "large_step_down_40cm": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.20,
      step_height_range=(0.40, 0.40),
      step_width=1.0,
      platform_width=4.0,
      border_width=1.0,
    ),
    "large_step_up_40cm": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.20,
      step_height_range=(0.40, 0.40),
      step_width=1.0,
      platform_width=4.0,
      border_width=1.0,
    ),
  },
  add_lights=True,
)

# Privileged capability-validation terrain set. Difficulty increases by row:
# stairs grow continuously from 2 cm to 20 cm, while grids and rough heightfields
# grow from nearly flat to their configured maximum.
_B2W_PRIVILEGED_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=10,
  num_cols=20,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=0.20),
    "stairs_up": terrain_gen.BoxInvertedPyramidStairsTerrainCfg(
      proportion=0.25,
      step_height_range=(0.02, 0.20),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "stairs_down": terrain_gen.BoxPyramidStairsTerrainCfg(
      proportion=0.25,
      step_height_range=(0.02, 0.20),
      step_width=0.30,
      platform_width=3.0,
      border_width=1.0,
    ),
    "random_grid": terrain_gen.BoxRandomGridTerrainCfg(
      proportion=0.15,
      grid_width=0.35,
      grid_height_range=(0.01, 0.12),
      platform_width=2.0,
      border_width=0.25,
    ),
    "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
      proportion=0.15,
      noise_range=(0.01, 0.08),
      noise_step=0.01,
      border_width=0.25,
    ),
  },
  add_lights=True,
)

_B2W_FLAT_GENERATOR_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=20.0,
  num_rows=2,
  num_cols=10,
  sub_terrains={
    "flat": terrain_gen.BoxFlatTerrainCfg(proportion=1.0),
  },
  add_lights=True,
)


def _leg_asset_cfg() -> SceneEntityCfg:
  """Return an unresolved leg-only entity config for one manager term."""
  return SceneEntityCfg("robot", joint_names=_LEG_JOINTS)


def _wheel_asset_cfg() -> SceneEntityCfg:
  """Return an unresolved wheel-only entity config for one manager term."""
  return SceneEntityCfg("robot", joint_names=_WHEEL_JOINTS)


def _ordered_wheel_asset_cfg() -> SceneEntityCfg:
  """Return aligned wheel joints and sites in MuJoCo model order."""
  return SceneEntityCfg(
    "robot",
    joint_names=_WHEEL_JOINT_NAMES,
    site_names=_ORDERED_WHEEL_SITES,
    preserve_order=True,
  )


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


def unitree_b2w_stairs_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the B2W fixed-height stair-specialist configuration."""
  cfg = unitree_b2w_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(_B2W_STAIRS_TERRAINS_CFG)

  if play:
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_b2w_privileged_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the wheel-legged privileged capability-validation task."""
  cfg = unitree_b2w_rough_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = deepcopy(_B2W_PRIVILEGED_TERRAINS_CFG)
  cfg.scene.terrain.max_init_terrain_level = 1

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      # Keep Unitree's deployment-facing 17 x 11 terrain layout. At 0.1 m
      # resolution this is exactly 187 samples, avoids adding extra rear-facing
      # rays through a symmetric 2.4 m footprint, and keeps the observation
      # contract smaller for later distillation.
      sensor.pattern.size = (1.6, 1.0)
      sensor.pattern.resolution = 0.1
      sensor.debug_vis = False

  # Privileged actor: exact base velocity, contact state/forces, and terrain scan.
  # Phase is retained because the pure-yaw reward requests an alternating diagonal
  # support pattern.
  cfg.observations["actor"].enable_corruption = False
  cfg.observations["actor"].terms["base_lin_vel"] = ObservationTermCfg(
    func=unitree_mdp.base_lin_vel,
  )
  cfg.observations["actor"].terms["wheel_contact"] = ObservationTermCfg(
    func=unitree_mdp.foot_contact,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["actor"].terms["wheel_contact_forces"] = ObservationTermCfg(
    func=unitree_mdp.foot_contact_forces,
    params={"sensor_name": "feet_ground_contact"},
  )
  cfg.observations["actor"].terms["height_scan"].noise = None

  # Use conservative wheel exploration and smaller hip-roll commands. The wheel
  # scale of 5 rad/s per policy unit matches M20-style training and avoids the
  # violent initial transients caused by the previous scale of 20.
  cfg.actions = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=_LEG_JOINTS,
      scale={
        r".*_hip_joint": 0.125,
        r".*_(thigh|calf)_joint": 0.25,
      },
      use_default_offset=True,
    ),
    "wheel_vel": JointVelocityActionCfg(
      entity_name="robot",
      actuator_names=_WHEEL_JOINTS,
      scale=5.0,
      use_default_offset=True,
    ),
  }

  cfg.commands["twist"] = unitree_mdp.ModeVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(6.0, 10.0),
    heading_command=False,
    rel_standing_envs=0.0,
    rel_zero_envs=0.15,
    rel_straight_envs=0.25,
    rel_yaw_envs=0.25,
    minimum_command=0.2,
    debug_vis=True,
    ranges=unitree_mdp.ModeVelocityCommandCfg.Ranges(
      lin_vel_x=(-0.5, 0.8),
      lin_vel_y=(0.0, 0.0),
      ang_vel_z=(-0.6, 0.6),
      heading=None,
    ),
  )

  # Wheel-legged reward set: velocity tracking plus correct rolling physics,
  # terrain-triggered clearance, and a dedicated pure-yaw stepping mode.
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["body_orientation_l2"].weight = -2.0
  cfg.rewards["pose"].weight = 0.3
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_joint": 0.20,
    r".*_thigh_joint": 0.45,
    r".*_calf_joint": 0.55,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_joint": 0.25,
    r".*_thigh_joint": 0.55,
    r".*_calf_joint": 0.65,
  }
  cfg.rewards["action_rate_l2"].weight = -0.01
  cfg.rewards["joint_pos_limits"].weight = -5.0
  cfg.rewards["wheel_stand_still"].weight = -0.05
  cfg.rewards.pop("angular_momentum", None)

  cfg.rewards["wheel_command_tracking"] = RewardTermCfg(
    func=unitree_mdp.wheel_command_tracking,
    weight=0.5,
    params={
      "command_name": "twist",
      "wheel_radius": 0.113,
      "std": 4.0,
      "terrain_sensor_name": "terrain_scan",
      "rough_terrain_scale": 0.25,
      "roughness_threshold": 0.05,
      "asset_cfg": _ordered_wheel_asset_cfg(),
    },
  )
  cfg.rewards["wheel_rolling_slip"] = RewardTermCfg(
    func=unitree_mdp.wheel_rolling_slip,
    weight=-1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "wheel_radius": 0.113,
      "lateral_weight": 0.5,
      "asset_cfg": _ordered_wheel_asset_cfg(),
    },
  )
  cfg.rewards["wheel_obstacle_clearance"] = RewardTermCfg(
    func=unitree_mdp.wheel_obstacle_clearance,
    weight=2.0,
    params={
      "sensor_name": "feet_ground_contact",
      "terrain_sensor_name": "terrain_scan",
      "command_name": "twist",
      "obstacle_threshold": 0.04,
      "clearance_margin": 0.04,
      "min_clearance": 0.06,
      "max_clearance": 0.25,
      "command_threshold": 0.15,
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=_ORDERED_WHEEL_SITES, preserve_order=True
      ),
    },
  )
  cfg.rewards["turning_diagonal_gait"] = RewardTermCfg(
    func=unitree_mdp.turning_diagonal_gait,
    weight=1.5,
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "period": 0.8,
      "target_height": 0.05,
      "linear_threshold": 0.15,
      "angular_threshold": 0.2,
      "group_a": (0, 3),
      "group_b": (1, 2),
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=_ORDERED_WHEEL_SITES, preserve_order=True
      ),
    },
  )
  cfg.rewards["nonwheel_contact"] = RewardTermCfg(
    func=unitree_mdp.self_collision_cost,
    weight=-2.0,
    params={
      "sensor_name": "nonfoot_ground_touch",
      "force_threshold": 20.0,
    },
  )

  # Contact is a learning signal on stairs, not an immediate failure. Only a
  # true fall terminates the episode.
  cfg.terminations.pop("illegal_contact", None)

  if not play:
    # Replace the generic velocity teleport with a physical side impact. B2W
    # uses a conventional serial thigh/calf chain rather than M20's straight
    # knee X-configuration, so recovery should emerge from B2W's own joint
    # geometry instead of copying M20-specific posture constraints.
    cfg.events.pop("push_robot", None)
    cfg.events["directional_body_impulse"] = EventTermCfg(
      func=unitree_mdp.apply_directional_body_impulse,
      mode="step",
      params={
        "force_stages": [
          {"step": 0, "magnitude_range": (60.0, 120.0)},
          {"step": 3000 * 24, "magnitude_range": (100.0, 180.0)},
          {"step": 8000 * 24, "magnitude_range": (140.0, 240.0)},
        ],
        "duration_s": (0.10, 0.18),
        "cooldown_s": (4.0, 8.0),
        "lateral_probability": 0.60,
        "vertical_probability": 0.15,
        "lateral_cone_deg": 25.0,
        "vertical_cone_deg": 20.0,
        "body_point_offset": (0.0, 0.0, 0.15),
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("base_link",),
        ),
      },
    )
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.5, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.6, 0.6),
      },
      {
        "step": 3000 * 24,
        "lin_vel_x": (-1.0, 1.5),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.0, 1.0),
      },
      {
        "step": 8000 * 24,
        "lin_vel_x": (-1.5, 2.0),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-1.5, 1.5),
      },
    ]
  else:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, unitree_mdp.ModeVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-1.5, 1.5)
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.num_rows = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_b2w_deployable_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create an asymmetric task whose actor uses only deployable observations.

  The critic retains simulator-only terrain and contact information. The actor
  receives five frames of proprioceptive history so terrain contact, wheel slip,
  and external disturbances can be inferred without relying on unavailable
  three-dimensional contact forces or base linear velocity.
  """
  cfg = unitree_b2w_privileged_env_cfg(play=play)

  actor = cfg.observations["actor"]
  for term_name in (
    "height_scan",
    "base_lin_vel",
    "wheel_contact",
    "wheel_contact_forces",
  ):
    actor.terms.pop(term_name, None)

  # Five policy frames at 50 Hz provide 100 ms of deployable temporal context.
  # MjLab flattens group history term-major, which is also supported by the
  # repository's C++ ObservationManager through per-term history buffers.
  actor.history_length = 5
  actor.flatten_history_dim = True
  actor.enable_corruption = not play

  # The critic remains single-frame and privileged.
  critic = cfg.observations["critic"]
  critic.history_length = 1
  critic.flatten_history_dim = True
  critic.enable_corruption = False

  if not play:
    # Sim-to-real randomization. The actor's five-frame history can infer the
    # resulting response changes without receiving privileged dynamics values.
    cfg.events["inertial_properties"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode="startup",
      params={
        # e^(2 alpha) gives approximately 0.90x--1.11x mass/inertia scale.
        "alpha_range": (-0.05, 0.05),
        "asset_cfg": SceneEntityCfg("robot", body_names=(r".*",)),
      },
    )
    cfg.events["leg_pd_gains"] = EventTermCfg(
      func=dr.pd_gains,
      mode="reset",
      params={
        "kp_range": (0.9, 1.1),
        "kd_range": (0.85, 1.15),
        "operation": "scale",
        # Hip, thigh, and calf actuators; XML wheel velocity gains are not PD.
        "asset_cfg": SceneEntityCfg("robot", actuator_ids=[0, 1, 2]),
      },
    )
    cfg.events["actuator_delay"] = EventTermCfg(
      func=dr.sync_actuator_delays,
      mode="reset",
      params={
        # Physics-step lag: 0--4 at 5 ms = 0--20 ms.
        "lag_range": (0, 4),
        "asset_cfg": SceneEntityCfg("robot"),
      },
    )

  return cfg


def unitree_b2w_deployable_stage2_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-tune the flat Precision policy on terrain and physical disturbances.

  This remains an asymmetric task: the actor contract is unchanged and
  deployable, while only the critic receives height/contact privileges.
  M20-specific X-stance and diagonal-lift targets stay disabled because B2W has
  conventional serial thigh/calf legs.
  """
  cfg = unitree_b2w_deployable_env_cfg(play=play)
  start_iteration = int(os.environ.get("B2W_STAGE2_START_ITERATION", "1397"))
  if start_iteration < 0:
    raise ValueError("B2W_STAGE2_START_ITERATION must be non-negative")
  stage2_step_offset = start_iteration * 24
  terrain_boundary = 3.25

  assert cfg.scene.terrain is not None
  assert cfg.scene.terrain.terrain_generator is not None
  cfg.scene.terrain.max_init_terrain_level = 0

  # Keep the rolling wheels away from seams between adjacent 8 m terrain
  # tiles. Reaching this safe boundary still counts as curriculum progress.
  cfg.terminations["terrain_patch_boundary"] = TerminationTermCfg(
    func=unitree_mdp.terrain_patch_boundary,
    params={
      "boundary": terrain_boundary,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  if "terrain_levels" in cfg.curriculum:
    cfg.curriculum["terrain_levels"].params["progress_distance"] = terrain_boundary

  # Report the exact term that first becomes non-finite. The simulator NaN
  # guard remains enabled by the launcher and still captures physics history.
  for group_name in ("actor", "critic"):
    observation_group = cfg.observations[group_name]
    observation_group.nan_check_per_term = True
    observation_group.nan_policy = "error"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, unitree_mdp.ModeVelocityCommandCfg)
  twist_cmd.rel_zero_envs = 0.15
  twist_cmd.rel_straight_envs = 0.60
  twist_cmd.rel_yaw_envs = 0.15
  twist_cmd.minimum_command = 0.15
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.4)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  # Preserve the validated flat objective while allowing more error during the
  # first contact-rich terrain stage. Vertical stabilization and terrain-
  # relative height are separate from planar velocity tracking.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=unitree_mdp.track_planar_velocity,
    weight=6.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.2),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  cfg.rewards["base_vertical_velocity_l2"] = RewardTermCfg(
    func=unitree_mdp.base_vertical_velocity_l2,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  cfg.rewards["base_height_above_terrain_l2"] = RewardTermCfg(
    func=unitree_mdp.base_height_above_terrain_l2,
    weight=-3.0,
    params={
      "target_height": 0.58,
      "terrain_sensor_name": "terrain_scan",
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_joint": 0.12,
    r".*_thigh_joint": 0.25,
    r".*_calf_joint": 0.35,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_joint": 0.18,
    r".*_thigh_joint": 0.40,
    r".*_calf_joint": 0.50,
  }
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["wheel_stand_still"].weight = -0.20
  cfg.rewards["wheel_command_tracking"].weight = 0.0
  cfg.rewards["wheel_rolling_slip"].weight = -0.25
  cfg.rewards["wheel_obstacle_clearance"].weight = 0.5
  cfg.rewards["turning_diagonal_gait"].weight = 0.0

  if not play:
    # Begin with moderate sim-to-real spread. Wider randomization belongs after
    # the terrain and disturbance recovery behavior is established.
    cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)
    cfg.events["base_com"].params["ranges"] = {
      0: (-0.02, 0.02),
      1: (-0.02, 0.02),
      2: (-0.02, 0.02),
    }
    cfg.events["leg_pd_gains"].params["kp_range"] = (0.95, 1.05)
    cfg.events["leg_pd_gains"].params["kd_range"] = (0.90, 1.10)
    cfg.events["actuator_delay"].params["lag_range"] = (0, 2)

    impulse = cfg.events["directional_body_impulse"]
    impulse.params["force_stages"] = [
      {"step": 0, "magnitude_range": (20.0, 50.0)},
      {"step": 400 * 24, "magnitude_range": (40.0, 80.0)},
      {"step": 1000 * 24, "magnitude_range": (60.0, 120.0)},
      {"step": 2500 * 24, "magnitude_range": (100.0, 180.0)},
    ]
    impulse.params["duration_s"] = (0.08, 0.16)
    impulse.params["cooldown_s"] = (6.0, 10.0)
    impulse.params["lateral_probability"] = 0.60
    impulse.params["vertical_probability"] = 0.15
    impulse.params["lateral_cone_deg"] = 25.0
    impulse.params["vertical_cone_deg"] = 20.0
    impulse.params["body_point_offset"] = (0.0, 0.0, 0.15)
    impulse.params["step_offset"] = stage2_step_offset

    cfg.curriculum["command_vel"].params["step_offset"] = stage2_step_offset
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.4, 0.4),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.3, 0.3),
      },
      {
        "step": 600 * 24,
        "lin_vel_x": (-0.6, 0.8),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.5, 0.5),
      },
      {
        "step": 1800 * 24,
        "lin_vel_x": (-0.8, 1.2),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.8, 0.8),
      },
    ]

  return cfg


def unitree_b2w_deployable_ablation_env_cfg(
  *,
  rolling_weight: float,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create an isolated flat-ground wheel-control experiment.

  The actor/critic observation contracts remain identical to the deployable
  rough task. Only reward shaping differs between the A/B variants.
  """
  cfg = unitree_b2w_deployable_env_cfg(play=play)

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "generator"
  cfg.scene.terrain.terrain_generator = deepcopy(_B2W_FLAT_GENERATOR_CFG)
  cfg.scene.terrain.terrain_generator.curriculum = False
  cfg.scene.terrain.max_init_terrain_level = 0
  cfg.curriculum.pop("terrain_levels", None)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, unitree_mdp.ModeVelocityCommandCfg)
  twist_cmd.rel_zero_envs = 0.10
  twist_cmd.rel_straight_envs = 0.75
  twist_cmd.rel_yaw_envs = 0.05
  twist_cmd.minimum_command = 0.15
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.6)
  twist_cmd.ranges.lin_vel_y = (0.0, 0.0)
  twist_cmd.ranges.ang_vel_z = (-0.2, 0.2)

  # M20's transferable principle is to optimize base motion from wheel-velocity
  # actions. X-stance-only diagonal lifting and analytic target wheel speeds are
  # deliberately disabled for B2W's conventional serial legs.
  cfg.rewards["track_linear_velocity"] = RewardTermCfg(
    func=unitree_mdp.track_planar_velocity,
    weight=5.0,
    params={
      "command_name": "twist",
      "std": math.sqrt(0.5),
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  cfg.rewards["base_vertical_velocity_l2"] = RewardTermCfg(
    func=unitree_mdp.base_vertical_velocity_l2,
    weight=-2.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )
  cfg.rewards["track_angular_velocity"].weight = 2.0
  cfg.rewards["wheel_command_tracking"].weight = 0.0
  cfg.rewards["wheel_rolling_slip"].weight = rolling_weight
  cfg.rewards["wheel_obstacle_clearance"].weight = 0.0
  cfg.rewards["turning_diagonal_gait"].weight = 0.0

  if not play:
    cfg.events.pop("directional_body_impulse", None)
    cfg.events.pop("encoder_bias", None)
    cfg.events.pop("base_com", None)
    cfg.events.pop("inertial_properties", None)
    cfg.events.pop("leg_pd_gains", None)
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.0)
    cfg.events["actuator_delay"].params["lag_range"] = (0, 1)
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.4, 0.6),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.2, 0.2),
      }
    ]
  else:
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 5
    cfg.scene.terrain.terrain_generator.border_width = 10.0

  return cfg


def unitree_b2w_deployable_base_ablation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation A: base velocity tracking without wheel regularization."""
  return unitree_b2w_deployable_ablation_env_cfg(
    rolling_weight=0.0,
    play=play,
  )


def unitree_b2w_deployable_rolling_ablation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Ablation B: base velocity tracking plus weak contact rolling cost."""
  return unitree_b2w_deployable_ablation_env_cfg(
    rolling_weight=-0.25,
    play=play,
  )


def unitree_b2w_deployable_stable_ablation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Flat validation task with balanced commands and B2W posture constraints."""
  cfg = unitree_b2w_deployable_ablation_env_cfg(
    rolling_weight=-0.25,
    play=play,
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, unitree_mdp.ModeVelocityCommandCfg)
  twist_cmd.rel_zero_envs = 0.20
  twist_cmd.rel_straight_envs = 0.50
  twist_cmd.rel_yaw_envs = 0.20
  twist_cmd.ranges.lin_vel_x = (-0.4, 0.4)
  twist_cmd.ranges.ang_vel_z = (-0.3, 0.3)

  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.25)
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].params["std"] = math.sqrt(0.1)
  cfg.rewards["body_orientation_l2"].weight = -5.0
  cfg.rewards["pose"].weight = 1.0
  cfg.rewards["pose"].params["std_walking"] = {
    r".*_hip_joint": 0.08,
    r".*_thigh_joint": 0.12,
    r".*_calf_joint": 0.18,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*_hip_joint": 0.12,
    r".*_thigh_joint": 0.18,
    r".*_calf_joint": 0.25,
  }
  cfg.rewards["stand_still"].weight = -2.0
  cfg.rewards["wheel_stand_still"].weight = -0.20
  cfg.rewards["base_height_l2"] = RewardTermCfg(
    func=unitree_mdp.base_height_l2,
    weight=-5.0,
    params={
      "target_height": 0.58,
      "asset_cfg": SceneEntityCfg("robot", body_names=("base_link",)),
    },
  )

  if not play:
    cfg.curriculum["command_vel"].params["velocity_stages"] = [
      {
        "step": 0,
        "lin_vel_x": (-0.4, 0.4),
        "lin_vel_y": (0.0, 0.0),
        "ang_vel_z": (-0.3, 0.3),
      }
    ]

  return cfg


def unitree_b2w_deployable_precision_ablation_env_cfg(
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Fine-tuning task that tightens low-speed longitudinal tracking."""
  cfg = unitree_b2w_deployable_stable_ablation_env_cfg(play=play)

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, unitree_mdp.ModeVelocityCommandCfg)
  twist_cmd.rel_zero_envs = 0.15
  twist_cmd.rel_straight_envs = 0.60
  twist_cmd.rel_yaw_envs = 0.15
  cfg.rewards["track_linear_velocity"].weight = 6.0
  cfg.rewards["track_linear_velocity"].params["std"] = math.sqrt(0.1)

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
