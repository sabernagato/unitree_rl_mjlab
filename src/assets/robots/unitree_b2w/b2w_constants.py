"""Unitree B2W constants and articulation configuration."""

from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg, XmlVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

from src import SRC_PATH

##
# MJCF and assets.
##

B2W_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_b2w" / "xmls" / "b2w.xml"
)
assert B2W_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, B2W_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(B2W_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

B2W_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_hip_joint",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_thigh_joint",),
  stiffness=100.0,
  damping=5.0,
  effort_limit=200.0,
  armature=0.1,
)
B2W_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_calf_joint",),
  stiffness=160.0,
  damping=8.0,
  effort_limit=320.0,
  armature=0.1,
)
B2W_ACTUATOR_WHEEL = XmlVelocityActuatorCfg(
  target_names_expr=(".*_wheel_joint",),
)


##
# Initial state.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.58),
  joint_pos={
    ".*_hip_joint": 0.0,
    ".*_thigh_joint": 0.9,
    ".*_calf_joint": -1.8,
    ".*_wheel_joint": 0.0,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)


##
# Collision config.
##

_wheel_regex = "^[FR][LR]_wheel_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_wheel_regex: 3, ".*_collision": 1},
  priority={_wheel_regex: 1},
  friction={_wheel_regex: (0.4, 0.005, 0.0001)},
  solimp={_wheel_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)


##
# Final config.
##

B2W_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    B2W_ACTUATOR_HIP,
    B2W_ACTUATOR_THIGH,
    B2W_ACTUATOR_CALF,
    B2W_ACTUATOR_WHEEL,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_b2w_robot_cfg() -> EntityCfg:
  """Get a fresh B2W robot configuration instance."""
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=B2W_ARTICULATION,
  )


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_b2w_robot_cfg())
  viewer.launch(robot.spec.compile())
