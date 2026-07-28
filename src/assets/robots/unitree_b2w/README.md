# Unitree B2W asset provenance

The B2W MJCF and mesh assets in this directory are adapted from the official
[`unitreerobotics/unitree_mujoco`](https://github.com/unitreerobotics/unitree_mujoco)
`unitree_robots/b2w` model at commit
`ae6a8403e272733e9996ef59990880330496177f`. The copied upstream license is
retained in `UNITREE_MUJOCO_LICENSE`.

The model was cross-checked against
[`unitreerobotics/unitree_ros`](https://github.com/unitreerobotics/unitree_ros)
`robots/b2w_description` at commit
`aa0f5c68b5aba347bad409e71b6430407da758d7`, especially its 12 leg joints, four
continuous wheel joints, joint limits, effort limits, inertial values, and
wheel axes.

Changes for mjlab:

- replaced the XML torque motors with mjlab position actuators for the legs and
  bounded XML velocity actuators for the continuous wheel joints;
- named all collision geoms and added wheel-center sites;
- exposed the standard mjlab IMU, velocimeter, accelerometer, and angular
  momentum sensors;
- separated visual wheel meshes from wheel collision meshes.
