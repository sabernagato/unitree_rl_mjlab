# B2-W deployment adapter

This controller is the deployment counterpart of `Unitree-B2W-Deployable`.
It intentionally does not consume simulator-only base linear velocity, height
scan, or contact-force observations.

## Policy contract

- control rate: 50 Hz;
- actor input: 275 values (55 deployable values x five frames);
- actor output: 16 values;
- output 0-11: leg position actions;
- output 12-15: wheel velocity actions;
- raw policy actions are clipped to `[-4, 4]`;
- wheel targets are therefore limited to `[-20, 20] rad/s`.

The observation and action order is defined in
`config/policy/velocity/v0/params/deploy.yaml`. Do not enable
`use_gym_history`: the default term-major history layout matches the exported
MjLab policy.

For the complete index-by-index layout of all B2-W Actor, Critic, history, and
Action variants, see
[`doc/b2w_policy_dimensions.md`](../../../doc/b2w_policy_dimensions.md).

## Prepare the policy

Each training checkpoint also exports `policy.onnx` in its run directory. Copy
a validated deployable policy to:

```text
config/policy/velocity/v0/exported/policy.onnx
```

Then build:

```bash
cd deploy/robots/b2w
mkdir -p build
cd build
cmake ..
make -j
```

## unitree_mujoco validation

Configure the official `unitree_mujoco` simulator for:

```yaml
robot: "b2w"
domain_id: 1
interface: "lo"
```

Start the simulator, then run:

```bash
./build/b2w_ctrl --domain=1 --network=lo
```

Use `L2 + Up` to enter `FixStand`, then `R2 + A` to enable the policy.

## Real B2-W

Only proceed after the same ONNX has passed flat-ground, steering, stopping,
push-recovery, and terrain tests in `unitree_mujoco`.

The robot must be suspended for the first low-level test, its built-in motion
service must be released, and an operator must be ready to enter passive mode.
With the robot Ethernet interface substituted below:

```bash
./build/b2w_ctrl --domain=0 --network=enp5s0
```

The mixed controller sends position targets to the 12 leg motors and velocity
targets to the four wheel motors. The Unitree SDK mapping is implemented in
`src/State_RLBase.cpp`; do not reuse a quadruped adapter that writes all 16
outputs as joint positions.
