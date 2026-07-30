# B2-W 策略观测与动作维度契约

本文档是 B2-W 训练、ONNX 导出、`unitree_mujoco` 验证和真机部署共同遵守的
张量契约。修改观测项、历史长度、关节顺序或动作缩放时，必须同步更新本文档和
`deploy/robots/b2w/config/policy/velocity/v0/params/deploy.yaml`。

## “新增 33 维”的准确含义

`275 - 242 = 33` 只是两个不同 Actor 契约之间的**净差值**，并不表示增加了
33 个新的传感器量：

```text
Rough/Stairs Actor = 55 个单帧本体量 + 187 点高度图 = 242
Deployable Actor   = 55 个单帧本体量 × 5 帧历史       = 275

净变化 = 增加 4 个历史帧（4 × 55 = 220）
       - 删除 Actor 高度图（187）
       = +33
```

Deployable Actor **不包含**高度图、精确机身线速度、轮端接触或三维接触力。
这些仿真特权量只交给 Critic，不进入导出的 Actor。

## 各任务总维度

| 任务 | Actor | Critic | Action | 说明 |
|---|---:|---:|---:|---|
| `Unitree-B2W-Flat` | 55 | 82 | 16 | 单帧本体量；无高度图 |
| `Unitree-B2W-Rough` | 242 | 269 | 16 | 55 + 187 点高度图 |
| `Unitree-B2W-Stairs` | 242 | 269 | 16 | 与 Rough 相同的张量契约 |
| `Unitree-B2W-Privileged` | 261 | 269 | 16 | Rough Actor 再增加 19 个特权量 |
| `Unitree-B2W-Deployable` | 275 | 269 | 16 | 55 个可部署量的 5 帧历史 |
| `Unitree-B2W-Deployable-Stage2` | 275 | 269 | 16 | 与 Deployable 完全兼容 |
| Deployable Flat Ablation 系列 | 275 | 269 | 16 | 仅奖励/地形不同，契约不变 |

Actor MLP 为 `input → 512 → 256 → 128 → 16`，Critic MLP 为
`input → 512 → 256 → 128 → 1`。Actor 和 Critic 都启用训练期经验归一化
（`obs_normalization=True`）。

20 kg 负载、地形类型和并行环境数不会改变任何张量维度。

## 坐标系、时间和通用顺序

- 控制频率：50 Hz，`step_dt = 0.02 s`。
- 机身角速度、机身线速度和投影重力均使用机身/IMU 局部坐标系：
  `x` 向前、`y` 向左、`z` 向上。
- 腿/轮的模型顺序统一为：`FL, FR, RL, RR`。
- Deployable 五帧历史按**观测项优先（term-major）**展平。
- 每个观测项内部按时间从旧到新排列：`t-4, t-3, t-2, t-1, t`。
- 环境重置后的第一帧会回填全部五个历史槽，不使用未初始化值。

## 单帧可部署观测：55 维

下面的 `[start:end]` 使用 Python 半开区间；例如 `[0:3]` 是索引
`0, 1, 2`。

| 单帧索引 | 项 | 维数 | 分量顺序 | 单位/变换 | Actor 训练噪声 |
|---|---|---:|---|---|---|
| `[0:3]` | `base_ang_vel` | 3 | `ωx, ωy, ωz` | rad/s，IMU 局部角速度 | 均匀 `[-0.2, 0.2]` |
| `[3:6]` | `projected_gravity` | 3 | `gx, gy, gz` | 重力单位向量投影到机身系 | 均匀 `[-0.05, 0.05]` |
| `[6:9]` | `command` | 3 | `vx_cmd, vy_cmd, wz_cmd` | m/s, m/s, rad/s；B2-W 的 `vy_cmd=0` | 无 |
| `[9:11]` | `phase` | 2 | `sin(2πφ), cos(2πφ)` | 周期 0.6 s；静止指令时两项均为 0 | 无 |
| `[11:23]` | `joint_pos` | 12 | 见下表 | 腿关节相对默认位姿，rad | 均匀 `[-0.01, 0.01]` |
| `[23:39]` | `joint_vel` | 16 | 见下表 | 全部腿/轮关节速度，rad/s | 均匀 `[-1.5, 1.5]` |
| `[39:55]` | `actions` | 16 | 与 Action 顺序相同 | 上一控制周期的原始策略动作，无量纲 | 无 |

训练观测处理顺序为：`compute → noise → clip → scale → history`。
Play/评估模式关闭 Actor 噪声。

### 12 维腿关节顺序

| 局部索引 | 关节 |
|---:|---|
| 0, 1, 2 | `FL_hip`, `FL_thigh`, `FL_calf` |
| 3, 4, 5 | `FR_hip`, `FR_thigh`, `FR_calf` |
| 6, 7, 8 | `RL_hip`, `RL_thigh`, `RL_calf` |
| 9, 10, 11 | `RR_hip`, `RR_thigh`, `RR_calf` |

默认腿位姿依次为：

```text
FL [-0.1, 0.9, -1.8]
FR [ 0.1, 0.9, -1.8]
RL [-0.1, 0.9, -1.8]
RR [ 0.1, 0.9, -1.8]
```

### 16 维全关节速度顺序

| 局部索引 | 关节 |
|---:|---|
| 0, 1, 2, 3 | `FL_hip`, `FL_thigh`, `FL_calf`, `FL_wheel` |
| 4, 5, 6, 7 | `FR_hip`, `FR_thigh`, `FR_calf`, `FR_wheel` |
| 8, 9, 10, 11 | `RL_hip`, `RL_thigh`, `RL_calf`, `RL_wheel` |
| 12, 13, 14, 15 | `RR_hip`, `RR_thigh`, `RR_calf`, `RR_wheel` |

## Deployable Actor：275 维精确索引

Deployable 不是把 55 维整帧连续堆叠五次，而是先堆叠每个观测项的五帧历史，
再连接下一个观测项。

| Actor 索引 | 项 | 计算 | 帧内分量 |
|---|---|---:|---|
| `[0:15]` | `base_ang_vel` | `3 × 5` | 每帧 `ωx, ωy, ωz` |
| `[15:30]` | `projected_gravity` | `3 × 5` | 每帧 `gx, gy, gz` |
| `[30:45]` | `command` | `3 × 5` | 每帧 `vx_cmd, vy_cmd, wz_cmd` |
| `[45:55]` | `phase` | `2 × 5` | 每帧 `sin, cos` |
| `[55:115]` | `joint_pos` | `12 × 5` | 每帧 12 个腿关节 |
| `[115:195]` | `joint_vel` | `16 × 5` | 每帧 16 个全关节速度 |
| `[195:275]` | `actions` | `16 × 5` | 每帧 16 个上一周期动作 |

任意项的索引公式：

```text
global_index = term_start + history_slot * term_dim + component
history_slot = 0,1,2,3,4  对应  t-4,t-3,t-2,t-1,t
```

例如：

- `[0:3]` 是 `t-4` 的机身角速度；
- `[12:15]` 是当前 `t` 的机身角速度；
- `[55:67]` 是 `t-4` 的 12 个腿关节位置；
- `[103:115]` 是当前 `t` 的 12 个腿关节位置；
- `[259:275]` 是当前 `t` 使用的上一周期动作。

部署 YAML 使用名称 `velocity_commands` 和 `gait_phase`，分别对应训练配置中的
`command` 和 `phase`；数值和顺序必须保持一致。部署端不得启用
`use_gym_history`，否则会变成按时间帧优先排列，与训练 Actor 不兼容。

## Rough/Stairs Actor：242 维

| Actor 索引 | 项 | 维数 |
|---|---|---:|
| `[0:55]` | 上述单帧可部署观测 | 55 |
| `[55:242]` | `height_scan` | 187 |

### 187 点高度图的精确顺序

地形扫描为随底盘偏航角旋转的 `17 × 11` 平行射线网格：

- 前后 `x ∈ [-0.8, 0.8] m`，17 点，间距 0.1 m；
- 左右 `y ∈ [-0.5, 0.5] m`，11 点，间距 0.1 m；
- `x` 变化最快，`y` 变化最慢；
- `ix ∈ [0,16]`，`iy ∈ [0,10]`。

```text
scan_local_index = iy * 17 + ix
actor_index      = 55 + scan_local_index
x                = -0.8 + 0.1 * ix
y                = -0.5 + 0.1 * iy
```

每个原始值为 `sensor_z - hit_z`，单位 m；射线未命中时使用 5 m。进入网络前
乘以 `1/5 = 0.2`。Rough/Stairs Actor 训练时先加入
`[-0.1, 0.1] m` 均匀噪声，再乘 0.2，因此网络输入上的噪声幅值为
`[-0.02, 0.02]`。Critic 和 Privileged Actor 的高度图不加此噪声。

## Privileged Actor：261 维

Privileged Actor 在 Rough 的 242 维之后追加 19 维：

| Actor 索引 | 项 | 维数 | 分量/变换 |
|---|---|---:|---|
| `[0:242]` | Rough Actor | 242 | 单帧本体量 + 高度图 |
| `[242:245]` | `base_lin_vel` | 3 | 机身系 `vx, vy, vz`，m/s |
| `[245:249]` | `wheel_contact` | 4 | `FL, FR, RL, RR`，接触为 1，否则 0 |
| `[249:261]` | `wheel_contact_forces` | 12 | 每轮世界系 `Fx,Fy,Fz`，顺序 `FL,FR,RL,RR` |

接触力不是直接牛顿值，而是逐分量变换：

```text
sign(force) * log(1 + abs(force))
```

Privileged Actor 关闭全部观测腐化，用于能力上限验证，不用于真机部署。

## Critic：269 维

Rough、Stairs、Privileged、Deployable、Stage2 和 Deployable Ablation 的
Critic 都是单帧 269 维：

| Critic 索引 | 项 | 维数 | 分量/变换 |
|---|---|---:|---|
| `[0:3]` | `base_ang_vel` | 3 | `ωx,ωy,ωz` |
| `[3:6]` | `projected_gravity` | 3 | `gx,gy,gz` |
| `[6:9]` | `command` | 3 | `vx_cmd,vy_cmd,wz_cmd` |
| `[9:11]` | `phase` | 2 | `sin,cos` |
| `[11:23]` | `joint_pos` | 12 | 腿关节相对位置 |
| `[23:39]` | `joint_vel` | 16 | 全关节速度 |
| `[39:55]` | `actions` | 16 | 上一周期动作 |
| `[55:242]` | `height_scan` | 187 | 无噪声，乘 0.2 |
| `[242:245]` | `base_lin_vel` | 3 | IMU 局部 `vx,vy,vz`，m/s |
| `[245:249]` | `foot_height` | 4 | 轮心世界坐标高度，`FL,FR,RL,RR`，m |
| `[249:253]` | `foot_air_time` | 4 | 各轮当前离地时间，s |
| `[253:257]` | `foot_contact` | 4 | 各轮二值接触 |
| `[257:269]` | `foot_contact_forces` | 12 | 各轮世界系 `Fx,Fy,Fz`，带 `sign·log1p` 变换 |

Flat Critic 删除 `[55:242]` 的 187 点高度图，因此为 `269 - 187 = 82`
维；删除后后续项左移，Flat Critic 的特权尾部位于 `[55:82]`。

Critic 只在 PPO 训练时使用，不随 Actor ONNX 部署到机器人。

## Action：16 维

策略输出顺序固定为 12 个腿位置动作，再接 4 个轮速度动作：

| Action 索引 | 目标 |
|---:|---|
| 0, 1, 2 | `FL_hip`, `FL_thigh`, `FL_calf` 位置 |
| 3, 4, 5 | `FR_hip`, `FR_thigh`, `FR_calf` 位置 |
| 6, 7, 8 | `RL_hip`, `RL_thigh`, `RL_calf` 位置 |
| 9, 10, 11 | `RR_hip`, `RR_thigh`, `RR_calf` 位置 |
| 12 | `FL_wheel` 速度 |
| 13 | `FR_wheel` 速度 |
| 14 | `RL_wheel` 速度 |
| 15 | `RR_wheel` 速度 |

### 动作缩放

| 任务族 | Hip | Thigh/Calf | Wheel |
|---|---:|---:|---:|
| Flat/Rough/Stairs | `q_default + 0.25·a` | `q_default + 0.25·a` | `20·a rad/s` |
| Privileged/Deployable/Stage2/Ablation | `q_default + 0.125·a` | `q_default + 0.25·a` | `5·a rad/s` |

Deployable 原始动作裁剪到 `[-4,4]`，所以轮速目标最终限制为
`[-20,20] rad/s`；腿位置目标还会按各关节物理范围裁剪。

### Unitree SDK 电机映射

策略顺序与 Unitree `LowCmd` 电机编号不同。部署适配器采用：

```text
腿：policy [0..11] -> SDK [3,4,5, 0,1,2, 9,10,11, 6,7,8]
轮：policy [12..15] -> SDK [13,12,15,14]
```

即策略使用 `FL,FR,RL,RR`，SDK 使用 `FR,FL,RR,RL`。不得把后四维当成
位置目标；它们必须以 `kp=0` 写入轮电机速度目标 `dq`。

## 检查点兼容性

- Flat Actor：55 维；Rough/Stairs Actor：242 维。
- Privileged Actor：261 维。
- Deployable/Stage2/Ablation Actor：275 维。
- 只有输入维度、项顺序和历史布局完全一致的检查点才能严格续训或直接部署。
- `scripts/convert_b2w_flat_to_rough.py` 只处理已定义的 Flat→Rough
  `55→242` 扩维，不适用于 Privileged 或 Deployable。
- Stage2 与 Deployable/Ablation 保持 `275/269/16` 契约，可以在模型形状上续接；
  是否适合续训仍需同时核对动作缩放、奖励和训练配置。

## 权威实现位置

- 训练观测定义：`src/tasks/velocity/velocity_env_cfg.py`
- B2-W 任务变体：`src/tasks/velocity/config/b2w/env_cfgs.py`
- B2-W 关节/MJCF 顺序：`src/assets/robots/unitree_b2w/xmls/b2w.xml`
- 部署张量配置：
  `deploy/robots/b2w/config/policy/velocity/v0/params/deploy.yaml`
- C++ 观测历史实现：`deploy/include/isaaclab/manager/observation_manager.h`
- C++ 电机映射：`deploy/robots/b2w/src/State_RLBase.cpp`
