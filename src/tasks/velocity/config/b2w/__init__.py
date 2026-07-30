from mjlab.tasks.registry import register_mjlab_task
from src.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  unitree_b2w_deployable_base_ablation_env_cfg,
  unitree_b2w_deployable_env_cfg,
  unitree_b2w_deployable_precision_ablation_env_cfg,
  unitree_b2w_deployable_rolling_ablation_env_cfg,
  unitree_b2w_deployable_stage2_env_cfg,
  unitree_b2w_deployable_stable_ablation_env_cfg,
  unitree_b2w_flat_env_cfg,
  unitree_b2w_privileged_env_cfg,
  unitree_b2w_rough_env_cfg,
  unitree_b2w_stairs_env_cfg,
)
from .rl_cfg import (
  unitree_b2w_ablation_ppo_runner_cfg,
  unitree_b2w_deployable_ppo_runner_cfg,
  unitree_b2w_ppo_runner_cfg,
  unitree_b2w_privileged_ppo_runner_cfg,
  unitree_b2w_stage2_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable-Stage2",
  env_cfg=unitree_b2w_deployable_stage2_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_stage2_env_cfg(play=True),
  rl_cfg=unitree_b2w_stage2_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable-Flat-Base-Ablation",
  env_cfg=unitree_b2w_deployable_base_ablation_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_base_ablation_env_cfg(play=True),
  rl_cfg=unitree_b2w_ablation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable-Flat-Rolling-Ablation",
  env_cfg=unitree_b2w_deployable_rolling_ablation_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_rolling_ablation_env_cfg(play=True),
  rl_cfg=unitree_b2w_ablation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable-Flat-Stable-Ablation",
  env_cfg=unitree_b2w_deployable_stable_ablation_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_stable_ablation_env_cfg(play=True),
  rl_cfg=unitree_b2w_ablation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable-Flat-Precision-Ablation",
  env_cfg=unitree_b2w_deployable_precision_ablation_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_precision_ablation_env_cfg(play=True),
  rl_cfg=unitree_b2w_ablation_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Deployable",
  env_cfg=unitree_b2w_deployable_env_cfg(),
  play_env_cfg=unitree_b2w_deployable_env_cfg(play=True),
  rl_cfg=unitree_b2w_deployable_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Privileged",
  env_cfg=unitree_b2w_privileged_env_cfg(),
  play_env_cfg=unitree_b2w_privileged_env_cfg(play=True),
  rl_cfg=unitree_b2w_privileged_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Rough",
  env_cfg=unitree_b2w_rough_env_cfg(),
  play_env_cfg=unitree_b2w_rough_env_cfg(play=True),
  rl_cfg=unitree_b2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Stairs",
  env_cfg=unitree_b2w_stairs_env_cfg(),
  play_env_cfg=unitree_b2w_stairs_env_cfg(play=True),
  rl_cfg=unitree_b2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Unitree-B2W-Flat",
  env_cfg=unitree_b2w_flat_env_cfg(),
  play_env_cfg=unitree_b2w_flat_env_cfg(play=True),
  rl_cfg=unitree_b2w_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
