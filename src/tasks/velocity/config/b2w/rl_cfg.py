"""RL configuration for the Unitree B2W velocity task."""

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)


def unitree_b2w_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the PPO runner configuration for B2W velocity tracking."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="b2w_velocity",
    logger="tensorboard",
    upload_model=False,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=10001,
  )


def unitree_b2w_privileged_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO configuration for privileged wheel-legged capability training."""
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="b2w_privileged",
    logger="tensorboard",
    upload_model=False,
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=20000,
    clip_actions=4.0,
  )


def unitree_b2w_deployable_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO configuration for a deployable actor and privileged critic."""
  cfg = unitree_b2w_privileged_ppo_runner_cfg()
  cfg.experiment_name = "b2w_deployable"
  return cfg


def unitree_b2w_stage2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create PPO configuration for terrain/disturbance fine-tuning."""
  cfg = unitree_b2w_deployable_ppo_runner_cfg()
  cfg.experiment_name = "b2w_deployable_stage2"
  # 512 * 24 / 2 = 6144 samples per mini-batch, matching the validated
  # 1024-environment flat run while respecting the heightfield world limit.
  cfg.algorithm.num_mini_batches = 2
  cfg.max_iterations = 5000
  return cfg


def unitree_b2w_ablation_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create the short-run PPO configuration used for wheel-control A/B tests."""
  cfg = unitree_b2w_deployable_ppo_runner_cfg()
  cfg.experiment_name = "b2w_deployable_ablation"
  cfg.max_iterations = 500
  return cfg
