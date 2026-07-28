"""Convert a trained B2W Flat checkpoint into a B2W Rough checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _reset_optimizer_state(optimizer_state: dict) -> dict:
  for state in optimizer_state.get("state", {}).values():
    for key, value in state.items():
      if torch.is_tensor(value):
        state[key] = torch.zeros_like(value)
  return optimizer_state


def _copy_common_state(source: dict, target: dict, skipped: set[str]) -> None:
  for key, value in source.items():
    if key in skipped:
      continue
    if key not in target:
      raise KeyError(f"Missing key in Rough checkpoint: {key}")
    if target[key].shape != value.shape:
      raise ValueError(
        f"Unexpected shape mismatch for {key}: "
        f"Flat {tuple(value.shape)}, Rough {tuple(target[key].shape)}"
      )
    target[key] = value.clone()


def _expand_actor(flat: dict, rough: dict) -> None:
  weight_key = "mlp.0.weight"
  normalizer_keys = {
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  }
  skipped = {weight_key, *normalizer_keys}
  _copy_common_state(flat, rough, skipped)

  flat_dim = flat[weight_key].shape[1]
  rough_dim = rough[weight_key].shape[1]
  if rough_dim <= flat_dim:
    raise ValueError(f"Rough actor input {rough_dim} must exceed Flat input {flat_dim}")

  expanded_weight = torch.zeros_like(rough[weight_key])
  expanded_weight[:, :flat_dim] = flat[weight_key]
  rough[weight_key] = expanded_weight

  for key in normalizer_keys:
    expanded = torch.ones_like(rough[key])
    if key.endswith("_mean"):
      expanded.zero_()
    expanded[..., :flat_dim] = flat[key]
    rough[key] = expanded


def _expand_critic(
  flat: dict,
  rough: dict,
  flat_actor_dim: int,
  rough_actor_dim: int,
) -> None:
  weight_key = "mlp.0.weight"
  normalizer_keys = {
    "obs_normalizer._mean",
    "obs_normalizer._var",
    "obs_normalizer._std",
  }
  skipped = {weight_key, *normalizer_keys}
  _copy_common_state(flat, rough, skipped)

  flat_dim = flat[weight_key].shape[1]
  rough_dim = rough[weight_key].shape[1]
  flat_privileged_dim = flat_dim - flat_actor_dim
  rough_privileged_dim = rough_dim - rough_actor_dim
  if flat_privileged_dim != rough_privileged_dim:
    raise ValueError(
      "Flat and Rough critic privileged observation sizes differ: "
      f"{flat_privileged_dim} != {rough_privileged_dim}"
    )

  expanded_weight = torch.zeros_like(rough[weight_key])
  expanded_weight[:, :flat_actor_dim] = flat[weight_key][:, :flat_actor_dim]
  expanded_weight[:, rough_actor_dim:] = flat[weight_key][:, flat_actor_dim:]
  rough[weight_key] = expanded_weight

  for key in normalizer_keys:
    expanded = torch.ones_like(rough[key])
    if key.endswith("_mean"):
      expanded.zero_()
    expanded[..., :flat_actor_dim] = flat[key][..., :flat_actor_dim]
    expanded[..., rough_actor_dim:] = flat[key][..., flat_actor_dim:]
    rough[key] = expanded


def convert(flat_path: Path, rough_template_path: Path, output_path: Path) -> None:
  flat_checkpoint = torch.load(flat_path, map_location="cpu", weights_only=False)
  rough_checkpoint = torch.load(
    rough_template_path, map_location="cpu", weights_only=False
  )

  flat_actor = flat_checkpoint["actor_state_dict"]
  rough_actor = rough_checkpoint["actor_state_dict"]
  flat_critic = flat_checkpoint["critic_state_dict"]
  rough_critic = rough_checkpoint["critic_state_dict"]

  flat_actor_dim = flat_actor["mlp.0.weight"].shape[1]
  rough_actor_dim = rough_actor["mlp.0.weight"].shape[1]
  _expand_actor(flat_actor, rough_actor)
  _expand_critic(flat_critic, rough_critic, flat_actor_dim, rough_actor_dim)

  rough_checkpoint["optimizer_state_dict"] = _reset_optimizer_state(
    rough_checkpoint["optimizer_state_dict"]
  )
  rough_checkpoint["iter"] = 0
  rough_checkpoint["infos"] = {
    **rough_checkpoint.get("infos", {}),
    "initialized_from": str(flat_path),
  }

  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(rough_checkpoint, output_path)
  print(
    f"Converted {flat_path} -> {output_path} "
    f"(actor {flat_actor_dim}->{rough_actor_dim}, "
    f"critic {flat_critic['mlp.0.weight'].shape[1]}"
    f"->{rough_critic['mlp.0.weight'].shape[1]})"
  )


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("flat_checkpoint", type=Path)
  parser.add_argument("rough_template_checkpoint", type=Path)
  parser.add_argument("output_checkpoint", type=Path)
  args = parser.parse_args()
  convert(
    args.flat_checkpoint,
    args.rough_template_checkpoint,
    args.output_checkpoint,
  )


if __name__ == "__main__":
  main()
