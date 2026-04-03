"""Postprocessing helpers for policy inference outputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SplitActionResult:
    action_vector: torch.Tensor
    arm_actions: torch.Tensor
    hand_actions: torch.Tensor


def split_actions(action_vector: torch.Tensor, clamp: float = 1.0) -> SplitActionResult:
    if action_vector.ndim == 1:
        action_vector = action_vector.unsqueeze(0)
    if action_vector.ndim != 2 or action_vector.shape[-1] != 36:
        raise ValueError("Expected action vector with shape [B, 36].")

    bounded = torch.clamp(action_vector, min=-clamp, max=clamp)
    arm_actions = bounded[:, :12].reshape(-1, 2, 6)
    hand_actions = bounded[:, 12:36].reshape(-1, 2, 12)
    return SplitActionResult(
        action_vector=bounded,
        arm_actions=arm_actions,
        hand_actions=hand_actions,
    )
