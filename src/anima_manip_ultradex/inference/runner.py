"""Checkpoint-backed inference runner."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from anima_manip_ultradex.config import ModuleConfig
from anima_manip_ultradex.inference.postprocess import SplitActionResult, split_actions
from anima_manip_ultradex.inference.preprocess import build_scene_input
from anima_manip_ultradex.policy.network import UltraDexPolicy


@dataclass
class InferenceResult:
    scene_input: torch.Tensor
    action_vector: torch.Tensor
    arm_actions: torch.Tensor
    hand_actions: torch.Tensor
    checkpoint_loaded: bool


class UltraDexRunner:
    def __init__(
        self,
        cfg: ModuleConfig,
        checkpoint_path: str | Path | None = None,
        *,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg
        self.device = torch.device(device)
        self.policy = UltraDexPolicy(
            cfg,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        ).to(self.device)
        self.checkpoint_loaded = False
        if checkpoint_path is not None and Path(checkpoint_path).exists():
            self.load_checkpoint(checkpoint_path)
        self.policy.eval()

    def load_checkpoint(self, checkpoint_path: str | Path) -> dict[str, list[str]]:
        payload = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(payload, dict):
            state_dict = payload.get("state_dict", payload)
        else:
            state_dict = payload

        current_state = self.policy.state_dict()
        remapped = {}
        for key, value in state_dict.items():
            normalized_key = key.removeprefix("model.").removeprefix("policy.")
            if normalized_key in current_state:
                remapped[normalized_key] = value

        missing, unexpected = self.policy.load_state_dict(remapped, strict=False)
        self.checkpoint_loaded = True
        return {"missing": list(missing), "unexpected": list(unexpected)}

    @torch.no_grad()
    def predict(
        self,
        raw_pc: Any,
        robot_pc: Any | None = None,
        *,
        apply_sor: bool = False,
    ) -> InferenceResult:
        scene_input = build_scene_input(
            raw_pc,
            robot_pc=robot_pc,
            apply_sor=apply_sor,
            target_points=self.cfg.paper.policy_input_points,
        ).to(self.device)
        output = self.policy(scene_input)
        actions: SplitActionResult = split_actions(output.sample)
        return InferenceResult(
            scene_input=scene_input.cpu(),
            action_vector=actions.action_vector.cpu(),
            arm_actions=actions.arm_actions.cpu(),
            hand_actions=actions.hand_actions.cpu(),
            checkpoint_loaded=self.checkpoint_loaded,
        )
