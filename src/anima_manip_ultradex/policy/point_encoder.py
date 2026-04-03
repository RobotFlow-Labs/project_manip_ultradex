"""Point-cloud encoder scaffold for UltraDexGrasp."""

from __future__ import annotations

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig


class PointEncoder(nn.Module):
    """PointNet-style encoder that preserves the paper's I/O contract."""

    def __init__(self, cfg: ModuleConfig, d_scene: int = 128) -> None:
        super().__init__()
        self.expected_points = cfg.paper.policy_input_points
        self.output_tokens = cfg.paper.abstraction_output_points
        self.scene_dim = d_scene
        self.point_mlp = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, d_scene),
            nn.GELU(),
        )
        self.token_fusion = nn.Sequential(
            nn.Linear(d_scene * 2, d_scene),
            nn.GELU(),
            nn.Linear(d_scene, d_scene),
        )
        self.norm = nn.LayerNorm(d_scene)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError("Expected points with shape [B, N, 3].")
        if points.shape[1] < self.output_tokens:
            raise ValueError(
                f"Need at least {self.output_tokens} input points, got {points.shape[1]}."
            )

        features = self.point_mlp(points)
        global_context = features.mean(dim=1, keepdim=True).expand_as(features)
        fused = self.token_fusion(torch.cat([features, global_context], dim=-1))

        index = torch.linspace(
            0,
            points.shape[1] - 1,
            self.output_tokens,
            device=points.device,
        ).round().long()
        tokens = fused.index_select(1, index)
        return self.norm(tokens)
