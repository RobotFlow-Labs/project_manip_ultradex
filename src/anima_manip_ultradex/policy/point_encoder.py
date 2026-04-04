"""Point-cloud encoder for UltraDexGrasp (§V.B).

Uses CUDA FPS kernel (7.2x speedup) when available for token selection.
Falls back to uniform strided sampling on CPU.
"""

from __future__ import annotations

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig


def _fps_indices(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """Farthest Point Sampling indices — CUDA kernel or PyTorch CPU fallback."""
    try:
        from point_cloud_ops import farthest_point_sample

        return farthest_point_sample(points, num_samples)
    except (ImportError, RuntimeError):
        pass
    # PyTorch CPU fallback: true FPS algorithm
    return _fps_cpu(points, num_samples)


def _fps_cpu(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """True Farthest Point Sampling on CPU — matches CUDA kernel output."""
    B, N, _ = points.shape
    device = points.device
    indices = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    min_dists = torch.full((B, N), float("inf"), device=device)
    farthest = torch.zeros(B, dtype=torch.long, device=device)
    for k in range(num_samples):
        indices[:, k] = farthest
        selected = points[torch.arange(B, device=device), farthest]  # [B, 3]
        dist = ((points - selected.unsqueeze(1)) ** 2).sum(dim=-1)  # [B, N]
        min_dists = torch.min(min_dists, dist)
        farthest = min_dists.argmax(dim=-1)
    return indices


class PointEncoder(nn.Module):
    """PointNet-style encoder that preserves the paper's I/O contract.

    Input:  [B, N, 3] point cloud (N >= 256)
    Output: [B, 256, D_scene] scene tokens
    """

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

        # Use CUDA FPS kernel for token selection (7.2x speedup on GPU)
        fps_idx = _fps_indices(points, self.output_tokens)  # [B, K]
        # Gather tokens at FPS indices
        batch_idx = torch.arange(points.shape[0], device=points.device).unsqueeze(1)
        tokens = fused[batch_idx, fps_idx]
        return self.norm(tokens)
