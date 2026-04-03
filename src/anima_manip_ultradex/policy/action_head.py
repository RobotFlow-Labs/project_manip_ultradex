"""Bounded Gaussian action head for dual-arm and dual-hand commands."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig
from anima_manip_ultradex.grasp.types import RobotEmbodiment


@dataclass
class PolicyOutput:
    arm_actions: torch.Tensor
    hand_actions: torch.Tensor
    action_mean: torch.Tensor
    action_log_std: torch.Tensor
    sample: torch.Tensor
    nll_loss: torch.Tensor | None


class BoundedGaussianActionHead(nn.Module):
    def __init__(
        self,
        cfg: ModuleConfig,
        d_model: int = 128,
        action_bound: float = 1.0,
    ) -> None:
        super().__init__()
        embodiment = RobotEmbodiment()
        self.layout = embodiment.action_query_layout
        self.action_bound = action_bound
        self.mean_heads = nn.ModuleList(nn.Linear(d_model, dims) for dims in self.layout)
        self.log_std_heads = nn.ModuleList(nn.Linear(d_model, dims) for dims in self.layout)
        self.total_dims = cfg.paper.arm_action_dims + cfg.paper.hand_action_dims

    def _gaussian_nll(
        self,
        targets: torch.Tensor,
        mean: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        variance = torch.exp(log_std * 2.0)
        return 0.5 * (((targets - mean) ** 2) / variance + 2.0 * log_std + torch.log(
            torch.full_like(log_std, 2.0 * pi)
        )).mean()

    def forward(
        self,
        query_tokens: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> PolicyOutput:
        if query_tokens.ndim != 3 or query_tokens.shape[1] != len(self.layout):
            raise ValueError(f"Expected query_tokens with shape [B, {len(self.layout)}, D].")

        mean_chunks = []
        log_std_chunks = []
        for query_index, dims in enumerate(self.layout):
            query = query_tokens[:, query_index, :]
            mean_chunks.append(torch.tanh(self.mean_heads[query_index](query)) * self.action_bound)
            log_std_chunks.append(self.log_std_heads[query_index](query).clamp(-5.0, 2.0))

        action_mean = torch.cat(mean_chunks, dim=-1)
        action_log_std = torch.cat(log_std_chunks, dim=-1)
        if action_mean.shape[-1] != self.total_dims:
            raise ValueError(f"Expected {self.total_dims} action dims, got {action_mean.shape[-1]}.")

        sample = action_mean
        arm_actions = torch.stack((action_mean[:, :6], action_mean[:, 6:12]), dim=1)
        hand_actions = torch.stack((action_mean[:, 12:24], action_mean[:, 24:36]), dim=1)

        nll_loss = None
        if targets is not None:
            if targets.shape != action_mean.shape:
                raise ValueError(
                    f"Targets must have shape {tuple(action_mean.shape)}, got {tuple(targets.shape)}."
                )
            nll_loss = self._gaussian_nll(targets, action_mean, action_log_std)

        return PolicyOutput(
            arm_actions=arm_actions,
            hand_actions=hand_actions,
            action_mean=action_mean,
            action_log_std=action_log_std,
            sample=sample,
            nll_loss=nll_loss,
        )
