"""Learnable action-query tokens for dual-arm dexterous control."""

from __future__ import annotations

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig


class ActionQueryBank(nn.Module):
    def __init__(self, cfg: ModuleConfig, d_model: int = 128) -> None:
        super().__init__()
        self.num_queries = cfg.paper.action_query_tokens
        self.d_model = d_model
        self.query_tokens = nn.Parameter(torch.randn(self.num_queries, d_model) * 0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        return self.query_tokens.unsqueeze(0).expand(batch_size, -1, -1)
