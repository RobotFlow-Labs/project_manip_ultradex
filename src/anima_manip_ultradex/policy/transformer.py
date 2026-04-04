"""Decoder-only transformer scaffold with query-to-scene attention."""

from __future__ import annotations

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig


class DecoderOnlyTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, num_queries: int = 4) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.norm_3 = nn.LayerNorm(d_model)
        # Cache causal mask as buffer — avoids allocation every forward pass
        self.register_buffer(
            "causal_mask",
            torch.ones(num_queries, num_queries, dtype=torch.bool).triu(diagonal=1),
        )

    def forward(self, query_tokens: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        causal_mask = self.causal_mask[: query_tokens.shape[1], : query_tokens.shape[1]]
        self_attended, _ = self.self_attn(
            query_tokens,
            query_tokens,
            query_tokens,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = self.norm_1(query_tokens + self_attended)
        cross_attended, _ = self.cross_attn(
            hidden,
            scene_tokens,
            scene_tokens,
            need_weights=False,
        )
        hidden = self.norm_2(hidden + cross_attended)
        return self.norm_3(hidden + self.feed_forward(hidden))


class DecoderOnlyTransformer(nn.Module):
    def __init__(
        self,
        cfg: ModuleConfig,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        if cfg.paper.action_query_tokens <= 0:
            raise ValueError("Config must expose a positive number of action query tokens.")
        num_queries = cfg.paper.action_query_tokens
        self.layers = nn.ModuleList(
            DecoderOnlyTransformerBlock(
                d_model=d_model, num_heads=num_heads, num_queries=num_queries
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, query_tokens: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        if query_tokens.ndim != 3 or scene_tokens.ndim != 3:
            raise ValueError("Expected query and scene tensors with shape [B, T, D].")
        if query_tokens.shape[0] != scene_tokens.shape[0]:
            raise ValueError("Query and scene batch sizes must match.")

        hidden = query_tokens
        for layer in self.layers:
            hidden = layer(hidden, scene_tokens)
        return self.output_norm(hidden)
