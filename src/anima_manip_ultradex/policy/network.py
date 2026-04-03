"""End-to-end UltraDexGrasp policy scaffold."""

from __future__ import annotations

import torch
from torch import nn

from anima_manip_ultradex.config import ModuleConfig
from anima_manip_ultradex.policy.action_head import BoundedGaussianActionHead, PolicyOutput
from anima_manip_ultradex.policy.action_queries import ActionQueryBank
from anima_manip_ultradex.policy.point_encoder import PointEncoder
from anima_manip_ultradex.policy.transformer import DecoderOnlyTransformer


class UltraDexPolicy(nn.Module):
    def __init__(
        self,
        cfg: ModuleConfig,
        d_model: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = PointEncoder(cfg, d_scene=d_model)
        self.queries = ActionQueryBank(cfg, d_model=d_model)
        self.backbone = DecoderOnlyTransformer(
            cfg,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
        )
        self.head = BoundedGaussianActionHead(cfg, d_model=d_model)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def forward(
        self,
        scene_pc: torch.Tensor,
        action_targets: torch.Tensor | None = None,
    ) -> PolicyOutput:
        scene_tokens = self.encoder(scene_pc)
        query_tokens = self.queries(scene_pc.shape[0])
        fused_tokens = self.backbone(query_tokens, scene_tokens)
        return self.head(fused_tokens, targets=action_targets)
