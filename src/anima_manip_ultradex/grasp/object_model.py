"""Object asset helpers for grasp synthesis and demo generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from anima_manip_ultradex.config import ModuleConfig
from anima_manip_ultradex.grasp.types import GraspStrategy


ALL_GRASP_STRATEGIES: tuple[GraspStrategy, ...] = (
    "pinch",
    "tripod",
    "whole_hand",
    "bimanual",
)


@dataclass(frozen=True)
class ObjectAssetSpec:
    object_id: str
    mesh_path: Path
    scale: float = 1.0
    category: str = "unknown"
    supported_strategies: tuple[GraspStrategy, ...] = ALL_GRASP_STRATEGIES

    def exists(self) -> bool:
        return self.mesh_path.exists()


def load_bowl_fixture(cfg: ModuleConfig) -> ObjectAssetSpec:
    return ObjectAssetSpec(
        object_id="bowl",
        mesh_path=cfg.bowl_mesh,
        scale=1.0,
        category="container",
        supported_strategies=ALL_GRASP_STRATEGIES,
    )
