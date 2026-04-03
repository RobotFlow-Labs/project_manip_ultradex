"""ANIMA-facing scene adapter for the public UltraDexGrasp environment."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from anima_manip_ultradex.config import ModuleConfig


class SceneEnvAdapter:
    """Reference-environment facade that avoids importing SAPIEN on unsupported hosts."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.reference_file = cfg.reference_repo_root / "env" / "base_env.py"

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "sapien": importlib.util.find_spec("sapien") is not None,
            "pytorch3d": importlib.util.find_spec("pytorch3d") is not None,
        }

    def fixture_paths(self) -> dict[str, Path]:
        return {
            "paper_pdf": self.cfg.paper_pdf,
            "bowl_mesh": self.cfg.bowl_mesh,
            "reference_env": self.reference_file,
        }
