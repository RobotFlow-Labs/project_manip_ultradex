"""Wrapper around cuRobo motion planning utilities (§IV.B).

Connects to the cuRobo repo at /mnt/forge-data/repos/curobo/ for
GPU-accelerated motion planning. Falls back gracefully when deps are missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from anima_manip_ultradex.config import ModuleConfig

CUROBO_REPO = Path("/mnt/forge-data/repos/curobo")
CUROBO_SRC = CUROBO_REPO / "src"


class CuroboAdapter:
    """Planner adapter that is import-safe and explicit about CUDA requirements."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.reference_file = cfg.reference_repo_root / "util" / "curobo_util.py"
        self._planner = None

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "curobo_repo": CUROBO_REPO.exists(),
            "curobo": importlib.util.find_spec("curobo") is not None or CUROBO_SRC.exists(),
            "torch": importlib.util.find_spec("torch") is not None,
        }

    def require_runtime(self) -> None:
        missing = []
        avail = self.availability()
        if not avail["curobo_repo"] and not avail["curobo"]:
            missing.append("curobo (repo or package)")
        if not avail["torch"]:
            missing.append("torch")
        if missing:
            raise RuntimeError(
                "cuRobo runtime is unavailable. Missing: "
                + ", ".join(missing)
                + f". Install from {CUROBO_REPO} or pip install curobo."
            )

    def _ensure_curobo_path(self) -> None:
        """Add cuRobo source to sys.path if not already importable."""
        if importlib.util.find_spec("curobo") is None and CUROBO_SRC.exists():
            src_str = str(CUROBO_SRC)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    def plan_trajectory(
        self,
        start_config: list[float],
        goal_config: list[float],
        *,
        robot_config: str | None = None,
    ) -> Any:
        """Plan a collision-free trajectory using cuRobo.

        Args:
            start_config: Joint angles at start [6 DoF per arm].
            goal_config: Joint angles at goal [6 DoF per arm].
            robot_config: Optional path to robot YAML config.

        Returns:
            Trajectory from cuRobo planner.
        """
        self.require_runtime()
        self._ensure_curobo_path()

        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuRobo requires CUDA. No GPU available.")

        from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig

        if self._planner is None:
            config = MotionGenConfig.load_from_robot_config(
                robot_config or "ur5e.yml",
                interpolation_dt=0.02,
            )
            self._planner = MotionGen(config)
            self._planner.warmup()

        start_t = torch.tensor(start_config, dtype=torch.float32, device="cuda").unsqueeze(0)
        goal_t = torch.tensor(goal_config, dtype=torch.float32, device="cuda").unsqueeze(0)

        return self._planner.plan_single(start_t, goal_t)
