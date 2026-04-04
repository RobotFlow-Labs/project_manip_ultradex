"""ANIMA-facing scene adapter for UltraDexGrasp simulation environments.

Supports:
- IsaacGym Preview 4 (paper-compatible, at /mnt/forge-data/shared_infra/simulators/isaacgym/)
- Isaac Lab (future-proof, at /mnt/forge-data/shared_infra/simulators/IsaacLab/)
- Mock sim (CPU-only testing without GPU sim dependencies)
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Literal

from anima_manip_ultradex.config import ModuleConfig

SimBackend = Literal["isaacgym", "isaac_lab", "mock"]

# Shared simulator paths
ISAACGYM_ROOT = Path("/mnt/forge-data/shared_infra/simulators/isaacgym")
ISAAC_LAB_ROOT = Path("/mnt/forge-data/shared_infra/simulators/IsaacLab")


class SceneEnvAdapter:
    """Reference-environment facade with multi-backend sim support."""

    def __init__(self, cfg: ModuleConfig, sim_backend: SimBackend = "mock") -> None:
        self.cfg = cfg
        self.sim_backend = sim_backend
        self.reference_file = cfg.reference_repo_root / "env" / "base_env.py"

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "isaacgym_installed": _has_isaacgym(),
            "isaacgym_on_disk": ISAACGYM_ROOT.exists(),
            "isaac_lab_installed": _has_isaac_lab(),
            "isaac_lab_on_disk": ISAAC_LAB_ROOT.exists(),
            "sapien": _safe_find("sapien"),
            "pytorch3d": _safe_find("pytorch3d"),
        }

    def fixture_paths(self) -> dict[str, Path]:
        return {
            "paper_pdf": self.cfg.paper_pdf,
            "bowl_mesh": self.cfg.bowl_mesh,
            "reference_env": self.reference_file,
            "isaacgym_root": ISAACGYM_ROOT,
            "isaac_lab_root": ISAAC_LAB_ROOT,
        }

    def select_backend(self) -> SimBackend:
        """Auto-select the best available simulation backend."""
        if self.sim_backend != "mock":
            return self.sim_backend
        if _has_isaacgym():
            return "isaacgym"
        if _has_isaac_lab():
            return "isaac_lab"
        return "mock"

    def create_env(self, num_envs: int = 1):
        """Create a simulation environment using the selected backend."""
        backend = self.select_backend()
        if backend == "isaacgym":
            return self._create_isaacgym_env(num_envs)
        if backend == "isaac_lab":
            return self._create_isaac_lab_env(num_envs)
        return self._create_mock_env(num_envs)

    def _create_isaacgym_env(self, num_envs: int):
        """Paper-compatible IsaacGym environment."""
        raise NotImplementedError(
            "IsaacGym env creation requires IsaacGym Preview 4 Python package. "
            f"Install from {ISAACGYM_ROOT}/python/ if available."
        )

    def _create_isaac_lab_env(self, num_envs: int):
        """Future-proof Isaac Lab environment with IsaacGym compat layer."""
        raise NotImplementedError(
            "Isaac Lab env creation requires Isaac Sim + Isaac Lab. "
            f"Install from {ISAAC_LAB_ROOT}/ if available."
        )

    def _create_mock_env(self, num_envs: int):
        """CPU-only mock environment for testing without GPU sim."""
        return MockSimEnv(self.cfg, num_envs=num_envs)


class MockSimEnv:
    """Lightweight mock sim environment for testing grasp pipelines without IsaacGym."""

    def __init__(self, cfg: ModuleConfig, num_envs: int = 1) -> None:
        self.cfg = cfg
        self.num_envs = num_envs
        self._step_count = 0

    def reset(self) -> dict:
        import numpy as np

        self._step_count = 0
        return {
            "scene_pc": np.random.default_rng(42)
            .normal(size=(self.num_envs, self.cfg.paper.policy_input_points, 3))
            .astype(np.float32),
            "robot_state": np.zeros((self.num_envs, 36), dtype=np.float32),
        }

    def step(self, actions) -> tuple[dict, float, bool, dict]:
        import numpy as np

        self._step_count += 1
        obs = {
            "scene_pc": np.random.default_rng(self._step_count)
            .normal(size=(self.num_envs, self.cfg.paper.policy_input_points, 3))
            .astype(np.float32),
            "robot_state": np.zeros((self.num_envs, 36), dtype=np.float32),
        }
        reward = 0.0
        done = self._step_count >= 100
        info = {"step": self._step_count, "backend": "mock"}
        return obs, reward, done, info


def _safe_find(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ModuleNotFoundError, ValueError):
        return False


def _has_isaacgym() -> bool:
    return _safe_find("isaacgym")


def _has_isaac_lab() -> bool:
    return _safe_find("omni.isaac.lab") or _safe_find("isaaclab")
