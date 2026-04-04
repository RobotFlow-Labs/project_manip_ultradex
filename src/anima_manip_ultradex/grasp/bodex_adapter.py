"""Wrapper around BODex-based grasp synthesizer (§IV.A).

Connects to the BODex_api repo at /mnt/forge-data/repos/BODex_api/ for
optimization-based grasp synthesis. Falls back gracefully when deps are missing.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from anima_manip_ultradex.config import ModuleConfig

BODEX_REPO = Path("/mnt/forge-data/repos/BODex_api")
BODEX_SRC = BODEX_REPO / "src"


class BodexAdapter:
    """Thin wrapper that keeps imports safe until CUDA-side deps exist."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.reference_file = cfg.reference_repo_root / "util" / "bodex_util.py"
        self._synthesizer = None

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "bodex_repo": BODEX_REPO.exists(),
            "bodex": importlib.util.find_spec("bodex") is not None or BODEX_SRC.exists(),
            "trimesh": importlib.util.find_spec("trimesh") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
        }

    def require_runtime(self) -> None:
        missing = []
        avail = self.availability()
        if not avail["bodex_repo"] and not avail["bodex"]:
            missing.append("bodex (repo or package)")
        if not avail["trimesh"]:
            missing.append("trimesh")
        if not avail["torch"]:
            missing.append("torch")
        if missing:
            raise RuntimeError(
                "BODex runtime is unavailable. Missing: "
                + ", ".join(missing)
                + f". Install from {BODEX_REPO} or pip install bodex."
            )

    def _ensure_bodex_path(self) -> None:
        """Add BODex source to sys.path if not already importable."""
        if importlib.util.find_spec("bodex") is None and BODEX_SRC.exists():
            src_str = str(BODEX_SRC)
            if src_str not in sys.path:
                sys.path.insert(0, src_str)

    def synthesize(
        self,
        object_path: str,
        object_pose: list[float],
        object_scale: float,
        config_path: str | None = None,
    ) -> Any:
        """Synthesize grasp candidates for an object using BODex optimization.

        Args:
            object_path: Path to object directory (contains mesh/, urdf/, info/).
            object_pose: [x, y, z, qw, qx, qy, qz] pose of object.
            object_scale: Scale factor for the object mesh.
            config_path: Optional path to BODex YAML config.

        Returns:
            Grasp results from BODex solver.
        """
        self.require_runtime()
        self._ensure_bodex_path()

        if self._synthesizer is None:
            from bodex.wrap.reacher.grasp_solver import GraspSolver  # noqa: F401

            # Use default config from BODex repo if not specified
            if config_path is None:
                config_path = str(BODEX_REPO / "example_grasp" / "xhand_left.yaml")
            self._synthesizer = _LazyGraspSynthesizer(config_path)

        return self._synthesizer.synthesize(object_path, object_pose, object_scale)


class _LazyGraspSynthesizer:
    """Deferred initialization wrapper for BODex GraspSynthesizer."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        self._inner = None

    def _init(self):
        # Import at call time so module loads without CUDA
        repo_str = str(BODEX_REPO)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)
        from synthesize_grasp import GraspSynthesizer

        self._inner = GraspSynthesizer(self.config_path)

    def synthesize(self, object_path: str, object_pose: list[float], object_scale: float):
        if self._inner is None:
            self._init()
        return self._inner.synthesize_grasp(object_path, object_pose, object_scale)
