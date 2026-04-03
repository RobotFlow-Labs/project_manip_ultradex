"""Lazy wrapper around the paper's BODex-based grasp synthesizer."""

from __future__ import annotations

import importlib.util

from anima_manip_ultradex.config import ModuleConfig


class BodexAdapter:
    """Thin wrapper that keeps macOS imports safe until CUDA-side deps exist."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.reference_file = cfg.reference_repo_root / "util" / "bodex_util.py"

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "bodex": importlib.util.find_spec("bodex") is not None,
            "trimesh": importlib.util.find_spec("trimesh") is not None,
        }

    def require_runtime(self) -> None:
        missing = [name for name, ok in self.availability().items() if not ok]
        if missing:
            raise RuntimeError(
                "BODex runtime is unavailable. Missing: "
                + ", ".join(missing)
                + ". Install CUDA-side reference dependencies before grasp synthesis."
            )

    def synthesize(self, object_path: str, object_pose: list[float], object_scale: float):
        self.require_runtime()
        raise NotImplementedError("CUDA-side BODex execution will be implemented in PRD-02.")
