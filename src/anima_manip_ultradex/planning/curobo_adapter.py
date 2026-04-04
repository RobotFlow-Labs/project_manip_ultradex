"""Lazy wrapper around the paper's cuRobo planning utilities."""

from __future__ import annotations

import importlib.util

from anima_manip_ultradex.config import ModuleConfig


class CuroboAdapter:
    """Planner adapter that is import-safe on macOS and explicit about CUDA requirements."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.reference_file = cfg.reference_repo_root / "util" / "curobo_util.py"

    def availability(self) -> dict[str, bool]:
        return {
            "reference_file": self.reference_file.exists(),
            "curobo": importlib.util.find_spec("curobo") is not None,
            "torch": importlib.util.find_spec("torch") is not None,
        }

    def require_runtime(self) -> None:
        missing = [name for name, ok in self.availability().items() if not ok]
        if missing:
            raise RuntimeError(
                "cuRobo runtime is unavailable. Missing: "
                + ", ".join(missing)
                + ". Install CUDA-side reference dependencies before motion planning."
            )

    def setup(self):
        self.require_runtime()
        raise NotImplementedError(
            "CUDA-side cuRobo planner integration will be implemented in PRD-02."
        )
