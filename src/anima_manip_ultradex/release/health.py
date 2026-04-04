"""Readiness and graceful degradation rules for production deployment."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HealthCheck:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class ProductionHealth:
    checks: list[HealthCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def critical_passed(self) -> bool:
        critical = {"torch_available", "model_loadable", "inference_runnable"}
        return all(c.passed for c in self.checks if c.name in critical)

    def summary(self) -> dict[str, object]:
        return {
            "all_passed": self.all_passed,
            "critical_passed": self.critical_passed,
            "checks": [
                {"name": c.name, "passed": c.passed, "detail": c.detail} for c in self.checks
            ],
        }


def run_health_checks(checkpoint_path: str | Path | None = None) -> ProductionHealth:
    health = ProductionHealth()

    # Check torch
    torch_ok = importlib.util.find_spec("torch") is not None
    health.checks.append(HealthCheck("torch_available", torch_ok))

    # Check CUDA
    cuda_ok = False
    if torch_ok:
        import torch

        cuda_ok = torch.cuda.is_available()
    health.checks.append(
        HealthCheck(
            "cuda_available",
            cuda_ok,
            detail="Required for full performance" if not cuda_ok else "",
        )
    )

    # Check model loadable
    model_ok = False
    if torch_ok:
        try:
            from anima_manip_ultradex.config import load_module_config
            from anima_manip_ultradex.policy.network import UltraDexPolicy

            cfg = load_module_config()
            policy = UltraDexPolicy(cfg, d_model=128, num_heads=4, num_layers=2)
            model_ok = policy.parameter_count > 0
        except Exception as e:
            health.checks.append(HealthCheck("model_loadable", False, str(e)))
    health.checks.append(HealthCheck("model_loadable", model_ok))

    # Check checkpoint exists
    ckpt_ok = checkpoint_path is not None and Path(checkpoint_path).exists()
    health.checks.append(
        HealthCheck(
            "checkpoint_exists",
            ckpt_ok,
            detail="No checkpoint — running with random weights" if not ckpt_ok else "",
        )
    )

    # Check ONNX runtime
    onnx_ok = importlib.util.find_spec("onnxruntime") is not None
    health.checks.append(HealthCheck("onnxruntime_available", onnx_ok))

    # Check inference runnable
    inference_ok = False
    if torch_ok and model_ok:
        try:
            from anima_manip_ultradex.inference.runner import UltraDexRunner

            cfg = load_module_config()
            runner = UltraDexRunner(cfg, d_model=128, num_heads=4, num_layers=2, device="cpu")
            import numpy as np

            pts = np.random.default_rng(0).normal(size=(2048, 3)).astype(np.float32)
            result = runner.predict(pts)
            inference_ok = result.action_vector.shape[-1] == 36
        except Exception as e:
            health.checks.append(HealthCheck("inference_runnable", False, str(e)))
    health.checks.append(HealthCheck("inference_runnable", inference_ok))

    return health
