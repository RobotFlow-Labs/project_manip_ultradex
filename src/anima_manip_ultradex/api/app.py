"""Minimal serving surface for gate-level infra checks."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.device import detect_backend

cfg = load_module_config()
app = FastAPI(title="MANIP-ULTRADEX")


class PredictRequest(BaseModel):
    raw_points: list[list[float]]
    robot_points: list[list[float]] | None = None
    apply_sor: bool = False


@lru_cache(maxsize=1)
def _get_runner():
    try:
        from anima_manip_ultradex.inference.runner import UltraDexRunner
    except ModuleNotFoundError:
        return None

    checkpoint_path = cfg.policy_checkpoint_path if cfg.policy_checkpoint_path.exists() else None
    return UltraDexRunner(cfg, checkpoint_path=checkpoint_path, d_model=64, num_heads=4, num_layers=2)


@app.get("/healthz")
def healthz() -> dict[str, object]:
    backend = detect_backend(cfg.compute.backend)
    return {"ok": True, "module": cfg.project.codename, "backend": backend.backend}


@app.get("/readyz")
def readyz() -> dict[str, object]:
    runner = _get_runner()
    return {
        "ready": runner is not None,
        "checkpoint_loaded": False if runner is None else runner.checkpoint_loaded,
        "reason": None if runner is not None else "Install training extras to enable inference.",
    }


@app.post("/predict")
def predict(request: PredictRequest) -> dict[str, object]:
    runner = _get_runner()
    if runner is None:
        raise HTTPException(status_code=503, detail="Inference dependencies are not installed.")

    result = runner.predict(
        np.asarray(request.raw_points, dtype=np.float32),
        robot_pc=None
        if request.robot_points is None
        else np.asarray(request.robot_points, dtype=np.float32),
        apply_sor=request.apply_sor,
    )
    return {
        "checkpoint_loaded": result.checkpoint_loaded,
        "action_vector": result.action_vector.tolist(),
        "arm_actions": result.arm_actions.tolist(),
        "hand_actions": result.hand_actions.tolist(),
    }
