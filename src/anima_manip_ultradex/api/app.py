"""FastAPI serving surface for MANIP-ULTRADEX policy inference."""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException

from anima_manip_ultradex.api.schemas import (
    DualArmHandActionResponse,
    HealthResponse,
    InfoResponse,
    ReadyResponse,
    ScenePointCloudRequest,
)
from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.device import detect_backend
from anima_manip_ultradex.version import __version__

_start_time = time.monotonic()
cfg = load_module_config()
app = FastAPI(title="MANIP-ULTRADEX", version=__version__)


@lru_cache(maxsize=1)
def _get_runner():
    try:
        from anima_manip_ultradex.inference.runner import UltraDexRunner
    except ModuleNotFoundError:
        return None

    backend = detect_backend(cfg.compute.backend)
    device = "cuda" if backend.cuda_available else "cpu"
    checkpoint_path = cfg.policy_checkpoint_path if cfg.policy_checkpoint_path.exists() else None
    return UltraDexRunner(
        cfg,
        checkpoint_path=checkpoint_path,
        d_model=128,
        num_heads=4,
        num_layers=2,
        device=device,
    )


# --- Health / Ready / Info ---------------------------------------------------


@app.get("/health", response_model=HealthResponse)
@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    backend = detect_backend(cfg.compute.backend)
    return HealthResponse(
        ok=True,
        module=cfg.project.codename,
        backend=backend.backend,
        cuda_available=backend.cuda_available,
        gpu_count=_gpu_count(backend),
    )


@app.get("/ready", response_model=ReadyResponse)
@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    runner = _get_runner()
    if runner is None:
        return ReadyResponse(
            ready=False,
            reason="Install training extras to enable inference.",
        )
    return ReadyResponse(
        ready=True,
        checkpoint_loaded=runner.checkpoint_loaded,
        model_params=runner.policy.parameter_count,
    )


@app.get("/info", response_model=InfoResponse)
def info() -> InfoResponse:
    return InfoResponse(
        module=cfg.project.codename,
        version=__version__,
        paper=f"arXiv:{cfg.project.paper_arxiv}",
        embodiment="2xUR5e + 2xXHand-12DoF",
        action_dims=cfg.paper.arm_action_dims + cfg.paper.hand_action_dims,
        input_points=cfg.paper.policy_input_points,
    )


# --- Predict ------------------------------------------------------------------


@app.post("/predict", response_model=DualArmHandActionResponse)
def predict(request: ScenePointCloudRequest) -> DualArmHandActionResponse:
    runner = _get_runner()
    if runner is None:
        raise HTTPException(status_code=503, detail="Inference unavailable.")

    raw = np.asarray(request.raw_points, dtype=np.float32)
    robot = (
        np.asarray(request.robot_points, dtype=np.float32)
        if request.robot_points is not None
        else None
    )
    result = runner.predict(raw, robot_pc=robot, apply_sor=request.apply_sor)
    return DualArmHandActionResponse(
        checkpoint_loaded=result.checkpoint_loaded,
        action_vector=result.action_vector.tolist(),
        arm_actions=result.arm_actions.tolist(),
        hand_actions=result.hand_actions.tolist(),
    )


# --- Helpers ------------------------------------------------------------------


def _gpu_count(backend) -> int:
    if backend.cuda_available:
        import torch

        return torch.cuda.device_count()
    return 0
