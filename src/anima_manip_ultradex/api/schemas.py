"""Request and response models for the MANIP-ULTRADEX API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScenePointCloudRequest(BaseModel):
    raw_points: list[list[float]] = Field(
        ..., description="Raw scene point cloud [[x,y,z], ...], at least 256 points."
    )
    robot_points: list[list[float]] | None = Field(
        None, description="Optional robot body point cloud."
    )
    apply_sor: bool = Field(False, description="Apply statistical outlier removal.")
    grasp_strategy_hint: str | None = Field(
        None, description="Optional hint: pinch, tripod, whole_hand, bimanual."
    )


class DualArmHandActionResponse(BaseModel):
    action_vector: list[list[float]] = Field(..., description="Full 36-DoF action vector [B, 36].")
    arm_actions: list[list[list[float]]] = Field(..., description="Arm actions [B, 2, 6].")
    hand_actions: list[list[list[float]]] = Field(..., description="Hand actions [B, 2, 12].")
    checkpoint_loaded: bool = Field(..., description="Whether a real checkpoint was loaded.")


class HealthResponse(BaseModel):
    ok: bool
    module: str
    backend: str
    cuda_available: bool = False
    gpu_count: int = 0


class ReadyResponse(BaseModel):
    ready: bool
    checkpoint_loaded: bool = False
    model_params: int = 0
    reason: str | None = None


class InfoResponse(BaseModel):
    module: str
    version: str
    paper: str
    embodiment: str
    action_dims: int
    input_points: int
