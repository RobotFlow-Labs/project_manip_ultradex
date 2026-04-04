"""Typed configuration and path registry for MANIP-ULTRADEX."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class ProjectMetadata(BaseModel):
    name: str = "anima-manip-ultradex"
    codename: str = "MANIP-ULTRADEX"
    functional_name: str = "MANIP-ultradex"
    wave: int = 7
    paper_arxiv: str = "2603.05312"
    package: str = "anima_manip_ultradex"
    python_version: str = "3.11"


class ComputeConfig(BaseModel):
    backend: Literal["auto", "mlx", "cuda", "cpu"] = "auto"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    mac_support: bool = True
    cuda_support: bool = True


class DataConfig(BaseModel):
    shared_volume: str = "/Volumes/AIFlowDev/RobotFlowLabs/datasets"
    repos_volume: str = "/Volumes/AIFlowDev/RobotFlowLabs/repos/wave7"
    ultradexgrasp_dataset: str = "datasets/manip-ultradex/ultradexgrasp_20m"
    dexgraspnet_assets: str = "datasets/dexgraspnet/selected_1000"
    sim_benchmark: str = "datasets/manip-ultradex/benchmarks/sim_600"
    real_benchmark: str = "datasets/manip-ultradex/benchmarks/real_25"
    policy_checkpoint: str = "models/manip-ultradex/policy/latest.ckpt"


class HardwareConfig(BaseModel):
    zed2i: bool = True
    unitree_l2_lidar: bool = True
    dual_ur5e: bool = True
    dual_xhand_12dof: bool = True
    azure_kinect_dk: bool = True
    cobot_xarm6: bool = False


class PaperConstants(BaseModel):
    candidate_grasps_per_object: int = 500
    pregrasp_offset_m: float = 0.1
    lift_target_m: float = 0.2
    lift_success_height_m: float = 0.17
    lift_hold_time_s: float = 1.0
    policy_input_points: int = 2048
    point_group_knn: int = 32
    abstraction_output_points: int = 256
    action_query_tokens: int = 4
    arm_action_dims: int = 12
    hand_action_dims: int = 24


class ModuleConfig(BaseModel):
    project: ProjectMetadata = Field(default_factory=ProjectMetadata)
    compute: ComputeConfig = Field(default_factory=ComputeConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    hardware: HardwareConfig = Field(default_factory=HardwareConfig)
    paper: PaperConstants = Field(default_factory=PaperConstants)
    module_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parents[2])

    @property
    def environment(self) -> Literal["GPU_SERVER", "MAC_LOCAL", "UNKNOWN"]:
        if Path("/mnt/forge-data").exists():
            return "GPU_SERVER"
        if Path("/Volumes/AIFlowDev").exists():
            return "MAC_LOCAL"
        return "UNKNOWN"

    @property
    def data_root(self) -> Path:
        if self.environment == "GPU_SERVER":
            return Path("/mnt/forge-data")
        if self.environment == "MAC_LOCAL":
            return Path(self.data.shared_volume)
        return self.module_root

    @property
    def package_root(self) -> Path:
        return self.module_root / "src" / self.project.package

    @property
    def reference_repo_root(self) -> Path:
        return self.module_root / "repositories" / "UltraDexGrasp"

    @property
    def paper_pdf(self) -> Path:
        return self.module_root / "papers" / f"{self.project.paper_arxiv}_UltraDexGrasp.pdf"

    @property
    def bowl_mesh(self) -> Path:
        return (
            self.reference_repo_root / "asset" / "object_mesh" / "bowl" / "mesh" / "simplified.obj"
        )

    @property
    def ultradexgrasp_dataset_root(self) -> Path:
        return self.data_root / self.data.ultradexgrasp_dataset

    @property
    def dexgraspnet_assets_root(self) -> Path:
        return self.data_root / self.data.dexgraspnet_assets

    @property
    def sim_benchmark_root(self) -> Path:
        return self.data_root / self.data.sim_benchmark

    @property
    def real_benchmark_root(self) -> Path:
        return self.data_root / self.data.real_benchmark

    @property
    def policy_checkpoint_path(self) -> Path:
        return self.data_root / self.data.policy_checkpoint


def load_module_config(path: str | os.PathLike[str] | None = None) -> ModuleConfig:
    config_path = (
        Path(path) if path else Path(__file__).resolve().parents[2] / "configs" / "default.toml"
    )
    raw = tomllib.loads(config_path.read_text())
    return ModuleConfig(**raw)
