"""Four-stage demonstration generator (§IV.B).

Stages: pregrasp → grasp → squeeze → lift.
Each stage produces a target pose and success criterion matching the paper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from anima_manip_ultradex.config import ModuleConfig
from anima_manip_ultradex.grasp.types import GraspCandidate, Pose7D


@dataclass(frozen=True)
class DemoStage:
    name: str
    target_pose: Pose7D
    duration_s: float = 1.0


@dataclass
class DemoTrajectory:
    stages: list[DemoStage] = field(default_factory=list)

    def stage_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stages)


class DemoGenerator:
    """Generates four-stage demonstration trajectories from a grasp candidate."""

    def __init__(self, cfg: ModuleConfig) -> None:
        self.cfg = cfg
        self.pregrasp_offset = cfg.paper.pregrasp_offset_m
        self.lift_target = cfg.paper.lift_target_m

    def generate(self, candidate: GraspCandidate) -> DemoTrajectory:
        grasp_xyz = tuple(candidate.wrist_pose.xyz)
        grasp_wxyz = tuple(candidate.wrist_pose.wxyz)

        pregrasp_xyz = (
            grasp_xyz[0],
            grasp_xyz[1],
            grasp_xyz[2] + self.pregrasp_offset,
        )
        squeeze_xyz = (
            grasp_xyz[0],
            grasp_xyz[1],
            grasp_xyz[2] - 0.01,
        )
        lift_xyz = (
            grasp_xyz[0],
            grasp_xyz[1],
            grasp_xyz[2] + self.lift_target,
        )

        return DemoTrajectory(stages=[
            DemoStage(
                name="pregrasp",
                target_pose=Pose7D(xyz=pregrasp_xyz, wxyz=grasp_wxyz),
                duration_s=1.0,
            ),
            DemoStage(
                name="grasp",
                target_pose=Pose7D(xyz=grasp_xyz, wxyz=grasp_wxyz),
                duration_s=0.5,
            ),
            DemoStage(
                name="squeeze",
                target_pose=Pose7D(xyz=squeeze_xyz, wxyz=grasp_wxyz),
                duration_s=0.5,
            ),
            DemoStage(
                name="lift",
                target_pose=Pose7D(xyz=lift_xyz, wxyz=grasp_wxyz),
                duration_s=1.0,
            ),
        ])
