"""Typed schemas for grasping and control contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


GraspStrategy = Literal["pinch", "tripod", "whole_hand", "bimanual"]
ActionQueryName = Literal["left_arm", "right_arm", "left_hand", "right_hand"]


@dataclass(frozen=True)
class RobotEmbodiment:
    num_arms: int = 2
    arm_dof: int = 6
    hand_dof: int = 12

    @property
    def total_action_dims(self) -> int:
        return self.num_arms * (self.arm_dof + self.hand_dof)

    @property
    def action_query_layout(self) -> tuple[int, int, int, int]:
        return (self.arm_dof, self.arm_dof, self.hand_dof, self.hand_dof)


@dataclass(frozen=True)
class Pose7D:
    xyz: Sequence[float]
    wxyz: Sequence[float]

    def as_vector(self) -> tuple[float, ...]:
        return tuple(self.xyz) + tuple(self.wxyz)


@dataclass(frozen=True)
class GraspCandidate:
    strategy: GraspStrategy
    object_id: str
    num_hands: int
    wrist_pose: Pose7D
    hand_joints: Sequence[float]
    score: float = 0.0


@dataclass(frozen=True)
class GraspCandidateSpec:
    num_hands: int
    num_keyposes: int = 3
    pose_dims: int = 7
    qpos_dims: int = 12

    @property
    def total_dims(self) -> int:
        return self.pose_dims + self.qpos_dims

    @property
    def tensor_rank(self) -> int:
        return 4

    def tensor_shape(self, num_candidates: int) -> tuple[int, int, int, int]:
        return (num_candidates, self.num_hands, self.num_keyposes, self.total_dims)


@dataclass(frozen=True)
class DualArmHandAction:
    left_arm: Sequence[float]
    right_arm: Sequence[float]
    left_hand: Sequence[float]
    right_hand: Sequence[float]

    def flattened(self) -> tuple[float, ...]:
        return (
            *self.left_arm,
            *self.right_arm,
            *self.left_hand,
            *self.right_hand,
        )
