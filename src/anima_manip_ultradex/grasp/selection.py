"""Preferred-grasp ranking helpers based on SE(3) proximity."""

from __future__ import annotations

from dataclasses import dataclass
from math import acos
from typing import Iterable

import numpy as np

from anima_manip_ultradex.grasp.types import GraspCandidate, GraspStrategy, Pose7D


def _as_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(tuple(values), dtype=np.float64)


def _normalize_quaternion(values: Iterable[float]) -> np.ndarray:
    quat = _as_array(values)
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        raise ValueError("Quaternion cannot have zero norm.")
    return quat / norm


def quaternion_angular_distance(lhs: Iterable[float], rhs: Iterable[float]) -> float:
    lhs_q = _normalize_quaternion(lhs)
    rhs_q = _normalize_quaternion(rhs)
    cosine = np.clip(abs(np.dot(lhs_q, rhs_q)), -1.0, 1.0)
    return 2.0 * acos(float(cosine))


def se3_distance(
    reference_pose: Pose7D,
    candidate_pose: Pose7D,
    translation_weight: float = 1.0,
    rotation_weight: float = 0.25,
) -> float:
    translation_delta = _as_array(reference_pose.xyz) - _as_array(candidate_pose.xyz)
    translation_error = float(np.linalg.norm(translation_delta))
    rotation_error = quaternion_angular_distance(reference_pose.wxyz, candidate_pose.wxyz)
    return translation_weight * translation_error + rotation_weight * rotation_error


@dataclass(frozen=True)
class RankedGraspCandidate:
    candidate: GraspCandidate
    distance: float


def rank_grasps_by_se3(
    reference_pose: Pose7D,
    candidates: Iterable[GraspCandidate],
) -> list[RankedGraspCandidate]:
    ranked = [
        RankedGraspCandidate(
            candidate=candidate,
            distance=se3_distance(reference_pose, candidate.wrist_pose),
        )
        for candidate in candidates
    ]
    return sorted(ranked, key=lambda item: (item.distance, -item.candidate.score))


def select_preferred_grasp(
    reference_pose: Pose7D,
    candidates: Iterable[GraspCandidate],
    strategy: GraspStrategy | None = None,
) -> GraspCandidate:
    filtered = [
        candidate for candidate in candidates if strategy is None or candidate.strategy == strategy
    ]
    if not filtered:
        raise ValueError("No grasp candidates available for selection.")
    return rank_grasps_by_se3(reference_pose, filtered)[0].candidate
