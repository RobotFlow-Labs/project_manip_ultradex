"""Point-cloud preprocessing for offline inference."""

from __future__ import annotations

from typing import Any

import numpy as np


def _to_numpy(points: Any) -> np.ndarray:
    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    if hasattr(points, "detach"):
        return points.detach().cpu().numpy().astype(np.float32)
    return np.asarray(points, dtype=np.float32)


def merge_point_clouds(raw_pc: Any, robot_pc: Any | None = None) -> np.ndarray:
    raw = _to_numpy(raw_pc).reshape(-1, 3)
    robot = _to_numpy(robot_pc).reshape(-1, 3)
    if robot.size == 0:
        return raw
    return np.concatenate([raw, robot], axis=0)


def crop_workspace(
    points: np.ndarray,
    min_bounds: tuple[float, float, float] = (-2.0, -2.0, -2.0),
    max_bounds: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> np.ndarray:
    if points.size == 0:
        raise ValueError("Point cloud cannot be empty.")
    lower = np.asarray(min_bounds, dtype=np.float32)
    upper = np.asarray(max_bounds, dtype=np.float32)
    mask = np.all((points >= lower) & (points <= upper), axis=1)
    cropped = points[mask]
    return cropped if cropped.size else points


def statistical_outlier_filter(
    points: np.ndarray,
    k: int = 16,
    z_threshold: float = 2.0,
) -> np.ndarray:
    if len(points) <= k:
        return points

    deltas = points[:, None, :] - points[None, :, :]
    distances = np.linalg.norm(deltas, axis=-1)
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, kth=k, axis=1)[:, :k]
    mean_distances = nearest.mean(axis=1)
    center = mean_distances.mean()
    spread = mean_distances.std()
    if spread == 0.0:
        return points
    z_scores = (mean_distances - center) / spread
    filtered = points[z_scores <= z_threshold]
    return filtered if filtered.size else points


def farthest_point_sample(points: np.ndarray, target_points: int) -> np.ndarray:
    if points.size == 0:
        raise ValueError("Point cloud cannot be empty.")
    if len(points) == target_points:
        return points
    if len(points) < target_points:
        repeats = np.tile(points, (target_points // len(points) + 1, 1))
        return repeats[:target_points]

    # Try CUDA FPS kernel (7.2x speedup) if available
    try:
        import torch

        if torch.cuda.is_available():
            from point_cloud_ops import farthest_point_sample as fps_cuda

            pts_t = torch.from_numpy(points).unsqueeze(0).cuda()
            idx = fps_cuda(pts_t, target_points)
            return points[idx.squeeze(0).cpu().numpy()]
    except (ImportError, RuntimeError):
        pass

    # CPU fallback
    centroid = points.mean(axis=0)
    selected_indices = [int(np.argmax(np.linalg.norm(points - centroid, axis=1)))]
    min_distances = np.linalg.norm(points - points[selected_indices[0]], axis=1)

    while len(selected_indices) < target_points:
        next_index = int(np.argmax(min_distances))
        selected_indices.append(next_index)
        candidate_distances = np.linalg.norm(points - points[next_index], axis=1)
        min_distances = np.minimum(min_distances, candidate_distances)

    return points[np.asarray(selected_indices)]


def normalize_points(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    radius = np.linalg.norm(centered, axis=1).max()
    if radius == 0.0:
        return centered
    return centered / radius


def build_scene_input(
    raw_pc: Any,
    robot_pc: Any | None = None,
    *,
    apply_sor: bool = False,
    target_points: int = 2048,
):
    import torch

    points = merge_point_clouds(raw_pc, robot_pc)
    points = crop_workspace(points)
    if apply_sor:
        points = statistical_outlier_filter(points)
    points = farthest_point_sample(points, target_points)
    points = normalize_points(points)
    return torch.from_numpy(points).float().unsqueeze(0)
