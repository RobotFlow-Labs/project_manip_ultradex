"""Internal message helpers for ROS2 bridge.

Provides pure-Python representations that can be converted to/from ROS2 messages
without importing rclpy at module level, so tests pass without a ROS2 install.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PointCloudMsg:
    """Lightweight stand-in for sensor_msgs/PointCloud2."""

    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3), dtype=np.float32))
    frame_id: str = "world"
    stamp_sec: int = 0
    stamp_nanosec: int = 0

    @classmethod
    def from_numpy(cls, arr: np.ndarray, frame_id: str = "world") -> PointCloudMsg:
        return cls(points=arr.reshape(-1, 3).astype(np.float32), frame_id=frame_id)


@dataclass
class DualArmActionMsg:
    """Lightweight stand-in for custom action message."""

    left_arm: list[float] = field(default_factory=lambda: [0.0] * 6)
    right_arm: list[float] = field(default_factory=lambda: [0.0] * 6)
    left_hand: list[float] = field(default_factory=lambda: [0.0] * 12)
    right_hand: list[float] = field(default_factory=lambda: [0.0] * 12)
    checkpoint_loaded: bool = False

    @property
    def flattened(self) -> list[float]:
        return self.left_arm + self.right_arm + self.left_hand + self.right_hand
