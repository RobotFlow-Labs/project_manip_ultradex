"""ROS2 policy node for UltraDexGrasp bimanual dexterous control.

Subscribes to scene point clouds, runs inference, publishes dual-arm + dual-hand actions.
Falls back to a standalone mode when rclpy is unavailable.
"""

from __future__ import annotations

import importlib.util
from typing import Any


from anima_manip_ultradex.config import ModuleConfig, load_module_config
from anima_manip_ultradex.ros2.messages import DualArmActionMsg, PointCloudMsg

_HAS_RCLPY = importlib.util.find_spec("rclpy") is not None


class UltraDexPolicyNode:
    """ROS2-compatible policy node with graceful degradation."""

    # Topic names
    TOPIC_SCENE_PC = "/manip_ultradex/scene_pointcloud"
    TOPIC_ROBOT_STATE = "/manip_ultradex/robot_state"
    TOPIC_ARM_ACTION = "/manip_ultradex/arm_action"
    TOPIC_HAND_ACTION = "/manip_ultradex/hand_action"
    TOPIC_STATUS = "/manip_ultradex/status"

    def __init__(
        self,
        cfg: ModuleConfig | None = None,
        *,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.cfg = cfg or load_module_config()
        self._device = device
        self._checkpoint_path = checkpoint_path
        self._runner = None
        self._ros_node = None
        self._published_actions: list[DualArmActionMsg] = []

    def setup_inference(self) -> None:
        from anima_manip_ultradex.inference.runner import UltraDexRunner

        self._runner = UltraDexRunner(
            self.cfg,
            checkpoint_path=self._checkpoint_path,
            d_model=128,
            num_heads=4,
            num_layers=2,
            device=self._device,
        )

    def setup_ros(self) -> None:
        if not _HAS_RCLPY:
            return
        import rclpy
        from rclpy.node import Node

        if not rclpy.ok():
            rclpy.init()
        self._ros_node = Node("ultradex_policy_node")

    def process_pointcloud(self, msg: PointCloudMsg) -> DualArmActionMsg:
        if self._runner is None:
            self.setup_inference()

        result = self._runner.predict(msg.points)
        action_vec = result.action_vector.squeeze(0).tolist()

        action_msg = DualArmActionMsg(
            left_arm=action_vec[:6],
            right_arm=action_vec[6:12],
            left_hand=action_vec[12:24],
            right_hand=action_vec[24:36],
            checkpoint_loaded=result.checkpoint_loaded,
        )
        self._published_actions.append(action_msg)
        return action_msg

    def get_status(self) -> dict[str, Any]:
        return {
            "node_name": "ultradex_policy_node",
            "runner_ready": self._runner is not None,
            "checkpoint_loaded": (self._runner.checkpoint_loaded if self._runner else False),
            "ros_available": _HAS_RCLPY,
            "ros_node_active": self._ros_node is not None,
            "actions_published": len(self._published_actions),
            "device": self._device,
        }

    @property
    def last_action(self) -> DualArmActionMsg | None:
        return self._published_actions[-1] if self._published_actions else None
