"""PRD-06 tests: ROS2 bridge node verification (no rclpy required)."""

import numpy as np

from anima_manip_ultradex.config import load_module_config
from anima_manip_ultradex.ros2.messages import DualArmActionMsg, PointCloudMsg
from anima_manip_ultradex.ros2.policy_node import UltraDexPolicyNode


def test_pointcloud_msg_from_numpy() -> None:
    pts = np.random.default_rng(1).normal(size=(1000, 3)).astype(np.float32)
    msg = PointCloudMsg.from_numpy(pts, frame_id="camera_link")
    assert msg.points.shape == (1000, 3)
    assert msg.frame_id == "camera_link"


def test_action_msg_flattened() -> None:
    action = DualArmActionMsg(
        left_arm=[0.1] * 6,
        right_arm=[0.2] * 6,
        left_hand=[0.3] * 12,
        right_hand=[0.4] * 12,
    )
    flat = action.flattened
    assert len(flat) == 36
    assert flat[0] == 0.1
    assert flat[6] == 0.2
    assert flat[12] == 0.3
    assert flat[24] == 0.4


def test_policy_node_processes_pointcloud() -> None:
    cfg = load_module_config()
    node = UltraDexPolicyNode(cfg, device="cpu")
    pts = np.random.default_rng(5).normal(size=(2500, 3)).astype(np.float32)
    msg = PointCloudMsg.from_numpy(pts)
    action = node.process_pointcloud(msg)

    assert len(action.left_arm) == 6
    assert len(action.right_arm) == 6
    assert len(action.left_hand) == 12
    assert len(action.right_hand) == 12
    assert len(action.flattened) == 36


def test_policy_node_status() -> None:
    cfg = load_module_config()
    node = UltraDexPolicyNode(cfg, device="cpu")
    status = node.get_status()
    assert status["node_name"] == "ultradex_policy_node"
    assert status["runner_ready"] is False

    # After processing, runner should be initialized
    pts = np.random.default_rng(9).normal(size=(2048, 3)).astype(np.float32)
    node.process_pointcloud(PointCloudMsg.from_numpy(pts))
    status = node.get_status()
    assert status["runner_ready"] is True
    assert status["actions_published"] == 1


def test_policy_node_topic_names() -> None:
    assert "/manip_ultradex/scene_pointcloud" == UltraDexPolicyNode.TOPIC_SCENE_PC
    assert "/manip_ultradex/arm_action" == UltraDexPolicyNode.TOPIC_ARM_ACTION
    assert "/manip_ultradex/hand_action" == UltraDexPolicyNode.TOPIC_HAND_ACTION
