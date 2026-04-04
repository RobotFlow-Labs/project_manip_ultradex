"""ROS2 launch file for MANIP-ULTRADEX policy node."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument("device", default_value="cuda"),
        DeclareLaunchArgument("checkpoint", default_value=""),
        DeclareLaunchArgument("config", default_value="configs/default.toml"),
        Node(
            package="anima_manip_ultradex",
            executable="ultradex_policy_node",
            name="ultradex_policy_node",
            output="screen",
            parameters=[{
                "device": LaunchConfiguration("device"),
                "checkpoint": LaunchConfiguration("checkpoint"),
                "config": LaunchConfiguration("config"),
            }],
            remappings=[
                ("/scene_pointcloud", "/manip_ultradex/scene_pointcloud"),
                ("/arm_action", "/manip_ultradex/arm_action"),
                ("/hand_action", "/manip_ultradex/hand_action"),
            ],
        ),
    ])
