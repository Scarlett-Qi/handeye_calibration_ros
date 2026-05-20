import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory("handeye_calibration_ros")
    default_config = os.path.join(pkg_share, "config", "handeye_collect.yaml")

    config_file = LaunchConfiguration("config_file")
    image_topic = LaunchConfiguration("image_topic")
    tool_pose_topic = LaunchConfiguration("tool_pose_topic")
    tool_pose_frame_id = LaunchConfiguration("tool_pose_frame_id")
    output_dir = LaunchConfiguration("output_dir")

    return LaunchDescription(
        [
            DeclareLaunchArgument("config_file", default_value=default_config),
            DeclareLaunchArgument("image_topic", default_value="/camera/color/image_raw"),
            DeclareLaunchArgument("tool_pose_topic", default_value="/handeye/tool_pose"),
            DeclareLaunchArgument("tool_pose_frame_id", default_value="base"),
            DeclareLaunchArgument("output_dir", default_value="./handeye_data"),
            Node(
                package="handeye_calibration_ros",
                executable="handeye_sample_collector",
                name="handeye_sample_collector",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "image_topic": image_topic,
                        "tool_pose_topic": tool_pose_topic,
                        "tool_pose_frame_id": tool_pose_frame_id,
                        "output_dir": output_dir,
                    },
                ],
            ),
        ]
    )
