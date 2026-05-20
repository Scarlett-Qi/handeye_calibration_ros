from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="handeye_calibration_ros",
                executable="handeye_gui.py",
                name="handeye_gui",
                output="screen",
            ),
        ]
    )
