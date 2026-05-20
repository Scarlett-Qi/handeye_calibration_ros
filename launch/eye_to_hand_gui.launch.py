from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            Node(
                package="handeye_calibration_ros",
                executable="eye_to_hand_gui.py",
                name="eye_to_hand_gui",
                output="screen",
            ),
        ]
    )
