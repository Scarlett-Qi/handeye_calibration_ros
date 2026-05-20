#!/usr/bin/env python3
# coding: utf-8

import argparse

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node


class PoseStampedToPose(Node):
    def __init__(self, args):
        super().__init__("pose_stamped_to_pose")
        self.pub = self.create_publisher(Pose, args.output_topic, 10)
        self.sub = self.create_subscription(PoseStamped, args.input_topic, self.on_pose_stamped, 10)
        self.get_logger().info(f"bridge {args.input_topic} (PoseStamped) -> {args.output_topic} (Pose)")

    def on_pose_stamped(self, msg: PoseStamped):
        self.pub.publish(msg.pose)


def main():
    parser = argparse.ArgumentParser(description="Bridge geometry_msgs/PoseStamped to Pose")
    parser.add_argument("--input-topic", default="/robot/tool_pose_stamped")
    parser.add_argument("--output-topic", default="/handeye/tool_pose")
    args = parser.parse_args()

    rclpy.init()
    node = PoseStampedToPose(args)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
