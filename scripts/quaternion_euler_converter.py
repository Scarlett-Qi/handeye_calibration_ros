#!/usr/bin/env python3
# coding: utf-8

import argparse
import math
from typing import Tuple

import numpy as np


def normalize_quat(x: float, y: float, z: float, w: float) -> Tuple[float, float, float, float]:
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        raise ValueError("四元数范数过小，无法归一化")
    return x / n, y / n, z / n, w / n


def quat_to_euler_zyx(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    # roll(X), pitch(Y), yaw(Z)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp)))

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def euler_zyx_to_quat(roll: float, pitch: float, yaw: float) -> Tuple[float, float, float, float]:
    # R = Rz(yaw) * Ry(pitch) * Rx(roll)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return normalize_quat(x, y, z, w)


def parse_csv_floats(text: str, n: int, name: str):
    vals = [v.strip() for v in text.split(",")]
    if len(vals) != n:
        raise ValueError(f"{name} 需要 {n} 个值，当前为 {len(vals)}")
    return [float(v) for v in vals]


def main():
    parser = argparse.ArgumentParser(description="四元数/欧拉角互转（ZYX）")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--quat",
        help="输入四元数 x,y,z,w，例如: --quat 0.1,0.2,0.3,0.9",
    )
    group.add_argument(
        "--euler",
        help="输入欧拉角 rx,ry,rz，例如: --euler 90,0,15",
    )
    parser.add_argument(
        "--unit",
        choices=["deg", "rad"],
        default="deg",
        help="欧拉角输入/输出单位，默认 deg",
    )
    args = parser.parse_args()

    if args.quat:
        qx, qy, qz, qw = parse_csv_floats(args.quat, 4, "--quat")
        qx, qy, qz, qw = normalize_quat(qx, qy, qz, qw)
        rx, ry, rz = quat_to_euler_zyx(qx, qy, qz, qw)

        if args.unit == "deg":
            rx, ry, rz = np.degrees([rx, ry, rz]).tolist()

        print("mode: quat_to_euler (ZYX)")
        print(f"quat(x,y,z,w): {qx:.12f}, {qy:.12f}, {qz:.12f}, {qw:.12f}")
        print(f"euler(rx,ry,rz) [{args.unit}]: {rx:.9f}, {ry:.9f}, {rz:.9f}")
        return

    rx, ry, rz = parse_csv_floats(args.euler, 3, "--euler")
    if args.unit == "deg":
        rx, ry, rz = np.radians([rx, ry, rz]).tolist()
    qx, qy, qz, qw = euler_zyx_to_quat(rx, ry, rz)

    print("mode: euler_to_quat (ZYX)")
    if args.unit == "deg":
        rin = np.degrees([rx, ry, rz]).tolist()
        print(f"euler(rx,ry,rz) [deg]: {rin[0]:.9f}, {rin[1]:.9f}, {rin[2]:.9f}")
    else:
        print(f"euler(rx,ry,rz) [rad]: {rx:.9f}, {ry:.9f}, {rz:.9f}")
    print(f"quat(x,y,z,w): {qx:.12f}, {qy:.12f}, {qz:.12f}, {qw:.12f}")


if __name__ == "__main__":
    main()

