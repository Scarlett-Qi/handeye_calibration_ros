#!/usr/bin/env python3
# coding: utf-8
"""
手眼角点测试脚本（眼在手上）

流程：
1) 从彩色图自动检测棋盘角点
2) 选定一个角点（corner_row/corner_col）
3) 用对齐深度求该角点3D（相机系）
4) 用 handeye 结果 + 当前末端位姿（topic订阅）求 base 目标点
5) 输出可直接发送给机械臂的 move 参数
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import Pose, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image


def rpy_to_rot(rx: float, ry: float, rz: float) -> np.ndarray:
    rxm = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float64
    )
    rym = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64
    )
    rzm = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float64
    )
    return rzm @ rym @ rxm


def make_transform(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    tf = np.eye(4, dtype=np.float64)
    tf[:3, :3] = r
    tf[:3, 3] = t.reshape(3)
    return tf


def invert_transform(tf: np.ndarray) -> np.ndarray:
    r = tf[:3, :3]
    t = tf[:3, 3].reshape(3, 1)
    r_inv = r.T
    t_inv = -r_inv @ t
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r_inv
    out[:3, 3] = t_inv.reshape(3)
    return out


def rot_to_rpy_deg_order_zyx(r: np.ndarray) -> Tuple[float, float, float]:
    # 与 rpy_to_rot 一致: R = Rz * Ry * Rx
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) > 1e-9:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])
    return tuple(np.degrees([roll, pitch, yaw]).tolist())


def load_handeye(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    r = np.array(data["R_cam2gripper"], dtype=np.float64)
    t = np.array(data["t_cam2gripper"], dtype=np.float64).reshape(3)
    return r, t


def quat_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def rot_to_rpy_deg(r: np.ndarray) -> Tuple[float, float, float]:
    pitch = np.arcsin(np.clip(-r[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)
    if abs(cp) > 1e-9:
        roll = np.arctan2(r[2, 1], r[2, 2])
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-r[0, 1], r[1, 1])
    return tuple(np.degrees([roll, pitch, yaw]).tolist())


class SnapshotNode(Node):
    def __init__(self, args, need_vision_inputs: bool):
        super().__init__("handeye_corner_pick_test_node")
        self.cam_info: Optional[CameraInfo] = None
        self.depth: Optional[Image] = None
        self.color: Optional[Image] = None
        self.pose: Optional[Pose] = None
        self.need_vision_inputs = need_vision_inputs
        if self.need_vision_inputs:
            self.create_subscription(CameraInfo, args.camera_info_topic, self.on_info, 10)
            self.create_subscription(Image, args.depth_topic, self.on_depth, 10)
            self.create_subscription(Image, args.color_topic, self.on_color, 10)
        if args.tool_pose_msg_type == "pose":
            self.create_subscription(Pose, args.tool_pose_topic, self.on_pose, 10)
        else:
            self.create_subscription(PoseStamped, args.tool_pose_topic, self.on_pose_stamped, 10)

    def on_info(self, msg: CameraInfo):
        self.cam_info = msg

    def on_depth(self, msg: Image):
        self.depth = msg

    def on_color(self, msg: Image):
        self.color = msg

    def on_pose(self, msg: Pose):
        self.pose = msg

    def on_pose_stamped(self, msg: PoseStamped):
        self.pose = msg.pose

    def ready(self) -> bool:
        if self.need_vision_inputs:
            return (
                self.cam_info is not None
                and self.depth is not None
                and self.color is not None
                and self.pose is not None
            )
        return self.pose is not None


def depth_to_numpy(depth_msg: Image, depth_scale: float) -> np.ndarray:
    h, w = depth_msg.height, depth_msg.width
    if depth_msg.encoding == "16UC1":
        arr = np.frombuffer(depth_msg.data, dtype=np.uint16).reshape(h, w)
        return arr.astype(np.float64) * depth_scale
    if depth_msg.encoding == "32FC1":
        return np.frombuffer(depth_msg.data, dtype=np.float32).reshape(h, w).astype(np.float64)
    raise ValueError(f"不支持的深度编码: {depth_msg.encoding}")


def color_to_gray(color_msg: Image) -> np.ndarray:
    h, w = color_msg.height, color_msg.width
    if color_msg.encoding == "bgr8":
        bgr = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if color_msg.encoding == "rgb8":
        rgb = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    if color_msg.encoding == "mono8":
        return np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w)
    if color_msg.encoding == "rgba8":
        rgba = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2GRAY)
    if color_msg.encoding == "bgra8":
        bgra = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"不支持的彩色编码: {color_msg.encoding}")


def color_to_bgr(color_msg: Image) -> np.ndarray:
    h, w = color_msg.height, color_msg.width
    if color_msg.encoding == "bgr8":
        return np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 3).copy()
    if color_msg.encoding == "rgb8":
        rgb = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if color_msg.encoding == "mono8":
        mono = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    if color_msg.encoding == "rgba8":
        rgba = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if color_msg.encoding == "bgra8":
        bgra = np.frombuffer(color_msg.data, dtype=np.uint8).reshape(h, w, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    raise ValueError(f"不支持的彩色编码: {color_msg.encoding}")


def robust_depth_m(depth_m: np.ndarray, u: int, v: int, win: int) -> float:
    h, w = depth_m.shape
    u0, u1 = max(0, u - win), min(w, u + win + 1)
    v0, v1 = max(0, v - win), min(h, v + win + 1)
    patch = depth_m[v0:v1, u0:u1]
    vals = patch[np.isfinite(patch) & (patch > 1e-6)]
    if vals.size == 0:
        raise ValueError("角点邻域没有有效深度")
    return float(np.median(vals))


def detect_corner_uv(
    gray: np.ndarray, cols: int, rows: int, c_col: int, c_row: int
) -> Tuple[int, int, np.ndarray]:
    found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
    if not found:
        # 使用更鲁棒的SB算法兜底
        found, corners = cv2.findChessboardCornersSB(gray, (cols, rows), None)
    if not found or corners is None:
        raise ValueError("未检测到棋盘角点，请检查棋盘是否完整可见")

    if corners.ndim == 3:
        corners2 = corners.reshape(-1, 2)
    else:
        corners2 = corners
    if c_col < 0 or c_col >= cols or c_row < 0 or c_row >= rows:
        raise ValueError(f"角点索引越界: col应在[0,{cols-1}] row应在[0,{rows-1}]")
    idx = c_row * cols + c_col
    uv = corners2[idx]
    return int(round(float(uv[0]))), int(round(float(uv[1]))), corners2


def solve_board_pose_cam(
    corners2: np.ndarray, cam_k: np.ndarray, cam_dist: np.ndarray, cols: int, rows: int, square: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    imgp = corners2.reshape(-1, 1, 2).astype(np.float32)
    ok, rvec, tvec = cv2.solvePnP(objp, imgp, cam_k, cam_dist, flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        raise ValueError("solvePnP 失败，无法估计标定板姿态")
    r_cam_board, _ = cv2.Rodrigues(rvec)
    return rvec, tvec, r_cam_board


def build_rotation_with_z(z_axis: np.ndarray, ref_x_axis: np.ndarray) -> np.ndarray:
    z = z_axis.astype(np.float64)
    z /= np.linalg.norm(z)

    x = ref_x_axis.astype(np.float64)
    x = x - np.dot(x, z) * z
    if np.linalg.norm(x) < 1e-9:
        # 退化时换一个参考轴
        alt = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(float(np.dot(alt, z))) > 0.9:
            alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        x = alt - np.dot(alt, z) * z
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    y /= np.linalg.norm(y)
    # 列向量为坐标轴
    return np.column_stack((x, y, z))


def draw_selected_corner(
    bgr: np.ndarray, u: int, v: int, row: int, col: int, z_m: float
) -> np.ndarray:
    out = bgr.copy()
    cv2.circle(out, (u, v), 10, (0, 255, 255), 2)
    cv2.drawMarker(out, (u, v), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=18, thickness=2)
    text1 = f"corner(row,col)=({row},{col}) pixel=({u},{v})"
    text2 = f"depth={z_m:.4f} m"
    cv2.putText(out, text1, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.putText(out, text2, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    return out


def draw_labeled_board_axes(
    img: np.ndarray,
    cam_k: np.ndarray,
    cam_dist: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    axis_len: float,
) -> np.ndarray:
    out = img.copy()
    pts3d = np.array(
        [
            [0.0, 0.0, 0.0],  # origin
            [axis_len, 0.0, 0.0],  # X
            [0.0, axis_len, 0.0],  # Y
            [0.0, 0.0, axis_len],  # Z
        ],
        dtype=np.float32,
    )
    pts2d, _ = cv2.projectPoints(pts3d, rvec, tvec, cam_k, cam_dist)
    p = pts2d.reshape(-1, 2).astype(int)
    o, px, py, pz = p[0], p[1], p[2], p[3]

    # OpenCV惯例：X红、Y绿、Z蓝
    cv2.arrowedLine(out, tuple(o), tuple(px), (0, 0, 255), 3, tipLength=0.15)
    cv2.arrowedLine(out, tuple(o), tuple(py), (0, 255, 0), 3, tipLength=0.15)
    cv2.arrowedLine(out, tuple(o), tuple(pz), (255, 0, 0), 4, tipLength=0.2)  # Z加粗
    cv2.putText(out, "X", tuple(px + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, "Y", tuple(py + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(out, "Z", tuple(pz + np.array([6, -6])), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    r_cam_board, _ = cv2.Rodrigues(rvec)
    z_cam = r_cam_board[:, 2]
    cv2.putText(
        out,
        f"+Z(cam)=[{z_cam[0]:+.2f},{z_cam[1]:+.2f},{z_cam[2]:+.2f}]",
        (20, 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 0, 0),
        2,
    )
    return out


def main():
    parser = argparse.ArgumentParser(description="手眼角点测试: 棋盘角点 -> 机器人目标点")
    parser.add_argument("--handeye-yaml", default="handeye_data/handeye_result.yaml")
    parser.add_argument("--camera-info-topic", default="/camera/color/camera_info")
    parser.add_argument("--color-topic", default="/camera/depth_to_color/image_raw")
    parser.add_argument("--depth-topic", default="/camera/depth/image_raw")
    parser.add_argument("--depth-scale", type=float, default=0.001, help="16UC1缩放(mm->m)")
    parser.add_argument("--depth-window", type=int, default=2, help="深度中值窗口半径")
    parser.add_argument("--timeout", type=float, default=8.0)
    parser.add_argument("--tool-pose-topic", default="/handeye/tool_pose")
    parser.add_argument(
        "--tool-pose-msg-type",
        choices=["pose_stamped", "pose"],
        default="pose",
        help="工具位姿话题类型，通用接口默认 geometry_msgs/Pose",
    )
    parser.add_argument("--cols", type=int, default=11, help="棋盘角点列数")
    parser.add_argument("--rows", type=int, default=7, help="棋盘角点行数")
    parser.add_argument("--square", type=float, default=0.035, help="棋盘格尺寸(米)，用于姿态估计")
    parser.add_argument("--corner-col", type=int, default=None, help="目标角点列索引(0-based)")
    parser.add_argument("--corner-row", type=int, default=None, help="目标角点行索引(0-based)")
    parser.add_argument(
        "--cam-xyz",
        default="",
        help="直接输入相机坐标，格式 x,y,z（例如 0.12,-0.03,0.45）",
    )
    parser.add_argument(
        "--cam-unit",
        choices=["m", "mm"],
        default="m",
        help="--cam-xyz 的单位",
    )
    parser.add_argument("--actuator-id", default="crp_robot")
    parser.add_argument("--speed", type=float, default=120.0)
    parser.add_argument(
        "--tool-offset",
        default="0.4125,-0.2395,0.1014",
        help="末端->工具 平移偏置(m)，格式 x,y,z",
    )
    parser.add_argument(
        "--tool-rpy-deg",
        default="90,0,-14.98",
        help="末端->工具 旋转角(度, ZYX组合)，格式 rx,ry,rz",
    )
    parser.add_argument(
        "--align-z-mode",
        choices=["none", "parallel", "anti_parallel"],
        default="none",
        help="目标工具Z轴与标定板Z轴的关系：none=沿用当前姿态，parallel=平行，anti_parallel=反向平行",
    )
    parser.add_argument("--visualize", action="store_true", help="弹窗显示选中的棋盘角点")
    parser.add_argument(
        "--vis-output",
        default="",
        help="保存标注结果图片路径，例如 handeye_data/pick_vis.png",
    )
    args = parser.parse_args()

    r_cam2tool, t_cam2tool = load_handeye(Path(args.handeye_yaml).expanduser().resolve())

    cam_xyz = None
    if args.cam_xyz:
        parts = [p.strip() for p in args.cam_xyz.split(",")]
        if len(parts) != 3:
            raise ValueError("--cam-xyz 格式错误，示例: --cam-xyz 0.12,-0.03,0.45")
        cam_xyz = np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)
    use_direct_cam = cam_xyz is not None
    if not use_direct_cam:
        if args.corner_col is None or args.corner_row is None:
            raise ValueError("像素模式需要提供 --corner-col 和 --corner-row")

    rclpy.init()
    node = SnapshotNode(args, need_vision_inputs=not use_direct_cam)
    deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
    while rclpy.ok() and not node.ready():
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.get_clock().now().nanoseconds > deadline:
            node.destroy_node()
            rclpy.shutdown()
            if use_direct_cam:
                raise TimeoutError("等待 tool_pose 超时")
            raise TimeoutError("等待 camera_info/color/depth/tool_pose 超时")

    cam = node.cam_info
    depth_msg = node.depth
    color_msg = node.color
    pose = node.pose
    assert pose is not None

    corners2 = None
    u = v = None
    z = None
    if use_direct_cam:
        scale = 1.0 if args.cam_unit == "m" else 0.001
        p_cam = cam_xyz * scale
    else:
        assert cam is not None and depth_msg is not None and color_msg is not None
        gray = color_to_gray(color_msg)
        u, v, corners2 = detect_corner_uv(gray, args.cols, args.rows, args.corner_col, args.corner_row)

        fx, fy = cam.k[0], cam.k[4]
        cx, cy = cam.k[2], cam.k[5]
        depth_m = depth_to_numpy(depth_msg, args.depth_scale)
        z = robust_depth_m(depth_m, u, v, args.depth_window)
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        p_cam = np.array([x, y, z], dtype=np.float64)

    r_base2tool = quat_to_rot(
        pose.orientation.x,
        pose.orientation.y,
        pose.orientation.z,
        pose.orientation.w,
    )
    t_base2tool = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)

    p_tool = r_cam2tool @ p_cam + t_cam2tool
    p_base = r_base2tool @ p_tool + t_base2tool
    x_mm, y_mm, z_mm = (p_base * 1000.0).tolist()

    print("\n=== Corner Pick Result ===")
    if use_direct_cam:
        print("mode: direct_cam_xyz")
        print(f"cam input unit: {args.cam_unit}")
    else:
        print("mode: pixel_depth")
        print(f"corner(row,col): ({args.corner_row}, {args.corner_col})")
        print(f"pixel(u,v): ({u}, {v})")
        print(f"depth(m): {z:.6f}")
    print(f"P_cam(m):  [{p_cam[0]:.6f}, {p_cam[1]:.6f}, {p_cam[2]:.6f}]")
    print(f"P_base(m): [{p_base[0]:.6f}, {p_base[1]:.6f}, {p_base[2]:.6f}]")
    print(f"P_base(mm): x={x_mm:.3f}, y={y_mm:.3f}, z={z_mm:.3f}")

    # 夹具目标点：希望“工具坐标系原点(夹具TCP)”到达 p_base
    # 需要反算末端目标: T_base_end = T_base_tool_target * inv(T_end_tool)
    off_xyz = [float(v) for v in args.tool_offset.split(",")]
    if len(off_xyz) != 3:
        raise ValueError("--tool-offset 格式错误，示例: 0.3015,-0.085,0.1454")
    off_rpy = [float(v) for v in args.tool_rpy_deg.split(",")]
    if len(off_rpy) != 3:
        raise ValueError("--tool-rpy-deg 格式错误，示例: 90,0,14.98")

    r_end_tool = rpy_to_rot(
        np.radians(off_rpy[0]),
        np.radians(off_rpy[1]),
        np.radians(off_rpy[2]),
    )
    t_end_tool = np.array(off_xyz, dtype=np.float64)
    t_end_tool_tf = make_transform(r_end_tool, t_end_tool)
    t_tool_end_tf = invert_transform(t_end_tool_tf)

    # 工具目标姿态：
    # - none: 沿用当前末端姿态
    # - parallel/anti_parallel: 令工具Z轴与标定板Z轴平行（或反向平行）
    r_base_tool_target = r_base2tool
    cam_k = None
    cam_dist = None
    if cam is not None:
        cam_k = np.array(cam.k, dtype=np.float64).reshape(3, 3)
        cam_dist = np.array(cam.d, dtype=np.float64).reshape(-1, 1)
    board_pose = None
    if (args.align_z_mode != "none" or args.visualize) and not use_direct_cam:
        rvec_board, tvec_board, r_cam_board = solve_board_pose_cam(
            corners2=corners2,
            cam_k=cam_k,
            cam_dist=cam_dist,
            cols=args.cols,
            rows=args.rows,
            square=args.square,
        )
        board_pose = (rvec_board, tvec_board)
    if args.align_z_mode != "none":
        if use_direct_cam:
            raise ValueError("direct_cam_xyz 模式不支持 --align-z-mode，请改为 --align-z-mode none")
        z_cam_board = r_cam_board[:, 2]
        z_base_board = r_base2tool @ (r_cam2tool @ z_cam_board)
        if args.align_z_mode == "anti_parallel":
            z_base_board = -z_base_board
        ref_x = r_base2tool[:, 0]
        r_base_tool_target = build_rotation_with_z(z_base_board, ref_x)

    t_base_tool_target = p_base.reshape(3)
    t_base_tool_target_tf = make_transform(r_base_tool_target, t_base_tool_target)

    t_base_end_target_tf = t_base_tool_target_tf @ t_tool_end_tf
    t_base_end_target = t_base_end_target_tf[:3, 3]
    r_base_end_target = t_base_end_target_tf[:3, :3]
    tx_mm, ty_mm, tz_mm = (t_base_end_target * 1000.0).tolist()
    trx, try_, trz = rot_to_rpy_deg_order_zyx(r_base_end_target)

    print("\n=== End Pose Target (Tool Tip To Point) ===")
    print(
        "tool_offset(m): "
        f"[{t_end_tool[0]:.6f}, {t_end_tool[1]:.6f}, {t_end_tool[2]:.6f}], "
        f"tool_rpy_deg(zyx): [{off_rpy[0]:.3f}, {off_rpy[1]:.3f}, {off_rpy[2]:.3f}]"
    )
    print(
        f"P_end_base(mm): x={tx_mm:.3f}, y={ty_mm:.3f}, z={tz_mm:.3f}, "
        f"rx={trx:.3f}, ry={try_:.3f}, rz={trz:.3f}"
    )
    print(f"align_z_mode: {args.align_z_mode}")

    if (args.visualize or args.vis_output) and use_direct_cam:
        print("\n[warn] direct_cam_xyz 模式下无图像输入，忽略可视化参数")

    if (args.visualize or args.vis_output) and not use_direct_cam:
        vis = draw_selected_corner(color_to_bgr(color_msg), u, v, args.corner_row, args.corner_col, z)
        if board_pose is not None:
            axis_len = max(0.01, args.square * 2.0)
            rvec_board, tvec_board = board_pose
            vis = draw_labeled_board_axes(vis, cam_k, cam_dist, rvec_board, tvec_board, axis_len)
        if args.vis_output:
            out_path = Path(args.vis_output).expanduser().resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(out_path), vis)
            if ok:
                print(f"\nvisualization saved: {out_path}")
            else:
                print(f"\n[warn] visualization save failed: {out_path}")
        if args.visualize:
            cv2.imshow("Selected Corner", vis)
            print("\nvisualization: 按任意键关闭窗口")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    rx, ry, rz = rot_to_rpy_deg(r_base2tool)
    print("\nmove parameters (姿态沿用当前末端位姿):")
    print(
        f"actuator_id={args.actuator_id}, command_type=move, "
        f"parameters=[x={x_mm:.3f}, y={y_mm:.3f}, z={z_mm:.3f}, "
        f"rx={rx:.3f}, ry={ry:.3f}, rz={rz:.3f}, speed={args.speed:.1f}]"
    )
    print("\nmove parameters (夹具TCP到目标点，已反算末端目标):")
    print(
        f"actuator_id={args.actuator_id}, command_type=move, "
        f"parameters=[x={tx_mm:.3f}, y={ty_mm:.3f}, z={tz_mm:.3f}, "
        f"rx={trx:.3f}, ry={try_:.3f}, rz={trz:.3f}, speed={args.speed:.1f}]"
    )

    print("\naction command example (原末端目标):")
    print(
        "ros2 action send_goal /crp_robot/execute_task actuator_msgs/action/ExecuteTask "
        f"\"{{command: {{actuator_id: '{args.actuator_id}', command_type: 'move', "
        f"parameters: ['x={x_mm:.3f}','y={y_mm:.3f}','z={z_mm:.3f}',"
        f"'rx={rx:.3f}','ry={ry:.3f}','rz={rz:.3f}','speed={args.speed:.1f}'], "
        "timeout: 30.0, is_async: false}}\""
    )
    print("\naction command example (夹具TCP到目标点，已反算末端目标):")
    print(
        "ros2 action send_goal /crp_robot/execute_task actuator_msgs/action/ExecuteTask "
        f"\"{{command: {{actuator_id: '{args.actuator_id}', command_type: 'move', "
        f"parameters: ['x={tx_mm:.3f}','y={ty_mm:.3f}','z={tz_mm:.3f}',"
        f"'rx={trx:.3f}','ry={try_:.3f}','rz={trz:.3f}','speed={args.speed:.1f}'], "
        "timeout: 30.0, is_async: false}}\""
    )

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
