#!/usr/bin/env python3
# coding=utf-8
"""
眼在手上手眼标定脚本（适配 handeye_data/poses.csv 格式）

输入：
1. 标定板图片（png/jpg）
2. poses.csv（列至少包含 image, wx, wy, wz, wrx, wry, wrz）

输出：
1. 控制台打印相机到末端变换
2. 可选保存到 yaml
"""

import argparse
import csv
import io
import re
from pathlib import Path

import cv2
import numpy as np
import yaml

np.set_printoptions(precision=8, suppress=True)
ALL_EULER_ORDERS = ["xyz", "xzy", "yxz", "yzx", "zxy", "zyx"]


def euler_to_rotation_matrix(rx: float, ry: float, rz: float, order: str = "zyx") -> np.ndarray:
    if order not in ALL_EULER_ORDERS:
        raise ValueError(f"不支持的欧拉顺序: {order}")

    rxm = np.array(
        [[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]], dtype=np.float64
    )
    rym = np.array(
        [[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]], dtype=np.float64
    )
    rzm = np.array(
        [[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]], dtype=np.float64
    )
    mat_map = {"x": rxm, "y": rym, "z": rzm}
    return mat_map[order[0]] @ mat_map[order[1]] @ mat_map[order[2]]


def invert_rt(r: np.ndarray, t: np.ndarray):
    r_inv = r.T
    t_inv = -r_inv @ t
    return r_inv, t_inv


def sanitize_csv_text(text: str) -> str:
    # 处理偶发“缺失换行导致两条记录粘连”的情况：
    # ...<stamp_ns>rgb_000001_xxx.png,...  -> ...<stamp_ns>\nrgb_000001_xxx.png,...
    fixed = re.sub(r"(?<=\d)(?=rgb_\d{6}_[0-9]+\.(?:png|jpg|jpeg),)", "\n", text)
    return fixed


def load_robot_poses(csv_path: Path):
    text = csv_path.read_text(encoding="utf-8")
    text = sanitize_csv_text(text)
    reader = csv.DictReader(io.StringIO(text))

    pose_map = {}
    required = ["image", "wx", "wy", "wz", "wrx", "wry", "wrz"]
    if reader.fieldnames is None:
        raise ValueError(f"CSV无表头: {csv_path}")
    for c in required:
        if c not in reader.fieldnames:
            raise ValueError(f"CSV缺少列: {c}")

    for row in reader:
        name = row["image"].strip()
        if not name:
            continue
        try:
            wx = float(row["wx"])
            wy = float(row["wy"])
            wz = float(row["wz"])
            wrx = float(row["wrx"])
            wry = float(row["wry"])
            wrz = float(row["wrz"])
        except Exception:
            continue
        pose_map[name] = (wx, wy, wz, wrx, wry, wrz)
    return pose_map


def collect_calib_data(
    image_dir: Path,
    pose_map,
    cols: int,
    rows: int,
    square: float,
    euler_order: str,
    invert_gripper_pose: bool,
    visualize: bool,
    vis_delay: int,
):
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square

    obj_points = []
    img_points = []
    r_gripper2base = []
    t_gripper2base = []
    image_size = None

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not images:
        raise ValueError(f"未找到图片: {image_dir}")

    ok_count = 0
    for img_path in images:
        img_name = img_path.name
        if img_name not in pose_map:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        if not found:
            if visualize:
                vis = img.copy()
                cv2.putText(
                    vis,
                    f"{img_name} corner not found",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("Chessboard", vis)
                cv2.waitKey(max(1, vis_delay))
            continue

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        if visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
            cv2.putText(
                vis,
                img_name,
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Chessboard", vis)
            cv2.waitKey(max(1, vis_delay))

        obj_points.append(objp)
        img_points.append(corners)

        wx, wy, wz, wrx, wry, wrz = pose_map[img_name]
        r = euler_to_rotation_matrix(wrx, wry, wrz, euler_order)
        t = np.array([[wx], [wy], [wz]], dtype=np.float64)
        if invert_gripper_pose:
            r, t = invert_rt(r, t)
        r_gripper2base.append(r)
        t_gripper2base.append(t)
        ok_count += 1

    if ok_count < 6:
        raise ValueError(f"有效样本太少: {ok_count}，建议至少 10 组")
    return obj_points, img_points, image_size, r_gripper2base, t_gripper2base


def run_hand_eye(args):
    image_dir = Path(args.image_dir).expanduser().resolve()
    pose_csv = Path(args.pose_csv).expanduser().resolve()

    pose_map = load_robot_poses(pose_csv)
    (
        obj_points,
        img_points,
        image_size,
        r_gripper2base,
        t_gripper2base,
    ) = collect_calib_data(
        image_dir,
        pose_map,
        args.cols,
        args.rows,
        args.square,
        args.euler_order,
        args.invert_gripper_pose,
        args.visualize,
        args.vis_delay,
    )

    ret, k, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)
    print(f"camera rms error: {ret:.6f}")
    print("camera K:\n", k)
    print("camera dist:\n", dist)

    method_map = {
        "tsai": cv2.CALIB_HAND_EYE_TSAI,
        "park": cv2.CALIB_HAND_EYE_PARK,
        "horaud": cv2.CALIB_HAND_EYE_HORAUD,
        "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
        "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
    }
    method = method_map[args.method]

    r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        r_gripper2base, t_gripper2base, rvecs, tvecs, method=method
    )

    print(f"\n[info] euler_order = {args.euler_order}")
    print(f"[info] invert_gripper_pose = {args.invert_gripper_pose}")
    print("\n=== Hand-Eye Result ===")
    print("R:\n", r_cam2gripper)
    print("t:\n", t_cam2gripper.reshape(3))
    print("||t|| (m):", float(np.linalg.norm(t_cam2gripper)))
    t_gripper2cam = (-r_cam2gripper.T @ t_cam2gripper).reshape(3)
    print("t_gripper2cam:\n", t_gripper2cam)
    print("||t_gripper2cam|| (m):", float(np.linalg.norm(t_gripper2cam)))

    if args.scan_orders:
        print("\n=== Euler Order Scan (translation norm only) ===")
        print("order\tnorm(m)")
        for order in ALL_EULER_ORDERS:
            (
                _obj_points,
                _img_points,
                _image_size,
                _r_gripper2base,
                _t_gripper2base,
            ) = collect_calib_data(
                image_dir,
                pose_map,
                args.cols,
                args.rows,
                args.square,
                order,
                args.invert_gripper_pose,
                args.visualize,
                args.vis_delay,
            )
            _, _, _, _rvecs, _tvecs = cv2.calibrateCamera(
                _obj_points, _img_points, _image_size, None, None
            )
            _, t = cv2.calibrateHandEye(
                _r_gripper2base, _t_gripper2base, _rvecs, _tvecs, method=method
            )
            n = float(np.linalg.norm(t))
            print(f"{order}\t{n:.6f}")

    if args.output:
        out = Path(args.output).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "method": args.method,
            "euler_order": args.euler_order,
            "invert_gripper_pose": bool(args.invert_gripper_pose),
            "camera_matrix": np.asarray(k).tolist(),
            "dist_coeffs": np.asarray(dist).reshape(-1).tolist(),
            "R_cam2gripper": np.asarray(r_cam2gripper).tolist(),
            "t_cam2gripper": np.asarray(t_cam2gripper).reshape(-1).tolist(),
            "t_gripper2cam": t_gripper2cam.tolist(),
            "samples": len(r_gripper2base),
        }
        out.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        print(f"saved result: {out}")


def parse_args():
    parser = argparse.ArgumentParser(description="眼在手上手眼标定")
    parser.add_argument(
        "--image-dir",
        default="handeye_data",
        help="图片目录，默认 handeye_data",
    )
    parser.add_argument(
        "--pose-csv",
        default="handeye_data/poses.csv",
        help="末端位姿CSV路径，默认 handeye_data/poses.csv",
    )
    parser.add_argument("--cols", type=int, default=11, help="棋盘角点列数，默认11")
    parser.add_argument("--rows", type=int, default=7, help="棋盘角点行数，默认7")
    parser.add_argument("--square", type=float, default=0.028, help="棋盘格尺寸(米)，默认0.028")
    parser.add_argument(
        "--method",
        choices=["tsai", "park", "horaud", "andreff", "daniilidis"],
        default="tsai",
        help="手眼标定算法",
    )
    parser.add_argument(
        "--output",
        default="handeye_data/handeye_result.yaml",
        help="结果yaml输出路径",
    )
    parser.add_argument(
        "--euler-order",
        choices=ALL_EULER_ORDERS,
        default="zyx",
        help="欧拉角组合顺序，默认 zyx (Rz*Ry*Rx)",
    )
    parser.add_argument(
        "--scan-orders",
        action="store_true",
        help="扫描全部欧拉顺序并打印平移模长对比",
    )
    parser.add_argument(
        "--invert-gripper-pose",
        action="store_true",
        help="将机械臂位姿由 base->gripper 取逆为 gripper->base（或反向）后再计算",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="显示每张图的棋盘角点检测结果窗口",
    )
    parser.add_argument(
        "--vis-delay",
        type=int,
        default=200,
        help="可视化时每帧停留毫秒数，默认200",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run_hand_eye(args)
    finally:
        if args.visualize:
            cv2.destroyAllWindows()
