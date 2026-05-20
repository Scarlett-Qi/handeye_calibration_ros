#!/usr/bin/env python3
# coding: utf-8
"""
无深度手眼验证脚本（眼在手上）

思路：
1) 每张图检测棋盘角点并 solvePnP，得到 T_cam_target
2) 由机器人位姿得到 T_base_tool
3) 由手眼结果读取 T_tool_cam (cam->tool)
4) 链式计算 T_base_target = T_base_tool * T_tool_cam * T_cam_target
5) 若手眼正确且标定板静止，所有样本的 T_base_target 应基本一致

输出：
- 样本数量、角点检测通过数
- 标定板原点在 base 下的离散度（mm）
- 标定板姿态离散度（deg）
- 每个样本相对均值的位置误差（mm）与姿态误差（deg）
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml

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


def invert_rt(r: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r_inv = r.T
    t_inv = -r_inv @ t
    return r_inv, t_inv


def rotation_angle_deg(r_a: np.ndarray, r_b: np.ndarray) -> float:
    r = r_a.T @ r_b
    v = np.clip((np.trace(r) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(v)))


def load_pose_map(csv_path: Path) -> Dict[str, Tuple[float, float, float, float, float, float]]:
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        need = ["image", "wx", "wy", "wz", "wrx", "wry", "wrz"]
        if reader.fieldnames is None:
            raise ValueError(f"CSV无表头: {csv_path}")
        for n in need:
            if n not in reader.fieldnames:
                raise ValueError(f"CSV缺少列: {n}")

        m = {}
        for row in reader:
            name = (row.get("image") or "").strip()
            if not name:
                continue
            try:
                m[name] = (
                    float(row["wx"]),
                    float(row["wy"]),
                    float(row["wz"]),
                    float(row["wrx"]),
                    float(row["wry"]),
                    float(row["wrz"]),
                )
            except Exception:
                continue
    return m


def load_handeye_yaml(path: Path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    r_tool_cam = np.array(data["R_cam2gripper"], dtype=np.float64)
    t_tool_cam = np.array(data["t_cam2gripper"], dtype=np.float64).reshape(3, 1)
    k = np.array(data["camera_matrix"], dtype=np.float64)
    dist = np.array(data["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    return r_tool_cam, t_tool_cam, k, dist


def build_obj_points(cols: int, rows: int, square: float) -> np.ndarray:
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square
    return objp


def main():
    parser = argparse.ArgumentParser(description="无深度手眼验证脚本")
    parser.add_argument("--image-dir", default="handeye_data", help="图片目录")
    parser.add_argument("--pose-csv", default="handeye_data/poses.csv", help="机器人位姿CSV")
    parser.add_argument("--handeye-yaml", default="handeye_data/handeye_result.yaml", help="手眼结果YAML")
    parser.add_argument("--cols", type=int, default=11, help="棋盘内角点列数")
    parser.add_argument("--rows", type=int, default=7, help="棋盘内角点行数")
    parser.add_argument("--square", type=float, default=0.028, help="棋盘格尺寸(米)")
    parser.add_argument(
        "--euler-order", choices=ALL_EULER_ORDERS, default="zyx", help="机器人欧拉角组合顺序"
    )
    parser.add_argument(
        "--invert-gripper-pose",
        action="store_true",
        help="将CSV位姿取逆（base->tool 与 tool->base 定义不一致时使用）",
    )
    parser.add_argument("--topk", type=int, default=10, help="打印误差最大的前K个样本")
    parser.add_argument("--visualize", action="store_true", help="显示角点检测结果")
    parser.add_argument(
        "--vis-output-dir",
        default="",
        help="保存可视化结果目录（会输出worst样本标注图和summary.csv）",
    )
    args = parser.parse_args()

    image_dir = Path(args.image_dir).expanduser().resolve()
    pose_csv = Path(args.pose_csv).expanduser().resolve()
    handeye_yaml = Path(args.handeye_yaml).expanduser().resolve()

    pose_map = load_pose_map(pose_csv)
    r_tool_cam, t_tool_cam, k, dist = load_handeye_yaml(handeye_yaml)
    objp = build_obj_points(args.cols, args.rows, args.square)

    images = sorted(list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg")))
    if not images:
        raise ValueError(f"图片目录为空: {image_dir}")

    t_base_target_list: List[np.ndarray] = []
    r_base_target_list: List[np.ndarray] = []
    names: List[str] = []
    reproj_errs: List[float] = []
    corners_map: Dict[str, np.ndarray] = {}
    name_to_path: Dict[str, Path] = {}
    used = 0
    for p in images:
        name = p.name
        if name not in pose_map:
            continue
        used += 1
        img = cv2.imread(str(p))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (args.cols, args.rows), None)
        if not found:
            found, corners = cv2.findChessboardCornersSB(gray, (args.cols, args.rows), None)
        if not found or corners is None:
            continue

        if corners.ndim == 3:
            corners = corners.reshape(-1, 1, 2)
        corners = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
        )

        ok, rvec, tvec = cv2.solvePnP(
            objp, corners, k, dist, flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not ok:
            continue
        r_cam_target, _ = cv2.Rodrigues(rvec)
        t_cam_target = np.array(tvec, dtype=np.float64).reshape(3, 1)

        wx, wy, wz, wrx, wry, wrz = pose_map[name]
        r_base_tool = euler_to_rotation_matrix(wrx, wry, wrz, args.euler_order)
        t_base_tool = np.array([[wx], [wy], [wz]], dtype=np.float64)
        if args.invert_gripper_pose:
            r_base_tool, t_base_tool = invert_rt(r_base_tool, t_base_tool)

        r_base_target = r_base_tool @ r_tool_cam @ r_cam_target
        t_base_target = r_base_tool @ (r_tool_cam @ t_cam_target + t_tool_cam) + t_base_tool

        proj, _ = cv2.projectPoints(objp, rvec, tvec, k, dist)
        err = float(np.mean(np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))

        names.append(name)
        r_base_target_list.append(r_base_target)
        t_base_target_list.append(t_base_target.reshape(3))
        reproj_errs.append(err)
        corners_map[name] = corners
        name_to_path[name] = p

        if args.visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (args.cols, args.rows), corners, True)
            cv2.putText(
                vis,
                f"{name} reproj={err:.3f}px",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("handeye_validate_no_depth", vis)
            key = cv2.waitKey(80)
            if key == 27:
                args.visualize = False
                cv2.destroyAllWindows()

    if args.visualize:
        cv2.destroyAllWindows()

    if len(names) < 6:
        raise ValueError(f"有效样本过少: {len(names)}（检测通过不足）")

    t_arr = np.array(t_base_target_list, dtype=np.float64)
    t_mean = np.mean(t_arr, axis=0)
    t_std = np.std(t_arr, axis=0)
    t_dev = np.linalg.norm(t_arr - t_mean, axis=1)

    # 旋转参考取第一帧，统计与第一帧夹角
    r_ref = r_base_target_list[0]
    ang_dev = np.array([rotation_angle_deg(r_ref, r) for r in r_base_target_list], dtype=np.float64)

    print("\n=== No-Depth Hand-Eye Validation ===")
    print(f"image_dir: {image_dir}")
    print(f"pose_csv: {pose_csv}")
    print(f"handeye_yaml: {handeye_yaml}")
    print(f"used(with pose): {used}")
    print(f"valid(chessboard+pnp): {len(names)}")
    print(f"euler_order: {args.euler_order}, invert_gripper_pose: {args.invert_gripper_pose}")

    print("\n[Target Origin In Base]")
    print(f"mean(m): [{t_mean[0]:.6f}, {t_mean[1]:.6f}, {t_mean[2]:.6f}]")
    print(f"std(mm): [{t_std[0]*1000:.3f}, {t_std[1]*1000:.3f}, {t_std[2]*1000:.3f}]")
    print(f"pos_dev mean/max (mm): {np.mean(t_dev)*1000:.3f} / {np.max(t_dev)*1000:.3f}")

    print("\n[Target Orientation In Base]")
    print(f"angle_dev mean/max (deg): {np.mean(ang_dev):.4f} / {np.max(ang_dev):.4f}")

    print("\n[PnP Reprojection]")
    print(f"reproj mean/max (px): {np.mean(reproj_errs):.4f} / {np.max(reproj_errs):.4f}")

    # 打印最差样本
    rank = np.argsort(-t_dev)[: max(1, args.topk)]
    print(f"\n[Worst {len(rank)} By Position Dev]")
    for idx in rank:
        print(
            f"{names[idx]}  pos_dev={t_dev[idx]*1000:.3f}mm  "
            f"ang_dev={ang_dev[idx]:.4f}deg  reproj={reproj_errs[idx]:.4f}px"
        )

    if args.vis_output_dir:
        out_dir = Path(args.vis_output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        summary_path = out_dir / "summary.csv"
        with summary_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "pos_dev_mm", "ang_dev_deg", "reproj_px"])
            order = np.argsort(-t_dev)
            for idx in order:
                w.writerow(
                    [
                        names[idx],
                        f"{t_dev[idx]*1000:.6f}",
                        f"{ang_dev[idx]:.6f}",
                        f"{reproj_errs[idx]:.6f}",
                    ]
                )

        for rank_i, idx in enumerate(rank.tolist(), start=1):
            name = names[idx]
            p = name_to_path[name]
            img = cv2.imread(str(p))
            if img is None:
                continue
            corners = corners_map[name]
            cv2.drawChessboardCorners(img, (args.cols, args.rows), corners, True)
            cv2.putText(
                img,
                f"#{rank_i} {name}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                img,
                f"pos_dev={t_dev[idx]*1000:.2f}mm ang_dev={ang_dev[idx]:.2f}deg reproj={reproj_errs[idx]:.3f}px",
                (20, 68),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )
            out_img = out_dir / f"{rank_i:02d}_{name}"
            cv2.imwrite(str(out_img), img)

        print(f"\n[Visualization Saved]")
        print(f"dir: {out_dir}")
        print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
