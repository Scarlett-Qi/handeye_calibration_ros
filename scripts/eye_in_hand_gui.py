#!/usr/bin/env python3
# coding: utf-8

import csv
import math
import os
import sys
import threading
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from PyQt5 import QtCore, QtGui, QtWidgets
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_srvs.srv import Trigger

import cv2
import yaml

from hand_eye_calibrate import ALL_EULER_ORDERS, collect_calib_data, load_robot_poses


os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)
os.environ["QT_PLUGIN_PATH"] = QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)


CSV_HEADER = ["image", "wx", "wy", "wz", "wrx", "wry", "wrz", "qx", "qy", "qz", "qw", "stamp_ns", "frame_id"]


def quat_to_rpy(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if not msg.data:
        raise ValueError("empty image")
    if msg.width <= 0 or msg.height <= 0:
        raise ValueError("invalid image size")

    height = int(msg.height)
    width = int(msg.width)
    step = int(msg.step)
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if msg.encoding == "bgr8":
        row = data.reshape((height, step))[:, : width * 3]
        return row.reshape((height, width, 3)).copy()
    if msg.encoding == "rgb8":
        row = data.reshape((height, step))[:, : width * 3]
        rgb = row.reshape((height, width, 3))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if msg.encoding == "mono8":
        row = data.reshape((height, step))[:, :width]
        return cv2.cvtColor(row, cv2.COLOR_GRAY2BGR)
    if msg.encoding == "bgra8":
        row = data.reshape((height, step))[:, : width * 4]
        bgra = row.reshape((height, width, 4))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if msg.encoding == "rgba8":
        row = data.reshape((height, step))[:, : width * 4]
        rgba = row.reshape((height, width, 4))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)

    raise ValueError(f"unsupported image encoding: {msg.encoding}")


class HandeyeQtNode(Node):
    def __init__(self):
        super().__init__("eye_in_hand_gui")
        self._lock = threading.Lock()
        self._image: Optional[Image] = None
        self._pose: Optional[Pose] = None
        self._image_count = 0
        self._pose_count = 0
        self._image_sub = None
        self._pose_sub = None
        self._next_client = None

    def connect_topics(self, image_topic: str, pose_topic: str):
        if self._image_sub is not None:
            self.destroy_subscription(self._image_sub)
        if self._pose_sub is not None:
            self.destroy_subscription(self._pose_sub)

        with self._lock:
            self._image = None
            self._pose = None
            self._image_count = 0
            self._pose_count = 0

        self._image_sub = self.create_subscription(Image, image_topic, self._on_image, 10)
        self._pose_sub = self.create_subscription(Pose, pose_topic, self._on_pose, 10)
        self.get_logger().info(f"subscribed image={image_topic}, pose={pose_topic}")

    def set_next_service(self, service_name: str):
        self._next_client = self.create_client(Trigger, service_name)
        self.get_logger().info(f"next-pose service={service_name}")

    def call_next_pose(self, callback):
        if self._next_client is None:
            callback(False, "next service is not configured")
            return
        if not self._next_client.service_is_ready():
            callback(False, "next service is not ready")
            return
        future = self._next_client.call_async(Trigger.Request())
        future.add_done_callback(lambda done: self._finish_trigger(done, callback))

    def snapshot(self):
        with self._lock:
            return self._image, self._pose, self._image_count, self._pose_count

    def _finish_trigger(self, future, callback):
        try:
            result = future.result()
            callback(bool(result.success), result.message)
        except Exception as exc:
            callback(False, str(exc))

    def _on_image(self, msg: Image):
        with self._lock:
            self._image = msg
            self._image_count += 1

    def _on_pose(self, msg: Pose):
        with self._lock:
            self._pose = msg
            self._pose_count += 1


class ImageView(QtWidgets.QLabel):
    def __init__(self):
        super().__init__("等待图像...")
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setMinimumSize(240, 135)
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Ignored)
        self.setStyleSheet("background:#202020;color:#e6e6e6;border:1px solid #404040;")
        self._pixmap = QtGui.QPixmap()

    def set_cv_image(self, bgr: np.ndarray):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        bytes_per_line = 3 * w
        image = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).copy()
        self._pixmap = QtGui.QPixmap.fromImage(image)
        self._update_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self):
        if self._pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 0 or target_size.height() <= 0:
            return
        scaled = self._pixmap.scaled(target_size, QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.setPixmap(scaled)


class MainWindow(QtWidgets.QMainWindow):
    next_done = QtCore.pyqtSignal(bool, str)
    calibrate_done = QtCore.pyqtSignal(bool, str)

    def __init__(self, node: HandeyeQtNode):
        super().__init__()
        self.node = node
        self.sample_index = 0
        self.last_image_count = -1

        self.setWindowTitle("Eye-in-Hand Calibration Collector")
        self.resize(980, 760)
        self.setMinimumSize(820, 620)

        self.output_dir = QtWidgets.QLineEdit("./handeye_data")
        self.image_prefix = QtWidgets.QLineEdit("rgb")
        self.csv_name = QtWidgets.QLineEdit("poses.csv")
        self.image_topic = QtWidgets.QLineEdit("/camera/color/image_raw")
        self.pose_topic = QtWidgets.QLineEdit("/handeye/tool_pose")
        self.pose_frame_id = QtWidgets.QLineEdit("base")
        self.next_service = QtWidgets.QLineEdit("/handeye/next_pose")
        self.calib_cols = QtWidgets.QSpinBox()
        self.calib_cols.setRange(2, 30)
        self.calib_cols.setValue(8)
        self.calib_rows = QtWidgets.QSpinBox()
        self.calib_rows.setRange(2, 30)
        self.calib_rows.setValue(8)
        self.calib_square = QtWidgets.QDoubleSpinBox()
        self.calib_square.setRange(0.001, 1.0)
        self.calib_square.setDecimals(4)
        self.calib_square.setSingleStep(0.001)
        self.calib_square.setValue(0.035)
        self.calib_method = QtWidgets.QComboBox()
        self.calib_method.addItems(["tsai", "park", "horaud", "andreff", "daniilidis"])
        self.calib_method.setCurrentText("tsai")
        self.calib_euler_order = QtWidgets.QComboBox()
        self.calib_euler_order.addItems(ALL_EULER_ORDERS)
        self.calib_euler_order.setCurrentText("zyx")
        self.calib_invert = QtWidgets.QCheckBox("反转机械臂位姿")
        self.calib_output = QtWidgets.QLineEdit("handeye_result.yaml")
        self.calib_result = QtWidgets.QPlainTextEdit()
        self.calib_result.setReadOnly(True)
        self.calib_result.setFixedHeight(90)
        self.calib_result.setPlaceholderText("标定结果")

        self.image_status = QtWidgets.QLabel("图像: 0")
        self.pose_status = QtWidgets.QLabel("姿态: 0")
        self.saved_status = QtWidgets.QLabel("已保存: 0")
        self.status = QtWidgets.QLabel("未连接")
        self.image_view = ImageView()
        self.saved_image_view = ImageView()
        self.saved_image_view.setText("等待保存图片...")

        self._build_ui()
        self.next_done.connect(self._on_next_done)
        self.calibrate_done.connect(self._on_calibrate_done)

        self.connect_topics()
        self.update_next_service()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(100)

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 12)
        root.setSpacing(10)

        save_group = QtWidgets.QGroupBox("保存设置")
        save_form = QtWidgets.QGridLayout(save_group)
        self._add_row(save_form, 0, "输出路径", self.output_dir)
        browse = QtWidgets.QPushButton("选择")
        browse.clicked.connect(self.choose_output_dir)
        save_form.addWidget(browse, 0, 2)
        self._add_row(save_form, 1, "图片前缀", self.image_prefix)
        self._add_row(save_form, 2, "CSV文件名", self.csv_name)
        root.addWidget(save_group)

        ros_group = QtWidgets.QGroupBox("ROS接口")
        ros_form = QtWidgets.QGridLayout(ros_group)
        self._add_row(ros_form, 0, "图像话题", self.image_topic)
        self._add_row(ros_form, 1, "姿态话题", self.pose_topic)
        self._add_row(ros_form, 2, "姿态坐标系", self.pose_frame_id)
        self._add_row(ros_form, 3, "下一步服务", self.next_service)
        reconnect = QtWidgets.QPushButton("重新连接话题")
        reconnect.clicked.connect(self.connect_topics)
        service = QtWidgets.QPushButton("更新移动服务")
        service.clicked.connect(self.update_next_service)
        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(reconnect)
        button_row.addWidget(service)
        button_row.addStretch(1)
        ros_form.addLayout(button_row, 4, 1, 1, 2)
        root.addWidget(ros_group)

        status_row = QtWidgets.QHBoxLayout()
        status_row.addWidget(self.image_status)
        status_row.addWidget(self.pose_status)
        status_row.addWidget(self.saved_status)
        status_row.addStretch(1)
        root.addLayout(status_row)

        action_row = QtWidgets.QHBoxLayout()
        next_button = QtWidgets.QPushButton("下一步移动")
        next_button.setMinimumHeight(32)
        next_button.clicked.connect(self.next_pose)
        save_button = QtWidgets.QPushButton("保存图片和姿态")
        save_button.setMinimumHeight(32)
        save_button.clicked.connect(self.save_sample)
        reset_button = QtWidgets.QPushButton("重置采集")
        reset_button.setMinimumHeight(32)
        reset_button.clicked.connect(self.reset_collection)
        close_button = QtWidgets.QPushButton("退出")
        close_button.setMinimumHeight(32)
        close_button.clicked.connect(self.close)
        action_row.addWidget(next_button)
        action_row.addWidget(save_button)
        action_row.addWidget(reset_button)
        action_row.addStretch(1)
        action_row.addWidget(close_button)
        root.addLayout(action_row)

        preview_group = QtWidgets.QGroupBox("图像预览")
        preview_group.setMaximumHeight(280)
        preview_group.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        preview_layout = QtWidgets.QHBoxLayout(preview_group)
        live_group = QtWidgets.QGroupBox("实时图像")
        live_layout = QtWidgets.QVBoxLayout(live_group)
        live_layout.addWidget(self.image_view)
        saved_group = QtWidgets.QGroupBox("最新保存")
        saved_layout = QtWidgets.QVBoxLayout(saved_group)
        saved_layout.addWidget(self.saved_image_view)
        preview_layout.addWidget(live_group)
        preview_layout.addWidget(saved_group)
        root.addWidget(preview_group, stretch=1)

        calib_group = QtWidgets.QGroupBox("标定设置 / 结果")
        calib_layout = QtWidgets.QVBoxLayout(calib_group)
        calib_top = QtWidgets.QGridLayout()
        calib_top.addWidget(QtWidgets.QLabel("角点列数"), 0, 0)
        calib_top.addWidget(self.calib_cols, 0, 1)
        calib_top.addWidget(QtWidgets.QLabel("角点行数"), 0, 2)
        calib_top.addWidget(self.calib_rows, 0, 3)
        calib_top.addWidget(QtWidgets.QLabel("格子边长(m)"), 0, 4)
        calib_top.addWidget(self.calib_square, 0, 5)
        calib_top.addWidget(QtWidgets.QLabel("算法"), 1, 0)
        calib_top.addWidget(self.calib_method, 1, 1)
        calib_top.addWidget(QtWidgets.QLabel("欧拉顺序"), 1, 2)
        calib_top.addWidget(self.calib_euler_order, 1, 3)
        calib_top.addWidget(self.calib_invert, 2, 1)
        start_calib = QtWidgets.QPushButton("开始标定")
        start_calib.setMinimumHeight(36)
        start_calib.clicked.connect(self.start_calibration)
        calib_top.addWidget(start_calib, 2, 5)
        calib_top.addWidget(QtWidgets.QLabel("结果文件"), 3, 0)
        calib_top.addWidget(self.calib_output, 3, 1, 1, 5)
        calib_top.setColumnStretch(5, 1)
        calib_layout.addLayout(calib_top)
        calib_layout.addWidget(self.calib_result)
        root.addWidget(calib_group)

        root.addWidget(self.status)

    def _add_row(self, layout, row: int, label: str, widget: QtWidgets.QWidget):
        layout.addWidget(QtWidgets.QLabel(label), row, 0)
        layout.addWidget(widget, row, 1)
        layout.setColumnStretch(1, 1)

    def choose_output_dir(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(self, "选择输出路径", self.output_dir.text() or ".")
        if selected:
            self.output_dir.setText(selected)

    def connect_topics(self):
        image_topic = self.image_topic.text().strip()
        pose_topic = self.pose_topic.text().strip()
        if not image_topic or not pose_topic:
            QtWidgets.QMessageBox.critical(self, "参数错误", "图像话题和姿态话题不能为空")
            return
        self.node.connect_topics(image_topic, pose_topic)
        self.status.setText(f"已连接: image={image_topic}, pose={pose_topic}")

    def update_next_service(self):
        service = self.next_service.text().strip()
        if not service:
            QtWidgets.QMessageBox.critical(self, "参数错误", "下一步服务名不能为空")
            return
        self.node.set_next_service(service)
        self.status.setText(f"移动服务: {service}")

    def next_pose(self):
        self.status.setText("正在请求机械臂移动到下一步...")
        self.node.call_next_pose(lambda ok, msg: self.next_done.emit(ok, msg))

    def _on_next_done(self, ok: bool, msg: str):
        state = "成功" if ok else "失败"
        text = f"下一步移动{state}: {msg}"
        self.status.setText(text)

    def save_sample(self):
        image, pose, _, _ = self.node.snapshot()
        if image is None:
            QtWidgets.QMessageBox.critical(self, "无法保存", "还没有收到图像")
            return
        if pose is None:
            QtWidgets.QMessageBox.critical(self, "无法保存", "还没有收到机械臂姿态")
            return

        output_dir = Path(self.output_dir.text()).expanduser()
        image_prefix = self.image_prefix.text().strip() or "rgb"
        csv_name = self.csv_name.text().strip() or "poses.csv"
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            bgr = image_msg_to_bgr(image)
        except Exception as exc:
            QtWidgets.QMessageBox.critical(self, "图像转换失败", str(exc))
            return

        stamp_ns = self.node.get_clock().now().nanoseconds
        image_name = f"{image_prefix}_{self.sample_index:06d}_{stamp_ns}.png"
        self.sample_index += 1
        image_path = output_dir / image_name
        csv_path = output_dir / csv_name

        if not cv2.imwrite(str(image_path), bgr):
            QtWidgets.QMessageBox.critical(self, "保存失败", f"无法写入图片: {image_path}")
            return

        self._append_pose(csv_path, image_name, pose, stamp_ns)
        self.saved_image_view.set_cv_image(bgr)
        self.saved_status.setText(f"已保存: {self.sample_index}")
        text = f"已保存样本: {image_path}"
        self.status.setText(text)

    def reset_collection(self):
        self.sample_index = 0
        self.saved_status.setText("已保存: 0")
        self.status.setText("已重置采集计数，已保存文件未删除")

    def start_calibration(self):
        image_dir = Path(self.output_dir.text()).expanduser()
        pose_csv = image_dir / (self.csv_name.text().strip() or "poses.csv")
        output_name = self.calib_output.text().strip() or "handeye_result.yaml"
        output_path = Path(output_name).expanduser()
        if not output_path.is_absolute():
            output_path = image_dir / output_path

        params = {
            "image_dir": image_dir,
            "pose_csv": pose_csv,
            "output_path": output_path,
            "cols": int(self.calib_cols.value()),
            "rows": int(self.calib_rows.value()),
            "square": float(self.calib_square.value()),
            "method": self.calib_method.currentText(),
            "euler_order": self.calib_euler_order.currentText(),
            "invert_gripper_pose": self.calib_invert.isChecked(),
        }
        self.status.setText("正在标定...")
        self.calib_result.setPlainText("正在检测棋盘角点并计算手眼结果，请稍等...")
        threading.Thread(target=self._run_calibration, args=(params,), daemon=True).start()

    def _run_calibration(self, params):
        try:
            result_text = self._calibrate(params)
            self.calibrate_done.emit(True, result_text)
        except Exception as exc:
            self.calibrate_done.emit(False, str(exc))

    def _calibrate(self, params) -> str:
        image_dir = params["image_dir"].resolve()
        pose_csv = params["pose_csv"].resolve()
        output_path = params["output_path"].resolve()

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
            params["cols"],
            params["rows"],
            params["square"],
            params["euler_order"],
            params["invert_gripper_pose"],
            False,
            1,
        )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        method_map = {
            "tsai": cv2.CALIB_HAND_EYE_TSAI,
            "park": cv2.CALIB_HAND_EYE_PARK,
            "horaud": cv2.CALIB_HAND_EYE_HORAUD,
            "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
            "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
        }
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            rvecs,
            tvecs,
            method=method_map[params["method"]],
        )
        t_cam2gripper_vec = np.asarray(t_cam2gripper).reshape(3)

        data = {
            "calibration_mode": "eye_in_hand",
            "method": params["method"],
            "euler_order": params["euler_order"],
            "invert_gripper_pose": bool(params["invert_gripper_pose"]),
            "camera_rms_error": float(rms),
            "camera_matrix": np.asarray(camera_matrix).tolist(),
            "dist_coeffs": np.asarray(dist_coeffs).reshape(-1).tolist(),
            "R_cam2gripper": np.asarray(r_cam2gripper).tolist(),
            "t_cam2gripper": t_cam2gripper_vec.tolist(),
            "samples": len(r_gripper2base),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

        r_text = np.array2string(np.asarray(r_cam2gripper), precision=8, suppress_small=True)
        t_text = np.array2string(t_cam2gripper_vec, precision=8, suppress_small=True)
        return (
            f"标定完成\n"
            f"有效样本: {len(r_gripper2base)}\n"
            f"模式: 眼在手上 eye-in-hand\n"
            f"算法: {params['method']}\n"
            f"camera RMS error: {float(rms):.6f}\n"
            f"R_cam2gripper:\n{r_text}\n"
            f"t_cam2gripper(m): {t_text}\n"
            f"||t_cam2gripper||: {float(np.linalg.norm(t_cam2gripper_vec)):.6f} m\n"
            f"结果文件: {output_path}"
        )

    def _on_calibrate_done(self, ok: bool, text: str):
        if ok:
            self.calib_result.setPlainText(text)
            self.status.setText("标定完成")
        else:
            self.calib_result.setPlainText(f"标定失败:\n{text}")
            self.status.setText("标定失败")

    def _append_pose(self, csv_path: Path, image_name: str, pose: Pose, stamp_ns: int):
        need_header = not csv_path.exists()
        q = pose.orientation
        p = pose.position
        roll, pitch, yaw = quat_to_rpy(q.x, q.y, q.z, q.w)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if need_header:
                writer.writerow(CSV_HEADER)
            writer.writerow(
                [
                    image_name,
                    f"{p.x:.12g}",
                    f"{p.y:.12g}",
                    f"{p.z:.12g}",
                    f"{roll:.12g}",
                    f"{pitch:.12g}",
                    f"{yaw:.12g}",
                    f"{q.x:.12g}",
                    f"{q.y:.12g}",
                    f"{q.z:.12g}",
                    f"{q.w:.12g}",
                    stamp_ns,
                    self.pose_frame_id.text().strip() or "base",
                ]
            )

    def refresh(self):
        image, pose, image_count, pose_count = self.node.snapshot()
        if image is None:
            self.image_status.setText(f"图像: {image_count}")
        else:
            self.image_status.setText(f"图像: {image_count}  {image.width}x{image.height}  {image.encoding}")

        if pose is None:
            self.pose_status.setText(f"姿态: {pose_count}")
        else:
            self.pose_status.setText(f"姿态: {pose_count}  frame={self.pose_frame_id.text().strip() or 'base'}")

        if image is not None and image_count != self.last_image_count:
            try:
                self.image_view.set_cv_image(image_msg_to_bgr(image))
                self.last_image_count = image_count
            except Exception as exc:
                self.image_view.setText(f"图像预览失败: {exc}")
                self.last_image_count = image_count

def main():
    rclpy.init()
    node = HandeyeQtNode()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(node)
    window.show()

    def spin_node():
        try:
            rclpy.spin(node)
        except ExternalShutdownException:
            pass

    spin_thread = threading.Thread(target=spin_node, daemon=True)
    spin_thread.start()

    try:
        code = app.exec_()
    finally:
        node.destroy_node()
        rclpy.shutdown()
        spin_thread.join(timeout=1.0)
    sys.exit(code)


if __name__ == "__main__":
    main()
