# handeye_calibration_ros

这是一个通用 ROS2 手眼标定工具包。核心逻辑不绑定具体机械臂，只要求外部提供标准图像话题和机械臂末端位姿话题。

## 依赖安装

```bash
sudo apt update
sudo apt install -y python3-opencv python3-yaml python3-pyqt5
```

GUI 使用 `PyQt5`。如果只使用命令行采集，可以不启动 GUI。

验证 `PyQt5`：

```bash
python3 -c "from PyQt5 import QtCore, QtGui, QtWidgets; print('PyQt5 ok')"
```

## 克隆编译

在工作空间根目录编译：

```bash
mkdir -p handeye_calibration_ws/src
cd handeye_calibration_ws/src
git clone https://github.com/Scarlett-Qi/handeye_calibration_ros.git
colcon build
```

编译完成后加载环境：
```bash
source install/setup.bash
```

## 接口约定

采集和标定流程主要依赖两个输入：
- 相机图像：`sensor_msgs/msg/Image`
- 末端位姿：`geometry_msgs/msg/Pose`

不同机械臂需要把当前 TCP/末端位姿统一发布为：
```text
/robot/current_pose geometry_msgs/msg/Pose
```

当前默认直接订阅 `geometry_msgs/msg/Pose`。如果机械臂发布的话题名称不是 `/robot/current_pose`，在 GUI 里把“姿态话题”改成机械臂实际话题即可。\
如果某些机械臂只发布 `PoseStamped`，可以用下面的脚本把 `PoseStamped` 转成 `Pose`：
```bash
ros2 run handeye_calibration_ros pose_to_pose_stamped.py \
  --input-topic /robot/current_pose_stamped \
  --output-topic /robot/current_pose
```

## GUI 采集

推荐使用 GUI 采集样本：
```bash
ros2 launch handeye_calibration_ros gui.launch.py
```

界面包含：

- 输出路径
- 图片前缀
- CSV 文件名
- 图像话题
- 姿态话题
- 姿态坐标系
- 下一步移动服务
- 图像预览
- 采集日志
- `下一步移动` 按钮
- `保存图片和姿态` 按钮

点击 `保存图片和姿态` 后，会保存当前图像和当前机械臂末端位姿。

点击 `下一步移动` 时，GUI 会调用：
```text
/handeye/next_pose  std_srvs/srv/Trigger
```
这个服务需要由具体机械臂的适配节点实现。它的作用是让机械臂移动到下一个标定位姿。如果暂时没有这个服务，也可以手动移动机械臂，然后只使用 GUI 的保存按钮。

采集结果目录示例：
```text
handeye_data/run_001/
  rgb_000000_<stamp>.png
  rgb_000001_<stamp>.png
  poses.csv
```

`poses.csv` 格式：
```text
image,wx,wy,wz,wrx,wry,wrz,qx,qy,qz,qw,stamp_ns,frame_id
```

说明：
- `wx, wy, wz`：末端位置，单位为米
- `wrx, wry, wrz`：RPY 欧拉角，单位为弧度
- `qx, qy, qz, qw`：四元数
- `frame_id`：位姿所在坐标系。由于 `Pose` 本身不带 `header`，这个值来自 GUI 的“姿态坐标系”输入框或命令行参数 `tool_pose_frame_id`

## 命令行采集

如果不使用 GUI，也可以启动命令行采集节点：
```bash
ros2 launch handeye_calibration_ros collect.launch.py \
  image_topic:=/camera/color/image_raw \
  tool_pose_topic:=/handeye/tool_pose \
  tool_pose_frame_id:=base \
  output_dir:=handeye_data/run_001
```

机械臂和标定板静止后，调用服务保存一组样本：
```bash
ros2 service call /handeye/save_sample std_srvs/srv/Trigger "{}"
```

## 标定

采集完成后运行：
```bash
ros2 run handeye_calibration_ros hand_eye_calibrate.py \
  --image-dir handeye_data/run_001 \
  --pose-csv handeye_data/run_001/poses.csv \
  --cols 8 --rows 8 --square 0.035 \
  --method daniilidis \
  --output handeye_data/run_001/handeye_result.yaml
```

参数说明：

- `--cols`：棋盘格内角点列数
- `--rows`：棋盘格内角点行数
- `--square`：棋盘格单格边长，单位为米
- `--method`：手眼标定方法，例如 `tsai`、`park`、`horaud`、`andreff`、`daniilidis`

## 验证

```bash
ros2 run handeye_calibration_ros handeye_validate_no_depth.py \
  --image-dir handeye_data/run_001 \
  --pose-csv handeye_data/run_001/poses.csv \
  --handeye-yaml handeye_data/run_001/handeye_result.yaml \
  --cols 8 --rows 8 --square 0.035
```

## 拾取测试

```bash
ros2 run handeye_calibration_ros handeye_pick_test.py \
  --handeye-yaml handeye_data/run_001/handeye_result.yaml \
  --tool-pose-topic /handeye/tool_pose \
  --cam-xyz 0.05,0.02,0.35
```

`--cam-xyz` 是相机坐标系下的目标点坐标，单位为米。

## 使用流程

1. 启动相机，确认图像话题正常。
2. 启动机械臂驱动或适配节点，确认 `/handeye/tool_pose` 正常发布。
3. 启动 GUI。
4. 移动机械臂到多个不同姿态，每个姿态下点击 `保存图片和姿态`。
5. 建议采集至少 15 组有效样本。
6. 运行标定脚本生成 `handeye_result.yaml`。
7. 运行验证脚本检查结果。
