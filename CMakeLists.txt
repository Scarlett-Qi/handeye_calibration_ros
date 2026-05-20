cmake_minimum_required(VERSION 3.8)
project(handeye_calibration_ros)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(handeye_sample_collector
  src/handeye_sample_collector.cpp
)
ament_target_dependencies(handeye_sample_collector
  rclcpp
  sensor_msgs
  geometry_msgs
  std_srvs
)
target_link_libraries(handeye_sample_collector
  ${OpenCV_LIBS}
)

install(TARGETS
  handeye_sample_collector
  DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/hand_eye_calibrate.py
  scripts/handeye_pick_test.py
  scripts/handeye_validate_no_depth.py
  scripts/quaternion_euler_converter.py
  scripts/rpy_to_matrix.py
  scripts/pose_to_pose_stamped.py
  scripts/handeye_gui.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
  config
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
