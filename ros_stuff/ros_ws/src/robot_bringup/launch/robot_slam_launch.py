#!/usr/bin/env python3
"""
robot_bringup/launch/bringup_slam.launch.py   (ROS 2)

► Fixes
  • Publishes the **missing static TF**  base_link → laser  *before* SLAM starts.
  • Removes the (incorrect) static TF  odom → base_link; odom should be dynamic or unused
    because `use_odom` is false in your slam_toolbox YAML.
  • Loads the xacro‑generated URDF into robot_state_publisher exactly once.
  • Keeps every original launch argument so nothing in your workflow breaks.
"""

import os
import xacro
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Package path
    bringup_dir = get_package_share_directory("robot_bringup")

    # -------------------------  Launch arguments  --------------------------
    use_sim_time     = LaunchConfiguration("use_sim_time",     default="false")

    channel_type     = LaunchConfiguration("channel_type",     default="serial")
    serial_port      = LaunchConfiguration("serial_port",      default="/dev/ttyUSB0")
    serial_baudrate  = LaunchConfiguration("serial_baudrate",  default="460800")
    frame_id         = LaunchConfiguration("frame_id",         default="laser")
    inverted         = LaunchConfiguration("inverted",         default="false")
    angle_compensate = LaunchConfiguration("angle_compensate", default="true")
    scan_mode        = LaunchConfiguration("scan_mode",        default="Standard")

    # ---------------------------  File paths  ------------------------------
    urdf_xacro_file  = os.path.join(bringup_dir, "urdf",  "robot.urdf.xacro")
    slam_params_path = os.path.join(bringup_dir, "config", "slam_toolbox_params.yaml")

    # ------------------  Process xacro → robot_description  ----------------
    robot_description = {
        "robot_description": xacro.process_file(urdf_xacro_file).toxml()
    }

    # ----------------------------------------------------------------------
    # Launch description
    # ----------------------------------------------------------------------
    return LaunchDescription([
        # ---- Declare CLI args (for ros2 launch … ) ------------------------
        DeclareLaunchArgument("use_sim_time",     default_value="false"),
        DeclareLaunchArgument("channel_type",     default_value="serial"),
        DeclareLaunchArgument("serial_port",      default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("serial_baudrate",  default_value="460800"),
        DeclareLaunchArgument("frame_id",         default_value="laser"),
        DeclareLaunchArgument("inverted",         default_value="false"),
        DeclareLaunchArgument("angle_compensate", default_value="true"),
        DeclareLaunchArgument("scan_mode",        default_value="Standard"),

        LogInfo(msg="Launching robot bring‑up with SLAM Toolbox (pure‑LiDAR)…"),

        # 1️⃣  **MUST** have laser→base_link TF BEFORE scans arrive
        Node(
            package="tf2_ros",
            executable="static_transform_publisher",
            name="static_tf_pub_base_to_laser",
            arguments=["0", "0", "0", "0", "0", "0", "base_link", "laser"],
            output="screen",
        ),

        # 2️⃣  Robot description → publishes all other URDF TFs
        Node(
            package     = "robot_state_publisher",
            executable  = "robot_state_publisher",
            name        = "robot_state_publisher",
            output      = "screen",
            parameters  = [{"use_sim_time": use_sim_time, **robot_description}],
        ),

        # 3️⃣  Joint‑state publisher (fake joints / sliders)
        Node(
            package     = "joint_state_publisher",
            executable  = "joint_state_publisher",
            name        = "joint_state_publisher",
            output      = "screen",
            parameters  = [{"use_sim_time": use_sim_time}],
        ),

        # 4️⃣  RPLidar driver
        Node(
            package     = "rplidar_ros",
            executable  = "rplidar_node",
            name        = "rplidar_node",
            output      = "screen",
            parameters  = [{
                "channel_type":     channel_type,
                "serial_port":      serial_port,
                "serial_baudrate":  serial_baudrate,
                "frame_id":         frame_id,
                "inverted":         inverted,
                "angle_compensate": angle_compensate,
                "scan_mode":        scan_mode,
            }],
        ),

        # 5️⃣  SLAM Toolbox – async / pure‑LiDAR
        Node(
            package     = "slam_toolbox",
            executable  = "async_slam_toolbox_node",
            name        = "slam_toolbox",
            output      = "screen",
            emulate_tty = True,               # pretty console formatting
            parameters  = [
                slam_params_path,             # YAML with use_odom:false, etc.
                {"use_sim_time": use_sim_time},
            ],
        ),

        LogInfo(msg="Robot SLAM launched successfully!"),
    ])
