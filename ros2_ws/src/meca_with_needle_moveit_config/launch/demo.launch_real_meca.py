from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_rsp_launch
import os


def generate_launch_description():
    # ─────────────────────────────────────────────
    # Launch arguments
    # ─────────────────────────────────────────────
    use_rviz = LaunchConfiguration("use_rviz")
    robot_ip = LaunchConfiguration("robot_ip")
    tcp_z_mm = LaunchConfiguration("tcp_z_mm")
    tcp_x_mm = LaunchConfiguration("tcp_x_mm")
    tcp_y_mm = LaunchConfiguration("tcp_y_mm")
    joint_vel_percent = LaunchConfiguration("joint_vel_percent")
    joint_acc_percent = LaunchConfiguration("joint_acc_percent")
    max_joint_step_deg = LaunchConfiguration("max_joint_step_deg")
    min_waypoint_delta_deg = LaunchConfiguration("min_waypoint_delta_deg")
    wait_idle_holds_lock = LaunchConfiguration("wait_idle_holds_lock")

    pkg = get_package_share_directory("meca_with_needle_moveit_config")

    mc = (
        MoveItConfigsBuilder(
            "meca_500_r3",
            package_name="meca_with_needle_moveit_config",
        )
        .to_moveit_configs()
    )

    static_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg, "launch", "static_virtual_joint_tfs.launch.py")
        )
    )

    rsp = generate_rsp_launch(mc)

    convert_to_meca_node = Node(
        package="meca_operate_real_robot",
        executable="convert_to_meca_node",
        name="convert_to_meca_node",
        output="screen",
        # respawn=False: with a real robot, a respawn loop on failure
        # (e.g. ActivateAndHome error, E-stop, TRF mismatch) is more
        # confusing than helpful during debugging.  Set to True only once
        # the system is fully stable and unattended operation is intended.
        respawn=False,
        parameters=[
            {
                "robot_ip": robot_ip,
                "use_needle_tcp": True,
                "tcp_z_mm": ParameterValue(tcp_z_mm, value_type=float),
                "tcp_x_mm": ParameterValue(tcp_x_mm, value_type=float),
                "tcp_y_mm": ParameterValue(tcp_y_mm, value_type=float),
                "joint_vel_percent": ParameterValue(joint_vel_percent, value_type=float),
                "joint_acc_percent": ParameterValue(joint_acc_percent, value_type=float),
                "max_joint_step_deg": ParameterValue(max_joint_step_deg, value_type=float),
                "min_waypoint_delta_deg": ParameterValue(min_waypoint_delta_deg, value_type=float),
                "wait_idle_holds_lock": ParameterValue(wait_idle_holds_lock, value_type=bool),
                "cart_lin_vel_mm_s": 5.0,
                "cart_acc_mm_s2": 20.0,
                "tf_parent_frame": "meca_base_link",
                "tf_child_frame": "needle_tcp",
                "tf_rate_hz": 60.0,
            }
        ],
    )

    # Manual move_group node instead of generate_move_group_launch(mc), so we can
    # override trajectory execution timeout parameters.
    #
    # Why: this V1 real-robot bridge executes MoveJoints + WaitIdle waypoint by
    # waypoint. That is intentionally conservative but slower than MoveIt's
    # internally time-parameterized trajectory. Without these parameters, MoveIt
    # may cancel the action even though the robot is still following the path.
    move_group = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            mc.to_dict(),
            {
                # Keep the simple controller manager from timing out the real
                # robot bridge too aggressively.
                "trajectory_execution.allowed_execution_duration_scaling": 100.0,
                "trajectory_execution.allowed_goal_duration_margin": 600.0,
                "trajectory_execution.allowed_start_tolerance": 0.05,

                # Most important switch for this V1 bridge: do not cancel just
                # because execution is slower than the time_from_start values.
                "trajectory_execution.execution_duration_monitoring": False,
            },
        ],
    )

    rviz = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", str(mc.package_path / "config" / "moveit.rviz")],
        parameters=[
            mc.robot_description,
            mc.robot_description_semantic,
            mc.robot_description_kinematics,
        ],
        output="screen",
        condition=IfCondition(use_rviz),
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_rviz", default_value="true"),
            DeclareLaunchArgument("robot_ip", default_value="192.168.0.100"),
            DeclareLaunchArgument("tcp_z_mm", default_value="108.751"),
            DeclareLaunchArgument("tcp_x_mm", default_value="-0.053"),
            DeclareLaunchArgument("tcp_y_mm", default_value="-0.457"),

            # Faster than 1%, but still conservative. Use 1.0 again if needed.
            DeclareLaunchArgument("joint_vel_percent", default_value="5.0"),
            DeclareLaunchArgument("joint_acc_percent", default_value="10.0"),

            # Densification prevents large gaps. Decimation avoids executing
            # hundreds of tiny points with WaitIdle.
            DeclareLaunchArgument("max_joint_step_deg", default_value="5.0"),
            DeclareLaunchArgument("min_waypoint_delta_deg", default_value="2.0"),

            # False allows /joint_states to update during motion for Unity MIRROR.
            # If mecademicpy thread-safety gives errors, temporarily set true.
            DeclareLaunchArgument("wait_idle_holds_lock", default_value="false"),

            static_tf,
            rsp,
            convert_to_meca_node,
            move_group,
            rviz,
        ]
    )
