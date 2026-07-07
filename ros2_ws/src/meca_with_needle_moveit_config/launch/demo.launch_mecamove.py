from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_rsp_launch
import os

def generate_launch_description():
    use_rviz = LaunchConfiguration("use_rviz")
    pkg = get_package_share_directory("meca_with_needle_moveit_config")

    mc = MoveItConfigsBuilder("meca_500_r3", package_name="meca_with_needle_moveit_config").to_moveit_configs()

    # 1) genau EIN statischer TF
    static_tf = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, "launch", "static_virtual_joint_tfs.launch.py"))
    )

    # 2) genau EIN robot_state_publisher
    rsp = generate_rsp_launch(mc)

    # 3) genau EIN ros2_control_node (FakeSystem) + Controller-Config
    ros2_control = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            mc.robot_description,                          # URDF
            os.path.join(pkg, "config", "ros2_controllers.yaml"),
        ],
        output="screen",
    )

    # 4) genau EIN Spawner-Set
    # spawner_js = Node(package="controller_manager", executable="spawner",
    #                   arguments=["joint_state_broadcaster"], output="screen")
    spawner_arm = Node(package="controller_manager", executable="spawner",
                       arguments=["meca_arm_controller"], output="screen")

    # 5) genau EIN move_group
    move_group = generate_move_group_launch(mc)

    # 6) optional EIN RViz mit Kinematics-Parametern
    rviz = Node(
        package="rviz2", executable="rviz2", name="rviz2",
        arguments=["-d", str(mc.package_path / "config" / "moveit.rviz")],
        parameters=[mc.robot_description, mc.robot_description_semantic, mc.robot_description_kinematics],
        output="screen", condition=IfCondition(use_rviz)
    )

    # CONVERT_TO_MECA_NODE /JOINT_STATES:
    meca_bridge = Node(
        package="meca_operate_real_robot",         
        executable="convert_to_meca_node",
        parameters=[{"robot_ip": "192.168.0.100"}],
        output="screen",
    )


    return LaunchDescription([
        DeclareLaunchArgument("use_rviz", default_value="true"),
        static_tf, rsp, meca_bridge, ros2_control, spawner_arm, move_group, rviz
    ])
