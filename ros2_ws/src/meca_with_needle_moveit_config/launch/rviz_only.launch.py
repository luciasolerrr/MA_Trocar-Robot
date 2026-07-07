from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    mc = MoveItConfigsBuilder(
        "meca_500_r3",
        package_name="meca_with_needle_moveit_config"
    ).to_moveit_configs()

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
    )

    return LaunchDescription([rviz])