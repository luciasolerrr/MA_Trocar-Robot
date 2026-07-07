from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder

def generate_launch_description():
    moveit_config = MoveItConfigsBuilder("meca", package_name="meca_moveit_config").to_moveit_configs()

    return LaunchDescription([
        Node(
            name="move_group_interface_tutorial",
            package="moveit2_tutorials",
            executable="move_group_interface_tutorial",
            output="screen",
            parameters=[
                moveit_config.to_dict(),
                # Global param set (most nodes accept this)
                {"planning_group": "meca_arm", "group_name": "meca_arm"},
                # Node-scoped param block (in case the node expects scoped YAML)
                {
                    "move_group_interface_tutorial": {
                        "ros__parameters": {
                            "planning_group": "meca_arm",
                            "group_name": "meca_arm",
                        }
                    }
                },
            ],
        )
    ])
