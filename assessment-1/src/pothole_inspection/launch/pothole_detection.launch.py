import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    start_object_detection_node = Node(
        parameters=[
            {
                "detection_model_path": os.path.join(
                    get_package_share_directory("pothole_inspection"),
                    "models",
                    "pothole_detector.pt",
                )
            }
        ],
        package="pothole_inspection",
        executable="object_detection_node.py",
        name="object_detection_node",
        output="screen",
    )

    start_detection_aggregation_node = Node(
        package="pothole_inspection",
        executable="detection_aggregation_node.py",
        name="detection_aggregation_node",
        output="screen",
    )

    ld = LaunchDescription()

    ld.add_action(start_object_detection_node)
    ld.add_action(start_detection_aggregation_node)

    return ld
