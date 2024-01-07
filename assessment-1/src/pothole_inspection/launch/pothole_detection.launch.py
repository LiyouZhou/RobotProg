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

    skip_localisation_init = LaunchConfiguration("skip_localisation_init")
    declare_skip_localisation_init = DeclareLaunchArgument(
        "skip_localisation_init",
        default_value="false",
        description="Skip initial localisation and start waypoint following.",
    )

    start_waypoint_mission_node = Node(
        parameters=[
            {
                "waypoint_file_path": os.path.join(
                    get_package_share_directory("pothole_inspection"),
                    "waypoints",
                    "waypoints.mcap",
                )
            },
            {"skip_localisation_init":  skip_localisation_init},
        ],
        package="pothole_inspection",
        executable="waypoint_mission_node.py",
        name="waypoint_mission_node",
        output="screen",
    )

    start_report_generator_node = Node(
        package="pothole_inspection",
        executable="report_generator_node.py",
        name="report_generator_node",
        output="screen",
    )

    ld = LaunchDescription()

    ld.add_action(declare_skip_localisation_init)
    ld.add_action(start_object_detection_node)
    ld.add_action(start_detection_aggregation_node)
    ld.add_action(start_waypoint_mission_node)
    ld.add_action(start_report_generator_node)

    return ld
