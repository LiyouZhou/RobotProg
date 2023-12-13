import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from pathlib import Path
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    ExecuteProcess,
    LogInfo,
    RegisterEventHandler,
    TimerAction,
)
from launch.event_handlers import (
    OnExecutionComplete,
    OnProcessExit,
    OnProcessIO,
    OnProcessStart,
    OnShutdown,
)
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import logging

# Set the logging level to DEBUG or desired level
logging.getLogger().setLevel(logging.DEBUG)


def generate_launch_description():
    launch_actions = []

    sim_world_file_param_name = "sim_world_file"
    sim_world_file = LaunchConfiguration(sim_world_file_param_name)
    declare_world_file = DeclareLaunchArgument(
        sim_world_file_param_name,
        default_value="/opt/ros/lcas_addons/src/limo_ros2/src/limo_gazebosim/worlds/potholes.world",
        description="Sim world file to launch",
    )
    launch_actions.append(declare_world_file)
    simulator = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("limo_gazebosim"),
                    "launch",
                    "limo_gazebo_diff.launch.py",
                )
            ]
        ),
        launch_arguments={"world": sim_world_file}.items(),
    )
    launch_actions.append(simulator)

    map_path = os.path.join(
        get_package_share_directory("pothole_inspection"), "maps", "potholes_20mm.yaml"
    )
    navigation = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                os.path.join(
                    get_package_share_directory("pothole_inspection"),
                    "launch",
                    "limo_navigation.launch.py",
                )
            ]
        ),
        launch_arguments={
            "params_file": os.path.join(
                get_package_share_directory("pothole_inspection"),
                "params",
                "nav2_params.yaml",
            ),
            "map": map_path,
        }.items(),
    )
    launch_actions.append(navigation)

    # navigation = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         [
    #             os.path.join(
    #                 get_package_share_directory("pothole_inspection"),
    #                 "launch",
    #                 "localization.launch.py",
    #             )
    #         ]
    #     ),
    #     launch_arguments={
    #         "slam_params_file": os.path.join(
    #             get_package_share_directory("pothole_inspection"),
    #             "params",
    #             "localization.yaml",
    #         ),
    #         "slam_toolbox.map_file_name": map_path,
    #     }.items(),
    # )
    # launch_actions.append(navigation)

    rviz_config_dir = os.path.join(
        get_package_share_directory("pothole_inspection"),
        "rviz",
        "pothole_inspection.rviz",
    )
    start_rviz2 = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        arguments=["-d", rviz_config_dir],
        parameters=[{"use_sim_time": True}],
        output="screen",
    )
    launch_actions.append(start_rviz2)

    # Declare the launch options
    ld = LaunchDescription(launch_actions)
    return ld
