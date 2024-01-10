import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    # Get the launch directory
    bringup_dir = get_package_share_directory("limo_navigation")

    namespace = LaunchConfiguration("namespace")
    use_sim_time = LaunchConfiguration("use_sim_time")
    autostart = LaunchConfiguration("autostart")
    params_file = LaunchConfiguration("params_file")
    nav_to_pose_bt_xml = LaunchConfiguration("default_nav_to_pose_bt_xml")
    nav_through_poses_bt_xml = LaunchConfiguration("default_nav_to_pose_bt_xml")
    use_lifecycle_mgr = LaunchConfiguration("use_lifecycle_mgr")

    remappings = [
        ((namespace, "/tf"), "/tf"),
        ((namespace, "/tf_static"), "/tf_static"),
        ("/tf", "tf"),
        ("/tf_static", "tf_static"),
    ]

    lifecycle_nodes = [
        "controller_server",
        "planner_server",
        "behavior_server",
        "bt_navigator",
        "waypoint_follower",
    ]

    param_substitutions = {
        "use_sim_time": use_sim_time,
        "default_nav_to_pose_bt_xml": nav_to_pose_bt_xml,
        "default_nav_through_poses_bt_xml": nav_through_poses_bt_xml,
        "autostart": autostart,
    }

    # default_nav_to_pose_bt_xml: replace/with/path/to/bt.xml
    # default_nav_through_poses_bt_xml: replace/with/path/to/bt.xml

    configured_params = RewrittenYaml(
        source_file=params_file,
        root_key=namespace,
        param_rewrites=param_substitutions,
        convert_types=True,
    )

    declare_namespace = DeclareLaunchArgument(
        "namespace", default_value="", description="Top-level namespace"
    )

    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true",
    )

    declare_autostart = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically startup the nav2 stack",
    )

    declare_params_file = DeclareLaunchArgument(
        "params_file",
        default_value=os.path.join(bringup_dir, "params", "nav2_params.yaml"),
        description="Full path to the ROS2 parameters file to use",
    )

    declare_nav_to_pose_bt_xml = DeclareLaunchArgument(
        "default_nav_to_pose_bt_xml",
        default_value=os.path.join(
            bringup_dir, "params", "navigate_w_replanning_and_recovery.xml"
        ),
        description="Full path to nav to pose behavior tree parameter files",
    )

    declare_nav_through_poses_bt_xml = DeclareLaunchArgument(
        "default_nav_through_poses_bt_xml",
        default_value=os.path.join(
            bringup_dir, "params", "navigate_w_replanning_and_recovery.xml"
        ),
        description="Full path to nav to pose behavior tree parameter files",
    )

    declare_use_lifecycle_mgr = DeclareLaunchArgument(
        "use_lifecycle_mgr",
        default_value="true",
        description="Whether to launch the lifecycle manager",
    )

    start_nav2_controller = Node(
        package="nav2_controller",
        executable="controller_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    start_nav2_planner = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    start_nav2_behaviors = Node(
        package="nav2_behaviors",
        executable="behavior_server",
        name="behavior_server",
        output="screen",
        parameters=[{"use_sim_time": use_sim_time}],
        remappings=remappings,
    )

    start_bt_navigator = Node(
        package="nav2_bt_navigator",
        executable="bt_navigator",
        name="bt_navigator",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    start_nav2_waypoint_follower = Node(
        package="nav2_waypoint_follower",
        executable="waypoint_follower",
        name="waypoint_follower",
        output="screen",
        parameters=[configured_params],
        remappings=remappings,
    )

    start_lifecycle_mgr = Node(
        condition=IfCondition(use_lifecycle_mgr),
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_coontroller",
        output="screen",
        parameters=[
            {"use_sim_time": use_sim_time},
            {"autostart": autostart},
            {"node_names": lifecycle_nodes},
        ],
    )

    # Create the launch description and populate
    ld = LaunchDescription()

    # Declare the launch options
    ld.add_action(declare_use_sim_time)
    ld.add_action(declare_autostart)
    ld.add_action(declare_nav_through_poses_bt_xml)
    ld.add_action(declare_nav_to_pose_bt_xml)
    ld.add_action(declare_namespace)
    ld.add_action(declare_params_file)
    ld.add_action(declare_use_lifecycle_mgr)

    ld.add_action(start_bt_navigator)
    ld.add_action(start_nav2_controller)
    ld.add_action(start_nav2_planner)
    ld.add_action(start_nav2_behaviors)
    ld.add_action(start_nav2_waypoint_follower)
    ld.add_action(start_lifecycle_mgr)

    return ld
