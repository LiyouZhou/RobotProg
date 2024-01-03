#! /usr/bin/env python3

# Python libs
import rclpy
from rclpy.node import Node
from rclpy import qos

from mcap_ros2.decoder import DecoderFactory
from mcap.reader import make_reader

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from sensor_msgs.msg import Imu
from vision_msgs.msg import Detection2DArray
from std_srvs.srv import Empty

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
import numpy as np
from enum import Enum


class State(Enum):
    INIT_LOCALISATION = 1
    START_SPIN = 2
    SPINING = 3
    SPIN_BACK = 4
    START_WAYPOINT_FOLLOWING = 5
    WAYPOINT_FOLLOWING = 6
    FNISHED = 7


class WaypointMissionNode(Node):
    def __init__(self):
        super().__init__("waypoint_mission_node")
        self.declare_parameter(
            "waypoint_file_path",
            "/volume/compose_dir/assessment-1/src/pothole_inspection/waypoints/waypoints.mcap",
        )
        self.waypoints = []

        with open(
            self.get_parameter("waypoint_file_path").get_parameter_value().string_value,
            "rb",
        ) as f:
            reader = make_reader(f, decoder_factories=[DecoderFactory()])

            topics = ["/waypoints"]
            for schema, channel, message, ros_msg in reader.iter_decoded_messages(
                topics=topics
            ):
                print(f"{channel.topic} {schema.name} [{message.log_time}]: ")
                for marker in ros_msg.markers:
                    p = PoseStamped()
                    p.header.frame_id = marker.header.frame_id
                    # print(type( p.header), type( marker.header))
                    # p.header = marker.header
                    p.pose.position.x = marker.pose.position.x
                    p.pose.position.y = marker.pose.position.y
                    p.pose.position.z = marker.pose.position.z
                    p.pose.orientation.x = marker.pose.orientation.x
                    p.pose.orientation.y = marker.pose.orientation.y
                    p.pose.orientation.z = marker.pose.orientation.z
                    p.pose.orientation.w = marker.pose.orientation.w
                    self.waypoints.append(p)

                break

        self.navigator = BasicNavigator()

        self.detection_counter = 0
        self.detection_subscriber = self.create_subscription(
            Detection2DArray, "/potholes/bbox", self.detection_subscriber_callback, 10
        )

        self.pose_subscriber = self.create_subscription(
            Imu, "/imu", self.imu_callback, 10
        )
        self.is_rotating = None

        self.pose_subscriber = self.create_subscription(
            PoseWithCovarianceStamped, "/amcl_pose", self.pose_callback, 10
        )
        self.pose_converged = None

        self.global_localisation_client = self.create_client(
            Empty, "/reinitialize_global_localization"
        )
        self.nomotion_update_client = self.create_client(
            Empty, "/request_nomotion_update"
        )

        self.state = State.INIT_LOCALISATION
        self.timer = self.create_timer(1, self.state_machine)

    def imu_callback(self, msg: Imu):
        """
        Deduce if the robot is rotating from the imu readings
        """
        if all(
            np.abs(i) < 0.01
            for i in [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ]
        ):
            self.is_rotating = False
        else:
            self.is_rotating = True

    def pose_callback(self, msg: PoseWithCovarianceStamped):
        """
        Deduce if amcl has converged from the convariance of the published poses
        """
        if all(np.abs(i) < 0.01 for i in msg.pose.covariance):
            self.pose_converged = True
        else:
            self.pose_converged = False

    def detection_subscriber_callback(self, msg):
        self.detection_counter += 1

    def state_machine(self):
        self.get_logger().info(f"current_state is {self.state}")

        # reset amcl so it does not assume initial position of robot
        if self.state == State.INIT_LOCALISATION:
            self.navigator.waitUntilNav2Active()
            self.global_localisation_client.wait_for_service()
            self.global_localisation_client.call_async(Empty.Request())
            self.state = State.START_SPIN
        # Spin robot a bit to help localise
        elif self.state == State.START_SPIN:
            self.navigator.spin(-np.pi / 8)
            self.state = State.SPINING
        # Wait for localisation convergance
        elif self.state == State.SPINING:
            # Check if localisation has converged
            if self.pose_converged == True:
                self.state = State.START_WAYPOINT_FOLLOWING
            elif (
                self.is_rotating is not None and self.is_rotating is False
            ):  # Check if spin has finished, if finished, spin the other way
                self.navigator.spin(np.pi / 4)
                self.state = State.SPIN_BACK

            # force the amcl localisation stack to update
            self.nomotion_update_client.call_async(Empty.Request())
        # Wait for localisation convergance
        elif self.state == State.SPIN_BACK:
            # Check if localisation has converged
            if self.pose_converged == True:
                self.state = State.START_WAYPOINT_FOLLOWING
            elif self.is_rotating is not None and self.is_rotating is False:
                self.navigator.spin(-np.pi / 4)
                self.state = State.SPINING
                self.get_logger().error("waiting for localisation convergance")

            # force the amcl localisation stack to update
            self.nomotion_update_client.call_async(Empty.Request())
        # Start to follow a list of waypoints
        elif self.state == State.START_WAYPOINT_FOLLOWING:
            # wait for detection stack to stablise and then start waypoint following
            if self.detection_counter > 5:
                self.navigator.waitUntilNav2Active()
                self.nav_start = self.navigator.get_clock().now()
                self.navigator.followWaypoints(self.waypoints)
                self.state = State.WAYPOINT_FOLLOWING
        # Continue to monitor the progress of waypoint following
        elif self.state == State.WAYPOINT_FOLLOWING:
            if not self.navigator.isTaskComplete():
                feedback = self.navigator.getFeedback()
                self.get_logger().info(
                    "Executing current waypoint: "
                    + str(feedback.current_waypoint + 1)
                    + "/"
                    + str(len(self.waypoints))
                )
            else:
                self.state = State.FNISHED
        elif self.state == State.FNISHED:
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                print("Goal succeeded!")
            elif result == TaskResult.CANCELED:
                print("Goal was canceled!")
            elif result == TaskResult.FAILED:
                print("Goal failed!")
            else:
                print("Goal has an invalid return status!")


def main(args=None):
    rclpy.init(args=args)
    waypoints_mission_node = WaypointMissionNode()
    rclpy.spin(waypoints_mission_node)
    waypoints_mission_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
