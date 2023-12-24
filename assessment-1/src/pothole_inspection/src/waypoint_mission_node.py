#! /usr/bin/env python3

# Python libs
import rclpy
from rclpy.node import Node
from rclpy import qos

from mcap_ros2.decoder import DecoderFactory
from mcap.reader import make_reader

from geometry_msgs.msg import PoseStamped
from vision_msgs.msg import Detection2DArray

from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult


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

    def detection_subscriber_callback(self, msg):
        self.detection_counter += 1

        # wait for detection stack to stablise and then start way point following
        if self.detection_counter == 5:
            self.navigator.waitUntilNav2Active()
            self.nav_start = self.navigator.get_clock().now()
            self.navigator.followWaypoints(self.waypoints)
            self.timer = self.create_timer(1, self.progress_monitor)

    def progress_monitor(self):
        if not self.navigator.isTaskComplete():
            feedback = self.navigator.getFeedback()
            self.get_logger().info(
                "Executing current waypoint: "
                + str(feedback.current_waypoint + 1)
                + "/"
                + str(len(self.waypoints))
            )
        else:
            # Do something depending on the return code
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
