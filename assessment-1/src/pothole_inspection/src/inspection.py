import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from std_msgs.msg import String
from std_srvs.srv import Empty
from builtin_interfaces.msg import Duration

from nav2_msgs.action import Spin
import numpy as np
from waypoints import waypoints
from nav2_msgs.action import FollowWaypoints
from threading import Condition

def callback_feedback(feedback):
    rclpy.loginfo("Feedback:%s" % str(feedback))


class SpinActionClient(Node):
    def __init__(self):
        super().__init__("SpinClient")
        self._action_client = ActionClient(self, Spin, "spin")
        self._action_client.wait_for_server()

        self.kick_amcl_service_client = self.create_client(
            Empty, "request_nomotion_update"
        )
        self.timer = self.create_timer(1, self.kick_amcl)

        self.follow_waypoints_client = ActionClient(
            self, FollowWaypoints, "follow_waypoints"
        )
        self.spin_finish = False
        self.send_goal_future = None

    def kick_amcl(self):
        print("kicking amcl")
        self.kick_amcl_service_client.call_async(Empty.Request())

    def spin_around_feedback_cb(self, feedback):
        print(f"feedback {feedback} {feedback.feedback.angular_distance_traveled} {self.target_yaw}")
        if feedback.feedback.angular_distance_traveled > self.target_yaw - .5:
            self.spin_finish = True
            self.follow_waypoints(waypoints)

    def spin_around(self, target_yaw):
        self.target_yaw = target_yaw
        goal_msg = Spin.Goal()
        goal_msg.target_yaw = float(target_yaw)
        time_allowance = Duration()
        time_allowance.sec = 10
        time_allowance.nanosec = 0
        goal_msg.time_allowance = time_allowance

        self._action_client.wait_for_server()

        return self._action_client.send_goal_async(
            goal_msg, feedback_callback=self.spin_around_feedback_cb)

    def follow_waypoints_feedback_cb(self, feedback):
        print(f"feedback {feedback}")

    def follow_waypoints(self, poses):
        if self.send_goal_future is not None:
            return self.send_goal_future

        goal_msg = FollowWaypoints.Goal()
        goal_msg.poses = poses

        print(f"Following {len(goal_msg.poses)} goals....")
        self.send_goal_future = self.follow_waypoints_client.send_goal_async(
            goal_msg, self.follow_waypoints_feedback_cb
        )

        return self.send_goal_future
        # rclpy.spin_until_future_complete(self, send_goal_future)
        # self.goal_handle = send_goal_future.result()

        # if not self.goal_handle.accepted:
        #     self.error(f"Following {len(poses)} waypoints request was rejected!")
        #     return False

        # self.result_future = self.goal_handle.get_result_async()
        # return True


def main(args=None):
    rclpy.init(args=args)

    action_client = SpinActionClient()

    future = action_client.spin_around(2 * np.pi)

    rclpy.spin_until_future_complete(action_client, future)

    print(f"spin result {future.result()}")

    rclpy.spin(action_client)


if __name__ == "__main__":
    main()
