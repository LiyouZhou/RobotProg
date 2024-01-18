#! /usr/bin/env python3

import rclpy
from rclpy.node import Node
import os
from datetime import datetime
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from threading import Event
from ament_index_python.packages import get_package_share_directory

from pothole_inspection.srv import ReportAggregatedDetections, GenerateReport
from geometry_msgs.msg import Pose

from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener
from tf2_geometry_msgs import do_transform_pose
import cv2
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

plt.switch_backend("agg")


class ReportGeneratorNode(Node):
    def __init__(self):
        super().__init__("detection_aggregation_node", parameter_overrides=[])

        self.callback_group = ReentrantCallbackGroup()
        self.cli = self.create_client(
            ReportAggregatedDetections,
            "/report_aggregated_detections",
            callback_group=self.callback_group,
        )
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self.generate_report_service = self.create_service(
            GenerateReport,
            "/generate_report",
            self.generate_report_callback,
            callback_group=self.callback_group,
        )
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

    def generate_report_callback(self, request, response):
        """
        Generates a report of the detected pothole.
        """
        self.get_logger().info("Generating report...")
        req = ReportAggregatedDetections.Request()

        self.get_logger().info("requesting aggregated detections...")
        aggregated_detections = self.cli.call(req)

        # request the latest transform from the map to the odom
        transform = None
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", "odom", rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return response

        # Create a folder for the report
        report_path = request.path
        os.makedirs(report_path, exist_ok=True)

        # write current date and time
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

        # Plot pothole locations on a map
        fig = plt.figure()
        maps_folder = os.path.join(
            get_package_share_directory("pothole_inspection"),
            "maps",
        )
        map_name = "potholes_20mm"
        play_area = plt.imread(f"{maps_folder}/{map_name}.pgm")
        play_area_yaml = None
        with open(f"{maps_folder}/{map_name}.yaml", "r") as fd:
            play_area_yaml = yaml.safe_load(fd)
        resolution = play_area_yaml["resolution"]  # m/pixel
        origin = np.array(play_area_yaml["origin"])
        plt.imshow(play_area, cmap="gray")
        pothole_map_path = f"{report_path}/pothole_map.png"

        # Generate a markdown report
        report = f"# Pothole Inspection Report\n\n{date_time}\n\n"
        report += "## Pothole Map\n\n"
        report += f"![](pothole_map.png)\n\n"

        # Generate a markdown table of pothole detections
        report += "## Potholes\n\n"
        report += "| ID | x | y | z | radius | image |\n"
        report += "| --- | --- | --- | --- | --- | --- |\n"
        for idx, pth in enumerate(aggregated_detections.potholes):
            self.get_logger().info(f"Pothole at {pth.x}, {pth.y}, {pth.z}")

            image = self.bridge.imgmsg_to_cv2(pth.image, desired_encoding="passthrough")
            image_path = f"{report_path}/{idx}.png"
            cv2.imwrite(image_path, image)

            p = Pose()
            p.position.x = pth.x
            p.position.y = pth.y
            p.position.z = pth.z
            p_map = do_transform_pose(p, transform)

            report += f"| {idx} | {p_map.position.x:.04f} | {p_map.position.y:.04f} | {p_map.position.z:.04f} | {pth.radius:.04f} | ![]({idx}.png) |\n"

            x = (p_map.position.x - origin[0]) / resolution
            y = (p_map.position.y - origin[1]) / resolution
            radius = pth.radius / resolution
            c = Circle((x, y), radius, color="g", fill=False)
            plt.gca().add_patch(c)

            plt.text(x + 1, y + 0.5, f"{idx}", fontsize=6, color="gray")

        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(pothole_map_path)

        report_path = os.path.abspath(f"{report_path}/report.md")
        self.get_logger().info(f"Writing report to {report_path}")
        with open(report_path, "w") as f:
            f.write(report)

        return response


def main(args=None):
    rclpy.init(args=args)
    executor = MultiThreadedExecutor()
    report_generator_node = ReportGeneratorNode()
    executor.add_node(report_generator_node)
    executor.spin()
    report_generator_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
