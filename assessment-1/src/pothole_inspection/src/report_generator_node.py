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
from cv_bridge import CvBridge
import cv2
import yaml
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import numpy as np

plt.switch_backend('agg')

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

    def generate_report_callback(self, request, response):
        self.get_logger().info("Generating report...")
        req = ReportAggregatedDetections.Request()

        self.get_logger().info("requesting aggregated detections...")
        aggregated_detections = self.cli.call(req)

        report_path = request.path
        os.makedirs(report_path, exist_ok=True)

        now = datetime.now()  # current date and time
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")

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

        report = f"# Pothole Inspection Report\n\n{date_time}\n\n"
        report += "## Pothole Map\n\n"
        report += f"![]({pothole_map_path})\n\n"
        report += "## Potholes\n\n"
        report += "| ID | x | y | z | radius | image |\n"
        report += "| --- | --- | --- | --- | --- | --- |\n"
        for idx, pth in enumerate(aggregated_detections.potholes):
            self.get_logger().info(f"Pothole at {pth.x}, {pth.y}, {pth.z}")

            image = self.bridge.imgmsg_to_cv2(pth.image, desired_encoding="passthrough")
            image_path = f"{report_path}/{idx}.png"
            cv2.imwrite(image_path, image)

            report += f"| {idx} | {pth.x:.04f} | {pth.y:.04f} | {pth.z:.04f} | {pth.radius:.04f} | ![]({idx}.png) |\n"

            x = (pth.x - origin[0]) / resolution
            y = (pth.y - origin[1]) / resolution
            radius = pth.radius / resolution
            c = Circle((x, y), radius, color="g", fill=False)
            plt.gca().add_patch(c)

            plt.text(x+1, y+0.5, f"{idx}", fontsize = 6, color="gray")

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
