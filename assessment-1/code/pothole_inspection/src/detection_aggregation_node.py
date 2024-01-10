#! /usr/bin/env python3

# Python libs
import rclpy
from rclpy.node import Node
from rclpy import qos

# ROS libraries
import image_geometry
from tf2_ros import Buffer, TransformListener

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray
from cv_bridge import CvBridge
from tf2_geometry_msgs import do_transform_pose
from vision_msgs.msg import Detection2DArray
from visualization_msgs.msg import Marker, MarkerArray
from pothole_inspection.msg import Detection2DArrayWithSourceImage
from pothole_inspection.srv import ReportAggregatedDetections

import numpy as np
import time
import os
from collections import deque
from threading import Lock

from pothole_tracker import PotholeTracker, Pothole
from utils import project3dToPixel, timestamp_to_float, distance

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from cv_bridge import CvBridge


class DetectionAggregationNode(Node):
    camera_model = None
    image_depth_ros = None

    visualisation = True
    # aspect ration between color and depth cameras
    # calculated as (color_horizontal_FOV/color_width) / (depth_horizontal_FOV/depth_width) from the dabai camera parameters
    # color2depth_aspect = (71.0 / 640) / (67.9 / 640)
    color2depth_aspect = 1.0  # for a simulated camera

    def __init__(self):
        super().__init__("detection_aggregation_node", parameter_overrides=[])
        self.bridge = CvBridge()

        self.camera_info_cbg = MutuallyExclusiveCallbackGroup()
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/limo/depth_camera_link/camera_info",
            self.camera_info_callback,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=self.camera_info_cbg,
        )

        self.object_location_pub = self.create_publisher(
            PoseArray, "/limo/object_location", 10
        )

        self.object_location_marker_pub = self.create_publisher(
            MarkerArray, "/limo/pothole_location/markers", 10
        )

        self.bbox_sub = self.create_subscription(
            Detection2DArrayWithSourceImage,
            "/potholes/bbox",
            self.pothole_bbox_callback,
            10,
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.depth_image_cbg = MutuallyExclusiveCallbackGroup()
        self.depth_image_sub = self.create_subscription(
            Image,
            "/limo/depth_camera_link/depth/image_raw",
            self.image_depth_callback,
            qos_profile=qos.qos_profile_sensor_data,
            callback_group=self.depth_image_cbg,
        )
        self.depth_image_buffer = deque(maxlen=12)

        self.color_image_shape = [640, 480]

        self.pothole_tracker = PotholeTracker()
        self.img_count = 0

        self.report_cbg = MutuallyExclusiveCallbackGroup()
        self.report_srv = self.create_service(
            ReportAggregatedDetections,
            "/report_aggregated_detections",
            self.report_aggregated_detections_callback,
            callback_group=self.report_cbg,
        )

        self.cv_bridge = CvBridge()
        self.potholes_mutex = Lock()

    def get_tf_transform(self, target_frame, source_frame, timestamp):
        """
        Get the transform from source_frame to target_frame at the given timestamp
        """
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp
            )
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None

    def camera_info_callback(self, data):
        """
        Callback for the camera info message, used to save the camera model
        """
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        """
        Callback for the depth image message, stores the depth image in a ring buffer
        """
        self.depth_image_buffer.append(data)

    def image_coords_to_camera_coords(self, x, y, image_depth) -> PoseStamped:
        """
        Convert image coordinates to camera coordinates
        """
        # make sure x, y does not go out of bounds
        x = max([x, 0])
        x = min([x, self.color_image_shape[0] - 1])
        y = max([y, 0])
        y = min([y, self.color_image_shape[1] - 1])

        # "map" from color to depth image
        depth_height, depth_width = image_depth.shape[:2]
        depth_coords = (
            depth_width / 2
            + (x - self.color_image_shape[0] / 2) * self.color2depth_aspect,
            depth_height / 2
            + (y - self.color_image_shape[1] / 2) * self.color2depth_aspect,
        )

        # get the depth reading at the centroid location
        depth_value = image_depth[int(depth_coords[1]), int(depth_coords[0])]

        # project the image coords (x,y) into 3D ray in camera coords
        rectified_point = self.camera_model.rectifyPoint((x, y))
        camera_coords = self.camera_model.projectPixelTo3dRay(rectified_point)

        # make a vector along the ray where z is 1
        camera_coords = [x / camera_coords[2] for x in camera_coords]

        # multiply the vector by depth
        camera_coords = [x * depth_value for x in camera_coords]

        # define a point in camera coordinates
        object_location = PoseStamped()
        object_location.header.frame_id = "depth_link"
        object_location.pose.orientation.w = 1.0
        object_location.pose.position.x = camera_coords[0]
        object_location.pose.position.y = camera_coords[1]
        object_location.pose.position.z = camera_coords[2]

        return object_location

    def get_depth_image_by_timestamp(self, timestamp):
        """
        Get the depth image closest to the given timestamp
        """
        if len(self.depth_image_buffer) == 0:
            self.get_logger().warning(f"No depth images in the buffer")
            return None

        if all(
            timestamp_to_float(timestamp) < timestamp_to_float(image.header.stamp)
            for image in self.depth_image_buffer
        ):
            self.get_logger().warning(
                f"Requested depth image with timestamp {timestamp} not in the buffer"
            )
            return None

        while all(
            timestamp_to_float(timestamp) > timestamp_to_float(image.header.stamp)
            for image in self.depth_image_buffer
        ):
            self.get_logger().warning(
                f"Requested depth image with timestamp {timestamp} in the future, waiting for it to arrive..."
            )
            time.sleep(1)

        for image in self.depth_image_buffer:
            if (
                np.abs(
                    timestamp_to_float(timestamp)
                    - timestamp_to_float(image.header.stamp)
                )
                < 1e-3
            ):
                return image

        self.get_logger().error(
            "Did not find a depth image with the requested timestamp {timestamp}"
        )
        return None

    def pothole_bbox_callback(self, msg: Detection2DArrayWithSourceImage):
        """
        Callback for the detection message,
        calculates the 3D location of the pothole and
        adds it to the pothole tracker. Publishes a
        list of pothole locations for visualisation.
        """
        # wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if len(msg.detection_array.detections) == 0:
            return

        # find the depth image matching exactly the timestamp of the detection
        image_depth_ros = self.get_depth_image_by_timestamp(
            msg.detection_array.header.stamp
        )
        if image_depth_ros is None:
            return
        image_depth = self.bridge.imgmsg_to_cv2(image_depth_ros, "32FC1")
        source_img = self.bridge.imgmsg_to_cv2(msg.source_image)

        pothole_detections = []
        for detection in msg.detection_array.detections:
            # calculate the camera coordinates of the center, top, bottom,  left and right of the bounding box
            object_location = self.image_coords_to_camera_coords(
                detection.bbox.center.position.x,
                detection.bbox.center.position.y,
                image_depth,
            )
            top = self.image_coords_to_camera_coords(
                detection.bbox.center.position.x + detection.bbox.size_x / 2,
                detection.bbox.center.position.y,
                image_depth,
            )
            bottom = self.image_coords_to_camera_coords(
                detection.bbox.center.position.x - +detection.bbox.size_x / 2,
                detection.bbox.center.position.y,
                image_depth,
            )
            left = self.image_coords_to_camera_coords(
                detection.bbox.center.position.x,
                detection.bbox.center.position.y - detection.bbox.size_y / 2,
                image_depth,
            )
            right = self.image_coords_to_camera_coords(
                detection.bbox.center.position.x,
                detection.bbox.center.position.y + detection.bbox.size_y / 2,
                image_depth,
            )

            pothole_radius = max(distance(top, bottom), distance(left, right)) / 2

            # transform the object location from the camera frame to the odom frame
            transform = self.get_tf_transform(
                "odom",
                object_location.header.frame_id,
                msg.detection_array.header.stamp,
            )

            if transform is not None:
                p_camera = do_transform_pose(object_location.pose, transform)

                object_location.header.frame_id = "odom"
                object_location.pose = p_camera

                # create a pothole object
                pothole = Pothole(
                    object_location.pose.position.x,
                    object_location.pose.position.y,
                    object_location.pose.position.z,
                    pothole_radius,
                    source_img,
                    self.get_tf_transform(
                        "depth_link", "odom", msg.detection_array.header.stamp
                    ),
                )
                pothole_detections.append(pothole)

        # update the pothole tracker with new detection, this will merge overlapping detections
        with self.potholes_mutex:
            self.pothole_tracker.update(pothole_detections)

        # create some visualisations
        ma = MarkerArray()
        pose_stamped_array = PoseArray()
        pose_stamped_array.poses = []
        for idx, pothole in enumerate(self.pothole_tracker.get_tracked_potholes()):
            # create a cylindrical marker for each pothole visualisation
            m = pothole.to_marker(idx + 1, msg.detection_array.header.stamp)
            ma.markers.append(m)

            # create a pose for each pothole for visualisation
            object_location = PoseStamped()
            object_location.header.frame_id = m.header.frame_id
            object_location.pose = m.pose

            pose_stamped_array.header.frame_id = object_location.header.frame_id
            pose_stamped_array.poses.append(object_location.pose)

        # create a text marker for the number of potholes detected
        m = Marker()
        m.header.frame_id = "odom"
        m.ns = "pothole_count"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = 0.0
        m.pose.position.y = -1.5
        m.pose.position.z = 0.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.action = Marker.ADD
        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0
        s = f"Potholes Detected {len(ma.markers)}"
        m.text = s
        self.get_logger().info(s)
        ma.markers.append(m)

        self.object_location_marker_pub.publish(ma)
        self.object_location_pub.publish(pose_stamped_array)

    def report_aggregated_detections_callback(self, request, response):
        """
        Service callback for reporting the aggregated detections
        Converts the pothole tracker objects to ROS message
        """
        with self.potholes_mutex:
            for idx, pth in enumerate(self.pothole_tracker.get_tracked_potholes()):
                self.get_logger().info(
                    f"converting pothole {idx} {pth.x} {pth.y} {pth.z} {pth.radius}"
                )
                response.potholes.append(pth.to_msg(self.camera_model, self.cv_bridge))

        self.get_logger().info(f"returning pothole count {len(response.potholes)}")
        return response


def main(args=None):
    rclpy.init(args=args)
    detection_aggregation_node = DetectionAggregationNode()
    executor = MultiThreadedExecutor()
    executor.add_node(detection_aggregation_node)
    executor.spin()
    detection_aggregation_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
