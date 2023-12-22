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

import numpy as np


class PotholeTracker:
    def __init__(self):
        self.tracked_potholes = []

    def update(self, new_detections):
        for new_det in new_detections:
            new_det_center = np.array(new_det[:2])
            new_det_radius = new_det[2]

            is_tracked = False
            for idx, tracked_det in enumerate(self.tracked_potholes):
                tracked_det_center = np.array(tracked_det[:2])
                tracked_det_radius = tracked_det[2]

                if (
                    np.linalg.norm(tracked_det_center - new_det_center)
                    < new_det_radius + tracked_det_radius
                ):
                    self.tracked_potholes[idx] = self.merge(tracked_det, new_det)
                    is_tracked = True
                    break

            if not is_tracked:
                self.add(new_det)

        return

    def add(self, det):
        self.tracked_potholes.append(det)
        return

    def merge(self, det1, det2):
        det1_center = np.array(det1[:2])
        det2_center = np.array(det2[:2])
        det1_radius = det1[2]
        det2_radius = det2[2]

        c1_c2_vec = det2_center - det1_center
        c1_c2_distance = np.linalg.norm(c1_c2_vec)
        c1_c2_unit_vec = c1_c2_vec / c1_c2_distance

        p1 = det2_center + c1_c2_unit_vec * det2_radius
        p2 = det1_center - c1_c2_unit_vec * det1_radius

        det3_radius = np.linalg.norm(p1 - p2) / 2
        det3_center = (p1 + p2) / 2

        det3 = [det3_center[0], det3_center[1], det3_radius]

        return det3

    def get_tracked_potholes(self):
        return self.tracked_potholes


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

        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/limo/depth_camera_link/camera_info",
            self.camera_info_callback,
            qos_profile=qos.qos_profile_sensor_data,
        )

        self.object_location_pub = self.create_publisher(
            PoseArray, "/limo/object_location", 10
        )

        self.object_location_marker_pub = self.create_publisher(
            MarkerArray, "/limo/pothole_location/markers", 10
        )

        self.image_sub = self.create_subscription(
            Detection2DArray, "/potholes/bbox", self.pothole_bbox_callback, 10
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image,
            "/limo/depth_camera_link/depth/image_raw",
            self.image_depth_callback,
            qos_profile=qos.qos_profile_sensor_data,
        )

        self.color_image_shape = [640, 480]

        self.pose_stamped_array = PoseArray()
        self.pothole_tracker = PotholeTracker()

    def get_tf_transform(self, target_frame, source_frame, timestamp):
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, source_frame, timestamp
            )
            return transform
        except Exception as e:
            self.get_logger().warning(f"Failed to lookup transform: {str(e)}")
            return None

    def camera_info_callback(self, data):
        if not self.camera_model:
            self.camera_model = image_geometry.PinholeCameraModel()
        self.camera_model.fromCameraInfo(data)

    def image_depth_callback(self, data):
        self.image_depth_ros = data

    def image_coords_to_world(self, x, y):
        # make sure x, y does not go out of bounds
        x = max([x, 0])
        x = min([x, self.color_image_shape[0] - 1])
        y = max([y, 0])
        y = min([y, self.color_image_shape[1] - 1])

        # "map" from color to depth image
        depth_height, depth_width = self.image_depth.shape[:2]
        depth_coords = (
            depth_width / 2
            + (x - self.color_image_shape[0] / 2) * self.color2depth_aspect,
            depth_height / 2
            + (y - self.color_image_shape[1] / 2) * self.color2depth_aspect,
        )

        # print(self.image_depth.shape[0], self.image_depth.shape[1])
        # print("image coords: ", x, y)
        # print("depth coords: ", depth_coords)

        # get the depth reading at the centroid location
        depth_value = self.image_depth[
            int(depth_coords[1]), int(depth_coords[0])
        ]  # you might need to do some boundary checking first!

        # print("depth value: ", depth_value)

        # calculate object's 3d location in camera coords
        camera_coords = self.camera_model.projectPixelTo3dRay(
            (x, y)
        )  # project the image coords (x,y) into 3D ray in camera coords
        camera_coords = [
            x / camera_coords[2] for x in camera_coords
        ]  # adjust the resulting vector so that z = 1
        camera_coords = [
            x * depth_value for x in camera_coords
        ]  # multiply the vector by depth

        # print("camera coords: ", camera_coords)

        # define a point in camera coordinates
        object_location = PoseStamped()
        object_location.header.frame_id = "depth_link"
        object_location.pose.orientation.w = 1.0
        object_location.pose.position.x = camera_coords[0]
        object_location.pose.position.y = camera_coords[1]
        object_location.pose.position.z = camera_coords[2]

        return object_location

    def distance(self, p1: PoseStamped, p2: PoseStamped):
        """
        Calculate the distance between 2 PoseStamped points in the x-y plane
        """
        # assume they are at the same z-level
        v1 = np.array([p1.pose.position.x, p1.pose.position.y])
        v2 = np.array([p2.pose.position.x, p2.pose.position.y])

        return np.linalg.norm(v1 - v2)

    def pothole_bbox_callback(self, msg: Detection2DArray):
        # wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if self.image_depth_ros is None:
            return

        self.image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")

        pothole_detections = []
        for detection in msg.detections:
            object_location = self.image_coords_to_world(
                detection.bbox.center.position.x, detection.bbox.center.position.y
            )
            top = self.image_coords_to_world(
                detection.bbox.center.position.x + detection.bbox.size_x / 2,
                detection.bbox.center.position.y,
            )
            bottom = self.image_coords_to_world(
                detection.bbox.center.position.x - +detection.bbox.size_x / 2,
                detection.bbox.center.position.y,
            )
            left = self.image_coords_to_world(
                detection.bbox.center.position.x,
                detection.bbox.center.position.y - detection.bbox.size_y / 2,
            )
            right = self.image_coords_to_world(
                detection.bbox.center.position.x,
                detection.bbox.center.position.y + detection.bbox.size_y / 2,
            )

            # print(
            #     "top, bottom",
            #     self.distance(top, bottom),
            #     self.distance(left, right),
            #     type(top),
            # )

            pothole_radius = max(self.distance(top, bottom), self.distance(left, right))

            transform = self.get_tf_transform(
                "odom", object_location.header.frame_id, msg.header.stamp
            )
            if transform is not None:
                p_camera = do_transform_pose(object_location.pose, transform)

                object_location.header.frame_id = "odom"
                object_location.pose = p_camera

                self.pose_stamped_array.header.frame_id = (
                    object_location.header.frame_id
                )
                self.pose_stamped_array.poses.append(object_location.pose)
                pothole_detections.append(
                    [
                        object_location.pose.position.x,
                        object_location.pose.position.y,
                        pothole_radius,
                    ]
                )

        self.pothole_tracker.update(pothole_detections)

        ma = MarkerArray()
        self.pose_stamped_array.poses = []
        for idx, pothole in enumerate(self.pothole_tracker.get_tracked_potholes()):
            m = Marker()
            m.header.frame_id = "odom"
            m.ns = "pothole"
            m.id = idx + 1
            m.type = Marker.CYLINDER
            m.pose.position.x = pothole[0]
            m.pose.position.y = pothole[1]
            m.pose.position.z = 0.0
            m.scale.x = m.scale.y = pothole[2] * 0.8
            m.scale.z = 0.03

            m.action = Marker.ADD

            m.color.r = 0.0;
            m.color.g = 1.0;
            m.color.b = 0.0;
            m.color.a = 1.0;

            ma.markers.append(m)

            object_location = PoseStamped()
            object_location.header.frame_id = m.header.frame_id
            object_location.pose = m.pose

            self.pose_stamped_array.header.frame_id = object_location.header.frame_id
            self.pose_stamped_array.poses.append(object_location.pose)

        m = Marker()
        m.header.frame_id = "odom"
        m.ns = "pothole_count"
        m.id = 0
        m.type = Marker.TEXT_VIEW_FACING
        m.pose.position.x = 0.0
        m.pose.position.y = -1.8
        m.pose.position.z = 0.0
        m.scale.x = m.scale.y = m.scale.z = 0.3
        m.action = Marker.ADD
        m.color.r = 0.0;
        m.color.g = 1.0;
        m.color.b = 0.0;
        m.color.a = 1.0;
        s = f"Potholes Detected {len(ma.markers)}"
        m.text = s;
        self.get_logger().info(s)

        ma.markers.append(m)

        self.object_location_marker_pub.publish(ma)
        self.object_location_pub.publish(self.pose_stamped_array)


def main(args=None):
    rclpy.init(args=args)
    image_projection = DetectionAggregationNode()
    rclpy.spin(image_projection)
    image_projection.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
