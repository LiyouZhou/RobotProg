# Python libs
import rclpy
from rclpy.node import Node
from rclpy import qos

# OpenCV
import cv2

# ROS libraries
import image_geometry
from tf2_ros import Buffer, TransformListener

# ROS Messages
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, PoseArray
from cv_bridge import CvBridge, CvBridgeError
from tf2_geometry_msgs import do_transform_pose
from vision_msgs.msg import Detection2D, Detection2DArray


class DetectionAggregationNode(Node):
    camera_model = None
    image_depth_ros = None

    visualisation = True
    # aspect ration between color and depth cameras
    # calculated as (color_horizontal_FOV/color_width) / (depth_horizontal_FOV/depth_width) from the dabai camera parameters
    # color2depth_aspect = (71.0 / 640) / (67.9 / 640)
    color2depth_aspect = 1.0 # for a simulated camera

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

        print(self.image_depth.shape[0], self.image_depth.shape[1])
        print("image coords: ", x, y)
        print("depth coords: ", depth_coords)

        # get the depth reading at the centroid location
        depth_value = self.image_depth[
            int(depth_coords[1]), int(depth_coords[0])
        ]  # you might need to do some boundary checking first!

        print("depth value: ", depth_value)

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

        print("camera coords: ", camera_coords)

        # define a point in camera coordinates
        object_location = PoseStamped()
        object_location.header.frame_id = "depth_link"
        object_location.pose.orientation.w = 1.0
        object_location.pose.position.x = camera_coords[0]
        object_location.pose.position.y = camera_coords[1]
        object_location.pose.position.z = camera_coords[2]

        return object_location

    def pothole_bbox_callback(self, msg: Detection2DArray):
        # wait for camera_model and depth image to arrive
        if self.camera_model is None:
            return

        if self.image_depth_ros is None:
            return

        self.image_depth = self.bridge.imgmsg_to_cv2(self.image_depth_ros, "32FC1")
        for detection in msg.detections:
            object_location = self.image_coords_to_world(
                detection.bbox.center.position.x, detection.bbox.center.position.y
            )

            transform = self.get_tf_transform(
                "odom", object_location.header.frame_id, msg.header.stamp
            )
            p_camera = do_transform_pose(object_location.pose, transform)

            object_location.header.frame_id = "odom"
            object_location.pose = p_camera

            self.pose_stamped_array.header.frame_id = object_location.header.frame_id
            self.pose_stamped_array.poses.append(object_location.pose)

        self.object_location_pub.publish(self.pose_stamped_array)


def main(args=None):
    rclpy.init(args=args)
    image_projection = DetectionAggregationNode()
    rclpy.spin(image_projection)
    image_projection.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
