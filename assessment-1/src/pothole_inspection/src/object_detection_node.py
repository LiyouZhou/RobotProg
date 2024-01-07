#! /usr/bin/env python3

# Python libs
import rclpy
from rclpy.node import Node
from rclpy import qos

# ROS Messages
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2D
from pothole_inspection.msg import Detection2DArrayWithSourceImage
from std_msgs.msg import Int32

from cv_bridge import CvBridge

import torch
import torchvision
import numpy as np

from torchvision.transforms import v2 as T

import cv2


def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)


class PotholeDetectionNode(Node):
    def __init__(self):
        super().__init__("pothole_detection_node")
        self.image_sub = self.create_subscription(
            Image,
            "/limo/depth_camera_link/image_raw",
            self.image_color_callback,
            qos_profile=qos.qos_profile_sensor_data,
        )

        self.declare_parameter(
            "detection_model_path",
            "/volume/compose_dir/assessment-1/src/pothole_inspection/models/pothole_detector.pt",
        )
        self.model = None
        self.detections_pub = self.create_publisher(
            Detection2DArrayWithSourceImage, "/potholes/bbox", 10
        )
        self.forward_pass_count_pub = self.create_publisher(
            Int32, "/object_detection_node/forward_pass_count", 10
        )
        self.forward_pass_count = 0
        self.debug_image_pub = self.create_publisher(Image, "/potholes/debug_image", 10)
        self.bridge = CvBridge()

    def image_color_callback(self, image):
        if self.model is None:
            model_path = (
                self.get_parameter("detection_model_path")
                .get_parameter_value()
                .string_value
            )
            self.model = torch.jit.load(model_path)
            self.model.eval()

        img_tensor = torch.tensor(np.frombuffer(image.data, dtype=np.uint8)).to("cuda")
        img_tensor = img_tensor.reshape([image.height, image.width, 3]).permute(
            [2, 0, 1]
        )

        transforms = get_transform(False)
        x = transforms(img_tensor)
        pred = self.model([x.to("cuda")])[1][0]

        self.forward_pass_count += 1
        self.forward_pass_count_pub.publish(Int32(data=self.forward_pass_count))

        # print(pred, x.size())
        boxes = pred["boxes"]
        scores = pred["scores"]

        # return early if no detections
        if boxes.size(dim=0) == 0:
            return

        detections_w_image = Detection2DArrayWithSourceImage()
        for idx in range(boxes.size(dim=0)):
            score = scores[idx]
            if score > 0.9:
                box = boxes[idx]
                xmin, ymin, xmax, ymax = box[0], box[1], box[2], box[3]
                center_x = (xmin + xmax) / 2.0
                center_y = (ymin + ymax) / 2.0
                width = xmax - xmin
                height = ymax - ymin

                # don't trust small detections
                if width < 10 or height < 10:
                    continue

                det = Detection2D()

                det.bbox.center.position.x = float(center_x)
                det.bbox.center.position.y = float(center_y)
                det.bbox.center.theta = 0.0

                det.bbox.size_x = float(width)
                det.bbox.size_y = float(height)

                detections_w_image.detection_array.detections.append(det)
        detections_w_image.detection_array.header.stamp = image.header.stamp

        if len(detections_w_image.detection_array.detections) > 0:
            detections_w_image.source_image = image

        self.detections_pub.publish(detections_w_image)

        image_color = self.bridge.imgmsg_to_cv2(image, "bgr8")
        for det in detections_w_image.detection_array.detections:
            center_x = det.bbox.center.position.x
            center_y = det.bbox.center.position.y
            width = det.bbox.size_x
            height = det.bbox.size_y
            cv2.circle(
                image_color,
                (int(det.bbox.center.position.x), int(det.bbox.center.position.y)),
                10,
                255,
                -1,
            )
            start_point = [
                int(a) for a in [center_x - width / 2, center_y - height / 2]
            ]
            end_point = [int(a) for a in [center_x + width / 2, center_y + height / 2]]
            image_color = cv2.rectangle(
                image_color, start_point, end_point, (255, 0, 0), 2
            )

        self.debug_image_pub.publish(self.bridge.cv2_to_imgmsg(image_color))


def main(args=None):
    rclpy.init(args=args)
    pd_node = PotholeDetectionNode()
    rclpy.spin(pd_node)
    pd_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
