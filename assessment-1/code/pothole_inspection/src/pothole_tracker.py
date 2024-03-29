import numpy as np
from collections import deque
import time
from geometry_msgs.msg import Pose
import itertools
import multiprocessing
import cv2
from utils import sample_pixel
from cv_bridge import CvBridge

from pothole_inspection.msg import Pothole as PotholeMsg
from visualization_msgs.msg import Marker, MarkerArray


class Pothole:
    def __init__(self, x: float, y: float, z: float, radius: float, image, tf):
        self.x = x
        self.y = y
        self.z = z
        self.radius = radius
        self.image = image
        self.tf = tf
        self.center = np.array((x, y, z))

        self.image_folder = "/volume/compose_dir/debug_images/"

    def merge(self, new_pothole):
        """
        Merges the current pothole with a new pothole such that
        the resulting pothole perfectly contains both potholes.
        """
        c1_c2_vec = new_pothole.center - self.center
        c1_c2_distance = np.linalg.norm(c1_c2_vec)
        c1_c2_unit_vec = c1_c2_vec / c1_c2_distance

        p1 = self.center - c1_c2_unit_vec * self.radius
        p2 = new_pothole.center + c1_c2_unit_vec * new_pothole.radius

        if np.linalg.norm(p2 - self.center) < self.radius:
            # new_pothole is entirely inside current pothole
            # no merge necessary
            return

        det3_radius = 0
        det3_center = np.array((0, 0, 0))
        if np.linalg.norm(p1 - new_pothole.center) < new_pothole.radius:
            # current pothole is entirely inside new pothole
            det3_radius = new_pothole.radius
            det3_center = new_pothole.center
        else:
            det3_radius = np.linalg.norm(p1 - p2) / 2
            det3_center = (p1 + p2) / 2

        self.__init__(
            det3_center[0],
            det3_center[1],
            det3_center[2],
            det3_radius,
            new_pothole.image,
            new_pothole.tf,
        )

    def get_pothole_image(self, camera_model):
        """
        Returns a cropped image of the pothole in the bird's eye view.
        """
        if camera_model is None or self.image is None or self.tf is None:
            return

        p = Pose()
        p.position.x = self.x
        p.position.y = self.y
        p.position.z = self.z

        # define a matrix of pixel coordinates that covers the pothole
        image_data = []
        pixel_step = self.radius / 64
        x_values = list(
            enumerate(np.arange(self.x - self.radius, self.x + self.radius, pixel_step))
        )
        y_values = list(
            enumerate(np.arange(self.y - self.radius, self.y + self.radius, pixel_step))
        )
        all_pixel_cases = itertools.product(x_values, y_values)
        all_pixel_cases = [
            [
                case[0][0],
                case[1][0],
                case[0][1],
                case[1][1],
                self.z,
                self.tf,
                camera_model,
                self.image,
            ]
            for case in all_pixel_cases
        ]

        # Project each 3d pixel point into the image coordinates to sample the color
        all_pixel_values = []
        with multiprocessing.Pool(3) as process_pool:
            all_pixel_values = process_pool.starmap(sample_pixel, all_pixel_cases)

        # combine the pixel color values into an image
        image_data = np.zeros([len(x_values), len(y_values), 3])
        for color, case in zip(all_pixel_values, all_pixel_cases):
            image_data[case[0], case[1]] = color

        return np.array(image_data)

    def to_msg(self, camera_model, cv_bridge):
        """
        Converts the pothole into a message that can be published.
        """
        msg = PotholeMsg()
        msg.x = self.x
        msg.y = self.y
        msg.z = self.z
        msg.radius = self.radius
        msg.image = cv_bridge.cv2_to_imgmsg(
            self.get_pothole_image(camera_model), encoding="passthrough"
        )
        return msg

    def to_marker(self, id, stamp):
        """
        Converts the pothole into a marker message that can be visualised.
        """
        m = Marker()
        m.header.frame_id = "odom"
        m.header.stamp = stamp
        m.ns = "pothole"
        m.id = id
        m.type = Marker.CYLINDER
        m.pose.position.x = self.x
        m.pose.position.y = self.y
        m.pose.position.z = self.z
        m.scale.x = m.scale.y = self.radius * 2
        m.scale.z = 0.03

        m.action = Marker.ADD

        m.color.r = 0.0
        m.color.g = 1.0
        m.color.b = 0.0
        m.color.a = 1.0

        return m


class PotholeTracker:
    """
    Maintain a list of detected potholes
    """
    def __init__(self):
        self.tracked_potholes: list[Pothole] = []

    def update(self, new_detections: list[Pothole]):
        """
        Update the list of tracked potholes with a list of new detections.
        """
        new_detections_queue = deque(new_detections)

        # loop through the list of new detections
        while len(new_detections_queue) > 0:
            new_det = new_detections_queue.popleft()

            # detect overlap with existing detections
            found_overlap = False
            for idx, tracked_det in enumerate(self.tracked_potholes):
                # if the centres are close together, merge the detections
                if (
                    np.linalg.norm(tracked_det.center - new_det.center)
                    < (new_det.radius + tracked_det.radius) * 0.9
                ):
                    # remove original from tacked list
                    original_det = self.tracked_potholes.pop(idx)
                    # merge original with new detections
                    original_det.merge(new_det)
                    # add merged detection to the new detections queue
                    new_detections_queue.append(original_det)

                    found_overlap = True
                    break

            if not found_overlap:
                self.add(new_det)

        return

    def add(self, det):
        """
        Add a new pothole to the list of tracked potholes.
        """
        self.tracked_potholes.append(det)
        return

    def get_tracked_potholes(self):
        return self.tracked_potholes
