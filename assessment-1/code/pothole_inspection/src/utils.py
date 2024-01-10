from geometry_msgs.msg import PoseStamped

import numpy as np
import math
from tf2_geometry_msgs import do_transform_pose
import cv2


def project3dToPixel(camera_model, point):
    """
    Projects a 3D point to pixel coordinates using the camera model
    """
    projection_matrix = camera_model.projectionMatrix()

    # make sure the translation component is zero
    projection_matrix[0, 3] = 0
    projection_matrix[1, 3] = 0
    projection_matrix[2, 3] = 0

    src = np.array([point[0], point[1], point[2], 1.0]).transpose()

    dst = np.matmul(projection_matrix, src)

    x = dst[0, 0]
    y = dst[0, 1]
    w = dst[0, 2]
    if w != 0:
        return (x / w, y / w)
    else:
        return (float("nan"), float("nan"))


def sample_pixel(i, j, x, y, z, tf, camera_model, image):
    """
    samples the pixel color in pixel coordinates given
    a 3d point in camera coordinate
    """
    p = PoseStamped()
    p.pose.position.x = x
    p.pose.position.y = y
    p.pose.position.z = z

    p_camera = do_transform_pose(p.pose, tf)

    pixel_coords = project3dToPixel(
        camera_model, (p_camera.position.x, p_camera.position.y, p_camera.position.z)
    )

    pixel_color = [0, 0, 0]
    if all(not math.isnan(val) for val in pixel_coords):
        pixel_color = cv2.getRectSubPix(image, (1, 1), pixel_coords)
        pixel_color = pixel_color[0][0]
    return pixel_color

def timestamp_to_float(timestamp):
    """
    Convert a ROS timestamp to a float
    """
    return timestamp.sec + timestamp.nanosec * 1e-9

def distance(p1: PoseStamped, p2: PoseStamped):
    """
    Calculate the distance between 2 PoseStamped points in the x-y plane
    """
    # assume they are at the same z-level
    v1 = np.array([p1.pose.position.x, p1.pose.position.y])
    v2 = np.array([p2.pose.position.x, p2.pose.position.y])

    return np.linalg.norm(v1 - v2)
