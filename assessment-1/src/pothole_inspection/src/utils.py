from geometry_msgs.msg import PoseStamped

import numpy as np
import math
from tf2_geometry_msgs import do_transform_pose
import cv2


def project3dToPixel(camera_model, point):
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
