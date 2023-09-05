import math

import numpy as np


def get_transformation_matrix(position, rotation):
    T = np.eye(4, 4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return np.linalg.inv(T.T)


def get_projection_matrix(fovx, fovy, znear=0.01, zfar=100.0):
    tanHalfFovX = math.tan(fovx * 0.5)
    tanHalfFovY = math.tan(fovy * 0.5)

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = np.zeros((4, 4))

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)

    return P.T
