import math

import numpy as np
import torch

from .utils import get_projection_matrix, get_transformation_matrix


class Camera:
    def __init__(self):
        self.width = None
        self.height = None
        self.image_width = None
        self.image_height = None
        self.fovX = None
        self.fovY = None
        self.transformation_matrix = None
        self.projection_matrix = None
        self.full_proj_transform = None
        self.camera_center = None
        self.cam_info = None

    def load(self, cam_info):
        self.cam_info = cam_info.copy()
        position = np.array(cam_info["position"])
        rotation = np.array(cam_info["rotation"])
        fx = cam_info["fx"]
        fy = cam_info["fy"]
        self.width = cam_info["width"]
        self.height = cam_info["height"]
        self.image_width = 8 * ((self.width // 4) // 8)
        self.image_height = 8 * ((self.height // 4) // 8)
        self.fovX = 2 * math.atan(self.width / (2 * fx))
        self.fovY = 2 * math.atan(self.height / (2 * fy))
        self.tanfovX = math.tan(self.fovX / 2)
        self.tanfovY = math.tan(self.fovY / 2)
        self.bg = torch.zeros(3).float().cuda()
        transformation_matrix = get_transformation_matrix(position, rotation)
        projection_matrix = get_projection_matrix(self.fovX, self.fovY)
        self.transformation_matrix = torch.from_numpy(transformation_matrix).float().cuda()
        self.projection_matrix = torch.from_numpy(projection_matrix).float().cuda()
        self.full_proj_transform = torch.from_numpy(transformation_matrix @ projection_matrix).float().cuda()
        self.camera_center = torch.from_numpy(position).float().cuda()
        return self

    def update(self, position, rotation):
        self.cam_info["position"] = position
        self.cam_info["rotation"] = rotation
        return self.load(self.cam_info)
