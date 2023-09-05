import numpy as np
import plyfile
import torch


class GaussianModel:
    def __init__(self):
        self.means3D = None
        self.means2D = None
        self.opacities = None
        self.rotations = None
        self.scales = None
        self.colors_precomp = None

    def load(self, path):
        plydata = plyfile.PlyData.read(path)
        xyz = np.stack(
            (np.asarray(plydata.elements[0]["x"]), np.asarray(plydata.elements[0]["y"]), np.asarray(plydata.elements[0]["z"])),
            axis=1,
        )
        opacity = np.asarray(plydata.elements[0]["opacity"], dtype=np.float32)[:, np.newaxis]
        rot = np.stack(
            (
                np.asarray(plydata.elements[0]["rot_0"]),
                np.asarray(plydata.elements[0]["rot_1"]),
                np.asarray(plydata.elements[0]["rot_2"]),
                np.asarray(plydata.elements[0]["rot_3"]),
            ),
            axis=1,
        )
        scale = np.stack(
            (
                np.asarray(plydata.elements[0]["scale_0"]),
                np.asarray(plydata.elements[0]["scale_1"]),
                np.asarray(plydata.elements[0]["scale_2"]),
            ),
            axis=1,
        )
        shs = np.stack(
            (
                np.asarray(plydata.elements[0]["f_dc_0"]),
                np.asarray(plydata.elements[0]["f_dc_1"]),
                np.asarray(plydata.elements[0]["f_dc_2"]),
            ),
            axis=1,
        )
        colors = np.clip(0.3 * shs + 0.5, 0.0, 1.0)

        self.means3D = torch.nn.Parameter(torch.from_numpy(xyz).float().cuda())
        self.means2D = torch.zeros_like(self.means3D, dtype=self.means3D.dtype, device="cuda")
        self.opacities = torch.nn.functional.sigmoid(torch.from_numpy(opacity).float().cuda())
        self.rotations = torch.nn.functional.normalize(torch.from_numpy(rot).float().cuda())
        self.scales = torch.exp(torch.from_numpy(scale).float().cuda())
        self.colors_precomp = torch.from_numpy(colors[:, :, np.newaxis]).float().cuda()

        return self
