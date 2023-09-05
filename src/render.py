import time

import numpy as np
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torch
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

from .camera import Camera
from .gaussian_model import GaussianModel


class Converter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.chain = []

    def add(self, pix_from, pix_to):
        self.chain.append(nvc.PySurfaceConverter(self.width, self.height, pix_from, pix_to, 0))

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        surf = src_surface
        cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)
        for cvt in self.chain:
            surf = cvt.Execute(surf, cc)
            if surf.Empty():
                raise RuntimeError("Failed to perform color conversion")

        return surf


class Renderer:
    def __init__(self, gaussian_model: GaussianModel, camera: Camera, logging: bool = False):
        self.gaussian_model = gaussian_model
        self.camera = camera
        self.logging = logging
        self.surface_converter = Converter(camera.image_width, camera.image_height)
        self.surface_converter.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
        self.surface_converter.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
        self.surface_converter.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)
        self.encoder_settings = {
            "preset": "P4",
            "codec": "h264",
            "s": f"{camera.image_width}x{camera.image_height}",
            "bitrate": "10M",
            "fps": "30",
        }
        self.encoder = nvc.PyNvEncoder(self.encoder_settings, 0)
        self.enc_frame = np.ndarray(shape=(0), dtype=np.uint8)

    def render(self):
        start_time = time.time()
        raster_settings = GaussianRasterizationSettings(
            image_height=self.camera.image_height,
            image_width=self.camera.image_width,
            tanfovx=self.camera.tanfovX,
            tanfovy=self.camera.tanfovY,
            bg=self.camera.bg,
            scale_modifier=1.0,
            viewmatrix=self.camera.transformation_matrix,
            projmatrix=self.camera.full_proj_transform,
            sh_degree=0,
            campos=self.camera.camera_center,
            prefiltered=False,
            debug=False,
        )
        if self.logging:
            print(f"Raster settings time: {time.time() - start_time}")
        start_time = time.time()
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        if self.logging:
            print(f"Rasterizer instantiation time: {time.time() - start_time}")
        start_time = time.time()
        rendered_image, _ = rasterizer(
            means3D=self.gaussian_model.means3D,
            means2D=self.gaussian_model.means2D,
            scales=self.gaussian_model.scales,
            rotations=self.gaussian_model.rotations,
            colors_precomp=self.gaussian_model.colors_precomp,
            opacities=self.gaussian_model.opacities,
            shs=None,
            cov3D_precomp=None,
        )
        if self.logging:
            print(f"Render time: {time.time() - start_time}")
        start_time = time.time()
        torch.cuda.synchronize()
        if self.logging:
            print(f"Sync time: {time.time() - start_time}")
        start_time = time.time()
        im = rendered_image.clamp(0.0, 1.0).multiply(255).reshape(-1).type(dtype=torch.cuda.ByteTensor)
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, self.camera.image_width, self.camera.image_height, 0)
        surf_plane = surface.PlanePtr()
        pnvc.TensorToDptr(
            im, surf_plane.GpuMem(), surf_plane.Width(), surf_plane.Height(), surf_plane.Pitch(), surf_plane.ElemSize()
        )
        dst_surface = self.surface_converter.run(surface)
        self.encoder.EncodeSingleSurface(dst_surface, self.enc_frame)
        if self.logging:
            print(f"Encoding time: {time.time() - start_time}")
        return self.enc_frame

    def update(self, position, rotation):
        self.camera.update(position, rotation)
