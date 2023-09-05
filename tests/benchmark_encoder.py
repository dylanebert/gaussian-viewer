import time
from io import BytesIO

import numpy as np
import PyNvCodec as nvc
import PytorchNvCodec as pnvc
import torch
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image

NUM_FRAMES = 300
width = 1236
height = 820


rgb_planar_to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB, 0)
rgb_to_yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420, 0)
yuv_to_nv12 = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12, 0)
cc = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)


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


nvEnc = nvc.PyNvEncoder({"preset": "P4", "codec": "h264", "s": f"{width}x{height}", "bitrate": "10M"}, 0)
to_nv12 = Converter(width, height)
to_nv12.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
to_nv12.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
to_nv12.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

image = read_image("test.png").float().cuda().multiply(1 / 255.0)

start_time = time.time()
for _ in range(NUM_FRAMES):
    # image = torch.rand(3, height, width).float().cuda()
    byte_buffer = BytesIO()
    im = to_pil_image(image)
    im.save(byte_buffer, format="JPEG")
    byte_data = byte_buffer.getvalue()
    byte_buffer.close()
print(f"Time to encode with to_pil_image: {time.time() - start_time}")

start_time = time.time()
encFrame = np.ndarray((0), dtype=np.uint8)
for i in range(NUM_FRAMES):
    # image = torch.rand(3, height, width).float().cuda()
    im = image.clamp(0.0, 1.0).multiply(255.0).reshape(-1).type(torch.cuda.ByteTensor)
    surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, width, height, 0)
    surf_plane = surface.PlanePtr()
    pnvc.TensorToDptr(
        im, surf_plane.GpuMem(), surf_plane.Width(), surf_plane.Height(), surf_plane.Pitch(), surf_plane.ElemSize()
    )
    dst_surface = to_nv12.run(surface)
    success = nvEnc.EncodeSingleSurface(dst_surface, encFrame)
    if success:
        byte_data = bytearray(encFrame)
        with open("test.h264", "ab") as f:
            f.write(byte_data)
print(f"Time to encode with nvEnc: {time.time() - start_time}")
