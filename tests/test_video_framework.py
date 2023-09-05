import numpy as np
import torch
import PyNvCodec as nvc
import PytorchNvCodec as pnvc


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


def test_video_framework():
    w, h = 256, 256
    encoder = nvc.PyNvEncoder({"preset": "P4", "codec": "h264", "s": f"{w}x{h}", "bitrate": "10M", "fps": "30"}, 0)
    surface_converter = Converter(w, h)
    surface_converter.add(nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB)
    surface_converter.add(nvc.PixelFormat.RGB, nvc.PixelFormat.YUV420)
    surface_converter.add(nvc.PixelFormat.YUV420, nvc.PixelFormat.NV12)

    for _ in range(10):
        im = torch.rand(3, h, w).float().cuda().multiply(255).reshape(-1).type(dtype=torch.cuda.ByteTensor)
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, w, h, 0)
        surf_plane = surface.PlanePtr()
        pnvc.TensorToDptr(
            im, surf_plane.GpuMem(), surf_plane.Width(), surf_plane.Height(), surf_plane.Pitch(), surf_plane.ElemSize()
        )
        dst = surface_converter.run(surface)
        enc_frame = np.ndarray(shape=(0), dtype=np.uint8)
        encoder.EncodeSingleSurface(dst, enc_frame)
        print(f"Encoded frame size: {len(enc_frame)}")


if __name__ == "__main__":
    test_video_framework()
