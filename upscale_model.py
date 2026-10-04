"""
Real-ESRGAN upscaler running on the GPU through ncnn (Vulkan).

Knows nothing about the emulator: it turns an R,G,B,A frame into a bigger RGB frame.
upscale_process.py runs it in its own process.

    python upscale_model.py [model name]     # quick speed check
"""

import cv2
import ncnn
import numpy as np


class UpscaleModel:
    # Measured on the GTX 1660 Ti (240x160 input, fp16):
    #   "realesr-animevideov3-x2"  ~12 ms  (480x320 output)
    #   "realesr-animevideov3-x4"  ~18 ms  (960x640 output)
    def __init__(self, name: str):
        ncnn.create_gpu_instance()
        self.net = ncnn.Net()
        opt = self.net.opt
        opt.use_vulkan_compute = True
        opt.use_fp16_packed = opt.use_fp16_storage = opt.use_fp16_arithmetic = True

        param, weights = rf"models\{name}.param", rf"models\{name}.bin"
        if self.net.load_param(param) != 0 or self.net.load_model(weights) != 0:
            self.close()
            raise RuntimeError(f"Failed to load {param} / {weights}")

        # The first inference compiles the Vulkan pipelines (~200 ms): do it now
        # instead of on a game frame. It also tells us the scale factor.
        out = self.upscale_rgba(np.zeros((160, 240, 4), dtype=np.uint8), 240, 160)
        self.scale = out.shape[1] // 240

    def upscale_rgba(self, rgba, w: int, h: int) -> np.ndarray:
        """
        Upscale a tightly packed R,G,B,A uint8 frame (any buffer, w*h*4 bytes).
        Returns a contiguous (H, W, 3) uint8 RGB array.

        Pre/post-processing runs in ncnn/OpenCV C++ code instead of numpy float
        math, which cost ~13 ms per 960x640 frame.
        """
        mat_in = ncnn.Mat.from_pixels(rgba, ncnn.Mat.PixelType.PIXEL_RGBA2RGB, w, h)
        mat_in.substract_mean_normalize([], [1 / 255.0] * 3)

        with self.net.create_extractor() as ex:
            ex.input("data", mat_in)
            ret, mat_out = ex.extract("output")
        if ret != 0:
            raise RuntimeError(f"Inference failed with error code: {ret}")

        out = np.asarray(mat_out)  # (3, H, W) float32 in 0..1, no copy
        c, oh, ow = out.shape
        planes = out.reshape(c * oh, ow)
        # x * 255, rounded and clamped to 0..255 in one pass
        u8 = cv2.addWeighted(planes, 255.0, planes, 0.0, 0.0, dtype=cv2.CV_8U)
        return cv2.merge([u8[:oh], u8[oh : 2 * oh], u8[2 * oh :]])

    def close(self) -> None:
        # GPU buffers must be released before the Vulkan instance goes away
        self.net.clear()
        ncnn.destroy_gpu_instance()


if __name__ == "__main__":
    import sys
    import time

    model = UpscaleModel(sys.argv[1] if len(sys.argv) > 1 else "realesr-animevideov3-x2")
    frame = np.zeros((160, 240, 4), dtype=np.uint8)
    t = time.perf_counter()
    for _ in range(100):
        model.upscale_rgba(frame, 240, 160)
    print(f"{model.scale}x: {10 * (time.perf_counter() - t):.2f} ms per frame")
    model.close()
