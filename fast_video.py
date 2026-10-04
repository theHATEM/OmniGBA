"""
Drop-in replacement for libretro.py's ArrayVideoDriver with a fast screenshot().

The stock ArrayVideoDriver.screenshot() converts every pixel in a pure-Python loop
(~36 ms per 240x160 frame on your machine). This subclass does the same conversion
with numpy in a fraction of a millisecond. The output format is identical:
a Screenshot whose data is tightly packed R, G, B, A (A = 255), 4 bytes per pixel.

Usage:
    from fast_video import FastArrayVideoDriver
    with Session(CORE_PATH, ROM_PATH, video=FastArrayVideoDriver) as session:
        shot = session.video.screenshot()

Note: this reads a few private attributes of ArrayVideoDriver (_frame, _frame_dims,
_pixel_format, _rotation), so it is tied to the current libretro.py internals.
"""

import numpy as np
from libretro.api.video import PixelFormat, Rotation
from libretro.drivers.video.driver import Screenshot
from libretro.drivers.video.software.array import ArrayVideoDriver


def _expand5(x: np.ndarray) -> np.ndarray:
    """Expand a 5-bit channel to 8 bits the same way the stock driver does."""
    return ((x << 3) | (x >> 2)).astype(np.uint8)


class FastArrayVideoDriver(ArrayVideoDriver):
    def screenshot(self, prerotate: bool = True) -> Screenshot | None:
        if not (self._frame and self._frame_dims):
            return None

        # Rotated outputs are rare (not used by GBA); let the stock code handle them.
        if prerotate and self._rotation != Rotation.NONE:
            return super().screenshot(prerotate)

        dims = self._frame_dims
        w, h, pitch = dims.width, dims.height, dims.pitch
        raw = np.frombuffer(self._frame, dtype=np.uint8, count=h * pitch).reshape(
            h, pitch
        )

        out = np.empty((h, w, 4), dtype=np.uint8)
        out[..., 3] = 255

        fmt = self._pixel_format
        if fmt == PixelFormat.XRGB8888:
            # Memory order is B, G, R, X
            px = raw[:, : w * 4].reshape(h, w, 4)
            out[..., 0] = px[..., 2]
            out[..., 1] = px[..., 1]
            out[..., 2] = px[..., 0]
        elif fmt == PixelFormat.RGB565:
            v = np.ascontiguousarray(raw[:, : w * 2]).view("<u2").reshape(h, w)
            r5 = (v >> 11) & 0x1F
            g6 = (v >> 5) & 0x3F
            b5 = v & 0x1F
            out[..., 0] = _expand5(r5)
            out[..., 1] = ((g6 << 2) | (g6 >> 4)).astype(np.uint8)
            out[..., 2] = _expand5(b5)
        elif fmt == PixelFormat.RGB1555:
            v = np.ascontiguousarray(raw[:, : w * 2]).view("<u2").reshape(h, w)
            out[..., 0] = _expand5((v >> 10) & 0x1F)
            out[..., 1] = _expand5((v >> 5) & 0x1F)
            out[..., 2] = _expand5(v & 0x1F)
        else:
            return super().screenshot(prerotate)

        return Screenshot(memoryview(out.reshape(-1)), w, h, self._rotation, fmt)


__all__ = ["FastArrayVideoDriver"]
