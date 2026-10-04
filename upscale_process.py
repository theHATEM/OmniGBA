"""
Runs UpscaleModel in a separate process, so emulation never waits on the GPU.

The emulator drops its newest frame into shared memory every frame and keeps
running. The worker upscales whichever frame is newest as soon as it is free, and
the emulator picks up the latest result whenever it draws. If the upscaler is
slower than 60 fps, the game still runs at full speed and only the upscaled picture
updates less often.

    with UpscalerProcess("realesr-animevideov3-x2", 240, 160) as upscaler:
        upscaler.submit(rgba)       # newest frame wins
        rgb = upscaler.poll()       # newest (H, W, 3) uint8 result, or None

The main process never imports ncnn: the model is only loaded in the worker.
"""

import multiprocessing as mp
import signal
import traceback
from multiprocessing import shared_memory

import numpy as np

MAX_SCALE = 4  # output buffer is sized for the biggest model


def _worker(conn, model_name, w, h, in_name, out_name, lock, new_frame, stop, out_info):
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is handled by the main process
    from upscale_model import UpscaleModel

    in_shm = shared_memory.SharedMemory(name=in_name)
    out_shm = shared_memory.SharedMemory(name=out_name)
    shared_in = np.ndarray(w * h * 4, dtype=np.uint8, buffer=in_shm.buf)
    shared_out = np.ndarray(w * h * 3 * MAX_SCALE**2, dtype=np.uint8, buffer=out_shm.buf)
    frame = np.empty((h, w, 4), dtype=np.uint8)
    model = None
    try:
        model = UpscaleModel(model_name)
        conn.send(("ready", model.scale))
        while not stop.is_set():
            if not new_frame.wait(timeout=0.1):
                continue
            with lock:
                new_frame.clear()
                frame.reshape(-1)[:] = shared_in

            rgb = model.upscale_rgba(frame, w, h)

            with lock:
                shared_out[: rgb.size] = rgb.reshape(-1)
                out_info[0], out_info[1] = rgb.shape[0], rgb.shape[1]
                out_info[2] += 1  # frame counter: tells the main process a result is new
    except Exception:
        try:
            conn.send(("error", traceback.format_exc()))
        except OSError:
            pass  # main process is gone
    finally:
        if model is not None:
            model.close()
        del shared_in, shared_out  # views must be released before the memory is closed
        in_shm.close()
        out_shm.close()


class UpscalerProcess:
    def __init__(self, model_name: str, width: int, height: int):
        self.width, self.height = width, height
        self.scale = None  # set once the worker has loaded the model
        self.error = None  # traceback text if the worker failed

        self._in_shm = shared_memory.SharedMemory(create=True, size=width * height * 4)
        self._out_shm = shared_memory.SharedMemory(
            create=True, size=width * height * 3 * MAX_SCALE**2
        )
        self._shared_in = np.ndarray(
            width * height * 4, dtype=np.uint8, buffer=self._in_shm.buf
        )
        self._shared_out = np.ndarray(
            width * height * 3 * MAX_SCALE**2, dtype=np.uint8, buffer=self._out_shm.buf
        )
        self._last_count = 0

        ctx = mp.get_context("spawn")
        self._lock = ctx.Lock()  # guards both shared buffers and _out_info
        self._new_frame = ctx.Event()
        self._stop = ctx.Event()
        self._out_info = ctx.Array("q", 3, lock=False)  # height, width, frame counter
        self._conn, child_conn = ctx.Pipe(duplex=False)
        self._proc = ctx.Process(
            target=_worker,
            args=(
                child_conn,
                model_name,
                width,
                height,
                self._in_shm.name,
                self._out_shm.name,
                self._lock,
                self._new_frame,
                self._stop,
                self._out_info,
            ),
            daemon=True,
        )
        self._proc.start()
        child_conn.close()

    @property
    def ready(self) -> bool:
        return self.scale is not None and self.error is None

    def submit(self, rgba) -> None:
        """Hand over a w*h*4 byte R,G,B,A frame. Replaces a frame the worker has not started yet."""
        with self._lock:
            self._shared_in[:] = np.frombuffer(rgba, dtype=np.uint8)
            self._new_frame.set()

    def poll(self) -> np.ndarray | None:
        """Return the newest upscaled frame as an (H, W, 3) uint8 array if there is a new one."""
        try:
            while self._conn.poll():
                kind, value = self._conn.recv()
                if kind == "ready":
                    self.scale = value
                elif kind == "error":
                    self.error = value
        except (EOFError, OSError):
            self.error = self.error or "upscaler process exited unexpectedly"

        if self._out_info[2] == self._last_count:
            return None
        with self._lock:
            h, w, self._last_count = self._out_info
            # Copy out: the worker overwrites the shared buffer with the next result
            return self._shared_out[: h * w * 3].reshape(h, w, 3).copy()

    def close(self) -> None:
        if self._proc is None:
            return
        self._stop.set()
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join()
        self._proc = None
        self._conn.close()
        self._shared_in = self._shared_out = None
        for shm in (self._in_shm, self._out_shm):
            shm.close()
            shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
