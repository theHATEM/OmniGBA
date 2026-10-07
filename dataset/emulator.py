"""
Headless mGBA for scripted play: no window, input set per frame from a button mask,
save states, cheats and direct views into GBA memory.
"""

import ctypes
import zlib

import numpy as np
from libretro import Session

from dataset.inputs import mask_to_joypad
from discard_audio import DiscardAudioDriver
from fast_video import FastArrayVideoDriver


class HeadlessGBA:
    def __init__(self, core_path, rom_path):
        self._core_path = str(core_path)
        self._rom_path = str(rom_path)
        self._mask = 0
        self._session = None
        self._views = {}

    def __enter__(self):
        self._session = Session(
            self._core_path, self._rom_path,
            input=self._poll, video=FastArrayVideoDriver, audio=DiscardAudioDriver,
        )
        self._session.__enter__()
        # The core publishes its memory map during the first frame
        self._session.run()
        return self

    def __exit__(self, *exc):
        self._views.clear()
        self._session.__exit__(*exc)

    def _poll(self):
        # libretro.py pulls the next pad state each time the core polls input
        while True:
            yield mask_to_joypad(self._mask)

    def step(self, mask: int, frames: int = 1) -> None:
        self._mask = mask
        for _ in range(frames):
            self._session.run()

    def frame(self) -> np.ndarray:
        """Current screen as a (160, 240, 3) uint8 RGB array."""
        shot = self._session.video.screenshot()
        rgba = np.frombuffer(shot.data, dtype=np.uint8).reshape(shot.height, shot.width, 4)
        return rgba[:, :, :3].copy()

    def save_state(self) -> bytes:
        core = self._session.core
        buf = bytearray(core.serialize_size())
        core.serialize(buf)
        return zlib.compress(bytes(buf), 1)  # ~400 KB -> ~16 KB

    def load_state(self, state: bytes) -> None:
        self._session.core.unserialize(zlib.decompress(state))

    def memory(self, start: int) -> np.ndarray:
        """Live uint8 view of the GBA memory region starting at `start` (e.g. 0x06000000)."""
        if start not in self._views:
            maps = self._session.memory_maps
            for i in range(maps.num_descriptors):
                d = maps.descriptors[i]
                if d.start == start:
                    raw = (ctypes.c_uint8 * d.len).from_address(d.ptr.value)
                    self._views[start] = np.ctypeslib.as_array(raw)
                    break
            else:
                raise KeyError(f"no memory region starts at {start:#010x}")
        return self._views[start]

    def apply_cheats(self, codes) -> None:
        core = self._session.core
        core.cheat_reset()
        for i, code in enumerate(codes):
            core.cheat_set(i, True, code)
