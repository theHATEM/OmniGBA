"""
Audio driver for libretro.py that throws the sound away.

libretro.py's default ArrayAudioDriver appends every sample to an in-memory array
and never clears it: about 460 MB per hour of game time at 60 fps, more when the
emulator runs faster than real time.

Usage:
    from discard_audio import DiscardAudioDriver
    with Session(CORE_PATH, ROM_PATH, audio=DiscardAudioDriver) as session:
        ...
"""

from libretro.drivers.audio.array import ArrayAudioDriver


class DiscardAudioDriver(ArrayAudioDriver):
    def sample(self, left: int, right: int):
        pass

    def sample_batch(self, frames: memoryview) -> int:
        return frames.nbytes // 4  # frames of two 16-bit samples


__all__ = ["DiscardAudioDriver"]
