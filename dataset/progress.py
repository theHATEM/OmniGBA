"""
Progress counters: a number in a game's RAM that keeps growing as the player gets
further, such as the score.

Some progress does not show on screen as a new scene. In Dragon Ball GT the screen
does not scroll until the current enemy wave (or an obstacle in the way) is beaten,
and every burst that restarts from a saved state loses the damage done so far. The
explorer adds the counter's level (value // step) to the scene key, so each higher
level becomes a cell of its own and later bursts continue from there.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Counter:
    address: int  # GBA address, e.g. 0x03001D0C
    size: int  # bytes, little-endian
    step: int  # values less than one step apart are the same progress level

    def level(self, emu) -> int:
        region = self.address & 0xFF000000
        offset = self.address - region
        raw = emu.memory(region)[offset : offset + self.size]
        return int.from_bytes(raw.tobytes(), "little") // self.step


PROGRESS = {
    # Score (from the cheat lists): about 50 per hit, a few hundred per enemy or obstacle.
    # Chained bursts collect far more points than steady play before a wave ends (the first
    # planet's first wave: ~40,000 vs ~8,300), so coarse steps keep the cell count down.
    "BT4E": Counter(0x03001D0C, 4, 1000),
}


def progress_for(game_code: str) -> Counter | None:
    return PROGRESS.get(game_code)
