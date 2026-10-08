import random
from pathlib import Path

import numpy as np
import pytest

from dataset.progress import PROGRESS, Counter, progress_for

IWRAM = 0x03000000


class FakeMemory:
    def __init__(self):
        self.iwram = np.zeros(0x8000, dtype=np.uint8)

    def memory(self, start):
        return {IWRAM: self.iwram}[start]


@pytest.mark.parametrize("raw, size, step, level", [
    (b"\xd2\x04\x00\x00", 4, 500, 2),  # 1234 // 500
    (b"\x00\x00\x01\x00", 4, 500, 131),  # 65536 // 500: the high bytes count
    (b"\x00\x00\x01\x00", 2, 1, 0),  # a 2-byte counter ignores what follows it
    (b"\xff\x00\x00\x00", 1, 16, 15),
])
def test_counter_level_buckets_little_endian_value(raw, size, step, level):
    emu = FakeMemory()
    emu.iwram[0x1D0C:0x1D10] = np.frombuffer(raw, np.uint8)
    assert Counter(0x03001D0C, size, step).level(emu) == level


def test_unknown_game_has_no_progress_counter():
    assert progress_for("ZZZZ") is None


CORE = Path("core/mgba_libretro.dll")


def test_dbgt_score_counter_rises_while_fighting():
    from dataset.cheats import cheats_for, rom_game_code
    from dataset.emulator import HeadlessGBA
    from dataset.inputs import InputPolicy
    from dataset.routes import dbgt_levels

    roms = sorted(Path("roms").rglob("*.gba")) if Path("roms").exists() else []
    rom = next((r for r in roms if rom_game_code(r) == "BT4E"), None)
    if not CORE.exists() or rom is None:
        pytest.skip("needs core and the BT4E ROM")
    counter = PROGRESS["BT4E"]
    with HeadlessGBA(CORE, rom) as emu:
        emu.apply_cheats(cheats_for("BT4E"))
        _, state = next(dbgt_levels(emu, random.Random(0), levels=[0]))
        emu.load_state(state)
        assert counter.level(emu) == 0
        for mask in InputPolicy(random.Random(1)).burst(1200, allow_start=False):
            emu.step(mask)
        assert counter.level(emu) >= 1
