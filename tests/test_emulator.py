"""Integration tests: need core\\mgba_libretro.dll and the Dragon Ball GT ROM in roms\\."""

from pathlib import Path

import numpy as np
import pytest

from dataset.cheats import rom_game_code
from dataset.inputs import A, RIGHT, START

CORE = Path("core/mgba_libretro.dll")


def _find_rom(code):
    for rom in Path("roms").rglob("*.gba"):
        if rom_game_code(rom) == code:
            return rom
    return None


DBGT = _find_rom("BT4E") if Path("roms").exists() else None
pytestmark = pytest.mark.skipif(not CORE.exists() or DBGT is None, reason="needs core and DBGT ROM")


@pytest.fixture
def emu():
    from dataset.emulator import HeadlessGBA

    with HeadlessGBA(CORE, DBGT) as gba:
        gba.step(0, 60)
        yield gba


def test_frame_is_rgb_gba_screen(emu):
    frame = emu.frame()
    assert frame.shape == (160, 240, 3)
    assert frame.dtype == np.uint8


def test_load_state_replays_identically(emu):
    emu.step(0, 300)
    state = emu.save_state()
    inputs = [START] * 5 + [0] * 30 + [A] * 5 + [RIGHT] * 60

    def play():
        for mask in inputs:
            emu.step(mask)
        return emu.frame()

    first = play()
    emu.load_state(state)
    assert np.array_equal(play(), first)


def test_memory_regions_have_gba_sizes(emu):
    assert emu.memory(0x06000000).size == 0x18000  # video RAM
    assert emu.memory(0x05000000).size == 0x400  # palette
    assert emu.memory(0x03000000).size == 0x8000  # fast work RAM


def test_memory_is_available_before_first_step():
    from dataset.emulator import HeadlessGBA

    with HeadlessGBA(CORE, DBGT) as gba:
        assert gba.memory(0x06000000).size == 0x18000


def test_audio_is_not_kept_in_memory(emu):
    emu.step(0, 600)  # ten seconds of game time
    assert len(emu._session.audio.buffer) == 0


def test_cheats_are_applied_every_frame(emu):
    emu.apply_cheats(["83001CDC+270F"])  # DBGT "Max Zennie"
    emu.step(0, 2)
    iwram = emu.memory(0x03000000)
    assert int(iwram[0x1CDC:0x1CDE].view("<u2")[0]) == 0x270F
