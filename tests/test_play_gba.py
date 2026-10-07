"""Integration test: needs core\\mgba_libretro.dll and the ROM set in play_gba.ROM_PATH."""

import os

import pytest
from libretro.api.input import JoypadState

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
try:
    import play_gba
except FileNotFoundError as e:  # play_gba checks the core and ROM paths on import
    pytest.skip(str(e), allow_module_level=True)


def _no_input():
    while True:
        yield JoypadState()


def test_session_does_not_keep_audio():
    with play_gba.open_session(input=_no_input) as session:
        for _ in range(600):  # ten seconds of game time
            session.run()
        assert len(session.audio.buffer) == 0
