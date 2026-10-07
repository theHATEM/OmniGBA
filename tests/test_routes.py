import os
import random
from pathlib import Path

import numpy as np
import pytest

from dataset.inputs import A, START
from dataset.routes import chapter_seeds, load_or_make_seeds, play, press


class StepLog:
    def __init__(self):
        self.steps = []

    def step(self, mask, frames=1):
        self.steps.append((mask, frames))


def test_press_holds_then_releases():
    assert press(A, 30) == [(A, 4), (0, 30)]


def test_play_steps_each_mask_for_its_frames():
    emu = StepLog()
    play(emu, [(START, 4), (0, 100)])
    assert emu.steps == [(START, 4), (0, 100)]


class FakeCampaign:
    """
    A game whose chapter byte (IWRAM 0x10) goes up every `chapter_frames` frames of
    play. Chapters listed in `fail_in` send the game back to the title (chapter 0)
    the first time they are played.
    """

    def __init__(self, chapter_frames, fail_in=(), last_chapter=99):
        self.iwram = np.zeros(0x8000, dtype=np.uint8)
        self.chapter_frames = chapter_frames
        self.fail_in = set(fail_in)
        self.last_chapter = last_chapter
        self.played = 0
        self.loads = 0
        self.masks = set()

    def step(self, mask, frames=1):
        self.masks.add(mask)
        for _ in range(frames):
            chapter = int(self.iwram[0x10])
            if chapter == 0:
                continue  # sitting at the title
            self.played += 1
            if chapter in self.fail_in and self.played == self.chapter_frames // 2:
                self.fail_in.discard(chapter)
                self.iwram[0x10] = 0
            elif self.played >= self.chapter_frames and chapter < self.last_chapter:
                self.iwram[0x10] = chapter + 1
                self.played = 0

    def memory(self, start):
        assert start == 0x03000000
        return self.iwram

    def save_state(self):
        return bytes([int(self.iwram[0x10]), self.played // 256, self.played % 256])

    def load_state(self, state):
        self.loads += 1
        self.iwram[0x10], self.played = state[0], state[1] * 256 + state[2]


def _start_chapter_1(emu):
    emu.iwram[0x10] = 1


def _campaign(emu, **kw):
    kw.setdefault("max_chapters", 5)
    kw.setdefault("frames_per_chapter", 10_000)
    return list(chapter_seeds(emu, random.Random(0), chapter_addr=0x10,
                              to_first_chapter=_start_chapter_1, **kw))


def test_chapter_seeds_saves_a_state_at_each_new_chapter():
    emu = FakeCampaign(chapter_frames=1200)
    seeds = _campaign(emu, max_chapters=3)
    assert [name for name, _ in seeds] == ["chapter01", "chapter02", "chapter03"]
    assert [state[0] for _, state in seeds] == [1, 2, 3]


def test_chapter_seeds_never_press_start():
    emu = FakeCampaign(chapter_frames=1200)
    _campaign(emu, max_chapters=3)
    assert not any(mask & START for mask in emu.masks)


def test_chapter_seeds_retry_a_chapter_that_falls_back_to_the_title():
    emu = FakeCampaign(chapter_frames=1200, fail_in={2})
    seeds = _campaign(emu, max_chapters=3)
    assert [name for name, _ in seeds] == ["chapter01", "chapter02", "chapter03"]
    assert emu.loads == 1


def test_chapter_seeds_stop_when_a_chapter_takes_too_long():
    emu = FakeCampaign(chapter_frames=1200, last_chapter=2)
    seeds = _campaign(emu, frames_per_chapter=6000)
    assert [name for name, _ in seeds] == ["chapter01", "chapter02"]


def test_seeds_are_made_once_then_loaded_from_disk(tmp_path):
    calls = []

    def route():
        calls.append(1)
        yield "level00", b"state-0"
        yield "level01", b"state-1"

    first = load_or_make_seeds(tmp_path, route)
    second = load_or_make_seeds(tmp_path, route)
    assert first == second == [("level00", b"state-0"), ("level01", b"state-1")]
    assert len(calls) == 1
    assert sorted(p.name for p in tmp_path.iterdir()) == ["level00.state", "level01.state"]


CORE = Path("core/mgba_libretro.dll")


def _rom(code):
    from dataset.cheats import rom_game_code

    roms = [r for r in Path("roms").rglob("*.gba")] if Path("roms").exists() else []
    return next((r for r in roms if rom_game_code(r) == code), None)


def _real_emulator(code):
    from dataset.cheats import cheats_for
    from dataset.emulator import HeadlessGBA

    rom = _rom(code)
    if not CORE.exists() or rom is None:
        pytest.skip(f"needs core and the {code} ROM")
    emu = HeadlessGBA(CORE, rom).__enter__()
    emu.apply_cheats(cheats_for(code))
    return emu


def _scene(emu, state):
    from dataset.novelty import scene_key

    emu.load_state(state)
    return scene_key(emu.memory(0x06000000), emu.memory(0x04000000), emu.frame())


def test_dbgt_levels_reach_different_stages():
    from dataset.routes import dbgt_levels

    emu = _real_emulator("BT4E")
    try:
        seeds = list(dbgt_levels(emu, random.Random(0), levels=[0, 4]))
        assert [name for name, _ in seeds] == ["level00", "level04"]
        assert len({_scene(emu, state) for _, state in seeds}) == 2
    finally:
        emu.__exit__(None, None, None)


def test_naruto_route_starts_chapter_1():
    from dataset.routes import NARUTO_CHAPTER, naruto_chapters

    emu = _real_emulator("BN2E")
    try:
        name, state = next(naruto_chapters(emu, random.Random(0)))
        emu.load_state(state)
        assert name == "chapter01"
        assert emu.memory(0x03000000)[NARUTO_CHAPTER] == 1
    finally:
        emu.__exit__(None, None, None)


@pytest.mark.skipif(not os.environ.get("GBA_SLOW_TESTS"), reason="set GBA_SLOW_TESTS=1 (about a minute)")
def test_naruto_campaign_reaches_chapter_2():
    from dataset.routes import naruto_chapters

    emu = _real_emulator("BN2E")
    try:
        names = [name for name, _ in naruto_chapters(emu, random.Random(0), max_chapters=2)]
        assert names == ["chapter01", "chapter02"]
    finally:
        emu.__exit__(None, None, None)
