import json
import pickle
import random
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.explore import Explorer
from dataset.inputs import A, LEFT, RIGHT, START, InputPolicy

VRAM, IO = 0x06000000, 0x04000000


class FakeGBA:
    """
    Tiny stand-in game. Right/Left walk through a stage: the screen brightness follows
    the position, and so does background 0's tile map (unless map_follows_position is
    False, i.e. only sprites move). A loads a new sprite tile.
    """

    def __init__(self, map_follows_position=True):
        self.pos = 0
        self.tiles_loaded = 0
        self.map_follows_position = map_follows_position
        self._vram = np.zeros(0x18000, dtype=np.uint8)
        self._io = np.zeros(0x400, dtype=np.uint8)
        self._io[0:2] = (0x00, 0x01)  # mode 0, background 0 on
        self._io[8:10] = (0x00, 31)  # background 0 map in screen block 31
        self.cheats = None
        self.pressed = set()  # every mask stepped with
        self.peak_tiles = 0  # most tiles ever loaded, across state loads

    def apply_cheats(self, codes):
        self.cheats = list(codes)

    def step(self, mask, frames=1):
        self.pressed.add(mask)
        for _ in range(frames):
            if mask & RIGHT:
                self.pos = min(self.pos + 1, 200)
            if mask & LEFT:
                self.pos = max(self.pos - 1, 0)
            if mask & A and self.tiles_loaded < 100:
                self.tiles_loaded += 1
                slot = 0x10000 + self.tiles_loaded * 32
                self._vram[slot:slot + 32] = self.tiles_loaded
                self.peak_tiles = max(self.peak_tiles, self.tiles_loaded)
            if self.map_follows_position:  # new scenery tiles every 8 steps
                self._vram[0xF800:0xF802] = np.frombuffer(np.uint16(self.pos // 8 * 16).tobytes(), np.uint8)

    def frame(self):
        gradient = np.linspace(0, 60, 240, dtype=np.float32)
        img = np.clip(gradient + self.pos, 0, 255).astype(np.uint8)
        return np.repeat(np.broadcast_to(img, (160, 240))[..., None], 3, axis=2).copy()

    def save_state(self):
        return pickle.dumps((self.pos, self.tiles_loaded, self._vram.copy()))

    def load_state(self, state):
        self.pos, self.tiles_loaded, vram = pickle.loads(state)
        self._vram[:] = vram

    def memory(self, start):
        return {VRAM: self._vram, IO: self._io}[start]


def _explorer(tmp_path, emu=None, seed=0):
    return Explorer(emu or FakeGBA(), tmp_path, random.Random(seed), cheats=["CODE+0001"])


def test_boot_applies_cheats_and_creates_first_cell(tmp_path):
    emu = FakeGBA()
    explorer = _explorer(tmp_path, emu)
    explorer.boot()
    assert emu.cheats == ["CODE+0001"]
    assert len(explorer.archive) == 1


def _rows(tmp_path):
    return [json.loads(line) for line in (tmp_path / "frames.jsonl").read_text().splitlines()]


def test_new_scenes_create_cells_to_explore_from(tmp_path):
    explorer = _explorer(tmp_path)
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    assert len(explorer.archive) > 3


def test_moving_sprites_create_no_cells(tmp_path):
    explorer = _explorer(tmp_path, FakeGBA(map_follows_position=False))
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    assert len(explorer.archive) == 1
    assert explorer.tracker.seen > 0  # sprite graphics still counted


def test_saves_frames_with_cell_ids(tmp_path):
    explorer = _explorer(tmp_path)
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    rows = _rows(tmp_path)
    assert len(rows) == explorer.recorder.count > 0
    assert all(0 <= row["cell"] < len(explorer.archive) for row in rows)
    assert any(row["new_graphics"] > 0 for row in rows)


def test_forced_saves_are_rate_limited(tmp_path):
    explorer = Explorer(FakeGBA(), tmp_path, random.Random(0), forced_save_gap=10_000)
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    assert sum(row["forced"] for row in _rows(tmp_path)) <= 1


def test_frames_per_scene_are_capped(tmp_path):
    emu = FakeGBA(map_follows_position=False)  # one scene, sprites moving
    explorer = Explorer(emu, tmp_path, random.Random(0), per_scene_cap=5, forced_save_gap=10_000)
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    rows = _rows(tmp_path)
    assert sum(not row["forced"] for row in rows) <= 5


def _start_happy_explorer(tmp_path, emu):
    return Explorer(emu, tmp_path, random.Random(0), policy=InputPolicy(random.Random(1), start_chance=0.5))


def test_add_root_starts_a_separate_root(tmp_path):
    emu = FakeGBA()
    explorer = _explorer(tmp_path, emu)
    explorer.boot()
    emu.step(RIGHT, 100)  # somewhere else in the game, e.g. reached by a scripted route
    explorer.add_root(emu.save_state(), allow_start=False)
    assert len(explorer.archive.roots) == 2


def test_bursts_from_no_start_roots_never_press_start(tmp_path):
    emu = FakeGBA()
    explorer = _start_happy_explorer(tmp_path, emu)
    explorer.add_root(emu.save_state(), allow_start=False)
    explorer.run(iterations=30)
    explorer.close()
    assert not any(mask & START for mask in emu.pressed)


def test_bursts_from_the_power_on_root_may_press_start(tmp_path):
    emu = FakeGBA()
    explorer = _start_happy_explorer(tmp_path, emu)
    explorer.boot()
    explorer.run(iterations=30)
    explorer.close()
    assert any(mask & START for mask in emu.pressed)


def test_saved_frames_record_their_root(tmp_path):
    explorer = _explorer(tmp_path)
    explorer.boot()
    explorer.run(iterations=20)
    explorer.close()
    assert all(row["root"] == 0 for row in _rows(tmp_path))


def _tiles_progress(emu):
    return emu.tiles_loaded // 5


def test_progress_carries_over_between_bursts(tmp_path):
    # one scene throughout; only the progress counter shows the game moving on, like a
    # stage that does not scroll until the enemy wave is beaten
    emu = FakeGBA(map_follows_position=False)
    explorer = Explorer(emu, tmp_path, random.Random(0), burst_frames=(20, 20), progress=_tiles_progress)
    explorer.boot()
    explorer.run(iterations=200)
    explorer.close()
    assert emu.peak_tiles > 40  # a single 20-frame burst loads at most 20


def test_saved_frames_record_progress(tmp_path):
    explorer = Explorer(FakeGBA(), tmp_path, random.Random(0), progress=_tiles_progress)
    explorer.boot()
    explorer.run(iterations=40)
    explorer.close()
    levels = [row["progress"] for row in _rows(tmp_path)]
    assert levels[0] == 0 and max(levels) > 0


def test_close_writes_cell_lineage(tmp_path):
    explorer = _explorer(tmp_path)
    explorer.boot()
    explorer.run(iterations=10)
    explorer.close()
    cells = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
    assert len(cells) == len(explorer.archive)
    assert cells[0]["parent"] is None
    assert all(c["parent"] < c["id"] for c in cells[1:])
    assert all(c["root"] == 0 for c in cells)


CORE = Path("core/mgba_libretro.dll")
ROMS = sorted(Path("roms").rglob("*.gba")) if Path("roms").exists() else []


@pytest.mark.skipif(not CORE.exists() or not ROMS, reason="needs core and a ROM")
def test_explores_real_game(tmp_path):
    from dataset.cheats import cheats_for, rom_game_code
    from dataset.emulator import HeadlessGBA

    with HeadlessGBA(CORE, ROMS[0]) as emu:
        cheats = cheats_for(rom_game_code(ROMS[0]))
        explorer = Explorer(emu, tmp_path, random.Random(0), cheats=cheats)
        explorer.boot()
        explorer.run(iterations=5)
        explorer.close()
    first = cv2.imread(str(tmp_path / "frames" / "000000.png"))
    assert first.shape == (160, 240, 3)
