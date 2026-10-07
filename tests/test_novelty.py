import numpy as np

from dataset.novelty import GraphicsTracker, screen_key, scene_key


def test_screen_key_same_frame_gives_same_key():
    frame = np.random.default_rng(0).integers(0, 256, (160, 240, 3), dtype=np.uint8)
    assert screen_key(frame) == screen_key(frame.copy())


def test_screen_key_ignores_small_local_change():
    frame = np.full((160, 240, 3), 100, dtype=np.uint8)
    changed = frame.copy()
    changed[10:14, 10:14] = 255  # a few pixels, e.g. a score digit changing
    assert screen_key(frame) == screen_key(changed)


def test_screen_key_differs_for_different_scene():
    dark = np.full((160, 240, 3), 20, dtype=np.uint8)
    bright = np.full((160, 240, 3), 220, dtype=np.uint8)
    assert screen_key(dark) != screen_key(bright)


def test_screen_key_resolution_is_configurable():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    assert len(screen_key(frame, width=12, height=8)) == 12 * 8
    assert len(screen_key(frame, width=6, height=4)) == 6 * 4


def _vram_with_tiles(*tiles, at=0):
    """96 KB of VRAM holding the given 32-byte tiles from offset `at`, zeros elsewhere."""
    vram = np.zeros(0x18000, dtype=np.uint8)
    for i, value in enumerate(tiles):
        vram[at + i * 32 : at + (i + 1) * 32] = value
    return vram


def _io(mode=0, backgrounds=()):
    """I/O registers: video mode plus enabled backgrounds as (bg, screen_block, size)."""
    io = np.zeros(0x400, dtype=np.uint8)
    dispcnt = mode
    for bg, screen_block, size in backgrounds:
        dispcnt |= 0x100 << bg
        io[8 + 2 * bg : 10 + 2 * bg] = np.frombuffer(
            np.uint16((screen_block << 8) | (size << 14)).tobytes(), np.uint8)
    io[0:2] = np.frombuffer(np.uint16(dispcnt).tobytes(), np.uint8)
    return io


TILE_MODE = _io(mode=0)


def test_tracker_counts_new_tiles_once():
    tracker = GraphicsTracker()
    vram = _vram_with_tiles(1, 2, 2)  # two distinct tiles, one repeated
    assert tracker.update(vram, TILE_MODE) == 2
    assert tracker.update(vram, TILE_MODE) == 0


def test_tracker_ignores_blank_tiles():
    tracker = GraphicsTracker()
    assert tracker.update(_vram_with_tiles(0, 0), TILE_MODE) == 0


def test_tracker_detects_tile_change():
    tracker = GraphicsTracker()
    tracker.update(_vram_with_tiles(1, 2), TILE_MODE)
    assert tracker.update(_vram_with_tiles(1, 3), TILE_MODE) == 1


def test_tracker_remembers_tiles_that_come_back():
    tracker = GraphicsTracker()
    tracker.update(_vram_with_tiles(1), TILE_MODE)
    tracker.update(_vram_with_tiles(2), TILE_MODE)
    assert tracker.update(_vram_with_tiles(1), TILE_MODE) == 0


def test_tracker_ignores_background_tile_maps():
    tracker = GraphicsTracker()
    io = _io(mode=0, backgrounds=[(0, 31, 0)])  # BG0 map in screen block 31 (0xF800)
    assert tracker.update(_vram_with_tiles(5, at=0xF800), io) == 0
    assert tracker.update(_vram_with_tiles(5, at=0xF000), io) == 1


def test_tracker_ignores_bitmap_screen_but_counts_sprites():
    tracker = GraphicsTracker()
    bitmap = _io(mode=4)
    assert tracker.update(_vram_with_tiles(9, at=0x0000), bitmap) == 0
    assert tracker.update(_vram_with_tiles(9, at=0x14000), bitmap) == 1


def test_tracker_total_seen():
    tracker = GraphicsTracker()
    tracker.update(_vram_with_tiles(1, 2, 3), TILE_MODE)
    assert tracker.seen == 3


def _scroll(io, bg, x, y):
    io = io.copy()
    io[0x10 + 4 * bg : 0x12 + 4 * bg] = np.frombuffer(np.uint16(x).tobytes(), np.uint8)
    io[0x12 + 4 * bg : 0x14 + 4 * bg] = np.frombuffer(np.uint16(y).tobytes(), np.uint8)
    return io


STAGE = _io(mode=0, backgrounds=[(0, 31, 0)])  # BG0 map in screen block 31 (0xF800)
FRAME = np.zeros((160, 240, 3), dtype=np.uint8)


def _map(*tile_indices, block=31):
    """VRAM whose screen block holds a map using the given tile indices, in order."""
    vram = np.zeros(0x18000, dtype=np.uint8)
    entries = np.array(tile_indices, dtype="<u2")
    vram[block * 0x800 : block * 0x800 + entries.nbytes] = entries.view(np.uint8)
    return vram


def test_scene_key_is_stable():
    assert scene_key(_map(3, 40), STAGE, FRAME) == scene_key(_map(3, 40), STAGE.copy(), FRAME)


def test_scene_key_ignores_rearranged_and_neighbouring_tiles():
    # scrolling streams the same scenery tiles into new map positions; HUD digits
    # are neighbouring tiles
    base = scene_key(_map(3, 40, 41), STAGE, FRAME)
    assert scene_key(_map(41, 3, 40), STAGE, FRAME) == base
    assert scene_key(_map(4, 41, 42), STAGE, FRAME) == base


def test_scene_key_changes_with_new_scenery_tiles():
    assert scene_key(_map(3, 40), STAGE, FRAME) != scene_key(_map(3, 40, 300), STAGE, FRAME)


def test_scene_key_ignores_sprites_and_unused_maps():
    base = scene_key(_map(3, 40), STAGE, FRAME)
    vram = _map(3, 40)
    vram[0x10000:0x10020] = 9  # sprite tiles
    vram[0xE000:0xE020] = 9  # a screen block no enabled background uses
    other_frame = np.full((160, 240, 3), 200, dtype=np.uint8)  # sprites move on screen
    assert scene_key(vram, STAGE, other_frame) == base


def test_scene_key_changes_with_background_setup():
    moved = _io(mode=0, backgrounds=[(0, 30, 0)])
    assert scene_key(_map(3, 40), STAGE, FRAME) != scene_key(_map(3, 40, block=30), moved, FRAME)


def test_scene_key_uses_coarse_scroll():
    vram = _map(3, 40)
    base = scene_key(vram, STAGE, FRAME)
    assert scene_key(vram, _scroll(STAGE, 0, 10, 5), FRAME) == base
    assert scene_key(vram, _scroll(STAGE, 0, 300, 0), FRAME) != base


def _picture(seed):
    return np.random.default_rng(seed).integers(40, 220, (160, 240, 3), dtype=np.uint8)


def test_scene_key_uses_the_picture_in_bitmap_modes():
    vram = np.zeros(0x18000, dtype=np.uint8)
    bitmap = _io(mode=4)
    smooth = np.repeat(np.linspace(0, 255, 240, dtype=np.uint8)[None, :, None], 160, 0).repeat(3, 2)
    assert scene_key(vram, bitmap, smooth) != scene_key(vram, bitmap, smooth[:, ::-1].copy())


def test_scene_key_ignores_fades_in_bitmap_modes():
    vram = np.zeros(0x18000, dtype=np.uint8)
    bitmap = _io(mode=4)
    smooth = np.repeat(np.linspace(0, 255, 240, dtype=np.uint8)[None, :, None], 160, 0).repeat(3, 2)
    faded = (smooth * 0.5).astype(np.uint8)
    assert scene_key(vram, bitmap, smooth) == scene_key(vram, bitmap, faded)
