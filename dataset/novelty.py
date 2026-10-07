"""
Novelty signals for exploration.

screen_key: a coarse fingerprint of the screen pixels (used to group similar frames).

scene_key: identifies "where the game is" from the background only: video mode,
background setup, which tiles each active background's map uses (in buckets of
TILE_BUCKET neighbouring tiles) and a coarse scroll position. Sprites are ignored,
so fighters moving around or an animated menu cursor do not count as a new scene.
Using the set of tiles rather than the exact map means scrolling (which streams the
same scenery into new map positions) and HUD digits changing do not count either,
while a new stage, area, menu page or scrolling far enough does. In bitmap modes
the picture itself is the background, so a brightness-normalised fingerprint of it
is used (fades do not count).

GraphicsTracker: remembers every 8x8 tile of graphics the game has ever had in video
memory. Games load a move's or effect's graphics the first time it appears, so new
tiles mean new graphics on screen, even when the effect is too small to change the
coarse fingerprint. Parts of video memory that change without new graphics are
skipped: background tile maps (rewritten while scrolling) and bitmap screens.
Palettes are not tracked, because fades change them every frame.
"""

import hashlib

import cv2
import numpy as np

TILE_BYTES = 32  # one 8x8 tile at 4 bits per pixel
VRAM_TILES = 0x18000 // TILE_BYTES
TILES_PER_SCREEN_BLOCK = 0x800 // TILE_BYTES  # tile maps live in 2 KB screen blocks
BITMAP_SPRITE_START = 0x14000  # in bitmap modes, sprite tiles start here
TEXT_MAP_BLOCKS = (1, 2, 2, 4)  # screen blocks per map, by BGxCNT size field
AFFINE_MAP_BLOCKS = (1, 1, 2, 8)
SCROLL_STEP = 256  # pixels of background scroll that count as a new scene
TILE_BUCKET = 16  # neighbouring tile numbers that count as the same kind of scenery


def screen_key(frame: np.ndarray, width: int = 12, height: int = 8, levels: int = 8) -> bytes:
    """Grey, shrink to width x height block averages, quantise to `levels` steps."""
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    small = cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA)
    return (small.astype(np.uint16) * levels // 256).astype(np.uint8).tobytes()


def _u16(io: np.ndarray, offset: int) -> int:
    return int(io[offset]) | int(io[offset + 1]) << 8


def _backgrounds(io: np.ndarray):
    """(bg, bgcnt, first screen block, block count, affine) for each enabled background."""
    dispcnt = _u16(io, 0)
    mode = dispcnt & 7
    for bg in range(4):
        if dispcnt & (0x100 << bg):
            bgcnt = _u16(io, 8 + 2 * bg)
            block, size = (bgcnt >> 8) & 31, bgcnt >> 14
            affine = (mode == 1 and bg == 2) or (mode == 2 and bg >= 2)
            yield bg, bgcnt, block, (AFFINE_MAP_BLOCKS if affine else TEXT_MAP_BLOCKS)[size], affine


def graphics_tiles(io: np.ndarray) -> np.ndarray:
    """Which 32-byte chunks of video memory hold tile graphics, from the video registers."""
    mask = np.ones(VRAM_TILES, dtype=bool)
    if _u16(io, 0) & 7 >= 3:  # bitmap modes: everything below the sprite tiles is the picture
        mask[: BITMAP_SPRITE_START // TILE_BYTES] = False
        return mask
    for _, _, block, blocks, _ in _backgrounds(io):
        mask[block * TILES_PER_SCREEN_BLOCK : (block + blocks) * TILES_PER_SCREEN_BLOCK] = False
    return mask


def _picture_fingerprint(frame: np.ndarray, width: int = 8, height: int = 6, levels: int = 4) -> bytes:
    """Like screen_key, but stretched to the full 0..1 range first, so fades look the same."""
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    small = cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA)
    lo, hi = small.min(), small.max()
    if hi - lo < 4:
        return bytes(width * height)
    return np.clip((small - lo) / (hi - lo) * levels, 0, levels - 1).astype(np.uint8).tobytes()


def scene_key(vram: np.ndarray, io: np.ndarray, frame: np.ndarray) -> bytes:
    mode = _u16(io, 0) & 7
    h = hashlib.blake2b(bytes([mode]), digest_size=12)
    if mode >= 3:
        h.update(_picture_fingerprint(frame))
        return h.digest()
    for bg, bgcnt, block, blocks, affine in _backgrounds(io):
        h.update(bytes([bg]) + bgcnt.to_bytes(2, "little"))
        tile_map = vram[block * 0x800 : (block + blocks) * 0x800]
        # text maps: 16-bit entries, low 10 bits = tile number; affine maps: 8-bit numbers
        tiles = tile_map if affine else tile_map.view("<u2") & 0x3FF
        h.update(np.unique(tiles // TILE_BUCKET).astype(np.uint16).tobytes())
        if not affine:
            x, y = _u16(io, 0x10 + 4 * bg) & 0x1FF, _u16(io, 0x12 + 4 * bg) & 0x1FF
            h.update(bytes([x // SCROLL_STEP, y // SCROLL_STEP]))
    return h.digest()


class GraphicsTracker:
    def __init__(self):
        self._seen = set()
        self._previous = None

    def update(self, vram: np.ndarray, io: np.ndarray) -> int:
        """Return how many tiles of graphics were never seen before."""
        tiles = vram.reshape(-1, TILE_BYTES)
        if self._previous is None:
            changed = np.ones(len(tiles), dtype=bool)
        else:
            changed = np.any(tiles != self._previous, axis=1)
        self._previous = tiles.copy()  # vram is a live view into the emulator

        before = len(self._seen)
        for i in np.flatnonzero(changed & graphics_tiles(io)):
            tile = tiles[i]
            if tile.any():
                self._seen.add(tile.tobytes())
        return len(self._seen) - before

    @property
    def seen(self) -> int:
        return len(self._seen)
