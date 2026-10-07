"""
Scripted routes from power-on into each game's stages.

Their end states become extra exploration roots (see explore.py), so stages that
random play from the title would rarely reach still get explored. Routes are frame
exact: from power-on the emulator is deterministic, so with the same ROM, core and
cheats they replay the same way every time. A route step is (button mask, frames to
hold it); mask 0 waits.

Generating the states takes a while (Naruto's chapters are reached by playing), so
load_or_make_seeds keeps them on disk.
"""

from pathlib import Path

from dataset.inputs import A, START, InputPolicy

IWRAM = 0x03000000


def press(mask: int, then: int = 16) -> list[tuple[int, int]]:
    """Hold `mask` for 4 frames, then release for `then` frames."""
    return [(mask, 4), (0, then)]


def play(emu, steps) -> None:
    for mask, frames in steps:
        emu.step(mask, frames)


# --- Dragon Ball GT: Transformation (USA, BT4E) --------------------------------------

# Title -> Story Mode -> Start New Game -> opening cutscene (skipped) -> star map
DBGT_TO_STAR_MAP = [
    (0, 1000),
    *press(A, 76), *press(A, 76), *press(A, 616),
    *press(A, 76), *press(A, 76), *press(A, 76), *press(A, 76),
    *press(START, 136),
]
DBGT_LEVEL = 0x1D04  # IWRAM: planet the star map cursor is on ("Level Modifier" cheat)
DBGT_LEVELS = 11  # planets on the star map; higher values wrap around
# Pressing A enters the planet; more A presses skip its intro and dialogue
DBGT_ENTER_LEVEL = [(0, 90), *press(A, 76) * 45]


def dbgt_levels(emu, rng, levels=range(DBGT_LEVELS)):
    """One state per planet, a few seconds into its first stage."""
    play(emu, DBGT_TO_STAR_MAP)
    star_map = emu.save_state()
    for level in levels:
        emu.load_state(star_map)
        emu.memory(IWRAM)[DBGT_LEVEL] = level
        play(emu, DBGT_ENTER_LEVEL)
        yield f"level{level:02d}", emu.save_state()


# --- Naruto: Ninja Council 2 (USA, BN2E) -----------------------------------------------

# Title -> Start -> main menu -> New Game -> chapter 1 title card. (VS and Co-op mode
# only wait for a link cable, so the story chapters are the only stages.)
NARUTO_TO_CHAPTER_1 = [(0, 2100), *press(START, 466), *press(A, 266)]
NARUTO_CHAPTER = 0x453F  # IWRAM: current story chapter, 0 at the title screen


def chapter_seeds(emu, rng, chapter_addr: int, to_first_chapter, max_chapters: int = 12,
                  frames_per_chapter: int = 30 * 60 * 60, burst_frames: int = 600):
    """
    Reach chapters by playing: random input (never Start, which pauses) until the
    chapter number in RAM changes, saving a state at the start of each chapter.
    If the game falls back to the title (time limit, "end game"), the chapter is
    retried from its start. Stops at max_chapters, or when a chapter is not beaten
    within frames_per_chapter frames of play.
    """
    to_first_chapter(emu)
    iwram = emu.memory(IWRAM)
    chapter, seed = int(iwram[chapter_addr]), emu.save_state()
    yield f"chapter{chapter:02d}", seed
    policy = InputPolicy(rng)
    spent = 0
    while chapter < max_chapters and spent < frames_per_chapter:
        for mask in policy.burst(burst_frames, allow_start=False):
            emu.step(mask)
        spent += burst_frames
        now = int(iwram[chapter_addr])
        if now == 0:
            emu.load_state(seed)
        elif now != chapter:
            chapter, seed, spent = now, emu.save_state(), 0
            yield f"chapter{chapter:02d}", seed


def naruto_chapters(emu, rng, max_chapters: int = 12, frames_per_chapter: int = 30 * 60 * 60):
    yield from chapter_seeds(emu, rng, NARUTO_CHAPTER, lambda e: play(e, NARUTO_TO_CHAPTER_1),
                             max_chapters, frames_per_chapter)


ROUTES = {"BT4E": dbgt_levels, "BN2E": naruto_chapters}


def load_or_make_seeds(seed_dir, route) -> list[tuple[str, bytes]]:
    """
    The (name, state) pairs saved in seed_dir, or, if there are none, the ones
    route() yields, saved there first. Files are renamed into place only at the end,
    so an interrupted run leaves nothing that looks complete.
    """
    seed_dir = Path(seed_dir)
    saved = sorted(seed_dir.glob("*.state"))
    if saved:
        return [(p.stem, p.read_bytes()) for p in saved]
    seed_dir.mkdir(parents=True, exist_ok=True)
    seeds = []
    for name, state in route():
        (seed_dir / f"{name}.state.tmp").write_bytes(state)
        seeds.append((name, state))
    for name, _ in seeds:
        (seed_dir / f"{name}.state.tmp").replace(seed_dir / f"{name}.state")
    return seeds
