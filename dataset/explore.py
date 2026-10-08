r"""
Automated frame collection: Go-Explore style exploration of a GBA game.

Loop: pick a remembered game state ("cell"), restore it, play a burst of random
input (walking, mashing, button chords, scripted special moves), and look at the
game every few frames. Two things count as progress:
  - a new scene: the backgrounds changed (new stage, menu page, dialogue line,
    scrolled further), see novelty.scene_key. The state becomes a new cell, so
    later bursts can continue from there. Sprites are ignored, so fighters moving
    around or an animated menu create nothing.
  - new graphics: tiles never seen in video memory before (a new enemy, pose,
    special move or effect), see novelty.GraphicsTracker.
Besides the power-on state, scripted routes (see routes.py) can add further roots,
e.g. one per stage; the archive shares bursts evenly between roots. Bursts from
those roots never press Start, because in gameplay Start only opens the pause menu
(and from there "end game" leads back to the title).
Games with a progress counter in RAM (e.g. the score, see progress.py) also make
a new cell each time the counter reaches a higher level, so bursts keep the damage
done to an enemy wave that has to be beaten before the screen scrolls on.
Cells whose bursts stop making progress get picked less (see archive.py). Frames
are saved selectively (see recorder.py), at most `per_scene_cap` per cell; progress
frames are always saved, at most one every `forced_save_gap` emulated frames.
Cheats (infinite health etc.) stop the run from getting stuck on game over.

    python -m dataset.explore --rom "roms\rom_d\Dragon Ball GT - Transformation.gba" --minutes 30

Output: data\raw\<game>\<timestamp>\ with frames\*.png, frames.jsonl (one line per
frame), cells.jsonl (cell lineage, for splitting train/validation by region) and run.json
(which root is which route seed). Route seeds are made on the first run for a game
and kept in data\seeds\<game>\; delete that folder to make them again.
"""

import argparse
import json
import random
import time
from dataclasses import asdict
from pathlib import Path

from dataset.archive import Archive
from dataset.inputs import InputPolicy
from dataset.novelty import GraphicsTracker, scene_key, screen_key
from dataset.recorder import FrameRecorder

VRAM, IO = 0x06000000, 0x04000000


class Explorer:
    def __init__(self, emu, out_dir, rng: random.Random, cheats=(),
                 burst_frames=(60, 360), check_every: int = 4, screen_shape=(8, 6, 4),
                 per_group_cap: int = 2, per_scene_cap: int = 20, min_diff: float = 4.0,
                 forced_save_gap: int = 300, policy: InputPolicy | None = None, progress=None):
        self.emu = emu
        self.out_dir = Path(out_dir)
        self.rng = rng
        self.cheats = list(cheats)
        self.burst_frames = burst_frames
        self.check_every = check_every
        self.screen_shape = screen_shape  # (width, height, levels) of the screen fingerprint
        self.per_scene_cap = per_scene_cap
        self.forced_save_gap = forced_save_gap
        self.progress = progress  # emu -> progress level (see progress.py), part of the cell key

        self.archive = Archive(rng)
        self.tracker = GraphicsTracker()
        self.policy = policy or InputPolicy(rng)
        self.recorder = FrameRecorder(self.out_dir, per_group_cap=per_group_cap, min_diff=min_diff)
        self.iterations = 0
        self.frames_run = 0
        self._last_forced_save = -forced_save_gap
        self._saved_per_cell: dict[int, int] = {}
        self._no_start_roots: set[int] = set()
        self.emu.apply_cheats(self.cheats)  # mGBA keeps them across state loads

    def _level(self) -> int:
        return self.progress(self.emu) if self.progress else 0

    def _look(self):
        """(frame, cell key, progress level, number of never-seen tiles) for the current moment."""
        frame = self.emu.frame()
        vram, io = self.emu.memory(VRAM), self.emu.memory(IO)
        level = self._level()
        key = scene_key(vram, io, frame) + level.to_bytes(4, "little")
        return frame, key, level, self.tracker.update(vram, io)

    def boot(self, frames: int = 10) -> None:
        """Start the power-on root: title screens and menus, where Start is needed."""
        self.emu.step(0, frames)
        self.frames_run += frames
        _, key, _, _ = self._look()
        self.archive.add(key, self.emu.save_state())

    def add_root(self, state: bytes, allow_start: bool = True):
        """Explore from `state` as a root of its own; None if that scene is already known."""
        self.emu.load_state(state)
        _, key, _, _ = self._look()
        cell = self.archive.add(key, state)
        if cell is not None and not allow_start:
            self._no_start_roots.add(cell.root)
        return cell

    def iterate(self) -> None:
        start = current = self.archive.choose()
        self.emu.load_state(start.state)
        masks = self.policy.burst(self.rng.randint(*self.burst_frames),
                                  allow_start=start.root not in self._no_start_roots)
        found_new = False
        for i, mask in enumerate(masks, 1):
            self.emu.step(mask)
            if i % self.check_every:
                continue
            frame, key, level, new_graphics = self._look()
            new_scene = key not in self.archive
            if new_scene:  # saving a state costs ~1 ms, so only for new scenes
                current = self.archive.add(key, self.emu.save_state(), parent=current.id)
            else:
                current = self.archive.get(key)
            progress = new_scene or new_graphics > 0
            found_new |= progress
            now = self.frames_run + i
            force = progress and now - self._last_forced_save >= self.forced_save_gap
            if not force and self._saved_per_cell.get(current.id, 0) >= self.per_scene_cap:
                continue
            if self.recorder.offer(frame, group=screen_key(frame, *self.screen_shape), force=force,
                                   cell=current.id, root=current.root,
                                   progress=level, new_scene=new_scene, new_graphics=new_graphics,
                                   forced=force, iteration=self.iterations):
                self._saved_per_cell[current.id] = self._saved_per_cell.get(current.id, 0) + 1
                if force:
                    self._last_forced_save = now
        self.archive.finish_burst(start, found_new)
        self.frames_run += len(masks)
        self.iterations += 1

    def run(self, iterations=None, seconds=None, target_frames=None, log_every=None) -> None:
        t0 = last_log = time.perf_counter()
        frames0 = self.frames_run
        try:
            while True:
                now = time.perf_counter()
                if iterations is not None and self.iterations >= iterations:
                    break
                if seconds is not None and now - t0 >= seconds:
                    break
                if target_frames is not None and self.recorder.count >= target_frames:
                    break
                if log_every and now - last_log >= log_every:
                    fps = (self.frames_run - frames0) / (now - t0)
                    print(f"[{now - t0:6.0f}s] {fps:5.0f} fps | iterations {self.iterations} | "
                          f"cells {len(self.archive)} | graphics {self.tracker.seen} | "
                          f"frames saved {self.recorder.count}", flush=True)
                    last_log = now
                self.iterate()
        except KeyboardInterrupt:
            print("stopped by user", flush=True)

    def close(self) -> None:
        self.recorder.close()
        with open(self.out_dir / "cells.jsonl", "w", encoding="utf-8") as f:
            for cell in self.archive.cells:
                f.write(json.dumps({"id": cell.id, "parent": cell.parent, "root": cell.root,
                                    "key": cell.key.hex(), "chosen": cell.chosen}) + "\n")


def main() -> None:
    from dataset.cheats import GAMES, cheats_for, rom_game_code
    from dataset.emulator import HeadlessGBA
    from dataset.progress import progress_for
    from dataset.routes import ROUTES, load_or_make_seeds

    ap = argparse.ArgumentParser(description="Collect GBA frames by automated exploration")
    ap.add_argument("--rom", required=True)
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--out", default=r"data\raw")
    ap.add_argument("--seeds", default=r"data\seeds", help="where route seed states are kept")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--target-frames", type=int, default=None, help="stop after saving this many frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cheats", action="store_true")
    ap.add_argument("--no-routes", action="store_true", help="explore from power-on only")
    ap.add_argument("--no-progress", action="store_true", help="ignore the game's progress counter")
    args = ap.parse_args()

    code = rom_game_code(args.rom)
    game = GAMES[code].name if code in GAMES else code
    cheats = [] if args.no_cheats else cheats_for(code)
    if not cheats and not args.no_cheats:
        print(f"no cheats known for game code {code}; exploring without them")
    counter = None if args.no_progress else progress_for(code)

    seeds = []
    if code in ROUTES and not args.no_routes:
        seed_dir = Path(args.seeds) / game

        def route():
            # routes are timed with the game's cheats on
            with HeadlessGBA(args.core, args.rom) as route_emu:
                route_emu.apply_cheats(cheats_for(code))
                yield from ROUTES[code](route_emu, random.Random(0))

        if not any(seed_dir.glob("*.state")):
            print(f"making route seeds in {seed_dir} (first run for this game, can take a few minutes)")
        seeds = load_or_make_seeds(seed_dir, route)
        print(f"{len(seeds)} route seeds: {', '.join(name for name, _ in seeds)}")

    out_dir = Path(args.out) / game / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    print(f"game {game} ({code}), {len(cheats)} cheats, writing to {out_dir}")

    with HeadlessGBA(args.core, args.rom) as emu:
        explorer = Explorer(emu, out_dir, random.Random(args.seed), cheats=cheats,
                            progress=counter.level if counter else None)
        explorer.boot()
        roots = {0: "power-on"}
        for name, state in seeds:
            cell = explorer.add_root(state, allow_start=False)
            if cell is not None:
                roots[cell.id] = name
        (out_dir / "run.json").write_text(json.dumps({
            "rom": str(args.rom), "game_code": code, "seed": args.seed,
            "minutes": args.minutes, "cheats": cheats, "roots": roots,
            "progress": {**asdict(counter), "address": f"{counter.address:#010x}"} if counter else None,
        }, indent=2))
        explorer.run(seconds=args.minutes * 60, target_frames=args.target_frames, log_every=10)
        explorer.close()
    print(f"done: {explorer.recorder.count} frames, {len(explorer.archive)} cells, "
          f"{explorer.tracker.seen} graphics items, {explorer.iterations} iterations")


if __name__ == "__main__":
    main()
