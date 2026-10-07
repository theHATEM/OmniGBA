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
Cells whose bursts stop making progress get picked less (see archive.py). Frames
are saved selectively (see recorder.py), at most `per_scene_cap` per scene; progress
frames are always saved, at most one every `forced_save_gap` emulated frames.
Cheats (infinite health etc.) stop the run from getting stuck on game over.

    python -m dataset.explore --rom "roms\rom_d\Dragon Ball GT - Transformation.gba" --minutes 30

Output: data\raw\<game>\<timestamp>\ with frames\*.png, frames.jsonl (one line per
frame), cells.jsonl (cell lineage, for splitting train/validation by region) and run.json.
"""

import argparse
import json
import random
import time
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
                 forced_save_gap: int = 300):
        self.emu = emu
        self.out_dir = Path(out_dir)
        self.rng = rng
        self.cheats = list(cheats)
        self.burst_frames = burst_frames
        self.check_every = check_every
        self.screen_shape = screen_shape  # (width, height, levels) of the screen fingerprint
        self.per_scene_cap = per_scene_cap
        self.forced_save_gap = forced_save_gap

        self.archive = Archive(rng)
        self.tracker = GraphicsTracker()
        self.policy = InputPolicy(rng)
        self.recorder = FrameRecorder(self.out_dir, per_group_cap=per_group_cap, min_diff=min_diff)
        self.iterations = 0
        self.frames_run = 0
        self._last_forced_save = -forced_save_gap
        self._saved_per_cell: dict[int, int] = {}

    def _look(self):
        """(frame, scene key, number of never-seen tiles) for the current moment."""
        frame = self.emu.frame()
        vram, io = self.emu.memory(VRAM), self.emu.memory(IO)
        return frame, scene_key(vram, io, frame), self.tracker.update(vram, io)

    def boot(self, frames: int = 10) -> None:
        self.emu.apply_cheats(self.cheats)
        self.emu.step(0, frames)
        self.frames_run += frames
        _, key, _ = self._look()
        self.archive.add(key, self.emu.save_state())

    def iterate(self) -> None:
        start = current = self.archive.choose()
        self.emu.load_state(start.state)
        masks = self.policy.burst(self.rng.randint(*self.burst_frames))
        found_new = False
        for i, mask in enumerate(masks, 1):
            self.emu.step(mask)
            if i % self.check_every:
                continue
            frame, key, new_graphics = self._look()
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
                                   cell=current.id, new_scene=new_scene, new_graphics=new_graphics,
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
                f.write(json.dumps({"id": cell.id, "parent": cell.parent, "key": cell.key.hex(),
                                    "chosen": cell.chosen}) + "\n")


def main() -> None:
    from dataset.cheats import GAMES, cheats_for, rom_game_code
    from dataset.emulator import HeadlessGBA

    ap = argparse.ArgumentParser(description="Collect GBA frames by automated exploration")
    ap.add_argument("--rom", required=True)
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--out", default=r"data\raw")
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--target-frames", type=int, default=None, help="stop after saving this many frames")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cheats", action="store_true")
    args = ap.parse_args()

    code = rom_game_code(args.rom)
    game = GAMES[code].name if code in GAMES else code
    cheats = [] if args.no_cheats else cheats_for(code)
    if not cheats and not args.no_cheats:
        print(f"no cheats known for game code {code}; exploring without them")

    out_dir = Path(args.out) / game / time.strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    (out_dir / "run.json").write_text(json.dumps({
        "rom": str(args.rom), "game_code": code, "seed": args.seed,
        "minutes": args.minutes, "cheats": cheats,
    }, indent=2))
    print(f"game {game} ({code}), {len(cheats)} cheats, writing to {out_dir}")

    with HeadlessGBA(args.core, args.rom) as emu:
        explorer = Explorer(emu, out_dir, random.Random(args.seed), cheats=cheats)
        explorer.boot()
        explorer.run(seconds=args.minutes * 60, target_frames=args.target_frames, log_every=10)
        explorer.close()
    print(f"done: {explorer.recorder.count} frames, {len(explorer.archive)} cells, "
          f"{explorer.tracker.seen} graphics items, {explorer.iterations} iterations")


if __name__ == "__main__":
    main()
