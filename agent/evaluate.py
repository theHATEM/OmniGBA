r"""
Compare play styles on Dragon Ball GT from the same start state over the same game time:

    ppo       a model from train.py
    explorer  the explorer's random input (dataset/inputs.py), one button mask per frame
    uniform   a random action every 4 frames, i.e. a model that learned nothing

Each is scored on points, new ground (pixels the camera got further right), waves beaten,
enemies killed, scenes (novelty.scene_key) and new graphics tiles (novelty.GraphicsTracker), checked
every 4 frames as the explorer does. A contact sheet per play style shows what it did.

    python -m agent.evaluate --model data\rl\<run>\model.zip --minutes 3 --episodes 5

Output in the model's folder (or --out): eval_<play style>.png and eval.json.
"""

import argparse
import json
import random
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from agent.env import IWRAM, DbgtEnv, mask_to_action, start_state
from dataset.inputs import InputPolicy
from dataset.novelty import GraphicsTracker, scene_key

VRAM, IO = 0x06000000, 0x04000000


class FrameStack:
    """The last n observations along the channel axis, as VecFrameStack does in training."""

    def __init__(self, n: int):
        self.n = n

    def reset(self, obs: np.ndarray) -> np.ndarray:
        self._frames = deque([np.zeros_like(obs)] * (self.n - 1) + [obs], maxlen=self.n)
        return np.concatenate(self._frames)

    def add(self, obs: np.ndarray) -> np.ndarray:
        self._frames.append(obs)
        return np.concatenate(self._frames)


class PPOPolicy:
    frames_per_step = 4

    def __init__(self, model, deterministic: bool = False):
        self.model = model
        self.deterministic = deterministic
        self._stack = FrameStack(model.observation_space.shape[0])

    def reset(self, obs):
        self._stacked = self._stack.reset(obs)

    def act(self, obs):
        if obs is not None:
            self._stacked = self._stack.add(obs)
        action, _ = self.model.predict(self._stacked, deterministic=self.deterministic)
        return action


class ExplorerInput:
    """The explorer's random bursts (60-360 frames each), one action per frame."""

    frames_per_step = 1

    def __init__(self, rng: random.Random, burst_frames=(60, 360)):
        self._rng = rng
        self._input = InputPolicy(rng)
        self.burst_frames = burst_frames

    def reset(self, obs):
        self._masks = deque()

    def act(self, obs):
        if not self._masks:
            frames = self._rng.randint(*self.burst_frames)
            self._masks.extend(self._input.burst(frames, allow_start=False))
        return mask_to_action(self._masks.popleft())


class UniformRandom:
    frames_per_step = 4

    def __init__(self, action_space, seed: int):
        self.space = action_space
        self.space.seed(seed)

    def reset(self, obs):
        pass

    def act(self, obs):
        return self.space.sample()


def play(env: DbgtEnv, policy, seconds: float, sheet_every: float = 10.0):
    """Play one episode of `seconds` game time; returns (stats, frames sampled every sheet_every s)."""
    env.frames_per_step = policy.frames_per_step
    env.max_steps = round(seconds * 60 / policy.frames_per_step)
    obs, _ = env.reset()
    policy.reset(obs)
    tracker, scenes, new_graphics, frames = GraphicsTracker(), set(), 0, []
    vram, io = env.emu.memory(VRAM), env.emu.memory(IO)
    tracker.update(vram, io)  # what the start state already has does not count
    obs = None  # the first action sees the reset observation
    sheet_frames = round(sheet_every * 60)
    truncated = False
    while not truncated:
        obs, _, _, truncated, info = env.step(policy.act(obs))
        if info["frames"] % 4 == 0:
            frame = env.emu.frame()
            scenes.add(scene_key(vram, io, frame))
            new_graphics += tracker.update(vram, io)
        if info["frames"] % sheet_frames == 0:
            frames.append(env.emu.frame())
    stats = {"frames": info["frames"], "score": info["score"], "ground": info["ground"],
             "waves": info["waves"], "kills": info["kills"], "scenes": len(scenes), "new_graphics": new_graphics}
    return stats, frames


def contact_sheet(frames, columns: int = 6) -> np.ndarray:
    """Frames in rows of `columns`, with a thin white line between them."""
    tiles = []
    for frame in frames:
        tile = cv2.copyMakeBorder(frame, 0, 2, 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        tiles.append(tile)
    while len(tiles) % columns:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i : i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


def main() -> None:
    from dataset.cheats import cheats_for

    ap = argparse.ArgumentParser(description="Compare a trained model with random play on Dragon Ball GT")
    ap.add_argument("--model", help="model.zip from agent.train; leave out to compare random play only")
    ap.add_argument("--rom", default=r"roms\rom_d\Dragon Ball GT - Transformation.gba")
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--seeds", default=r"data\seeds")
    ap.add_argument("--root", default="level00", help="route seed to start from (level00 = planet 0)")
    ap.add_argument("--minutes", type=float, default=3.0, help="game time per episode")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--deterministic", action="store_true", help="the model always picks its likeliest action")
    ap.add_argument("--out", help="folder for eval.json and the contact sheets (default: the model's folder)")
    args = ap.parse_args()

    out = Path(args.out or (Path(args.model).parent if args.model else r"data\rl\eval"))
    out.mkdir(parents=True, exist_ok=True)
    env = DbgtEnv(args.core, args.rom, start_state(args.core, args.rom, args.seeds, args.root),
                  cheats=cheats_for("BT4E"))
    styles = {}
    if args.model:
        from stable_baselines3 import PPO

        model = PPO.load(args.model, device="cpu")
        styles["ppo"] = lambda episode: PPOPolicy(model, args.deterministic)
    styles["explorer"] = lambda episode: ExplorerInput(random.Random(episode))
    styles["uniform"] = lambda episode: UniformRandom(env.action_space, episode)

    results = {}
    for name, make in styles.items():
        runs = []
        for episode in range(args.episodes):
            stats, frames = play(env, make(episode), args.minutes * 60)
            runs.append(stats)
            print(f"{name:8s} episode {episode}: {stats}", flush=True)
            if episode == 0:
                sheet = cv2.cvtColor(contact_sheet(frames), cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(out / f"eval_{name}.png"), sheet)
        results[name] = {"episodes": runs,
                         "mean": {k: float(np.mean([r[k] for r in runs])) for k in runs[0]}}
    env.close()

    (out / "eval.json").write_text(json.dumps({"model": args.model, "root": args.root,
                                               "minutes": args.minutes, "results": results}, indent=2))
    print(f"\nmean over {args.episodes} episodes of {args.minutes:g} game minutes from {args.root}:")
    keys = ["score", "ground", "waves", "kills", "scenes", "new_graphics"]
    print(f"{'':10s}" + "".join(f"{k:>14s}" for k in keys))
    for name, result in results.items():
        print(f"{name:10s}" + "".join(f"{result['mean'][k]:14.1f}" for k in keys))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
