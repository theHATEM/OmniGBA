r"""
Train a PPO agent on Dragon Ball GT (see env.py), with several emulators playing at once.

    python -m agent.train --timesteps 1000000

One timestep is one action (4 game frames), so 1,000,000 timesteps are about 18 hours of
game time. Output in data\rl\<timestamp>\:

    model.zip                  the trained model (checkpoints\ has earlier ones)
    progress.csv               the training log; dbgt/score, dbgt/ground and dbgt/waves are
                               means over the episodes that ended since the row before
    monitor\<n>.monitor.csv    one line per episode played by emulator n
    run.json                   the settings

Ctrl+C stops early and still saves model.zip. Then compare it with random play:

    python -m agent.evaluate --model data\rl\<timestamp>\model.zip
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from agent.env import DbgtEnv, start_state

INFO_KEYS = ("score", "ground", "waves")


def _make_env(core, rom, state, cheats, max_steps, monitor_file):
    env = DbgtEnv(core, rom, state, cheats=cheats, max_steps=max_steps)
    return Monitor(env, monitor_file, info_keywords=INFO_KEYS)


class EpisodeStats(BaseCallback):
    """Logs the mean score, new ground and waves of the episodes that ended since the last log row."""

    def _on_training_start(self) -> None:
        self._episodes = []

    def _on_step(self) -> bool:
        self._episodes += [info["episode"] for info in self.locals["infos"] if "episode" in info]
        return True

    def _on_rollout_end(self) -> None:
        if self._episodes:
            for key in INFO_KEYS:
                self.logger.record(f"dbgt/{key}", float(np.mean([e[key] for e in self._episodes])))
            self._episodes = []


def train(core, rom, state: bytes, cheats, out_dir, timesteps: int, envs: int = 8,
          max_steps: int = 1800, n_steps: int = 128, seed: int = 0, device: str = "auto",
          checkpoint_every: int = 100_000) -> Path:
    """Train from `state` for `timesteps` actions in total; returns the path of the saved model."""
    out_dir = Path(out_dir)
    (out_dir / "monitor").mkdir(parents=True, exist_ok=True)
    factories = [partial(_make_env, str(core), str(rom), state, list(cheats), max_steps,
                         str(out_dir / "monitor" / str(i))) for i in range(envs)]
    # each emulator in its own process: one mGBA core per process
    vec = SubprocVecEnv(factories) if envs > 1 else DummyVecEnv(factories)
    vec = VecFrameStack(vec, 4, channels_order="first")
    # Atari settings from the PPO paper / rl-baselines3-zoo
    model = PPO("CnnPolicy", vec, learning_rate=2.5e-4, n_steps=n_steps,
                batch_size=min(256, n_steps * envs), n_epochs=4, gamma=0.99, gae_lambda=0.95,
                clip_range=0.1, ent_coef=0.01, vf_coef=0.5, seed=seed, device=device, verbose=1)
    model.set_logger(configure(str(out_dir), ["stdout", "csv"]))
    checkpoints = CheckpointCallback(max(checkpoint_every // envs, 1), str(out_dir / "checkpoints"),
                                     name_prefix="ppo")
    try:
        model.learn(timesteps, callback=[EpisodeStats(), checkpoints], log_interval=4)
    except KeyboardInterrupt:
        print("stopped by user; saving the model so far", flush=True)
    path = out_dir / "model.zip"
    model.save(path)
    try:
        vec.close()
    except (EOFError, BrokenPipeError, ConnectionError):
        pass  # after Ctrl+C the emulator processes may already be gone
    return path


def main() -> None:
    from dataset.cheats import cheats_for

    ap = argparse.ArgumentParser(description="Train a PPO agent on Dragon Ball GT")
    ap.add_argument("--rom", default=r"roms\rom_d\Dragon Ball GT - Transformation.gba")
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--seeds", default=r"data\seeds")
    ap.add_argument("--root", default="level00", help="route seed to start from (level00 = planet 0)")
    ap.add_argument("--timesteps", type=int, default=1_000_000, help="actions in total (4 game frames each)")
    ap.add_argument("--envs", type=int, default=8, help="emulators playing at once")
    ap.add_argument("--minutes", type=float, default=2.0, help="game time per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto (GPU if PyTorch has CUDA), cuda or cpu")
    ap.add_argument("--out", default=r"data\rl")
    args = ap.parse_args()

    out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
    out.mkdir(parents=True)
    state = start_state(args.core, args.rom, args.seeds, args.root)
    max_steps = round(args.minutes * 60 * 60 / 4)
    (out / "run.json").write_text(json.dumps({**vars(args), "max_steps": max_steps}, indent=2))
    print(f"training from {args.root} for {args.timesteps} timesteps on {args.envs} emulators, "
          f"writing to {out}", flush=True)
    path = train(args.core, args.rom, state, cheats_for("BT4E"), out, args.timesteps,
                 envs=args.envs, max_steps=max_steps, seed=args.seed, device=args.device)
    print(f"model saved to {path}\ncompare with random play: python -m agent.evaluate --model {path}")


if __name__ == "__main__":
    main()
