r"""
Train a PPO agent on Dragon Ball GT (see env.py), with several emulators playing at once.

    python -m agent.train --timesteps 1000000

One timestep is one action (4 game frames), so 1,000,000 timesteps are about 18 hours of
game time. Output in data\rl\<timestamp>\:

    model.zip                  the trained model (checkpoints\ has earlier ones)
    progress.csv               the training log; dbgt/score, dbgt/ground and dbgt/waves are
                               means over the episodes that ended since the row before
    monitor\<n>.monitor.csv    one line per episode played by emulator n
    best\                      the best episode of the run (see BestEpisode)
    run.json                   the settings

Ctrl+C stops early and still saves model.zip. Then compare it with random play:

    python -m agent.evaluate --model data\rl\<timestamp>\model.zip

A run that was stopped (or a longer run from a finished one, with a larger --timesteps in
run.json) continues in its folder from its latest checkpoint, with the settings in run.json:

    python -m agent.train --resume data\rl\<timestamp>

--per-process N runs N emulators in each worker process (see vec_env.py): faster when the
emulators outnumber the CPU's threads, e.g. --envs 24 --per-process 2 on 12 threads.
"""

import argparse
import csv
import json
import math
import sys
import time
import zipfile
from functools import partial
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import CSVOutputFormat, HumanOutputFormat, Logger
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from agent.env import DbgtEnv, start_state
from agent.replay import record, save_episode
from agent.vec_env import SubprocGroupVecEnv, core_copies

INFO_KEYS = ("score", "ground", "waves")
TASK_SETTINGS = ("rom", "core", "seeds", "root", "timesteps", "minutes", "seed")  # kept on --resume
SPEED_SETTINGS = {"envs": 8, "per_process": 1, "device": "auto"}  # defaults; --resume may change them


def _make_env(core, rom, state, cheats, max_steps, monitor_file, append=False):
    env = DbgtEnv(core, rom, state, cheats=cheats, max_steps=max_steps)
    append = append and Path(f"{monitor_file}.monitor.csv").exists()
    return Monitor(env, monitor_file, info_keywords=INFO_KEYS, override_existing=not append)


class ContinuedCSV(CSVOutputFormat):
    """Stable-Baselines3's CSV log, continuing the rows and columns already in the file."""

    def __init__(self, filename):
        old = Path(filename).read_text() if Path(filename).exists() else ""
        super().__init__(filename)  # starts the file again...
        if old:  # ...so put the old rows back
            self.keys = old.splitlines()[0].split(self.separator)
            self.file.write(old)
            self.file.flush()


def saved_timesteps(path) -> int:
    """The timesteps a model saved by Stable-Baselines3 was trained for."""
    with zipfile.ZipFile(path) as archive:
        return json.loads(archive.read("data"))["num_timesteps"]


def _saved_models(run_dir) -> list[Path]:
    run_dir = Path(run_dir)
    return [p for p in [*(run_dir / "checkpoints").glob("*.zip"), run_dir / "model.zip"] if p.exists()]


def latest_model(run_dir) -> Path:
    """The run's saved model with the most timesteps: a checkpoint, or model.zip if the run ended."""
    saved = _saved_models(run_dir)
    if not saved:
        raise FileNotFoundError(f"no checkpoint or model.zip in {run_dir}")
    return max(saved, key=saved_timesteps)


def resume_point(path) -> tuple[Path, Path]:
    """(run folder, model to continue from) for --resume: a run folder (its latest model) or a model in it."""
    path = Path(path)
    if path.is_dir():
        return path, latest_model(path)
    run_dir = next(p for p in path.parents if (p / "run.json").exists())
    return run_dir, path


def set_aside_after(run_dir, timesteps: int) -> None:
    """Moves what the run saved after `timesteps` aside, to continue it from an earlier checkpoint.

    Later models go to checkpoints\\dropped\\ (so latest_model no longer finds them) and later
    progress.csv rows to progress_dropped.csv. The monitor CSVs keep their episodes."""
    run_dir = Path(run_dir)
    for model in _saved_models(run_dir):
        if saved_timesteps(model) > timesteps:
            (run_dir / "checkpoints" / "dropped").mkdir(parents=True, exist_ok=True)
            model.replace(run_dir / "checkpoints" / "dropped" / model.name)
    progress = run_dir / "progress.csv"
    if progress.exists():
        header, *rows = progress.read_text().splitlines(keepends=True)
        column = header.rstrip("\n").split(",").index("time/total_timesteps")
        later = [int(row.split(",")[column]) > timesteps for row in rows]
        if any(later):
            dropped = run_dir / "progress_dropped.csv"
            old = dropped.read_text() if dropped.exists() else header
            dropped.write_text(old + "".join(row for row, drop in zip(rows, later) if drop))
            progress.write_text(header + "".join(row for row, drop in zip(rows, later) if not drop))


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


class BestEpisode(BaseCallback):
    """Keeps the highest-reward episode any emulator played as a recording in `folder` (see replay.py).

    Records actions, not frames: every episode starts from `state` and the emulator is
    deterministic, so the actions replay the whole episode. A tie keeps the earlier episode.
    history.csv gets a line per new best. In a resumed run, a new best must beat the one saved."""

    def __init__(self, folder, state: bytes, frames_per_step: int = 4, rom=None, cheats=(), root=None):
        super().__init__()
        self.folder = Path(folder)
        self.state, self.frames_per_step = state, frames_per_step
        self.about = {"rom": None if rom is None else str(rom), "cheats": list(cheats), "root": root}
        saved = self.folder / "episode.json"
        self.reward = json.loads(saved.read_text())["reward"] if saved.exists() else -math.inf
        self._actions = None  # per emulator, the actions of its current episode

    def _on_step(self) -> bool:
        self.add(self.locals["actions"], self.locals["dones"], self.locals["infos"])
        return True

    def add(self, actions, dones, infos) -> None:
        """One step of every emulator; the step that ends an episode belongs to it."""
        if self._actions is None:
            self._actions = [[] for _ in actions]
        for i, (action, done, info) in enumerate(zip(actions, dones, infos)):
            self._actions[i].append(np.array(action, np.uint8))
            if done:
                if info["episode"]["r"] > self.reward:
                    self._save(i, self._actions[i], info["episode"])
                self._actions[i] = []  # the emulator has reset; its next step starts a new episode

    def _save(self, emulator: int, actions, episode: dict) -> None:
        self.reward = episode["r"]
        stats = {"reward": episode["r"], **{k: episode[k] for k in INFO_KEYS},
                 "timesteps": self.num_timesteps, "emulator": emulator, **self.about}
        save_episode(self.folder, np.stack(actions), self.state, self.frames_per_step, stats)
        history = self.folder / "history.csv"
        new = not history.exists()
        with open(history, "a", newline="") as f:
            writer = csv.writer(f)
            if new:
                writer.writerow(["timesteps", "reward", *INFO_KEYS])
            writer.writerow([self.num_timesteps, episode["r"], *(episode[k] for k in INFO_KEYS)])
        print(f"new best episode at {self.num_timesteps} timesteps: reward {episode['r']:.1f}, "
              + ", ".join(f"{k} {episode[k]}" for k in INFO_KEYS) + f" (emulator {emulator})", flush=True)


def train(core, rom, state: bytes, cheats, out_dir, timesteps: int, envs: int = 8, per_process: int = 1,
          max_steps: int = 1800, n_steps: int = 128, seed: int = 0, device: str = "auto",
          checkpoint_every: int = 100_000, root=None, resume=None, batch_size=None, target_kl=None) -> Path:
    """Train from `state` for `timesteps` actions in total; returns the path of the saved model.

    `envs` emulators play at once, `per_process` of them in each worker process (see vec_env.py).
    The best episode is kept in out_dir\\best\\ (see BestEpisode); `root` names the start state there.
    `resume` is a model saved in out_dir (see latest_model) to continue from: training goes on up
    to `timesteps` in total, and the logs and the best episode continue too.
    `batch_size` is the minibatch size (default 256) and `target_kl` stops an update's epochs early
    once the policy moved that far (default: never); when resuming, None keeps the model's values."""
    # Checking the arguments of every action distribution took about a third of PPO's own time per
    # step (each check waits for the GPU); the distributions come from the policy, so skip the checks.
    torch.distributions.Distribution.set_default_validate_args(False)
    out_dir = Path(out_dir)
    (out_dir / "monitor").mkdir(parents=True, exist_ok=True)
    per_process = min(per_process, envs)
    cores = core_copies(core, per_process)  # one mGBA core per DLL file in a process
    factories = [partial(_make_env, str(cores[i % per_process]), str(rom), state, list(cheats), max_steps,
                         str(out_dir / "monitor" / str(i)), resume is not None) for i in range(envs)]
    if envs == 1:
        vec = DummyVecEnv(factories)
    elif per_process == 1:
        vec = SubprocVecEnv(factories)
    else:
        vec = SubprocGroupVecEnv(factories, per_process)
    vec = VecFrameStack(vec, 4, channels_order="first")
    if resume is None:
        # Atari settings from the PPO paper / rl-baselines3-zoo
        model = PPO("CnnPolicy", vec, learning_rate=2.5e-4, n_steps=n_steps,
                    batch_size=min(batch_size or 256, n_steps * envs), n_epochs=4, gamma=0.99,
                    gae_lambda=0.95, clip_range=0.1, ent_coef=0.01, vf_coef=0.5, target_kl=target_kl,
                    seed=seed, device=device, verbose=1)
    else:
        changes = {k: v for k, v in {"batch_size": batch_size, "target_kl": target_kl}.items() if v is not None}
        model = PPO.load(resume, env=vec, device=device, custom_objects=changes)
        print(f"continuing from {resume} at {model.num_timesteps} timesteps "
              f"(minibatch {model.batch_size}, target_kl {model.target_kl})", flush=True)
    model.set_logger(Logger(str(out_dir), [HumanOutputFormat(sys.stdout),
                                           ContinuedCSV(str(out_dir / "progress.csv"))]))
    checkpoints = CheckpointCallback(max(checkpoint_every // envs, 1), str(out_dir / "checkpoints"),
                                     name_prefix="ppo")
    best = BestEpisode(out_dir / "best", state, rom=rom, cheats=cheats, root=root)
    try:
        model.learn(timesteps - model.num_timesteps, callback=[EpisodeStats(), best, checkpoints],
                    log_interval=4, reset_num_timesteps=resume is None)
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
    ap.add_argument("--envs", type=int, help=f"emulators playing at once (default {SPEED_SETTINGS['envs']})")
    ap.add_argument("--per-process", type=int, dest="per_process",
                    help="emulators per worker process (default 1); the processes are envs / per-process")
    ap.add_argument("--minutes", type=float, default=2.0, help="game time per episode")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", help="auto (GPU if PyTorch has CUDA, the default), cuda or cpu")
    ap.add_argument("--batch-size", type=int, dest="batch_size", help="PPO minibatch size (default 256)")
    ap.add_argument("--target-kl", type=float, dest="target_kl",
                    help="stop an update's epochs early once the policy moved this far (default: never)")
    ap.add_argument("--out", default=r"data\rl")
    ap.add_argument("--resume", metavar="RUN", help="continue the run in this folder from its latest "
                    "checkpoint (or from a given checkpoint file in it; what was saved after it is set "
                    "aside), with the settings in its run.json; only --envs, --per-process, --device, "
                    "--batch-size and --target-kl can be changed")
    args = ap.parse_args()

    resume, run = None, {}
    if args.resume:
        out, resume = resume_point(args.resume)
        run = json.loads((out / "run.json").read_text())
        args = argparse.Namespace(**{**vars(args), **{k: run[k] for k in TASK_SETTINGS}})
    latest = {**run, **(run.get("resumed") or [{}])[-1]}  # the settings the run used last
    for key, default in SPEED_SETTINGS.items():
        if getattr(args, key) is None:
            setattr(args, key, latest.get(key, default))
    if resume is not None:
        steps = saved_timesteps(resume)
        if steps < saved_timesteps(latest_model(out)):
            print(f"setting aside what the run saved after {steps} timesteps", flush=True)
            set_aside_after(out, steps)
        run.setdefault("resumed", []).append({"from": str(resume), "timesteps": steps,
                                              **{k: getattr(args, k) for k in SPEED_SETTINGS},
                                              "batch_size": args.batch_size, "target_kl": args.target_kl,
                                              "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        (out / "run.json").write_text(json.dumps(run, indent=2))
    else:
        out = Path(args.out) / time.strftime("%Y%m%d_%H%M%S")
        out.mkdir(parents=True)
    state = start_state(args.core, args.rom, args.seeds, args.root)
    max_steps = round(args.minutes * 60 * 60 / 4)
    if resume is None:
        (out / "run.json").write_text(json.dumps({**vars(args), "max_steps": max_steps}, indent=2))
    print(f"training from {args.root} for {args.timesteps} timesteps on {args.envs} emulators "
          f"({args.per_process} per process), writing to {out}", flush=True)
    path = train(args.core, args.rom, state, cheats_for("BT4E"), out, args.timesteps, envs=args.envs,
                 per_process=args.per_process, max_steps=max_steps, seed=args.seed, device=args.device,
                 root=args.root, resume=resume, batch_size=args.batch_size, target_kl=args.target_kl)
    print(f"model saved to {path}", flush=True)
    best = out / "best" / "episode.npz"
    if best.exists():
        print("replaying the best episode into a video...", flush=True)
        info = record(best, args.core, args.rom)
        print(f"best episode: score {info['score']}, ground {info['ground']}, waves {info['waves']}; "
              f"video {best.with_suffix('.mp4')}", flush=True)
    print(f"compare with random play: python -m agent.evaluate --model {path}")


if __name__ == "__main__":
    main()
