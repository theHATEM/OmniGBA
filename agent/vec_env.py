"""
Several emulators per worker process, for training.

Stable-Baselines3 steps all emulators in lockstep, and the network runs in the main process
between steps while the emulators wait; that costs about the same for 12 emulators as for 24.
So more emulators per step make each step cheaper per emulator, but more processes than the
CPU has threads only add overhead (and each process takes ~0.5 GB for PyTorch). This runs
several emulators in each process, one after another.

An mGBA core keeps its state in the DLL, so two cores loaded from the same file in one process
share it (and crash when closed). Each emulator in a process therefore loads its own copy of
the DLL (core_copies).
"""

import multiprocessing as mp
import shutil
from pathlib import Path

import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import CloudpickleWrapper, VecEnv
from stable_baselines3.common.vec_env.patch_gym import _patch_env


def core_copies(core, n: int, folder=None) -> list[Path]:
    """`core` and n - 1 copies of it under other names (in `folder`, default <core dir>\\copies).

    A copy is made again when the core has changed (size or modification time)."""
    core = Path(core)
    folder = Path(folder) if folder else core.parent / "copies"
    paths = [core]
    for i in range(1, n):
        copy = folder / f"{core.stem}_{i}{core.suffix}"
        source, target = core.stat(), copy.stat() if copy.exists() else None
        if target is None or (target.st_size, target.st_mtime_ns) != (source.st_size, source.st_mtime_ns):
            folder.mkdir(parents=True, exist_ok=True)
            shutil.copy2(core, copy)  # keeps the modification time
        paths.append(copy)
    return paths


def _worker(remote, parent_remote, env_fns_wrapper) -> None:
    """Runs a group of environments; like Stable-Baselines3's SubprocVecEnv worker, per environment."""
    from stable_baselines3.common.env_util import is_wrapped

    parent_remote.close()
    envs = [_patch_env(fn()) for fn in env_fns_wrapper.var]
    reset_infos = [{} for _ in envs]
    while True:
        try:
            cmd, data = remote.recv()
            if cmd == "step":
                results = []
                for i, (env, action) in enumerate(zip(envs, data)):
                    observation, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                    info["TimeLimit.truncated"] = truncated and not terminated
                    if done:  # save the final observation, then reset
                        info["terminal_observation"] = observation
                        observation, reset_infos[i] = env.reset()
                    results.append((observation, reward, done, info, reset_infos[i]))
                remote.send(results)
            elif cmd == "reset":
                results = []
                for i, (env, (seed, options)) in enumerate(zip(envs, data)):
                    observation, reset_infos[i] = env.reset(seed=seed, **({"options": options} if options else {}))
                    results.append((observation, reset_infos[i]))
                remote.send(results)
            elif cmd == "close":
                for env in envs:
                    env.close()
                remote.close()
                break
            elif cmd == "get_spaces":
                remote.send((envs[0].observation_space, envs[0].action_space))
            else:  # commands for one environment of the group: data = (index in the group, *arguments)
                env, args = envs[data[0]], data[1:]
                if cmd == "render":
                    remote.send(env.render())
                elif cmd == "env_method":
                    remote.send(env.get_wrapper_attr(args[0])(*args[1], **args[2]))
                elif cmd == "get_attr":
                    remote.send(env.get_wrapper_attr(args[0]))
                elif cmd == "has_attr":
                    try:
                        env.get_wrapper_attr(args[0])
                        remote.send(True)
                    except AttributeError:
                        remote.send(False)
                elif cmd == "set_attr":
                    remote.send(setattr(env, args[0], args[1]))
                elif cmd == "is_wrapped":
                    remote.send(is_wrapped(env, args[0]))
                else:
                    raise NotImplementedError(f"`{cmd}` is not implemented in the worker")
        except (EOFError, KeyboardInterrupt):
            break


class SubprocGroupVecEnv(VecEnv):
    """Like SubprocVecEnv, but each process runs `per_process` environments (the last one maybe fewer).

    Environments i * per_process ... (i + 1) * per_process - 1 share process i."""

    def __init__(self, env_fns, per_process: int, start_method: str = "spawn"):
        self.waiting = False
        self.closed = False
        groups = [env_fns[i : i + per_process] for i in range(0, len(env_fns), per_process)]
        self._sizes = [len(group) for group in groups]
        self._where = [(p, j) for p, size in enumerate(self._sizes) for j in range(size)]  # env -> (process, index)
        ctx = mp.get_context(start_method)
        self.remotes, work_remotes = zip(*[ctx.Pipe() for _ in groups])
        self.processes = []
        for work_remote, remote, group in zip(work_remotes, self.remotes, groups):
            # daemon=True: if the main process crashes, the workers must not keep it waiting
            process = ctx.Process(target=_worker, args=(work_remote, remote, CloudpickleWrapper(group)), daemon=True)
            process.start()
            self.processes.append(process)
            work_remote.close()
        self.remotes[0].send(("get_spaces", None))
        observation_space, action_space = self.remotes[0].recv()
        super().__init__(len(env_fns), observation_space, action_space)

    def _per_process(self, values):
        """Splits one value per environment into one list per process."""
        starts = np.cumsum([0, *self._sizes])
        return [values[a:b] for a, b in zip(starts[:-1], starts[1:])]

    def step_async(self, actions: np.ndarray) -> None:
        for remote, group_actions in zip(self.remotes, self._per_process(actions)):
            remote.send(("step", group_actions))
        self.waiting = True

    def step_wait(self):
        results = [result for remote in self.remotes for result in remote.recv()]
        self.waiting = False
        obs, rewards, dones, infos, self.reset_infos = zip(*results)
        return np.stack(obs), np.stack(rewards), np.stack(dones), infos

    def reset(self):
        seeds_options = list(zip(self._seeds, self._options))
        for remote, group in zip(self.remotes, self._per_process(seeds_options)):
            remote.send(("reset", group))
        results = [result for remote in self.remotes for result in remote.recv()]
        obs, self.reset_infos = zip(*results)
        # seeds and options are only used once
        self._reset_seeds()
        self._reset_options()
        return np.stack(obs)

    def close(self) -> None:
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(("close", None))
        for process in self.processes:
            process.join()
        self.closed = True

    def _call(self, cmd, indices, *args) -> list:
        """Sends a command to each environment in `indices` and returns the answers in that order."""
        answers = []
        for i in self._get_indices(indices):
            process, index = self._where[i]
            self.remotes[process].send((cmd, (index, *args)))
            answers.append(self.remotes[process].recv())
        return answers

    def get_images(self):
        return self._call("render", None) if self.render_mode == "rgb_array" else [None] * self.num_envs

    def has_attr(self, attr_name: str) -> bool:
        return all(self._call("has_attr", None, attr_name))

    def get_attr(self, attr_name: str, indices=None) -> list:
        return self._call("get_attr", indices, attr_name)

    def set_attr(self, attr_name: str, value, indices=None) -> None:
        self._call("set_attr", indices, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs) -> list:
        return self._call("env_method", indices, method_name, method_args, method_kwargs)

    def env_is_wrapped(self, wrapper_class, indices=None) -> list[bool]:
        return self._call("is_wrapped", indices, wrapper_class)
