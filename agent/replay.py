r"""
Replay a recorded episode (the best episode of a training run, see train.py) and make a video
of it. A recording is the start state plus the action of every step: the emulator is
deterministic, so that plays the same game again.

    python -m agent.replay data\rl\<run>\best\episode.npz

Checks that the replay scores the same points, new ground and waves as the recording (and
stops with an error if not), then writes next to it:

    episode.mp4   every frame at 60 fps, scaled 3x
    episode.png   a contact sheet, one frame every 10 game seconds
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

from agent.env import DbgtEnv
from agent.evaluate import contact_sheet

CHECKED = ("score", "ground", "waves")


def save_episode(folder, actions: np.ndarray, state: bytes, frames_per_step: int, stats: dict) -> None:
    """Writes episode.npz (stands alone: actions, start state and stats) and episode.json (the stats).

    Each file goes to a temporary name first, so stopping the program never leaves half a file.
    `stats` needs score, ground and waves (checked on replay) and the cheats; rom and anything
    else are kept for reference."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stats = {**stats, "steps": len(actions)}
    temp = folder / "episode.npz.tmp"
    with open(temp, "wb") as f:
        np.savez_compressed(f, actions=np.asarray(actions, np.uint8), state=np.frombuffer(state, np.uint8),
                            frames_per_step=frames_per_step, stats=json.dumps(stats))
    os.replace(temp, folder / "episode.npz")
    temp = folder / "episode.json.tmp"
    temp.write_text(json.dumps({**stats, "frames_per_step": frames_per_step}, indent=2))
    os.replace(temp, folder / "episode.json")


def load_episode(path) -> dict:
    """The stats of a recording plus its actions, state (bytes) and frames_per_step."""
    with np.load(path) as data:
        return {**json.loads(str(data["stats"])), "actions": data["actions"],
                "state": data["state"].tobytes(), "frames_per_step": int(data["frames_per_step"])}


def replay(core, rom, episode: dict, on_frame=None) -> dict:
    """Plays a recording in a new emulator; returns the last step's info.

    `on_frame` is called with every game frame. Raises RuntimeError if the points, new ground
    or waves differ from the recording."""
    actions = episode["actions"]
    env = DbgtEnv(core, rom, episode["state"], cheats=episode["cheats"],
                  frames_per_step=episode["frames_per_step"], max_steps=len(actions), on_frame=on_frame)
    try:
        env.reset()
        for action in actions:
            *_, info = env.step(action)
    finally:
        env.close()
    got, recorded = {k: info[k] for k in CHECKED}, {k: episode[k] for k in CHECKED}
    if got != recorded:
        raise RuntimeError(f"the recording is not reproducible: the replay got {got}, the recording {recorded}")
    return info


def record(path, core, rom=None, scale: int = 3, sheet_every: float = 10.0) -> dict:
    """Replays the recording at `path` into a video and a contact sheet next to it; returns the last info.

    `rom` defaults to the ROM path saved in the recording."""
    path = Path(path)
    episode = load_episode(path)
    size = (240 * scale, 160 * scale)
    video = cv2.VideoWriter(str(path.with_suffix(".mp4")), cv2.VideoWriter_fourcc(*"mp4v"), 60, size)
    sheet, sheet_frames, count = [], round(sheet_every * 60), 0

    def on_frame(frame):
        nonlocal count
        count += 1
        video.write(cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_NEAREST), cv2.COLOR_RGB2BGR))
        if count % sheet_frames == 0:
            sheet.append(frame)

    try:
        info = replay(core, rom or episode["rom"], episode, on_frame)
    finally:
        video.release()
    if sheet:
        cv2.imwrite(str(path.with_suffix(".png")), cv2.cvtColor(contact_sheet(sheet), cv2.COLOR_RGB2BGR))
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay a recorded Dragon Ball GT episode into a video")
    ap.add_argument("episode", help="episode.npz, e.g. data\\rl\\<run>\\best\\episode.npz")
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--rom", help="the game (default: the ROM path saved in the recording)")
    args = ap.parse_args()

    info = record(args.episode, args.core, args.rom)
    path = Path(args.episode)
    print(f"replayed {info['frames']} frames: score {info['score']}, ground {info['ground']}, "
          f"waves {info['waves']}\nwritten {path.with_suffix('.mp4')} and {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
