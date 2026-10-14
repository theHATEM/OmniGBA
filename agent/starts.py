r"""
Make a practice start state from a recorded episode (see replay.py): the state the recording
reaches when a wave starts at a given camera position, e.g. planet 0's gun-turret wave:

    python -m agent.starts data\rl\<run>\best\episode.npz --camera 1320 --out data\starts\dbgt\turrets.state

Training can then start a share of its episodes there (agent.train --practice), so the model
practises a part of the stage it otherwise only reaches near the end of an episode.
"""

import argparse
from pathlib import Path

from agent.env import MAX_WAVE_SIZE, DbgtEnv
from agent.replay import load_episode


def state_at(core, rom, episode: dict, until) -> tuple[bytes, dict]:
    """Replays a recording until `until(info)` holds after a step; returns (save state, that info).

    Raises ValueError if it never holds."""
    env = DbgtEnv(core, rom, episode["state"], cheats=episode["cheats"],
                  frames_per_step=episode["frames_per_step"], max_steps=len(episode["actions"]))
    try:
        env.reset()
        for action in episode["actions"]:
            *_, info = env.step(action)
            if until(info):
                return env.emu.save_state(), info
    finally:
        env.close()
    raise ValueError("the recording never reaches that point")


def wave_starts(camera: int):
    """True once the camera is at `camera` or further and a wave's enemy count has been set."""
    return lambda info: info["camera_x"] >= camera and 0 < info["enemies_left"] <= MAX_WAVE_SIZE


def main() -> None:
    ap = argparse.ArgumentParser(description="Save the state a recorded episode reaches when a wave starts")
    ap.add_argument("episode", help="episode.npz, e.g. data\\rl\\<run>\\best\\episode.npz")
    ap.add_argument("--camera", type=int, required=True,
                    help="camera x of the wave's arena (planet 0's turret wave: 1320)")
    ap.add_argument("--out", required=True, help="the state file to write")
    ap.add_argument("--core", default=r"core\mgba_libretro.dll")
    ap.add_argument("--rom", help="the game (default: the ROM path saved in the recording)")
    args = ap.parse_args()

    episode = load_episode(args.episode)
    state, info = state_at(args.core, args.rom or episode["rom"], episode, wave_starts(args.camera))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(state)
    print(f"saved {args.out}: after {info['frames']} frames, camera {info['camera_x']}, "
          f"{info['enemies_left']} enemies left, score {info['score']}")


if __name__ == "__main__":
    main()
