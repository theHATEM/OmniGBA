"""
Gymnasium environment for reinforcement learning on Dragon Ball GT: Transformation.

Each episode starts from a saved state (a route seed: planet 0 a few seconds into its
first stage, by default) with the game's cheats on, so the fighter cannot die and an
episode only ends at its time limit.

    observation  the screen in grey at half size, (1, 80, 120) uint8; stack several
                 for motion (train.py stacks 4)
    action       d-pad (one of 9: none or 8 directions) plus A, B, L, R on or off,
                 held for 4 game frames. Start (pause menu) and Select are left out.
    reward       points scored / 100 (about 0.5 per hit), plus new ground: how far right
                 the camera got beyond its best so far this episode, / 16 pixels.
                 The screen only scrolls once a wave of enemies is beaten, so the
                 score drives the fighting and the camera term drives moving on.
"""

from pathlib import Path

import cv2
import gymnasium as gym
import numpy as np

from dataset.inputs import A, B, DIRECTIONS, DOWN, L, LEFT, R, RIGHT, UP

IWRAM = 0x03000000
DPAD = [0, *DIRECTIONS]
BUTTONS = [A, B, L, R]
OBS_SHAPE = (1, 80, 120)

# Dragon Ball GT addresses in IWRAM (see HANDOFF.md, game facts)
SCORE = 0x1D0C  # 32-bit, 0 at the start of each planet's route seed
CAMERA_X = 0x24D8  # 16-bit, grows through a stage, stops at each wave's arena
GO_ARROW = 0x1CEA  # 1 while the "GO ->" arrow shows (wave beaten, walk right)


def start_state(core, rom, seeds_root, root: str = "level00") -> bytes:
    """A Dragon Ball GT route seed (see dataset/routes.py), made on first use."""
    from dataset.cheats import GAMES, rom_game_code
    from dataset.routes import route_seeds

    code = rom_game_code(rom)
    if code != "BT4E":
        raise ValueError(f"{rom} is not Dragon Ball GT: Transformation (USA), game code {code}")
    states = dict(route_seeds(core, rom, code, Path(seeds_root) / GAMES[code].name))
    if root not in states:
        raise ValueError(f"no route seed {root!r}; there are {', '.join(states)}")
    return states[root]


def action_to_mask(action) -> int:
    dpad, *pressed = (int(a) for a in action)
    mask = DPAD[dpad]
    for button, on in zip(BUTTONS, pressed):
        if on:
            mask |= button
    return mask


def mask_to_action(mask: int) -> np.ndarray:
    """The action pressing the same buttons as `mask` (Start and Select are dropped)."""
    direction = mask & (UP | DOWN | LEFT | RIGHT)
    if direction not in DPAD:
        raise ValueError(f"a d-pad cannot press {direction:#x}")
    return np.array([DPAD.index(direction), *(int(bool(mask & b)) for b in BUTTONS)], dtype=np.int64)


def observe(frame: np.ndarray) -> np.ndarray:
    grey = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, height, width = OBS_SHAPE
    return cv2.resize(grey, (width, height), interpolation=cv2.INTER_AREA)[None]


class DbgtReward:
    """Reward from RAM: points scored plus new ground to the right; also counts waves beaten."""

    def __init__(self, iwram: np.ndarray, score_scale: float = 100.0, camera_scale: float = 16.0):
        self.iwram = iwram  # live view, read on every step
        self.score_scale = score_scale
        self.camera_scale = camera_scale

    @property
    def score(self) -> int:
        return int(self.iwram[SCORE : SCORE + 4].view("<u4")[0])

    @property
    def camera_x(self) -> int:
        return int(self.iwram[CAMERA_X : CAMERA_X + 2].view("<u2")[0])

    def reset(self) -> None:
        self._score = self.score
        self.camera_start = self.camera_max = self.camera_x
        self._arrow = bool(self.iwram[GO_ARROW])
        self.waves = 0

    @property
    def ground(self) -> int:
        """Pixels the camera got further right than where the episode started."""
        return self.camera_max - self.camera_start

    def step(self) -> float:
        score, camera, arrow = self.score, self.camera_x, bool(self.iwram[GO_ARROW])
        points = max(score - self._score, 0)
        ground = max(camera - self.camera_max, 0)
        self.waves += arrow and not self._arrow
        self._score, self._arrow = score, arrow
        self.camera_max = max(self.camera_max, camera)
        return points / self.score_scale + ground / self.camera_scale


class DbgtEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, core, rom, state: bytes, cheats=(), frames_per_step: int = 4,
                 max_steps: int = 1800, render_mode=None):
        self.core, self.rom = str(core), str(rom)
        self.state = state
        self.cheats = list(cheats)
        self.frames_per_step = frames_per_step
        self.max_steps = max_steps  # 1800 steps of 4 frames = 2 game minutes
        self.render_mode = render_mode
        self.observation_space = gym.spaces.Box(0, 255, OBS_SHAPE, np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([len(DPAD), *[2] * len(BUTTONS)])
        self.emu = None  # started on the first reset, so each worker process makes its own

    def _start(self):
        from dataset.emulator import HeadlessGBA

        self.emu = HeadlessGBA(self.core, self.rom).__enter__()
        self.emu.apply_cheats(self.cheats)  # mGBA keeps them across state loads
        self.reward = DbgtReward(self.emu.memory(IWRAM))

    def _info(self) -> dict:
        return {"score": self.reward.score, "camera_x": self.reward.camera_x,
                "camera_max": self.reward.camera_max, "ground": self.reward.ground,
                "waves": self.reward.waves,
                "frames": self.steps * self.frames_per_step}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.emu is None:
            self._start()
        self.emu.load_state(self.state)
        self.reward.reset()
        self.steps = 0
        return observe(self.emu.frame()), self._info()

    def step(self, action):
        self.emu.step(action_to_mask(action), self.frames_per_step)
        reward = self.reward.step()
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return observe(self.emu.frame()), reward, False, truncated, self._info()

    def render(self):
        return self.emu.frame()

    def close(self):
        if self.emu is not None:
            self.emu.__exit__(None, None, None)
            self.emu = None
