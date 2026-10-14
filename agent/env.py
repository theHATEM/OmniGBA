"""
Gymnasium environment for reinforcement learning on Dragon Ball GT: Transformation.

Each episode starts from a saved state (a route seed: planet 0 a few seconds into its
first stage, by default) with the game's cheats on, so the fighter cannot die and an
episode only ends at its time limit.

    observation  the screen in grey at half size, (1, 80, 120) uint8; stack several
                 for motion (train.py stacks 4)
    action       d-pad (one of 9: none or 8 directions) plus A, B, L, R on or off,
                 held for 4 game frames. Start (pause menu) and Select are left out.
    reward       "points" (the default): points scored / 100 (about 0.5 per hit), plus new
                 ground: how far right the camera got beyond its best so far this episode,
                 / 16 pixels. The screen only scrolls once a wave of enemies is beaten, so the
                 score drives the fighting and the camera term drives moving on.
                 "progress": 1 per drop of the wave's enemies-left count (as enemies are
                 beaten), 10 per wave beaten, plus the same new ground, and nothing for points
                 except 0.2 per step that scores while the count is 0 and the wave goes on (its
                 last enemies are still on screen then, and the turret wave also needs its
                 turrets destroyed). A model trained on points learned to keep enemies alive in
                 long combos (a kill worth ~10,000 points could take 10-16 seconds) and ran out
                 of time at the turret wave; without the 0.2 a model never destroyed the turrets.
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
ENEMIES_LEFT = 0x19B0  # counts down as a wave's enemies are beaten, 0 while its last ones are still on
# screen; unrelated values (40, 120, 192, 224) between waves

# reward per point scored, per step that scores while the enemies-left count is 0 and the wave goes on
# (its last enemies, and the turret wave's turrets, which are not counted), per pixel of new ground,
# per drop of the enemies-left count and per wave beaten
REWARDS = {
    "points": {"points": 1 / 100, "cleared_hits": 0.0, "ground": 1 / 16, "kills": 0.0, "waves": 0.0},
    "progress": {"points": 0.0, "cleared_hits": 0.2, "ground": 1 / 16, "kills": 1.0, "waves": 10.0},
}
MAX_WAVE_SIZE = 30  # a larger enemies-left value is not a count
MAX_KILLS_PER_STEP = 4  # a larger drop is the counter being set, not kills


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
    """Reward from RAM: points scored, new ground to the right, enemies killed and waves beaten,
    each with its own weight (see REWARDS; the defaults are the "points" reward)."""

    def __init__(self, iwram: np.ndarray, points: float = 1 / 100, cleared_hits: float = 0.0,
                 ground: float = 1 / 16, kills: float = 0.0, waves: float = 0.0):
        self.iwram = iwram  # live view, read on every step
        self.weights = {"points": points, "cleared_hits": cleared_hits, "ground": ground,
                        "kills": kills, "waves": waves}
        self.kills = self.waves = self.cleared_hits = 0

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
        self._left = int(self.iwram[ENEMIES_LEFT])
        self.kills = self.waves = self.cleared_hits = 0

    @property
    def ground(self) -> int:
        """Pixels the camera got further right than where the episode started."""
        return self.camera_max - self.camera_start

    def step(self) -> float:
        score, camera, arrow = self.score, self.camera_x, bool(self.iwram[GO_ARROW])
        left = int(self.iwram[ENEMIES_LEFT])
        points = max(score - self._score, 0)
        ground = max(camera - self.camera_max, 0)
        waves = int(arrow and not self._arrow)
        drop = self._left - left
        kills = drop if self._left <= MAX_WAVE_SIZE and 0 < drop <= MAX_KILLS_PER_STEP else 0
        # a hit while the count has been 0 since the step before (so not the step of its last drop)
        # and the wave is not over: on its last enemies, or the turret wave's turrets
        cleared = int(points > 0 and self._left == 0 and left == 0 and not arrow)
        self.waves += waves
        self.kills += kills
        self.cleared_hits += cleared
        self._score, self._arrow, self._left = score, arrow, left
        self.camera_max = max(self.camera_max, camera)
        w = self.weights
        return (points * w["points"] + cleared * w["cleared_hits"] + ground * w["ground"]
                + kills * w["kills"] + waves * w["waves"])


class DbgtEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, core, rom, state: bytes, cheats=(), frames_per_step: int = 4,
                 max_steps: int = 1800, render_mode=None, on_frame=None, reward: str = "points",
                 practice: bytes | None = None, practice_share: float = 0.0):
        self.core, self.rom = str(core), str(rom)
        self.state = state
        # a second start state (e.g. a later wave) that a share of the episodes start from instead
        self.practice, self.practice_share = practice, practice_share if practice else 0.0
        self.start = 0  # the current episode's start: 0 = state, 1 = practice
        self.cheats = list(cheats)
        self.reward_weights = REWARDS[reward]
        self.frames_per_step = frames_per_step
        self.max_steps = max_steps  # 1800 steps of 4 frames = 2 game minutes
        self.render_mode = render_mode
        self.on_frame = on_frame  # called with every game frame of a step (e.g. to write a video)
        self.observation_space = gym.spaces.Box(0, 255, OBS_SHAPE, np.uint8)
        self.action_space = gym.spaces.MultiDiscrete([len(DPAD), *[2] * len(BUTTONS)])
        self.emu = None  # started on the first reset, so each worker process makes its own

    def _start(self):
        from dataset.emulator import HeadlessGBA

        self.emu = HeadlessGBA(self.core, self.rom).__enter__()
        self.emu.apply_cheats(self.cheats)  # mGBA keeps them across state loads
        self.reward = DbgtReward(self.emu.memory(IWRAM), **self.reward_weights)

    def _info(self) -> dict:
        return {"score": self.reward.score, "camera_x": self.reward.camera_x,
                "camera_max": self.reward.camera_max, "ground": self.reward.ground,
                "waves": self.reward.waves, "kills": self.reward.kills, "start": self.start,
                "enemies_left": int(self.reward.iwram[ENEMIES_LEFT]),
                "frames": self.steps * self.frames_per_step}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.emu is None:
            self._start()
        self.start = int(self.np_random.random() < self.practice_share)
        self.emu.load_state(self.practice if self.start else self.state)
        self.reward.reset()
        self.steps = 0
        return observe(self.emu.frame()), self._info()

    def step(self, action):
        mask = action_to_mask(action)
        if self.on_frame is None:
            self.emu.step(mask, self.frames_per_step)
        else:
            for _ in range(self.frames_per_step):
                self.emu.step(mask)
                self.on_frame(self.emu.frame())
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
