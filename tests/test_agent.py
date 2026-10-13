import importlib.util
import random
from pathlib import Path

import numpy as np
import pytest

gym = pytest.importorskip("gymnasium", reason="needs requirements-rl.txt")

from agent.env import DPAD, DbgtReward, action_to_mask, mask_to_action, observe  # noqa: E402
from dataset.inputs import A, B, DOWN, L, LEFT, R, RIGHT, SELECT, START, UP, InputPolicy  # noqa: E402

needs_sb3 = pytest.mark.skipif(not importlib.util.find_spec("stable_baselines3"),
                               reason="needs requirements-rl.txt")
CORE = Path("core/mgba_libretro.dll")
SEED = Path("data/seeds/dbgt/level00.state")  # planet 0, made by dataset.explore's first run
IWRAM_SIZE = 0x8000


def test_action_to_mask_combines_dpad_and_buttons():
    assert action_to_mask([0, 0, 0, 0, 0]) == 0
    assert action_to_mask([DPAD.index(RIGHT), 1, 0, 0, 1]) == RIGHT | A | R
    assert action_to_mask([DPAD.index(DOWN | LEFT), 0, 1, 1, 0]) == DOWN | LEFT | B | L


def test_actions_never_press_start_or_select():
    for dpad in range(9):
        for buttons in range(16):
            mask = action_to_mask([dpad, *((buttons >> i) & 1 for i in range(4))])
            assert not mask & (START | SELECT)


def test_mask_to_action_round_trips_the_explorers_random_input():
    masks = InputPolicy(random.Random(1)).burst(20000, allow_start=False)
    for mask in masks:
        assert action_to_mask(mask_to_action(mask)) == mask & ~SELECT


def test_mask_to_action_rejects_opposite_directions():
    with pytest.raises(ValueError):
        mask_to_action(UP | DOWN)


def test_observe_makes_a_small_grey_picture():
    frame = np.zeros((160, 240, 3), dtype=np.uint8)
    frame[:, :120] = 255
    obs = observe(frame)
    assert obs.shape == (1, 80, 120)
    assert obs.dtype == np.uint8
    assert (obs[0, :, :60] == 255).all()
    assert (obs[0, :, 60:] == 0).all()


def _iwram(score=0, camera=138, go_arrow=0):
    iwram = np.zeros(IWRAM_SIZE, dtype=np.uint8)
    _set(iwram, score, camera, go_arrow)
    return iwram


def _set(iwram, score=None, camera=None, go_arrow=None):
    if score is not None:
        iwram[0x1D0C:0x1D10] = np.frombuffer(np.uint32(score).tobytes(), np.uint8)
    if camera is not None:
        iwram[0x24D8:0x24DA] = np.frombuffer(np.uint16(camera).tobytes(), np.uint8)
    if go_arrow is not None:
        iwram[0x1CEA] = go_arrow


def test_reward_counts_points_scored():
    iwram = _iwram(score=1000)
    reward = DbgtReward(iwram)
    reward.reset()
    _set(iwram, score=1500)
    assert reward.step() == pytest.approx(5.0)  # 100 points = 1
    assert reward.step() == 0.0


def test_reward_counts_only_new_ground_to_the_right():
    iwram = _iwram(camera=200)
    reward = DbgtReward(iwram)
    reward.reset()
    _set(iwram, camera=232)
    assert reward.step() == pytest.approx(2.0)  # 16 pixels = 1
    _set(iwram, camera=150)
    assert reward.step() == 0.0  # walking back costs nothing
    _set(iwram, camera=232)
    assert reward.step() == 0.0  # ...and earns nothing on the way forward again
    _set(iwram, camera=248)
    assert reward.step() == pytest.approx(1.0)
    assert reward.ground == 48


def test_reward_counts_waves_beaten():
    iwram = _iwram()
    reward = DbgtReward(iwram)
    reward.reset()
    assert reward.waves == 0
    for arrow in (0, 1, 1, 0, 0, 1):  # the "GO ->" arrow appears after each wave
        _set(iwram, go_arrow=arrow)
        reward.step()
    assert reward.waves == 2


def test_reset_starts_counting_from_the_current_state():
    iwram = _iwram(score=5000, camera=900, go_arrow=1)
    reward = DbgtReward(iwram)
    reward.reset()
    assert reward.step() == 0.0
    assert reward.waves == 0


needs_game = pytest.mark.skipif(not CORE.exists() or not SEED.exists() or not Path("roms").exists(),
                                reason="needs core, DBGT ROM and its planet 0 seed state")


def _dbgt_rom():
    from dataset.cheats import rom_game_code

    return next(p for p in Path("roms").rglob("*.gba") if rom_game_code(p) == "BT4E")


@pytest.fixture
def env():
    from agent.env import DbgtEnv
    from dataset.cheats import cheats_for

    env = DbgtEnv(CORE, _dbgt_rom(), SEED.read_bytes(), cheats=cheats_for("BT4E"), max_steps=50)
    yield env
    env.close()


@needs_game
def test_env_passes_gymnasium_checks(env):
    from gymnasium.utils.env_checker import check_env

    check_env(env, skip_render_check=True)


@needs_game
def test_episode_starts_at_the_seed_and_ends_at_the_time_limit(env):
    obs, info = env.reset(seed=0)
    assert obs.shape == (1, 80, 120)
    assert info["score"] == 0
    for step in range(1, 51):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        assert not terminated
        assert truncated == (step == 50)
    assert info["frames"] == 200  # 4 game frames per step


@needs_game
def test_same_actions_play_the_same_game(env):
    actions = [mask_to_action(m) for m in InputPolicy(random.Random(3)).burst(40, allow_start=False)]
    runs = []
    for _ in range(2):
        obs, _ = env.reset()
        for action in actions:
            obs, *_ = env.step(action)
        runs.append(obs)
    assert np.array_equal(*runs)


@needs_game
def test_random_play_scores_and_rewards_add_up(env):
    env.max_steps = 2250  # 2.5 game minutes
    _, start = env.reset()
    assert start["ground"] == 0
    masks = InputPolicy(random.Random(0)).burst(env.max_steps * 4, allow_start=False)
    total, truncated, step = 0.0, False, 0
    while not truncated:
        _, reward, _, truncated, info = env.step(mask_to_action(masks[step * 4]))
        total += reward
        step += 1
    assert info["score"] > 0
    assert info["ground"] == info["camera_max"] - start["camera_x"]
    assert total == pytest.approx(info["score"] / 100 + info["ground"] / 16)


# --- evaluation and training ----------------------------------------------------------


class Counter(gym.Env):
    """Observations count the steps: 1, 2, 3, ..."""

    observation_space = gym.spaces.Box(0, 255, (1, 2, 3), np.uint8)
    action_space = gym.spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return self._obs(), {}

    def step(self, action):
        self.t += 1
        return self._obs(), 0.0, False, self.t >= 10, {}

    def _obs(self):
        return np.full((1, 2, 3), self.t + 1, np.uint8)


@needs_sb3
def test_frame_stack_matches_the_one_used_in_training():
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    from agent.evaluate import FrameStack

    training = VecFrameStack(DummyVecEnv([Counter]), 4, channels_order="first")
    stack, env = FrameStack(4), Counter()
    assert np.array_equal(stack.reset(env.reset()[0]), training.reset()[0])
    for _ in range(6):
        assert np.array_equal(stack.add(env.step(0)[0]), training.step(np.array([0]))[0][0])


def test_explorer_input_plays_the_explorers_bursts_frame_by_frame():
    from agent.evaluate import ExplorerInput

    rng = random.Random(5)
    explorer_input, expected = InputPolicy(rng), []
    while len(expected) < 1500:  # as Explorer.iterate picks them
        expected += explorer_input.burst(rng.randint(60, 360), allow_start=False)
    policy = ExplorerInput(random.Random(5))
    assert policy.frames_per_step == 1
    policy.reset(None)
    got = [action_to_mask(policy.act(None)) for _ in range(1500)]
    assert got == [m & ~SELECT for m in expected[:1500]]


@needs_game
def test_play_measures_progress_and_samples_frames(env):
    from agent.evaluate import ExplorerInput, play

    env.frames_per_step = 1
    stats, frames = play(env, ExplorerInput(random.Random(0)), seconds=150, sheet_every=10)
    assert stats["frames"] == 150 * 60
    assert stats["score"] > 0
    assert stats["scenes"] >= 1
    assert stats["new_graphics"] > 0
    assert len(frames) == 15
    assert frames[0].shape == (160, 240, 3)


@needs_game
@needs_sb3
def test_training_runs_on_two_emulators_then_the_model_plays(tmp_path, env):
    from stable_baselines3 import PPO

    from agent.evaluate import PPOPolicy, play
    from agent.train import train
    from dataset.cheats import cheats_for

    model_path = train(CORE, _dbgt_rom(), SEED.read_bytes(), cheats_for("BT4E"), tmp_path,
                       timesteps=512, envs=2, max_steps=64, n_steps=64, seed=0)  # 4 updates: one log row
    assert model_path == tmp_path / "model.zip"
    episodes = (tmp_path / "monitor" / "0.monitor.csv").read_text().splitlines()
    assert "score" in episodes[1] and len(episodes) >= 3  # a comment line, the header, episodes
    assert "dbgt/score" in (tmp_path / "progress.csv").read_text().splitlines()[0]

    stats, _ = play(env, PPOPolicy(PPO.load(model_path)), seconds=10, sheet_every=5)
    assert stats["frames"] == 600
