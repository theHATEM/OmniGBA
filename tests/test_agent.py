import csv
import importlib.util
import json
import random
from collections import deque
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


def _set(iwram, score=None, camera=None, go_arrow=None, enemies_left=None):
    if score is not None:
        iwram[0x1D0C:0x1D10] = np.frombuffer(np.uint32(score).tobytes(), np.uint8)
    if camera is not None:
        iwram[0x24D8:0x24DA] = np.frombuffer(np.uint16(camera).tobytes(), np.uint8)
    if go_arrow is not None:
        iwram[0x1CEA] = go_arrow
    if enemies_left is not None:
        iwram[0x19B0] = enemies_left


def _kills(reward, iwram, values):
    """Steps through enemies-left values (or (enemies left, GO arrow) pairs); returns the kills counted."""
    for value in values:
        left, arrow = value if isinstance(value, tuple) else (value, 0)
        _set(iwram, enemies_left=left, go_arrow=arrow)
        reward.step()
    return reward.kills


def test_reward_counts_kills_as_the_enemies_left_go_down():
    iwram = _iwram()
    _set(iwram, enemies_left=8)
    reward = DbgtReward(iwram)
    reward.reset()
    assert _kills(reward, iwram, [8, 7, 7, 5, 4]) == 4  # two in one step
    assert DbgtReward(iwram).kills == 0


def test_kills_ignore_the_counter_between_waves():
    # values seen in a real game (planet 0): while the "GO ->" arrow shows and while walking to the
    # next arena the counter holds unrelated values, then the next wave's enemies are set
    iwram = _iwram()
    _set(iwram, enemies_left=1)
    reward = DbgtReward(iwram)
    reward.reset()
    between = [0, 120, (120, 1), (120, 1), (120, 0), 0, 4,  # wave 1 -> wave 2
               3, 2, 1, 0, 128, (224, 1), (224, 0), 2,  # wave 2 -> wave 3
               1, 0, 224, (192, 1), (192, 0), 9,  # wave 3 -> the turret wave
               8, 0, 40, (40, 1), (40, 0), 0]  # (8 -> 0 is a reset, not 8 kills)
    assert _kills(reward, iwram, between) == 1 + 4 + 2 + 1


def test_progress_reward_pays_for_kills_waves_and_new_ground_not_points():
    from agent.env import REWARDS

    iwram = _iwram(score=1000, camera=200)
    _set(iwram, enemies_left=3)
    reward = DbgtReward(iwram, **REWARDS["progress"])
    reward.reset()
    _set(iwram, score=30_000)  # a long combo
    assert reward.step() == 0.0
    _set(iwram, enemies_left=2)
    assert reward.step() == pytest.approx(1.0)  # a kill
    _set(iwram, enemies_left=0)
    assert reward.step() == pytest.approx(2.0)
    _set(iwram, enemies_left=120, go_arrow=1)
    assert reward.step() == pytest.approx(10.0)  # a wave beaten
    _set(iwram, camera=232)
    assert reward.step() == pytest.approx(2.0)  # 16 pixels = 1, as with the points reward


def test_progress_reward_pays_hits_only_once_the_enemies_left_count_is_zero():
    # the counter reaches 0 while the wave's last enemies are still on screen, and the turret wave
    # also needs its turrets destroyed: a small reward per step that scores, whatever the points
    # (combos would pay ever more for the same enemy), while the count is 0 and the wave goes on
    from agent.env import REWARDS

    iwram = _iwram(score=1000)
    _set(iwram, enemies_left=1)
    reward = DbgtReward(iwram, **REWARDS["progress"])
    reward.reset()
    _set(iwram, score=1600)
    assert reward.step() == 0.0  # hitting an enemy
    _set(iwram, score=2100, enemies_left=0)
    assert reward.step() == pytest.approx(1.0)  # the count drops: 1, not the points
    assert reward.step() == 0.0  # nothing scored
    _set(iwram, score=2250)
    assert reward.step() == pytest.approx(0.2)  # a hit on what is left
    _set(iwram, score=2900)
    assert reward.step() == pytest.approx(0.2)  # a combo hit pays the same
    _set(iwram, score=3400, go_arrow=1, enemies_left=120)
    assert reward.step() == pytest.approx(10.0)  # the wave beaten; no hits once the arrow shows
    _set(iwram, score=3900, go_arrow=0)
    assert reward.step() == 0.0  # walking on, the counter holding 120
    _set(iwram, score=4400, enemies_left=0)
    assert reward.step() == 0.0  # the counter just reset to 0 in the next arena
    assert (reward.kills, reward.cleared_hits) == (1, 2)


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


@needs_game
def test_random_play_kills_and_the_progress_reward_adds_up():
    from agent.env import DbgtEnv
    from dataset.cheats import cheats_for

    env = DbgtEnv(CORE, _dbgt_rom(), SEED.read_bytes(), cheats=cheats_for("BT4E"), max_steps=2250,
                  reward="progress")
    try:
        _, start = env.reset()
        assert start["kills"] == 0
        masks = InputPolicy(random.Random(0)).burst(env.max_steps * 4, allow_start=False)
        total, truncated, step = 0.0, False, 0
        while not truncated:
            _, reward, _, truncated, info = env.step(mask_to_action(masks[step * 4]))
            total += reward
            step += 1
        enemies_left = int(env.emu.memory(0x03000000)[0x19B0])
    finally:
        env.close()
    # 2.5 minutes of this random play kill some of planet 0's first wave (8 enemies), not all of it
    assert info["waves"] == 0
    assert 0 < info["kills"] < 8
    assert info["kills"] + enemies_left == 8
    assert total == pytest.approx(info["kills"] + 10 * info["waves"] + info["ground"] / 16)


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
    assert stats["kills"] > 0
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

    from agent.replay import load_episode, replay

    best = json.loads((tmp_path / "best" / "episode.json").read_text())
    rewards = [float(row["r"]) for i in range(2)
               for row in csv.DictReader((tmp_path / "monitor" / f"{i}.monitor.csv").read_text().splitlines()[1:])]
    assert best["reward"] == max(rewards)
    env.close()  # one mGBA core per process
    replay(CORE, _dbgt_rom(), load_episode(tmp_path / "best" / "episode.npz"))  # raises if it plays differently


# --- practice starts ------------------------------------------------------------------------


@needs_game
def test_state_at_saves_the_state_a_recording_reaches_when_the_condition_first_holds(tmp_path, env):
    from agent.env import DbgtEnv
    from agent.replay import load_episode
    from agent.starts import state_at
    from dataset.cheats import cheats_for

    _record_random_play(env, tmp_path, steps=300, seed=5)
    episode = load_episode(tmp_path / "episode.npz")
    state, info = state_at(CORE, _dbgt_rom(), episode, lambda info: info["frames"] >= 400)
    assert info["frames"] == 400

    # the game RAM after 100 steps of the recording, and after loading the saved state in a new
    # emulator (the screen is only redrawn by the next frame after a state load, so compare RAM)
    rams = []
    for start, actions in ((episode["state"], episode["actions"][:100]), (state, [])):
        check = DbgtEnv(CORE, _dbgt_rom(), start, cheats=episode["cheats"])
        check.reset()
        for action in actions:
            check.step(action)
        rams.append(check.emu.memory(0x03000000).copy())
        check.close()
    assert np.array_equal(*rams)
    with pytest.raises(ValueError):
        state_at(CORE, _dbgt_rom(), episode, lambda info: False)


@needs_game
def test_episodes_start_from_the_practice_state_with_its_share(tmp_path, env):
    from agent.env import DbgtEnv
    from dataset.cheats import cheats_for

    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
    later, later_ram = env.emu.save_state(), env.emu.memory(0x03000000).copy()
    env.close()

    practice = DbgtEnv(CORE, _dbgt_rom(), SEED.read_bytes(), cheats=cheats_for("BT4E"), practice=later,
                       practice_share=0.25)
    starts = []
    try:
        for i in range(400):
            obs, info = practice.reset(seed=1 if i == 0 else None)
            starts.append(info["start"])
            ram = practice.emu.memory(0x03000000)
            assert np.array_equal(ram, later_ram) == (info["start"] == 1)
            assert info["ground"] == 0  # new ground counts from where the episode starts
        *_, info = practice.step(practice.action_space.sample())
        assert info["start"] == starts[-1]  # every step says which start its episode had
    finally:
        practice.close()
    assert set(starts) == {0, 1}
    assert 0.18 < starts.count(1) / len(starts) < 0.32


@needs_sb3
def test_monitor_files_of_a_resumed_run_get_the_columns_added_since(tmp_path):
    from agent.train import add_monitor_columns

    old = tmp_path / "0.monitor.csv"
    old.write_text('#{"t_start": 1.0, "env_id": "None"}\nr,l,t,score,ground,waves,kills\n9.5,4500,1.2,100,16,1,8\n')
    new = tmp_path / "1.monitor.csv"
    new.write_text('#{"t_start": 1.0, "env_id": "None"}\nr,l,t,score,ground,waves,kills,start\n9.5,4500,1.2,100,16,1,8,1\n')
    add_monitor_columns(tmp_path, {"start": 0})
    for path, start in ((old, "0"), (new, "1")):
        lines = path.read_text().splitlines()
        assert lines[0].startswith("#")
        [row] = csv.DictReader(lines[1:])
        assert (row["kills"], row["start"]) == ("8", start)


@needs_sb3
def test_best_episode_only_keeps_episodes_from_the_main_start(tmp_path):
    from agent.replay import load_episode
    from agent.train import BestEpisode

    best = BestEpisode(tmp_path, b"main start")
    practice = _episode(50.0)
    practice["start"] = 1
    main = _episode(9.0)
    main["start"] = 0
    best.add(np.array([[1, 0, 0, 0, 0], [2, 0, 0, 0, 0]]), np.array([True, True]), [practice, main])
    assert load_episode(tmp_path / "episode.npz")["actions"].tolist() == [[2, 0, 0, 0, 0]]
    assert json.loads((tmp_path / "episode.json").read_text())["reward"] == 9.0


# --- several emulators per process --------------------------------------------------------


def test_core_copies_give_each_emulator_its_own_up_to_date_file(tmp_path):
    from agent.vec_env import core_copies

    core = tmp_path / "core.dll"
    core.write_bytes(b"core v1")
    cores = core_copies(core, 3, tmp_path / "copies")
    assert cores[0] == core
    assert len(set(cores)) == 3
    assert all(Path(c).read_bytes() == b"core v1" for c in cores)
    core.write_bytes(b"core v2, a new build")
    assert all(Path(c).read_bytes() == b"core v2, a new build" for c in core_copies(core, 3, tmp_path / "copies"))


@needs_game
def test_two_emulators_in_one_process_play_independently_with_their_own_core_copy(tmp_path):
    from agent.env import DbgtEnv
    from agent.vec_env import core_copies
    from dataset.cheats import cheats_for

    cores = core_copies(CORE, 2, tmp_path)
    space = gym.spaces.MultiDiscrete([9, 2, 2, 2, 2], seed=4)
    plays = [[space.sample() for _ in range(200)] for _ in cores]

    def new_env(core):
        env = DbgtEnv(core, _dbgt_rom(), SEED.read_bytes(), cheats=cheats_for("BT4E"))
        env.reset()
        return env

    envs = [new_env(core) for core in cores]  # both at once, steps interleaved
    for actions in zip(*plays):
        infos = [env.step(a)[4] for env, a in zip(envs, actions)]
    together = [(info["score"], info["ground"], env.render()) for env, info in zip(envs, infos)]
    for env in envs:
        env.close()

    for core, actions, (score, ground, frame) in zip(cores, plays, together):
        env = new_env(core)  # alone
        for action in actions:
            *_, info = env.step(action)
        assert (info["score"], info["ground"]) == (score, ground)
        assert np.array_equal(env.render(), frame)
        env.close()
    assert together[0][0] != together[1][0] or not np.array_equal(together[0][2], together[1][2])


@needs_sb3
def test_group_vec_env_matches_one_process_per_env():
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agent.vec_env import SubprocGroupVecEnv

    class Toy(gym.Env):
        """Env i: observations 10 i + t, reward action x (i + 1), episodes of 3 + i steps."""

        observation_space = gym.spaces.Box(0, 255, (1, 2, 3), np.uint8)
        action_space = gym.spaces.Discrete(3)

        def __init__(self, i):
            self.i = i

        def reset(self, *, seed=None, options=None):
            self.t = 0
            return self._obs(), {"start": self.i}

        def step(self, action):
            self.t += 1
            return self._obs(), float(action * (self.i + 1)), False, self.t >= 3 + self.i, {"t": self.t}

        def _obs(self):
            return np.full((1, 2, 3), 10 * self.i + self.t, np.uint8)

    factories = [lambda i=i: gym.Wrapper(Toy(i)) for i in range(5)]
    grouped, single = SubprocGroupVecEnv(factories, per_process=2), DummyVecEnv(factories)  # processes of 2, 2, 1
    try:
        assert np.array_equal(grouped.reset(), single.reset())
        assert list(grouped.reset_infos) == list(single.reset_infos)
        rng = np.random.default_rng(0)
        for _ in range(9):
            actions = rng.integers(0, 3, 5)
            (obs_g, rew_g, done_g, info_g), (obs_s, rew_s, done_s, info_s) = (
                grouped.step(actions), single.step(actions))
            assert np.array_equal(obs_g, obs_s)
            assert rew_g.tolist() == rew_s.tolist()
            assert done_g.tolist() == done_s.tolist()
            for g, s in zip(info_g, info_s):
                assert g.keys() == s.keys()
                assert all(np.array_equal(g[k], s[k]) if k == "terminal_observation" else g[k] == s[k] for k in g)
        assert grouped.get_attr("i", indices=[4, 1, 2]) == [4, 1, 2]
        assert grouped.env_is_wrapped(gym.Wrapper) == [True] * 5
    finally:
        grouped.close()


@needs_game
@needs_sb3
def test_training_runs_with_two_emulators_per_process(tmp_path):
    from agent.replay import load_episode, replay
    from agent.train import train
    from dataset.cheats import cheats_for

    train(CORE, _dbgt_rom(), SEED.read_bytes(), cheats_for("BT4E"), tmp_path, timesteps=512, envs=4,
          per_process=2, max_steps=64, n_steps=64, reward="progress")
    for i in range(4):  # 128 steps each: 2 episodes
        lines = (tmp_path / "monitor" / f"{i}.monitor.csv").read_text().splitlines()
        assert len(lines) == 2 + 2
        for row in csv.DictReader(lines[1:]):  # the episodes were paid with the progress reward
            assert float(row["r"]) == pytest.approx(int(row["kills"]) + 10 * int(row["waves"])
                                                    + int(row["ground"]) / 16, abs=1e-5)
    # recorded by an emulator that used a copy of the core; replays the same with the original
    episode = load_episode(tmp_path / "best" / "episode.npz")
    assert "kills" in episode
    replay(CORE, _dbgt_rom(), episode)


@needs_game
@needs_sb3
def test_training_with_a_practice_start_logs_its_episodes_apart(tmp_path, env):
    from agent.replay import load_episode, replay
    from agent.train import train
    from dataset.cheats import cheats_for

    env.reset()
    for _ in range(50):
        env.step(env.action_space.sample())
    later = env.emu.save_state()
    env.close()

    train(CORE, _dbgt_rom(), SEED.read_bytes(), cheats_for("BT4E"), tmp_path, timesteps=1024, envs=2,
          max_steps=64, n_steps=64, reward="progress", practice=later, practice_share=0.5, seed=3)
    starts = [int(row["start"]) for i in range(2) for row in
              csv.DictReader((tmp_path / "monitor" / f"{i}.monitor.csv").read_text().splitlines()[1:])]
    assert len(starts) == 16 and set(starts) == {0, 1}
    header = (tmp_path / "progress.csv").read_text().splitlines()[0].split(",")
    assert "dbgt/kills" in header and "dbgt_practice/kills" in header
    replay(CORE, _dbgt_rom(), load_episode(tmp_path / "best" / "episode.npz"))  # a main-start episode


# --- recording the best episode ---------------------------------------------------------


def _episode(r, score=0, ground=0, waves=0, length=0, kills=0):
    return {"episode": {"r": r, "l": length, "t": 1.0, "score": score, "ground": ground, "waves": waves,
                        "kills": kills}}


@needs_sb3
def test_best_episode_keeps_each_emulators_actions_and_only_saves_a_better_episode(tmp_path):
    from agent.replay import load_episode
    from agent.train import BestEpisode

    best = BestEpisode(tmp_path / "best", b"start state", frames_per_step=4, rom="game.gba",
                       cheats=["code"], root="level00")
    saved = tmp_path / "best" / "episode.npz"

    def step(t, dones, infos):  # emulator i presses [t, i, 0, 0, 1] at step t
        best.add(np.array([[t, 0, 0, 0, 1], [t, 1, 0, 0, 1]]), np.array(dones), infos)

    step(1, [False, False], [{}, {}])
    step(2, [True, False], [_episode(5.0, score=400, ground=16, waves=1, length=2), {}])
    episode = load_episode(saved)
    assert episode["actions"].tolist() == [[1, 0, 0, 0, 1], [2, 0, 0, 0, 1]]  # the last step included
    assert episode["actions"].dtype == np.uint8
    assert episode["state"] == b"start state"
    assert episode["frames_per_step"] == 4
    assert (episode["reward"], episode["score"], episode["ground"], episode["waves"]) == (5.0, 400, 16, 1)
    assert (episode["steps"], episode["emulator"]) == (2, 0)
    assert (episode["rom"], episode["cheats"], episode["root"]) == ("game.gba", ["code"], "level00")
    assert json.loads((tmp_path / "best" / "episode.json").read_text())["reward"] == 5.0

    step(3, [False, True], [{}, _episode(3.0)])  # worse: emulator 1's steps 1-3 are not saved
    step(4, [True, False], [_episode(5.0), {}])  # a tie keeps the earlier episode
    assert load_episode(saved)["actions"].tolist() == [[1, 0, 0, 0, 1], [2, 0, 0, 0, 1]]

    step(5, [False, True], [{}, _episode(9.0, score=900)])  # emulator 1's new episode: steps 4-5
    episode = load_episode(saved)
    assert episode["actions"].tolist() == [[4, 1, 0, 0, 1], [5, 1, 0, 0, 1]]
    assert (episode["reward"], episode["score"], episode["emulator"]) == (9.0, 900, 1)

    history = list(csv.DictReader((tmp_path / "best" / "history.csv").read_text().splitlines()))
    assert [float(row["reward"]) for row in history] == [5.0, 9.0]
    assert [int(row["score"]) for row in history] == [400, 900]
    assert sorted(p.name for p in (tmp_path / "best").iterdir()) == ["episode.json", "episode.npz", "history.csv"]


@needs_sb3
def test_best_episode_of_a_resumed_run_must_beat_the_best_already_saved(tmp_path):
    from agent.replay import load_episode
    from agent.train import BestEpisode

    first = BestEpisode(tmp_path, b"state")
    first.add(np.array([[1, 0, 0, 0, 0]]), np.array([True]), [_episode(9.0)])
    resumed = BestEpisode(tmp_path, b"state")  # a new process continuing the run
    resumed.add(np.array([[2, 0, 0, 0, 0]]), np.array([True]), [_episode(5.0)])
    assert load_episode(tmp_path / "episode.npz")["actions"].tolist() == [[1, 0, 0, 0, 0]]
    resumed.add(np.array([[3, 0, 0, 0, 0]]), np.array([True]), [_episode(10.0)])
    assert load_episode(tmp_path / "episode.npz")["actions"].tolist() == [[3, 0, 0, 0, 0]]


def _fake_model(path, timesteps):
    """A zip with the "data" member Stable-Baselines3 saves, holding num_timesteps."""
    import zipfile

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("data", json.dumps({"num_timesteps": timesteps, "n_envs": 12}))


@needs_sb3
def test_latest_model_is_the_one_with_the_most_timesteps(tmp_path):
    from agent.train import latest_model

    _fake_model(tmp_path / "checkpoints" / "ppo_900000_steps.zip", 900_000)
    _fake_model(tmp_path / "checkpoints" / "ppo_1000000_steps.zip", 1_000_000)
    assert latest_model(tmp_path) == tmp_path / "checkpoints" / "ppo_1000000_steps.zip"
    _fake_model(tmp_path / "model.zip", 1_040_000)  # saved when the run stopped
    assert latest_model(tmp_path) == tmp_path / "model.zip"
    with pytest.raises(FileNotFoundError):
        latest_model(tmp_path / "nothing")


def _fake_progress(path, steps):
    path.write_text("time/fps,time/total_timesteps\n" + "".join(f"600,{s}\n" for s in steps))


@needs_sb3
def test_resume_point_is_the_latest_model_or_the_checkpoint_given(tmp_path):
    from agent.train import resume_point

    (tmp_path / "run.json").write_text("{}")
    for steps in (100, 200, 300):
        _fake_model(tmp_path / "checkpoints" / f"ppo_{steps}_steps.zip", steps)
    assert resume_point(tmp_path) == (tmp_path, tmp_path / "checkpoints" / "ppo_300_steps.zip")
    checkpoint = tmp_path / "checkpoints" / "ppo_200_steps.zip"
    assert resume_point(checkpoint) == (tmp_path, checkpoint)


@needs_sb3
def test_continuing_from_an_earlier_checkpoint_sets_later_files_aside(tmp_path):
    from agent.train import latest_model, set_aside_after

    for steps in (100, 200, 300):
        _fake_model(tmp_path / "checkpoints" / f"ppo_{steps}_steps.zip", steps)
    _fake_model(tmp_path / "model.zip", 350)
    _fake_progress(tmp_path / "progress.csv", [100, 200, 300, 350])
    set_aside_after(tmp_path, 200)
    assert latest_model(tmp_path) == tmp_path / "checkpoints" / "ppo_200_steps.zip"
    dropped = tmp_path / "checkpoints" / "dropped"
    assert sorted(p.name for p in dropped.iterdir()) == ["model.zip", "ppo_300_steps.zip"]
    rows = lambda name: [r["time/total_timesteps"] for r in csv.DictReader(open(tmp_path / name))]  # noqa: E731
    assert rows("progress.csv") == ["100", "200"]
    assert rows("progress_dropped.csv") == ["300", "350"]


@needs_game
@needs_sb3
def test_a_resumed_run_continues_its_model_logs_and_best_episode(tmp_path):
    from agent.train import latest_model, saved_timesteps, train
    from dataset.cheats import cheats_for

    settings = dict(core=CORE, rom=_dbgt_rom(), state=SEED.read_bytes(), cheats=cheats_for("BT4E"),
                    out_dir=tmp_path, envs=2, max_steps=64, n_steps=64, checkpoint_every=256)
    train(timesteps=512, **settings)
    first_best = json.loads((tmp_path / "best" / "episode.json").read_text())["reward"]
    checkpoint = latest_model(tmp_path)
    assert saved_timesteps(checkpoint) == 512

    train(timesteps=1024, resume=checkpoint, **settings)
    assert saved_timesteps(tmp_path / "model.zip") == 1024

    progress = list(csv.DictReader((tmp_path / "progress.csv").read_text().splitlines()))
    steps = [int(row["time/total_timesteps"]) for row in progress]
    assert 512 in steps and steps[-1] == 1024 and steps == sorted(steps)  # the first part's rows are kept
    episodes = (tmp_path / "monitor" / "0.monitor.csv").read_text().splitlines()
    assert episodes[0].startswith("#") and episodes[1].startswith("r,")
    assert len(episodes) == 2 + 8  # 4 episodes of 64 steps per part, one header
    best = json.loads((tmp_path / "best" / "episode.json").read_text())["reward"]
    assert best >= first_best
    history = (tmp_path / "best" / "history.csv").read_text().splitlines()
    assert history.count(history[0]) == 1  # one header


@needs_game
@needs_sb3
def test_resuming_can_change_the_batch_size_and_target_kl_which_later_resumes_keep(tmp_path):
    from stable_baselines3 import PPO

    from agent.train import latest_model, train
    from dataset.cheats import cheats_for

    settings = dict(core=CORE, rom=_dbgt_rom(), state=SEED.read_bytes(), cheats=cheats_for("BT4E"),
                    out_dir=tmp_path, envs=2, max_steps=64, n_steps=64)
    train(timesteps=128, **settings)
    assert (PPO.load(tmp_path / "model.zip").batch_size, PPO.load(tmp_path / "model.zip").target_kl) == (128, None)
    train(timesteps=256, resume=latest_model(tmp_path), batch_size=64, target_kl=0.05, **settings)
    assert (PPO.load(tmp_path / "model.zip").batch_size, PPO.load(tmp_path / "model.zip").target_kl) == (64, 0.05)
    train(timesteps=384, resume=latest_model(tmp_path), **settings)
    model = PPO.load(tmp_path / "model.zip")
    assert (model.num_timesteps, model.batch_size, model.target_kl) == (384, 64, 0.05)


def _record_random_play(env, folder, steps, seed):
    """Plays `steps` random actions from the start state and saves them as a recording.

    Closes the emulator afterwards: mGBA cores in one process share their state, so a replay
    can only start a new one once this one is gone."""
    from agent.replay import save_episode
    from dataset.cheats import cheats_for

    env.max_steps = steps
    env.action_space.seed(seed)
    env.reset()
    actions = [env.action_space.sample() for _ in range(steps)]
    for action in actions:
        *_, info = env.step(action)
    last_frame = env.render()
    env.close()
    save_episode(folder, np.array(actions, np.uint8), env.state, env.frames_per_step,
                 {"reward": 0.0, "score": info["score"], "ground": info["ground"], "waves": info["waves"],
                  "rom": str(_dbgt_rom()), "cheats": cheats_for("BT4E")})
    return info, last_frame


@needs_game
def test_a_saved_episode_plays_the_same_game_in_a_new_emulator(tmp_path, env):
    from agent.replay import load_episode, replay

    env.reset()
    for _ in range(100):  # training plays many episodes on one emulator; the state load must reset it all
        env.step(env.action_space.sample())
    info, last_frame = _record_random_play(env, tmp_path, steps=300, seed=1)
    assert info["score"] > 0

    frames = deque(maxlen=1)
    got = replay(CORE, _dbgt_rom(), load_episode(tmp_path / "episode.npz"), on_frame=frames.append)
    assert (got["score"], got["ground"], got["waves"]) == (info["score"], info["ground"], info["waves"])
    assert np.array_equal(frames[0], last_frame)


@needs_game
def test_replay_stops_when_the_recording_does_not_play_the_same(tmp_path, env):
    from agent.replay import load_episode, replay

    _record_random_play(env, tmp_path, steps=30, seed=2)
    episode = load_episode(tmp_path / "episode.npz")
    episode["score"] += 1
    with pytest.raises(RuntimeError):
        replay(CORE, _dbgt_rom(), episode)


@needs_game
def test_record_writes_every_frame_to_a_video_and_a_contact_sheet(tmp_path, env):
    import cv2

    from agent.replay import record

    _record_random_play(env, tmp_path, steps=300, seed=3)  # 1200 frames, 20 game seconds
    record(tmp_path / "episode.npz", CORE)  # the ROM path saved in the recording
    video = cv2.VideoCapture(str(tmp_path / "episode.mp4"))
    try:
        assert video.get(cv2.CAP_PROP_FRAME_COUNT) == 1200
        assert (video.get(cv2.CAP_PROP_FRAME_WIDTH), video.get(cv2.CAP_PROP_FRAME_HEIGHT)) == (720, 480)
        assert video.get(cv2.CAP_PROP_FPS) == 60
    finally:
        video.release()
    sheet = cv2.imread(str(tmp_path / "episode.png"))
    assert sheet.shape[0] == 162  # one row: frames at 10 s and 20 s, 160 pixels plus a 2-pixel line
