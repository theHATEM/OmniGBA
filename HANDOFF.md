# Handoff

Where the project stands and what to do next, so the work can continue in a new chat.
Last updated 2026-10-13.

**Next task:** the GPU task (below the reinforcement learning test section) is done: the 10M-step run
finished on 2026-10-13 and its results are under "Second run: 10M steps on the GPU". It scores 4x
random play but every model stops at the gun-turret wave or the one after; the next step is a run
that rewards progress instead of points (ideas at the end of that section), or the upscaler work in
"Open issues and next steps".

## Goal

OmniGBA plays GBA games (mGBA libretro core, driven from Python) and upscales the picture with a
neural network in real time. The long-term goal is to run on phones. The current upscaler is too
heavy for that, so the plan is to **train a much smaller "student" model that copies the current
one** (distillation), using frames collected from the two games we care about.

A second, newer goal: a model that plays the games, built like NVIDIA's NitroGen (behaviour cloning:
screen in, buttons out, learned from recorded play). The reinforcement learning test below checks
how far a trained player gets compared with random input.

## What exists today

| Part | Files | State |
|---|---|---|
| Player | `play_gba.py`, `fast_video.py`, `discard_audio.py` | Works. Game at 60 fps. |
| Upscaler | `upscale_model.py` (ncnn, Vulkan), `upscale_process.py` (separate process, shared memory) | Works. See numbers below. |
| Frame collection | `dataset/` (explore, archive, novelty, inputs, recorder, cheats, routes, progress, emulator) | Works; coverage still improving (see "Open issues"). |
| Frames from TAS movies | `dataset/tas.py` | Works for `.bk2`, `.gbmv`, `.vbm`; real `.bk2` and `.gbmv` TASes verified in sync (see below). |
| Reinforcement learning test | `agent/` (env, train, evaluate, replay), `requirements-rl.txt` | Works; PPO on Dragon Ball GT planet 0 clearly beats random input; training keeps its best episode as a replayable recording (see below). |
| Tests | `tests/` (pytest) | 143 pass (~2.5 min); 1 slow test opt-in, 2 need BizHawk/GBAHawk (env vars). |
| Upscaler training | nothing yet | Design agreed (below). |

Setup and usage are in `README.md`. The emulator core, models and ROMs are not in git
(`core/`, `models/`, `roms/`); collected data goes to `data/` (also not in git).

### Upscaler performance (GTX 1660 Ti, 240x160 input, fp16)

| Model | Per frame | Notes |
|---|---|---|
| `realesrgan-x4plus-anime` (old "digital-art-4x") | ~100 ms | RRDBNet, 438 GFLOP/frame; removed |
| `realesr-animevideov3-x4` | ~20 ms | 960x640 output; `play_gba.py` default (`UPSCALE_MODEL`) |
| `realesr-animevideov3-x2` | ~13 ms | Same weights as x4 plus a bicubic 0.5x; 480x320 output |

The upscaler runs in its own process, so the game always runs at 60 fps; the upscaled picture
updates at ~55-60 fps (x2) or ~36-40 fps (x4). `play_gba.py` currently loads
`roms\rom_d\Dragon Ball GT - Transformation.gba`.

## Decisions already made (training plan)

- **Games:** only the two we have: Dragon Ball GT: Transformation (USA, game code `BT4E`) and
  Naruto: Ninja Council 2 (USA, `BN2E`).
- **Teacher:** `realesr-animevideov3-x2`, run at **fp32** on **full frames** (never on crops; its
  outputs depend on a 37x37 neighbourhood). The x2 file is the x4 net plus a bicubic downscale, so
  bicubic is the resize filter.
- **Student output:** x2 (480x320); SDL stretches it to the window.
- **Data:** input 160x240x3 uint8 RGB exactly as `FastArrayVideoDriver` produces it, saved as PNG;
  target 320x480x3 from the teacher. Aim for ~10k unique frames (~5k per game). Train on random
  64x64 input crops paired with the matching 128x128 target crops; horizontal flip is the only
  augmentation.
- **Split:** train/validation/test by starting point or scene lineage (`root` / `cells.jsonl`),
  never by individual frame. Keep a few runs of consecutive frames in validation to measure flicker.
- **Student design:** SRVGGNetCompact-like, plain ops only (3x3 conv, PReLU, pixel shuffle x2, plus
  a nearest-neighbour x2 copy of the input added to the output). Train three sizes and keep the
  smallest that passes: F=16/N=4 (0.9 GFLOP/frame), F=24/N=6 (2.6), F=32/N=8 (6.0); the teacher is
  47.6.
- **Loss:** L1 or Charbonnier against the teacher. Add an edge-weighted term only if text comes out
  soft. No GAN or VGG loss.
- **Training:** Adam ~5e-4 with cosine decay, ~100k steps, batch 32-64. Needs PyTorch with CUDA; the
  current venv has the CPU-only build.
- **Judging:** PSNR against the teacher (bicubic as the floor), separate scores for text-heavy and
  effect-heavy frames, flicker on consecutive frames, side-by-side sheets, then speed.
- **Export:** PyTorch -> ONNX -> ncnn (`pnnx`); the same weights can later become a GPU shader or a
  Core ML model.

## Frame collection: how it works

```powershell
python -m dataset.explore --rom "roms\rom_d\Dragon Ball GT - Transformation.gba" --minutes 30
```

- Go-Explore style: keeps a save state per new *scene* (identified from the background layers,
  ignoring sprites) and keeps returning to the least-explored ones with bursts of random input
  (walking, mashing, chords, scripted special moves).
- New graphics (tiles never seen in video memory) count as progress; those frames are always saved.
- Several **roots** (starting points): power-on, plus scripted routes into each stage
  (`dataset/routes.py`, cached in `data\seeds\<game>\`). Bursts are split evenly between roots;
  bursts from route roots never press Start (it pauses).
- **Progress counters** (`dataset/progress.py`): a RAM value that grows with progress, bucketed, is
  added to the cell key, so each higher level is a new cell and later bursts continue from it. Only
  Dragon Ball GT has one: score at `0x03001D0C`, steps of 1000. `--no-progress` turns it off.
- Frames are capped per cell and deduplicated. Each saved frame's metadata in `frames.jsonl` includes
  its `root` and `progress` level.

### Last 10-minute run per game

| | Roots | Frames saved | Memory after 10 min |
|---|---|---|---|
| Dragon Ball GT | power-on + 11 planets | 5,567 (205 MB) | 142 MB |
| Naruto | power-on + chapters 1-2 | 3,460 (101 MB) | 82 MB |

Speed is ~350-430 emulated fps during gameplay; the profile shows most of that time is the mGBA core
itself (gameplay costs ~1.15 ms/frame, menus ~0.65 ms).

## Progress counter results (Dragon Ball GT)

The score counter (`dataset/progress.py`) fixes the stall, but only with enough run time per stage.
Runs on a laptop (Intel Core 5 120U, two or three runs side by side, ~800-1000 emulated fps each),
seed 0:

| 10-minute run, all 11 planets | Frames | Cells | Graphics items | Scenes per planet 0-10 |
|---|---|---|---|---|
| no progress counter | 10,219 | 1,394 | 36,057 | 5, 372, 600, 8, 6, 9, 4, 10, 9, 5, 5 |
| score, steps of 500 | 19,097 | 3,549 | 36,350 | 5, 330, 586, 19, 21, 9, 4, 10, 9, 43, 5 |

| 10-minute run, planet 0 only (~64 game minutes) | Cells | Scenes |
|---|---|---|
| no progress counter | 5 | 5 |
| score, steps of 1000 | 361 | 61 (wave 1 beaten, blocking machine broken, mid-boss reached) |
| score + camera x + enemies left | 468 | 58 |

- With 12 roots, a 10-minute run gives each planet ~5 game minutes; one wave needs far more. Steady
  random play beat planet 0's first wave in about 160 s at 8,306 points; chained bursts needed ~40,000
  points, hence steps of 1000, not 500 (fewer cells to spread the bursts over).
- Camera x and enemies-left (addresses below) as extra counters added nothing over score alone in
  these runs: bursts of 1-6 s rarely kill an enemy, and the camera only moves after a wave is beaten.
- Score cells multiply the frames kept per scene (per-cell cap), so frame counts go up a lot; the
  per-root balancing planned for sampling (issue 2) matters even more now.

## Frames from TAS movies (`dataset/tas.py`)

A TAS movie is the buttons pressed on every frame, not video. `dataset/tas.py` replays movies in an
emulator, saves every frame as a lossless 240x160 PNG, and passes them through `FrameRecorder` with
the explorer's settings (8x6x4 screen groups, 2 per group, min diff 4). Output:
`data\raw\<game>\tas_<movie name>_<extension>\`, same files as the explorer; `frames.jsonl` has
`movie_frame`, `run.json` the movie, ROM and emulator versions.

```powershell
python -m dataset.tas "C:\tas"
```

Defaults: `bizhawk\EmuHawk.exe` (BizHawk 2.11.1), `gbahawk\GBAHawk.exe` (GBAHawk 3.0.0) and
`BIOS\gba_bios.bin` (official BIOS) in the project folder, all git-ignored.

- Why: whole playthroughs of other GBA games, exact frames, no per-game work. Neither of our two games
  is on TASVideos at all (not even user files), so this only adds other games, which the training plan
  above does not include yet (decide before using it).
- Three movie kinds on TASVideos (first page of GBA publications: 68 `.bk2`, 22 `.vbm`, 10 `.gbmv`):
  - `.bk2` (BizHawk, mGBA core): played in BizHawk. BizHawk 2.11.1 has only mGBA for GBA.
  - `.gbmv` (GBAHawk): played in GBAHawk. GBAHawk is a cut-down BizHawk by Alyosha; a `.gbmv` is a
    `.bk2` under another name with core `GBAHawk` and `P1 Up|...|P1 Power` input columns.
  - `.vbm` (VBA-rr): converted to an mGBA `.bk2` in Python (port of BizHawk's `VbmImport`; BizHawk's
    command line cannot take a `.vbm`, it opens every `--movie` as a `.bk2` first) and played in BizHawk.
    The ROM is found by game code and internal title (a `.vbm` has no ROM hash). Unlike BizHawk's
    importer, the BIOS intro is kept when the movie was recorded with it; VBA resets become Power.
- Needs per movie: the ROM (by the SHA1 in the header, searched under `--roms`), a GBA BIOS (the
  emulators refuse GBA movies without one; most movies name the official BIOS
  `300C20DF6731A33952DED8C436F7F186D25D3492` and are checked against it), and ideally the emulator
  version the movie was made with (a mismatch only prints a note; GBAHawk movies often say
  "Version 1.0.0", which is not the real version).

Verified (2026-10-08, official BIOS, homebrew ROMs from the movies' TASVideos pages and Homebrew Hub):

| Movie | Made with | Played in | Result |
|---|---|---|---|
| Dangerous Xmas "all gifts" (#5770, `.bk2`, 8,639 frames) | BizHawk 2.9.1 | BizHawk 2.11.1 | In sync to the end: "THE END" and score screen; 2,267 frames kept, 95 s |
| Anguna (#5531, `.gbmv`, 46,885 frames) | GBAHawk 1.6 | GBAHawk 3.0.0 | In sync to the end: dragon beaten, ending text, credits; ~77 frames/s |
| Same random Dragon Ball GT input as `.bk2` and as `.vbm` (18,104 frames) | (made here) | BizHawk 2.11.1 | All 18,104 frames identical, so the converter maps buttons right |

- Not verified: a real VBA-rr movie. Every GBA `.vbm` on TASVideos is for a commercial game (no
  homebrew `.vbm` exists), and VBA-rr's timing differs from mGBA's, so real `.vbm` movies may well
  desync. Check the last frames of each one before using its frames.
- BizHawk and GBAHawk are deterministic here: the same movie dumped twice gave identical frames.

Emulator automation gotchas (all handled in `dump_movie`; any modal dialog blocks a run until the
timeout, which then prints the command to run by hand):

- Command: `<exe> --config=<ini> --movie=<file> --dump-type=imagesequence --dump-name=<dir>\f.png
  --dump-length=<movie frames> --dump-close <rom>` (ROM last). GBAHawk's parser only knows the
  `--flag=value` form; BizHawk takes it too. Frames come out as `f_0.png`, `f_1.png`, ...
- `--dump-length` is needed for GBAHawk: its command-line code never loads the movie's length, so
  without it the dump runs forever (seen: 108k frames for a 47k-frame movie). BizHawk works it out.
- GBAHawk always dumps through a "video stretcher" (no audio-sync switch), so a frame may now and then
  be repeated or dropped; harmless for training frames.
- The dump folder must exist; otherwise saving the first frame fails with an error dialog.
- Own config file (JSON) per run:
  - `LastWrittenFrom` must equal the emulator's version, or a version dialog appears. GBAHawk's .exe
    does not carry its version (ProductVersion says 1.0.0), so `emulator_version` starts the emulator
    once with an empty config and a Lua script that runs `client.exit()`, and reads the version from
    the config it saves on exit.
  - BIOS in `FirmwareUserSpecifications["GBA+Bios"]`, `Unthrottled: true` (otherwise 60 fps),
    `VideoWriterAudioSync: false` (BizHawk would drop or repeat frames to match the sound).
  - `PathEntries` sets the GBA base folder to the temporary work folder. Without it the emulators
    write save RAM into `<emulator>\GBA\SaveRAM\` (it does not affect movie playback, but it clutters
    the install).
  - The emulator rewrites the file in full on exit, so one config per run in the temporary folder.
- No BIOS gives the dialog "A BIOS is required for deterministic recordings!".
- A movie without a `Core` line makes BizHawk ask which core to use; converted movies always set it.
- BizHawk reads the input columns from the `LogKey` line; a wrong `LogKey` silently misreads buttons.
- ROMs are matched before starting the emulator. A `.bk2` whose SHA1 differs from the loaded ROM's
  plays without asking (only `.tasproj` files prompt).
- Homebrew ROMs often have an empty game code; their output folder is named after the ROM file.

## Reinforcement learning test (`agent/`)

Question: does a trained player get further through a stage than the explorer's random input? If so,
reinforcement learning (RL) is worth adding on top of the behaviour cloning plan. Setup and commands
are in `README.md`.

- `agent/env.py`: `DbgtEnv`, a Gymnasium environment around `HeadlessGBA`. Starts from the planet 0
  route seed (`route_seeds` in `dataset/routes.py`, shared with the explorer) with the explorer's
  cheats, so the fighter never dies. Observation: grey 120x80 screen, 4 stacked. Action:
  `MultiDiscrete([9, 2, 2, 2, 2])` = d-pad (none or one of 8 directions) and A, B, L, R, held 4
  frames; never Start or Select. Episodes are 2 game minutes (1800 steps), never ended early.
- Reward (`DbgtReward`): score / 100 plus new ground (camera x beyond the episode's best so far) / 16.
  Also counts waves beaten (rising edges of the "GO ->" flag) for logging only.
- `agent/train.py`: PPO (Stable-Baselines3), NatureCNN (2.2M parameters), 8 emulators in subprocesses,
  Atari-style settings (lr 2.5e-4, 128 steps per emulator per update, 4 epochs, clip 0.1, entropy
  0.01). Logs episode means of score, ground and waves to `progress.csv`; checkpoint every 100k steps.
- `agent/evaluate.py`: plays the model, the explorer's random bursts (exactly as `Explorer.iterate`
  picks them) and uniform random actions (a model that learned nothing) from the same start, and
  reports points, new ground, waves, scenes (`scene_key`) and new graphics tiles, plus contact sheets.

First run (2026-10-12, laptop CPU, 1,000,000 steps = about 18 hours of game time): 5,785 s (1.6 hours),
172 steps/s on average (237 at the start, slowing as the CPU heated). `ep_rew_mean` rose 119 -> 145
(300k) -> 166 (600k) -> 169 (1M), so it had mostly levelled off; policy entropy 4.96 -> 4.42 (4.97 is
fully random). Model in `data\rl\20261012_175329\` (git-ignored).

Means over 5 episodes from the planet 0 start:

| Play style | Game minutes | Points | New ground (px) | Waves beaten | Scenes |
|---|---|---|---|---|---|
| PPO, 300k steps (4 episodes) | 2 | 11,603 | 260 | 1.0 | 15.5 |
| PPO, 1M steps | 2 | 13,903 | 411 | 1.0 | 18.6 |
| Explorer's random bursts | 2 | 10,690 | 26 | 0.2 | 5.0 |
| Uniform random, 4-frame steps | 2 | 10,918 | 112 | 1.0 | 9.2 |
| PPO, 1M steps | 5 | 31,190 | 1,012 | 2.6 | 46.0 |
| Explorer's random bursts | 5 | 18,340 | 388 | 1.0 | 21.6 |
| Uniform random, 4-frame steps | 5 | 16,148 | 415 | 1.6 | 25.2 |

- Within the 2 minutes it trained on, every play style but the explorer beats wave 1; the model gets
  further right and scores more.
- Over 5 minutes (longer than any training episode) the model keeps going: in the first episode's
  contact sheet it beats wave 1, then the wave with the purple robot mid-boss, and reaches an area
  with gun turrets (score 28,336). Its points and ground are about double either kind of random input.
- The explorer's long random bursts are worst at fighting: they beat wave 1 in only 1 of 5 two-minute
  episodes.
- At 300k steps the contact sheet showed the model idling and firing after the "GO ->" arrow instead of
  walking right; by 1M it walks on.
- Evaluating 3 play styles x 5 episodes x 5 game minutes takes about 8 minutes.

### Second run: 10M steps on the GPU (2026-10-13)

GTX 1660 Ti PC, 5-minute episodes, 10,000,644 steps in `data\rl\20261013_112630\` (11:26-16:23,
about 5 hours of wall time including ~1.5M steps of a dropped branch; see part 3 of the GPU task for
the resumes, the speed fixes and why the run went back from 5.4M to 3.9M). Speed ~535 steps/s at the
start (12 emulators), ~710-730 at the end (24 emulators, 2 per process, distribution checks off).

Learning curve (training episodes of 5 game minutes; means of the log rows around each point):

| Steps | 1M | 2M | 3M | 4M | 5M | 6M | 7M | 8M | 9M | 10M |
|---|---|---|---|---|---|---|---|---|---|---|
| `ep_rew_mean` | 359 | 435 | 517 | 598 | 528 | 559 | 591 | 629 | 687 | 734 |
| points | 30,845 | 36,429 | 44,240 | 50,314 | 43,950 | 48,997 | 55,389 | 54,249 | 59,325 | 70,240 |
| new ground (px) | 990 | 1,121 | 1,152 | 1,188 | 1,098 | 1,189 | 1,130 | 1,171 | 1,177 | 1,274 |
| waves | 2.58 | 2.83 | 2.88 | 3.01 | 2.76 | 3.00 | 2.90 | 3.03 | 3.00 | 3.40 |

(5M is the branch after going back to 3.9M; the dropped branch had fallen to ~470.)

Evaluation, means over 5 episodes of 5 game minutes from the planet 0 start (`agent.evaluate` for
the final model and the random play; the checkpoints with the same `play`, model only):

| Play style | Points | New ground (px) | Waves beaten | Scenes | New graphics |
|---|---|---|---|---|---|
| PPO 10M (final `model.zip`) | 74,250 | 1,248 | 3.2 | 41.0 | 5,581 |
| PPO 7.1M (`ppo_7099140_steps.zip`) | 59,080 | 1,267 | 3.4 | 39.8 | 5,289 |
| PPO 3.9M (`ppo_3899652_steps.zip`) | 58,404 | 1,279 | 3.6 | 36.4 | 5,621 |
| PPO 1M (first run, laptop) | 31,190 | 1,012 | 2.6 | 46.0 | |
| Explorer's random bursts | 18,340 | 388 | 1.0 | 21.6 | 3,807 |
| Uniform random, 4-frame steps | 16,148 | 415 | 1.6 | 25.2 | 3,821 |

- The random play gave exactly the first run's numbers (same seeds, same start state), so the
  comparison is like for like.
- The final model scores 2.4x the first run's 1M model and 4x random play, with 23% more new ground
  and 3.2 instead of 2.6 waves. Results are in `eval.json` and `eval_ppo.png` (final),
  `eval_3.9M\` and `eval_7.1M\` in the run folder.
- **All PPO models stop at the gun-turret wave or the one after it.** In every evaluation episode of
  all three models the new ground is 1,200 px (3 waves: stuck at the turret wave) or 1,440 px (4
  waves: turrets beaten, stuck at the next wave). None of the 2,520 training episodes (all monitor
  CSVs, dropped branch included) got past 1,440 px: 221 beat 4 waves, none beat 5. The final model's
  contact sheet: wave 1, the robot mid-boss at ~1:40 (a 21-hit combo), the dark corridor, a second
  blocking object, then the turret area from ~3:00 to the end, fighting the soldiers that keep
  coming there.
- **More points, not more progress.** From 3.9M to 10M the points rose 27% while ground and waves did
  not (the 3.9M checkpoint even beat slightly more waves). The model learned combos: points per kill
  grow with the combo, from ~550-600 to ~2,800-2,950 (the best episode scores 20,400 in its first 9
  seconds, holding B with R or diagonals). Points are ~85-90% of the reward, so the reward now
  mostly pays for combo farming.
- The best episode of the run (`best\`, reward 1,282 at 8.76M steps) is a farming one: 120,666 points
  but only 1,200 px and 3 waves. The best by progress are the earlier bests with 1,440 px and 4 waves
  (3.70M-8.33M in `best\history.csv`; only the last best is kept as a recording).

Ideas for the next run (in order): reward progress, not points: e.g. a bonus per wave beaten, points
clipped or scaled down, or points only while the camera can move on; start some episodes from a save
state at the turret wave (an explorer cell or a state saved from the model's play) so it practises
the part it is stuck on; normalise rewards (`VecNormalize`; the value loss reached 5-9 with raw
rewards) and shrink the learning rate and clip range linearly to 0 as rl-baselines3-zoo does; keep
minibatches of 512 with `target_kl` 0.02 for 24 emulators. Also still open: other planets
(`--root levelNN`); recording the model's play (frames plus buttons) as training data for the
behaviour cloning model.

## Next task: GPU training and recording the best episode

Planned 2026-10-13 in three parts: set up CUDA PyTorch and measure the speed; record the best episode
the model plays during a training run (test-first); then a long training run that uses it. Parts 1
and 2 were done on 2026-10-13; part 3 is running.

### Part 1 (done): the GPU PC and its speed

The PC with the GTX 1660 Ti is an Intel Core i7-9750H laptop (6 cores, 12 threads, 16 GB RAM); see
"Environments" for its setup. PyTorch `2.14.1+cu126` sees the GPU (driver 610.88).

Install PyTorch with CUDA **before** `requirements-rl.txt`; otherwise pip takes the CPU-only build:

```powershell
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements-rl.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Speed, `python -u -m agent.train --device cuda --envs N`, steps per second for all emulators together
(`time/fps`) after ~25k steps:

| `--envs` | 6 | 8 | 10 | 12 | 16 |
|---|---|---|---|---|---|
| steps/s | 363 | 440 | 523 | 575 | 585 |

- The emulators (CPU) limit the speed: it grows up to the 12 hardware threads and stops there. The GPU
  was about 33% busy with 8 emulators. Over a 100k-step run with 8 emulators the speed only dropped
  from 440 to 420.
- Each emulator process takes ~0.5 GB of RAM (it imports PyTorch); 16 emulators used 9.7 GB, leaving
  2.4 GB free. **12 is the setting for this PC**: as fast as 16, less memory. The laptop CPU did 172
  with 8.
- More emulators also means more data per update (128 steps x emulators: 1,536 at 12 instead of
  1,024 at 8), with the same 4 epochs and minibatches of 256.
- On 100k steps with 8 emulators, `ep_rew_mean` rose 113 -> 134 and the best episode reached the robot
  mid-boss (below).

### Part 2 (done): the best episode of a training run

`data\rl\<run>\best\` holds the best episode any emulator played in the run (highest episode reward,
the training objective; a tie keeps the earlier one), written whenever a better one ends:

- `episode.npz`: the actions (uint8, steps x 5), the start state (compressed save state), the frames
  per step, and the stats as a JSON string, so the file stands alone. Written to a temporary name and
  then renamed, so Ctrl+C never leaves half a file. `episode.json` has the same stats for reading:
  reward, score, ground, waves, steps, training timesteps when it was played, emulator, ROM, cheats,
  root.
- `history.csv`: one line per new best (timesteps, reward, score, ground, waves); each also prints a
  line in the training log.
- `episode.mp4` and `episode.png`: written by `agent.train` at the end (also after Ctrl+C) by
  replaying the recording: every frame at 60 fps scaled 3x to 720x480 (`mp4v`, ~90 MB per 2 game
  minutes), and a contact sheet with one frame every 10 game seconds.
- `python -m agent.replay data\rl\<run>\best\episode.npz` does the same by hand. It replays the actions
  in a fresh `DbgtEnv` and stops with an error if score, ground or waves differ from the recording.

Code: `BestEpisode` (a Stable-Baselines3 callback in `agent/train.py`; `add(actions, dones, infos)`
does the bookkeeping, one action list per emulator, the step that ends an episode belongs to it),
`agent/replay.py` (`save_episode`, `load_episode`, `replay`, `record`) and an `on_frame` callback on
`DbgtEnv` that gets every game frame of a step (for the video; the reward still samples RAM once per
step). Tests in `tests/test_agent.py`: bookkeeping with made-up steps for 2 emulators; a recording
replays the same score, ground, waves and last frame in a new emulator after the first one played
other episodes; a changed recording is refused; the video has every frame; and the training test
checks `best\episode.json` against the monitor CSVs and replays it.

Checked for real on the 100k-step run with 8 emulators (`data\rl\speed_envs8\20261013_110816\`): the
best episode (reward 194.1 at 86,400 steps: 16,406 points, 480 px new ground, 1 wave) replayed with the
same numbers; its contact sheet shows wave 1 beaten, the blocking machine broken and the robot
mid-boss reached within the 2 minutes. The 8 emulators' episodes all have the same length, so they end
together and new bests come in groups every 14,400 steps (8 x 1,800).

Found on the way:

- **Python 3.12.0 cannot load Stable-Baselines3 models.** Its `zipfile` fast-seek for uncompressed
  members runs before the read-buffer check and leaves the end-of-file flag set, so after a short
  forward seek near the end of a member reads return nothing. SB3 saves models uncompressed, and
  `PPO.load` fails with `PytorchStreamReader failed reading file .data/serialization_id: file read
  failed`. Fixed in later 3.12 releases (3.14's code checks the buffer first). The venv here was moved
  to 3.12.13 with `python -m venv --upgrade .venv`, keeping the installed packages.
- **One mGBA core per process.** A second `HeadlessGBA` in the same process shares the core's state
  (it is one DLL), and closing both crashes with an access violation. Training already uses one
  process per emulator; the tests close their emulator before replaying, and `agent.train` replays
  after the emulator processes are closed.
- Not done: the optional `--frames` flag (every frame as PNG plus `buttons.csv`) for the behaviour
  cloning goal.

### Part 3 (done): the long run

Started 2026-10-13 11:26 in its own minimized console window, output in `data\rl\20261013_112630\`:

```powershell
python -u -m agent.train --device cuda --envs 12 --minutes 5 --timesteps 10000000
```

It was stopped and resumed three times, always in its own minimized window (Ctrl+C there stops it
and still saves `model.zip` and replays the best episode): twice for the speed fixes below, right
after a checkpoint (so only a few thousand steps were lost), and once back to an earlier checkpoint
because the training had got worse:

| From | Command | Speed (steps/s) | Log |
|---|---|---|---|
| 11:26, step 0 | `--device cuda --envs 12 --minutes 5 --timesteps 10000000` | 576 cold, ~535 warm | `data\rl\long_run.log` |
| 12:05, step 1,199,952 | `--resume data\rl\20261013_112630` (distribution checks off) | ~585-604 | `data\rl\long_run2.log` |
| 12:44, step 2,299,908 | `--resume data\rl\20261013_112630 --envs 24 --per-process 2` | ~710 | `data\rl\long_run3.log` |
| 14:01, back to step 3,899,652 | `--resume data\rl\20261013_112630\checkpoints\ppo_3899652_steps.zip --batch-size 512 --target-kl 0.02` | ~726 | `data\rl\long_run4.log` |

Why the third resume went back: after the peak at ~3.9M steps (`ep_rew_mean` 614, 54k points, 1,260
px, 3.3 waves) the training got worse and stayed worse up to 5.4M (`ep_rew_mean` 462-488, ~40k
points, ~1,000 px, ~2.5 waves). Each update had started changing the policy too much: `clip_fraction`
(healthy around 0.1-0.2) rose 0.19 (first 0.5M) -> 0.32 (by 2.3M) -> 0.44 after the switch to 24
emulators -> 0.53, and `approx_kl` 0.004 -> 0.008 -> 0.015 -> 0.03. The switch made it worse:
with minibatches of 256, twice the samples per update meant twice the gradient steps (48 instead of
24). Minibatches of 512 bring it back to 24, and `target_kl` 0.02 stops an update once the KL passes
0.03 (SB3 uses 1.5 x target_kl). In the first 200k steps after it: `approx_kl` ~0.016, `clip_fraction`
~0.33, and some updates stop early. The dropped branch's checkpoints are in `checkpoints\dropped\` and
its log rows in `progress_dropped.csv`; the monitor CSVs still hold its episodes, and `best\` keeps
the episode from 3.92M (reward 1,027: 93,748 points, 1,440 px, 4 waves), played after the checkpoint
by the dropped branch. The learning rate (2.5e-4) and clip range (0.1) are constant here;
rl-baselines3-zoo's Atari settings shrink both linearly to 0, which would also calm the late updates
(not tried).

Errors and the core's `%s: %s` lines are in the matching `.err` files; the process id is in
`data\rl\long_run.pid`. `run.json` lists the resumes under `resumed`. `progress.csv` and the monitor
CSVs simply continue (`time/time_elapsed` starts again from 0 at each resume; monitor files 12-23
start at the second resume). From the second resume on, each PPO update uses 3,072 samples
(128 steps x 24 emulators) instead of 1,536.

- `--minutes 5`: 2-minute episodes spent most of their time on wave 1, yet the 1M model went further
  in 5-minute evaluations. Longer episodes let it practise the later parts of planet 0 (mid-boss,
  turrets).
- 10M steps is the usual Atari budget. It ended at 16:23 with 10,000,644 steps; the replay at the end
  reproduced the best episode and wrote `best\episode.mp4` (~210 MB for 5 game minutes) and
  `best\episode.png`. Results: "Second run: 10M steps on the GPU" in the reinforcement learning
  section. Windows sleep and hibernate are off on mains power on this PC; during a run do not close
  the lid, and do not run `play_gba.py` (it also uses the GPU) at the same time.

Where the training time goes (12 emulators, measured with PPO and the emulators separately on
2026-10-13, both while the run was going, so absolute numbers are a little high). Stable-Baselines3
is synchronous: the emulators wait while the main process runs the network and the PPO update.

| Part, per timestep (~1.87 ms at 535 steps/s) | Time |
|---|---|
| main process: network on the GPU and bookkeeping (9.5 ms per step of all 12 emulators) | 0.79 ms |
| emulators in parallel (4 frames 4.4 ms + frame to observation 1.0 ms, waiting for the slowest) | ~0.55 ms |
| PPO update (emulators idle) | 0.50 ms |

- **Done:** `torch.distributions` checked the arguments of every action distribution (5 categorical
  distributions per step for the `MultiDiscrete` action, each check waiting for the GPU), about a
  third of the main process's time per step. `train` now turns that off
  (`Distribution.set_default_validate_args(False)`); main process 9.5 -> 6.3 ms per step, update
  0.50 -> 0.40 ms per timestep, the run 535 -> 604 steps/s. It changes no numbers.
- **Done:** several emulators per worker process (`--per-process`, `agent/vec_env.py`). The main
  process's cost per step barely depends on the number of emulators, so more emulators per step make
  it cheaper per emulator, but more processes than hardware threads did not help (16 processes: same
  speed as 12, and 9.7 GB RAM, ~0.5 GB per process for PyTorch). `SubprocGroupVecEnv` runs several
  emulators in each process, one after another; it behaves like SB3's `SubprocVecEnv` (tested
  against `DummyVecEnv`). Each emulator in a process loads its own copy of the core DLL
  (`core_copies`, in `core\copies\`): a test shows two emulators in one process with their own copies
  play exactly as they do alone, and a best episode recorded with a copy replays with the original.
  Measured back to back (warm laptop, 45k steps each): 12 x 1 ~655, 24 x 2 ~766 (+17%), 36 x 3
  ~790-810 (+20%) steps/s. 24 x 2 was chosen: most of the gain while doubling, not tripling, the
  samples per update.
- Not done, in order of payoff: (1) sample the 5 action parts in one go instead of 5 separate
  `Categorical` distributions (their sampling and normalising are ~36% of what is left of the main
  process's time per step; a custom SB3 action distribution, which must give the same log
  probabilities and entropy), ~+10%; (2) turn the emulator's 16-bit frame straight into the grey
  observation with a lookup table (identical result; `FastArrayVideoDriver.screenshot` plus the copy
  cost ~0.95 of the 1.0 ms), ~+3-5%. Overlapping emulation with the network in two groups of
  emulators would help less than it seems: the network's time per call hardly shrinks with fewer
  emulators, so two groups mean twice the calls. (Minibatches of 512, also on the list, came in with
  the third resume for stability; they made the update cheaper too: ~710 -> ~726 steps/s.)
- Measured and not worth it: more than 12 emulators (16: same speed, 9.7 GB RAM); a higher process
  priority for the main process (no change); `mgba_idle_optimization = Detect and Remove` (no gain,
  same game). The core option `mgba_frameskip = fixed_interval` (interval 3) saves 15% of emulation
  and plays the same game, but it skips rendering the wrong frames for 4-frame steps, and libretro.py
  0.12 crashes on skipped frames: its `video_refresh` matches a NULL frame as `0`, but ctypes gives
  `None`.
- Follow `ep_rew_mean` and `dbgt/*` in the log or `progress.csv`, and `best\history.csv`. With 12
  emulators x 4,500 steps, episodes end together every 54,000 steps.

Evaluated after the run with
`python -m agent.evaluate --model data\rl\20261013_112630\model.zip --minutes 5 --episodes 5`
(~15 minutes with two checkpoint evaluations running alongside). `agent.evaluate` loads models on
the CPU, so a GPU-trained model can be evaluated on any machine. It always plays the two random
styles too; the checkpoints were evaluated with a small script around `evaluate.play` and
`PPOPolicy` (model only). A `--model-only` flag or a way to evaluate several checkpoints would save
that.

## Open issues and next steps (in suggested order)

1. **Run time per stage.** Getting through Dragon Ball GT stages needs on the order of an hour of game
   time per planet. Either run much longer (60+ min), or add an option to pick which route seeds to
   explore (e.g. `--roots level00,level01`) and run several processes in parallel on different
   planets, each with its own output folder.
2. **Frame imbalance across roots.** Levels 1 and 2 produce most Dragon Ball GT frames: their parallax
   backgrounds stream new tiles on small camera moves, which look like new scenes. Plan: balance per
   root when sampling training data (metadata is already there), not in collection.
3. **Naruto beyond chapter 2.** The route reaches chapter 2 (Forest of Death) and stops; random play
   cannot finish that stage within 30 game-minutes. Find a progress counter in RAM (score, enemies
   left, mission timer...). What worked for Dragon Ball GT: replay deterministic random play from a
   seed state, snapshot IWRAM and EWRAM every 30 frames, and search for bytes that count up or down
   or flip exactly when a known event happens (see the game facts below).
4. Run longer collections, look at contact sheets, check the balance.
5. **TAS movies for other games (optional).** Decide whether other games belong in the training data.
   If so: collect movies and their exact ROMs, run `dataset.tas`, and check each movie reached its
   ending (contact sheet of the last minutes), especially `.vbm` ones.
6. Generate teacher targets (x2, fp32, full frames), build the split, write the training script
   (needs CUDA PyTorch), then evaluate and export as planned above.

## Game facts found so far

**Dragon Ball GT: Transformation (BT4E)**
- Menus and dialogue all respond to A; Start is never needed.
- Route: title -> Story Mode -> Start New Game -> skip opening -> star map (timings in
  `routes.py`, `DBGT_TO_STAR_MAP`).
- Star-map cursor / planet: IWRAM `0x03001D04`, values 0-10 (11 planets; 11 wraps to 0). Set it,
  wait 90 frames, then ~45 A presses skip each intro into gameplay.
- Score: `0x03001D0C`, 32-bit binary (not BCD), 0 at the start of each planet's route seed; ~50 per
  hit, a few hundred per enemy, 800 for breaking the machine that blocks planet 0.
- Fighter slots: `0x03001AF0` + i * `0x2C`, i = 0-8 (the id is at `+4`); the health cheats write `+6`
  (health) and `+8` (max health). Enemies are not among them: they still die with the cheats on.
- Enemies left in the current wave: `0x030019B0` (e.g. 7 -> 0 for planet 0's first wave). While the
  green "GO ->" arrow shows it holds an unrelated value (120, 192, 224).
- "GO ->" arrow shown (wave beaten, walk right): `0x03001CEA` = 1.
- Camera x: `0x030024D8` (16-bit) = BG3 horizontal scroll + 320, mod 512. Grows through a stage (planet
  0: 138 -> 1560 over the first five waves), stops at each wave's arena, rarely drops a few pixels.
- Stage timer: `0x03001CE4`, one byte, counts seconds and wraps at 256.
- Planet 0: after the first wave a large machine blocks the way right; it has a health bar and breaks
  under attacks (B). Then comes a robot mid-boss.
- Pause menu (Continue / View Stats / Exit) is not closed by Start.

**Naruto: Ninja Council 2 (BN2E)**
- The title screen only responds to Start; menus and dialogue respond to A.
- Main menu: New Game (top-left), Continue (bottom-left), Co-op (top-right), VS (bottom-right). Co-op
  and VS only show "Communicating. Please wait." (link cable), so story chapters are the only stages.
- Chapter number: IWRAM `0x0300453F` (0 at the title). Forcing it only changes the chapter title card,
  not the stage loaded.
- Mission cleared flag: `0x03004526` goes 0 -> 1 at the results screen.
- `0x0300454A-D` are the four ninjas' energy. Do **not** lock them with cheats: they are also opponents
  (mission 1 is a fight against Rock Lee), which made the mission unwinnable.
- After a mission the game sometimes goes straight to the next chapter and sometimes back to the title,
  depending on the input. Continue opens a load screen with 3 slots; the cursor starts on slot 3 (empty),
  so press Left twice for slot 1, then A.
- Saves are 32 KB SRAM (type detected on the first write); emulator save states include the save data.
- Random play (no Start) beats mission 1 in ~7-11 minutes of game time.

## libretro.py / mGBA gotchas

- `session.memory_maps` is only available after the first `run()`; `HeadlessGBA` runs one frame on enter.
- The default `ArrayAudioDriver` keeps every audio sample forever (~460 MB per hour at 60 fps); always
  pass `audio=DiscardAudioDriver`.
- `core.get_memory_data(0)` (save RAM) reads back as all 0xFF; use RAM diffs instead.
- Cheats: `core.cheat_set(i, True, "AAAAAAAA+VVVV+...")` with CodeBreaker codes works and is re-applied
  every frame; direct writes through the memory-map views also work.
- mGBA prints `%s: %s` log lines; harmless.
- The ncnn `UpscaleModel` destroys the GPU instance in `close()`; only one per process.

## Environments

- The game has been run with `D:\development\python\GBA\.venv` (Python 3.12; has ncnn, libretro.py,
  pygame). `README.md` describes creating a fresh `.venv` from `requirements.txt` (verified to work).
- GPU PC (Intel Core i7-9750H laptop, 6 cores / 12 threads, 16 GB RAM, GTX 1660 Ti 6 GB, driver
  610.88), project in `D:\development\python\gba_capture`: `.venv` made on 2026-10-13 from
  `requirements-dev.txt`, then CUDA PyTorch (`cu126`) and `requirements-rl.txt`; all tests pass. It
  was made with Python 3.12.0 and moved to 3.12.13 (installed with `uv python install 3.12`) because of
  3.12.0's zipfile bug (see part 2 of the GPU task). The core, both ROMs (`roms\rom_d\`,
  `roms\rom_naruto\`) and the models are in place, and `data\seeds\` has both games' route seeds.
  BizHawk, GBAHawk and the BIOS are not (only their `DOWNLOAD.txt`), so the TAS tests skip here. PyTorch downloaded at only ~0.5 MB/s here (ProtonVPN was running).
- Second machine (laptop, Intel Core 5 120U, Intel graphics only, so no CUDA for training): `.venv` made
  with Python 3.13.14 from `requirements-dev.txt`, plus CPU PyTorch and `requirements-rl.txt`; all
  tests pass. The ROMs there are in `Downloads\ROMS_FOLDER` and were copied to `roms\rom_d\`.
- Laptop: Windows Smart App Control is on. On 2026-10-11 it blocked the unsigned
  `core\mgba_libretro.dll` and PyTorch's `torch_python.dll` (`WinError 4551`, "An Application Control
  policy has blocked this file"); a day later both loaded fine. If it comes back, either turn Smart App
  Control off (Windows Security > App & browser control; it cannot be turned back on without
  reinstalling Windows) or run on the desktop PC.
- Tests: `pip install -r requirements-dev.txt`, then `python -m pytest`. Tests needing the core or a
  ROM skip when those are missing. `GBA_SLOW_TESTS=1` enables the Naruto campaign test (~1 min).
  `GBA_BIZHAWK` (path of EmuHawk.exe), `GBA_GBAHAWK` (path of GBAHawk.exe) and `GBA_BIOS` (any GBA
  BIOS file) enable the real emulator tests (their windows show for a few seconds).
- On the laptop: `bizhawk\` (BizHawk 2.11.1), `gbahawk\` (GBAHawk 3.0.0, from its GitHub release
  `GBAHawk.v3.0.0.zip`) and `BIOS\gba_bios.bin` (official, SHA1 `300C20DF...`) are in the project
  folder and git-ignored. Both emulators are unsigned and run without installing.

## Conventions

- Commit messages: a single line, no `Co-Authored-By` or other attribution trailer.
- Changes are made test-first (pytest), and verified on the real games with contact sheets of
  sampled frames before calling them done.
- When editing files through a shell, text containing backslashes (Windows paths, `\n`) has been
  mangled by heredocs before; editing files directly is safer.
