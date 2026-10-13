# Handoff

Where the project stands and what to do next, so the work can continue in a new chat.
Last updated 2026-10-13.

**Next task:** see "Next task: GPU training and recording the best episode" (after the
reinforcement learning test section).

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
| Reinforcement learning test | `agent/` (env, train, evaluate), `requirements-rl.txt` | Works; PPO on Dragon Ball GT planet 0 clearly beats random input (see below). |
| Tests | `tests/` (pytest) | 129 pass (~1 min); 1 slow test opt-in, 2 need BizHawk/GBAHawk (env vars). |
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

Ideas not tried yet: longer training episodes, or starting episodes from explorer cells further into
the stage (2-minute episodes spend most of their time on wave 1); more steps on the GTX 1660 Ti
(`--device cuda`); other planets (`--root levelNN`); recording the model's play (frames plus
buttons) as extra training data for the behaviour cloning model.

## Next task: GPU training and recording the best episode

Handoff written 2026-10-13 for the next session, on the desktop PC with the GTX 1660 Ti. Three parts,
in this order: set up CUDA PyTorch and measure the speed; add a recording of the best episode the
model plays during a training run (test-first); then start a long training run that uses it.

### Part 1: set up the desktop

1. Get the repo at commit `a662f61` or later (it has `agent/`). The core, the ROM and `data\` are not
   in git: `core\mgba_libretro.dll` and `roms\rom_d\Dragon Ball GT - Transformation.gba` must be in
   place (the game already ran on this PC; see "Environments"). The planet 0 start state
   `data\seeds\dbgt\level00.state` is made by the first `agent.train` run if missing (a one-time wait
   while it plays the route from power-on).
2. Install PyTorch with CUDA **before** `requirements-rl.txt`; otherwise pip takes the CPU-only build
   from PyPI:

   ```powershell
   pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
   pip install -r requirements-rl.txt
   ```

   The `torch==2.14.1` line in `requirements-rl.txt` is satisfied by `2.14.1+cu126`, so pip keeps the
   CUDA build. If a CPU build got in first, `pip uninstall torch` and repeat. The cu126 build needs an
   NVIDIA driver for CUDA 12.6 or newer (`nvidia-smi` shows "CUDA Version"); update the driver if it
   is older. The GTX 1660 Ti (Turing, compute capability 7.5) is supported by this build.
3. Check:

   ```powershell
   python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   python -m pytest tests/test_agent.py
   ```

   Expect `2.14.1+cu126 True NVIDIA GeForce GTX 1660 Ti`, and all 17 agent tests passing.
4. Measure the speed. The emulators run on the CPU (one process each; one alone does ~290 steps/s on
   the laptop), the network on the GPU. Try `--envs` values around the number of physical CPU cores
   (`(Get-CimInstance Win32_Processor).NumberOfCores`):

   ```powershell
   python -u -m agent.train --device cuda --envs 8 --timesteps 100000 --out data\rl\speed
   ```

   `time/fps` in the log is steps per second for all emulators together; `nvidia-smi` should list the
   python process. The laptop did 172 steps/s on its CPU with 8 emulators, CPU at 100%. Pick the
   `--envs` with the highest fps and keep it for the main run (it also sets the batch per update:
   128 steps x emulators).

### Part 2: record the best episode of a training run

Goal: during and after a training run, `data\rl\<run>\best\` holds the single best episode any
emulator played in the whole run, as a file that replays exactly, plus a video and a contact sheet
made from it.

Design:

- **Record actions, not frames.** The emulator is deterministic and every episode starts from the
  same state, so the start state plus each step's action reproduces the whole game (see
  `test_same_actions_play_the_same_game`). An episode is 1,800-4,500 actions of 5 bytes; sending
  frames back from 8 emulator processes would cost far more.
- **Where:** a Stable-Baselines3 callback in `agent/train.py` next to `EpisodeStats`, e.g.
  `BestEpisode(out_dir / "best", state, frames_per_step)`. In `_on_step`:
  - `self.locals["actions"]` (shape `(n_envs, 5)`) holds the actions just taken;
  - `self.locals["dones"]` says which emulators just finished an episode;
  - `self.locals["infos"][i]["episode"]` holds that episode's Monitor stats (`r` reward, `l` length,
    `score`, `ground`, `waves`).

  Keep one action list per emulator and append every step's actions. When an emulator's episode
  ends, compare its reward with the best so far, save it if higher, then clear that list. The step
  that ends an episode belongs to the old episode; the next step is the first of the new one (SB3
  resets automatically). Put the bookkeeping in a plain method (e.g. `add(actions, dones, infos)`)
  so it can be tested without PPO.
- **Best = highest episode reward** (`r`, the training objective: score / 100 + new ground / 16).
  A tie keeps the earlier episode.
- **Files:**
  - `best\episode.npz` (`np.savez_compressed`): `actions` (uint8, steps x 5), `state` (the start
    state, a ~16 KB compressed save state, so the file stands alone) and `frames_per_step`.
  - `best\episode.json`: reward, score, ground, waves, steps, training timesteps so far
    (`self.num_timesteps`), emulator index, root, ROM file name and cheats.
  - Write both to a temporary name and then rename, so Ctrl+C never leaves half a file.
  - Append a line per new best to `best\history.csv` (timesteps, reward, score, ground, waves), and
    print one line, to show how the best improves over the run.
- **Replay:** a new `agent/replay.py`, `python -m agent.replay data\rl\<run>\best\episode.npz`. It
  works as follows:
  - Make a fresh `DbgtEnv` with the same cheats, the saved state and `max_steps` = number of actions,
    and play the actions.
  - Check that score, ground and waves equal the recorded values. If they differ, stop with an
    error: the recording is not reproducible.
  - Write `episode.mp4`: every frame at 60 fps, scaled 3x with nearest neighbour to 720x480
    (`cv2.VideoWriter`, `mp4v`).
  - Write `episode.png`: a contact sheet with one frame every 10 s (`evaluate.contact_sheet`).
  - For the video it needs every frame while keeping 4-frame steps (the reward and wave count sample
    RAM once per step). So give `DbgtEnv` a way to hand out each frame of a step, such as an
    optional callback called after every frame.
  - Write frames to the video as they come instead of keeping them: 5 minutes is 18,000 frames
    (~2 GB in memory).
  - `agent.train`'s `main` runs the replay at the end of training, also after Ctrl+C.
- Optional, for the behaviour cloning goal: a `--frames` flag on the replay that writes every frame
  as PNG plus `buttons.csv` (frame number, button mask), a ready demonstration of good play.

Tests first, in `tests/test_agent.py` (it already has the `env` fixture, `needs_game`, `needs_sb3`
and the toy `Counter` env):

1. Bookkeeping with made-up `actions`, `dones` and `infos` for 2 emulators: lists kept apart, only a
   better episode replaces the saved one, the step that ends an episode is included.
2. A saved episode replays: play ~300 random steps in `env`, save them, replay them in a **new**
   `DbgtEnv` (fresh emulator), and get the same score, ground, waves and last frame. Training reuses
   one emulator per process across episodes, so this checks that loading the state resets
   everything the game uses.
3. The replay writes a video with the right number of frames (`cv2.VideoCapture`,
   `CAP_PROP_FRAME_COUNT`) and a contact sheet.
4. Extend `test_training_runs_on_two_emulators_then_the_model_plays`: `best\episode.json` exists and
   its reward equals the highest `r` in the monitor CSVs.

Then check it for real: a 100k-step run, `agent.replay` on its best episode, and look at the video and
the contact sheet. Add the replay command and the `best\` files to the README's reinforcement learning
section.

### Part 3: the long run

```powershell
python -u -m agent.train --device cuda --envs <N> --minutes 5 --timesteps 10000000
```

- `--minutes 5`: 2-minute episodes spent most of their time on wave 1, yet the 1M model went further
  in 5-minute evaluations. Longer episodes let it practise the later parts of planet 0 (mid-boss,
  turrets).
- 10M steps is the usual Atari budget. Work out the hours from the measured fps first (at the
  laptop's 172 steps/s it would be 16 hours). Checkpoints come every 100k steps. Ctrl+C stops the run
  and still saves `model.zip`, and `best\` is kept.
- Keep Windows from sleeping during the run, and do not run `play_gba.py` (it also uses the GPU) at
  the same time.
- From a Claude session, start it as a background command with its output going to a log file, as
  the first run did. Follow `ep_rew_mean` and `dbgt/*` in the log or `progress.csv`, and
  `best\history.csv`.

After the run:

```powershell
python -m agent.evaluate --model data\rl\<run>\model.zip --minutes 5 --episodes 5
```

Compare with the first run's table above (PPO 1M over 5 minutes: 31,190 points, 1,012 ground,
2.6 waves, 46 scenes) and watch `best\episode.mp4`. Then add the results to the "Reinforcement
learning test" section: speed, learning curve, evaluation table and what the best episode reached.
`agent.evaluate` loads models on the CPU, so a GPU-trained model can be evaluated on any machine.

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
