# Handoff

Where the project stands and what to do next, so the work can continue in a new chat.
Last updated 2026-10-08.

## Goal

OmniGBA plays GBA games (mGBA libretro core, driven from Python) and upscales the picture with a
neural network in real time. The long-term goal is to run on phones. The current upscaler is too
heavy for that, so the plan is to **train a much smaller "student" model that copies the current
one** (distillation), using frames collected from the two games we care about.

## What exists today

| Part | Files | State |
|---|---|---|
| Player | `play_gba.py`, `fast_video.py`, `discard_audio.py` | Works. Game at 60 fps. |
| Upscaler | `upscale_model.py` (ncnn, Vulkan), `upscale_process.py` (separate process, shared memory) | Works. See numbers below. |
| Frame collection | `dataset/` (explore, archive, novelty, inputs, recorder, cheats, routes, progress, emulator) | Works; coverage still improving (see "Open issues"). |
| Frames from TAS movies | `dataset/tas.py` | Works for `.bk2`, `.gbmv`, `.vbm`; real `.bk2` and `.gbmv` TASes verified in sync (see below). |
| Tests | `tests/` (pytest) | 111 pass; 1 slow test opt-in, 2 need BizHawk/GBAHawk (env vars). |
| Training | nothing yet | Design agreed (below). |

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
  with Python 3.13.14 from `requirements-dev.txt`; all tests pass. The ROMs there are in
  `Downloads\ROMS_FOLDER` and were copied to `roms\rom_d\`.
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
