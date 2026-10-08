# OmniGBA

A Game Boy Advance player that upscales the picture with a neural network in real time.

- Emulation: [mGBA](https://mgba.io/) libretro core, driven from Python by [libretro.py](https://pypi.org/project/libretro.py/), displayed with pygame.
- Upscaling: Real-ESRGAN `realesr-animevideov3` running on the GPU through [ncnn](https://github.com/Tencent/ncnn) (Vulkan).
- The upscaler runs in its own process, so the game always runs at full speed (60 fps); the upscaled picture updates as fast as the GPU allows.

## Requirements

- Windows 10 or 11, 64-bit. Paths in the code use Windows separators and the emulator core is a `.dll`.
- Python 3.12 or newer (libretro.py and numpy 2.5 need it). Tested with Python 3.12.0.
- A GPU with Vulkan drivers for the upscaler. Tested on an NVIDIA GeForce GTX 1660 Ti.
  If the upscaler cannot start, the game still runs and shows the original picture.

## 1. Get the code

```powershell
git clone https://github.com/theHATEM/OmniGBA.git
cd OmniGBA
```

## 2. Create the virtual environment

In PowerShell, from the project folder:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run `Activate.ps1` ("running scripts is disabled on this system"),
allow scripts for the current window only and try again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

In Command Prompt use `.venv\Scripts\activate.bat` instead.

Check that everything installed:

```powershell
python -c "import libretro, pygame, ncnn, cv2; print('ok')"
```

Run `deactivate` to leave the virtual environment. Activate it again in every new terminal before running the project.

## 3. Add the files that are not in the repository

`core/`, `models/` and `roms/` are excluded from git (large binaries, and ROMs are copyrighted).
Create them and add:

| Folder | Files | Where to get them |
|---|---|---|
| `core/` | `mgba_libretro.dll` | [mgba_libretro.dll.zip](https://buildbot.libretro.com/nightly/windows/x86_64/latest/mgba_libretro.dll.zip) from the libretro buildbot; extract the `.dll` |
| `models/` | `realesr-animevideov3-x2.param`, `realesr-animevideov3-x2.bin`, `realesr-animevideov3-x4.param`, `realesr-animevideov3-x4.bin` | `realesrgan-ncnn-vulkan-20220424-windows.zip` from the [Real-ESRGAN v0.2.5.0 release](https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0); copy them from its `models` folder |
| `roms/` | your `.gba` files | dumps of games you own |

Expected layout:

```
OmniGBA/
├── core/
│   └── mgba_libretro.dll
├── models/
│   ├── realesr-animevideov3-x2.bin
│   ├── realesr-animevideov3-x2.param
│   ├── realesr-animevideov3-x4.bin
│   └── realesr-animevideov3-x4.param
├── roms/
│   └── <your game>.gba
├── fast_video.py
├── play_gba.py
├── upscale_model.py
├── upscale_process.py
└── requirements.txt
```

## 4. Configure and run

Settings are at the top of `play_gba.py`:

| Setting | Meaning |
|---|---|
| `ROM_PATH` | the game to load, e.g. `r"roms\My Game.gba"` |
| `UPSCALE_MODEL` | `"realesr-animevideov3-x2"` (480x320, faster) or `"realesr-animevideov3-x4"` (960x640, more detail) |
| `WINDOW_SCALE` | initial window size as a multiple of the GBA screen (240x160) |
| `PROFILE` | `True` prints time spent per stage once a second |

Run from the project folder (paths are relative to it):

```powershell
python play_gba.py
```

The window title shows the game frame rate and the upscaled frame rate, e.g. `GBA | 60 fps | upscaled 56 fps`.

### Controls

| Key | Action | Key | Action |
|---|---|---|---|
| Arrow keys | D-pad | Enter | Start |
| X | A | Right Shift | Select |
| Z | B | Esc | Quit |
| S | R (shoulder) | P | Pause / resume |
| A | L (shoulder) | R | Reset the game |
| Tab (hold) | Fast-forward | F12 | Save screenshot to `captured_frames/` |

### Check the upscaler speed

```powershell
python upscale_model.py realesr-animevideov3-x2
```

Prints the time per frame. On a GTX 1660 Ti: x2 about 13 ms, x4 about 20 ms.

## Project files

| File | Purpose |
|---|---|
| `play_gba.py` | window, keyboard input, emulation loop |
| `fast_video.py` | fast frame conversion for libretro.py's video driver |
| `discard_audio.py` | audio driver that drops the sound (libretro.py's default keeps every sample in memory) |
| `upscale_model.py` | the Real-ESRGAN model on the GPU (ncnn) |
| `upscale_process.py` | runs the model in a separate process; frames pass through shared memory |
| `dataset/` | automated frame collection for training a smaller upscaler (see below) |
| `tests/` | tests for `dataset/` |

## Collecting training frames

`dataset/` plays a game by itself and saves a varied set of frames, as training data for a smaller
model that copies the current upscaler. It needs no human play:

- **Exploration (Go-Explore style):** it remembers save states of every new *scene* (identified
  from the background layers, ignoring sprites), keeps returning to the least-explored ones and plays
  bursts of random input from there: walking, button mashing, buttons held together, and scripted
  special-move inputs (quarter circles, dashes, charges, combo strings).
- **New graphics:** tiles that never appeared in video memory before (a new enemy, pose or effect)
  count as progress, and those frames are always saved.
- **Cheats:** infinite health (and similar codes) for the two supported games, so runs never end in a
  game over. Codes live in `dataset/cheats.py`, keyed by the ROM's game code.
- **Progress counters:** some progress does not show on screen. In Dragon Ball GT the screen does not
  scroll until the enemy wave is beaten, and a burst that starts again from a saved state loses the
  damage done. So every 1000 points of score (read from RAM) also count as a new scene, and later
  bursts continue from there. Counters live in `dataset/progress.py`, keyed by game code (Naruto has
  none yet). It is slow: on the first planet, getting past the first wave and up to the mid-boss took about an
  hour of game time from that one stage, while a 10-minute run with all 11 planets gives each about 5
  minutes. Long runs are needed for the explorer to get far into the stages.
- **Routes into the stages:** `dataset/routes.py` scripts the way from power-on into each stage, and
  exploration also starts from those states. Bursts are shared evenly between these starting points,
  and bursts from a stage never press Start (in gameplay it only opens the pause menu).
  - Dragon Ball GT: Story Mode, then each of the 11 planets on the star map (picked by setting the
    star map cursor in RAM).
  - Naruto: New Game, then later chapters reached by playing (random input until the chapter number in
    RAM changes; retried if the game falls back to the title). It stops at a chapter it cannot beat
    within 30 minutes of game time; the explorer usually gets further from there.

  The route states are made on the first run for a game (about 30 s for Dragon Ball GT, 2-3 minutes
  for Naruto) and kept in `data\seeds\<game>\`. Delete that folder to make them again, for example
  after changing a route or the cheats.

```powershell
python -m dataset.explore --rom "roms\rom_d\Dragon Ball GT - Transformation.gba" --minutes 30
```

Options: `--target-frames N` stops after saving N frames, `--seed` changes the random play,
`--no-cheats` disables cheats, `--no-routes` explores from power-on only, `--no-progress` ignores the
progress counter. Output goes to `data\raw\<game>\<timestamp>\`:

| File | Contents |
|---|---|
| `frames\*.png` | saved frames, 240x160, lossless |
| `frames.jsonl` | one line per frame: scene id, starting point (root), progress level, whether it showed a new scene or new graphics |
| `cells.jsonl` | every scene found, the scene it was reached from and its root (for splitting train/validation by area) |
| `run.json` | ROM, seed, cheats used, which root is which route state |

### Frames from TAS movies

[TASVideos](https://tasvideos.org/) publishes tool-assisted speedruns as movie files: the buttons
pressed on every frame, not video. `dataset/tas.py` replays BizHawk movies in BizHawk, saves every
frame losslessly at 240x160, and keeps the varied ones with the same selection as the explorer. That
gives whole playthroughs of other games with no per-game work. (Neither Dragon Ball GT
Transformation nor Naruto: Ninja Council 2 has a TAS.)

| What | Where to get it |
|---|---|
| BizHawk | [BizHawk releases](https://github.com/TASEmulators/BizHawk/releases): extract the `win-x64` zip anywhere. Best is the version a movie was made with (its TASVideos page says which); newer versions usually work but can desync. |
| GBA BIOS | a dump from your own console. BizHawk does not play GBA movies without one, and most movies need the official BIOS (SHA1 `300C20DF6731A33952DED8C436F7F186D25D3492`). |
| Movies | from a game's publication page on TASVideos: the `.bk2` file, or the `.zip` the site downloads. Only movies made with BizHawk's mGBA core work; VBA-rr (`.vbm`) and GBAHawk (`.gbmv`) movies are skipped. |
| ROMs | the exact ROMs the movies were made for, anywhere under `roms\` (found by checksum). |

```powershell
python -m dataset.tas --bizhawk "C:\BizHawk\EmuHawk.exe" --bios "C:\bios\gba_bios.bin" "C:\tas"
```

Movies can be given one by one or as folders. Output goes to `data\raw\<game>\tas_<movie name>\`
with the same files as the explorer's (no `cells.jsonl`); `frames.jsonl` records the movie frame of
each saved frame and `run.json` the movie, ROM and BizHawk versions. Movies already done are skipped,
and a movie that cannot be played is skipped with the reason (missing ROM, different BIOS, other
emulator). BizHawk's window shows while it runs; it uses a settings file of its own, so your BizHawk
settings are not changed. A 10-minute movie takes under 3 minutes.

### Running the tests

```powershell
pip install -r requirements-dev.txt
python -m pytest
```

Tests that need the emulator core and a ROM are skipped when those files are missing. One slow test
(playing Naruto's first chapter, about a minute) only runs with `GBA_SLOW_TESTS` set:

```powershell
$env:GBA_SLOW_TESTS = "1"; python -m pytest tests/test_routes.py
```

The test that plays a movie in real BizHawk needs the paths of `EmuHawk.exe` and a GBA BIOS file
(any BIOS works for it):

```powershell
$env:GBA_BIZHAWK = "C:\BizHawk\EmuHawk.exe"; $env:GBA_BIOS = "C:\bios\gba_bios.bin"; python -m pytest tests/test_tas.py
```

## Troubleshooting

- **`FileNotFoundError: Please verify your Windows paths for the DLL and ROM.`**
  `core\mgba_libretro.dll` or the file in `ROM_PATH` is missing. Check `ROM_PATH` and run from the project folder.
- **Window title says `upscaler failed`.** The reason is printed in the console. Usually the `models/` files are missing or the GPU has no Vulkan driver.
- **`%s: %s` lines in the console.** Log output from the mGBA core; harmless.
- **The pygame welcome message appears twice.** The upscaler process imports `play_gba.py` again when it starts; harmless.
