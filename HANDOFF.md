# Handoff

Where the project stands and what to do next, so the work can continue in a new chat.
Last updated 2026-10-07.

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
| Frame collection | `dataset/` (explore, archive, novelty, inputs, recorder, cheats, routes, emulator) | Works; coverage still improving (see "Open issues"). |
| Tests | `tests/` (pytest) | 74 pass, 1 slow test opt-in. |
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
- Frames are capped per scene and deduplicated. Each saved frame's metadata in `frames.jsonl` includes
  its `root`.

### Last 10-minute run per game

| | Roots | Frames saved | Memory after 10 min |
|---|---|---|---|
| Dragon Ball GT | power-on + 11 planets | 5,567 (205 MB) | 142 MB |
| Naruto | power-on + chapters 1-2 | 3,460 (101 MB) | 82 MB |

Speed is ~350-430 emulated fps during gameplay; the profile shows most of that time is the mGBA core
itself (gameplay costs ~1.15 ms/frame, menus ~0.65 ms).

## Open issues and next steps (in suggested order)

1. **Dragon Ball GT barely progresses inside a stage.** Most planets reach only 5-10 scenes: the screen
   does not scroll until the current enemy wave is beaten, and short bursts that restart from saved
   states lose the damage done. Proposed fix: treat a higher score as progress (new cell), using the
   score at IWRAM `0x03001D0C` (from the cheat list), bucketed. Needs a test first, like everything else.
2. **Frame imbalance across roots.** Levels 1 and 2 produced 74% of the Dragon Ball GT frames: their
   parallax backgrounds stream new tiles on small camera moves, which look like new scenes. Plan:
   balance per root when sampling training data (metadata is already there), not in collection.
3. **Naruto beyond chapter 2.** The route reaches chapter 2 (Forest of Death) and stops; random play
   cannot finish that stage within 30 game-minutes. Options: let long explorer runs push through it,
   or find a progress signal in RAM for Naruto too.
4. Run longer collections (e.g. 30-60 min per game), look at contact sheets, check the balance.
5. Generate teacher targets (x2, fp32, full frames), build the split, write the training script
   (needs CUDA PyTorch), then evaluate and export as planned above.

## Game facts found so far

**Dragon Ball GT: Transformation (BT4E)**
- Menus and dialogue all respond to A; Start is never needed.
- Route: title -> Story Mode -> Start New Game -> skip opening -> star map (timings in
  `routes.py`, `DBGT_TO_STAR_MAP`).
- Star-map cursor / planet: IWRAM `0x03001D04`, values 0-10 (11 planets; 11 wraps to 0). Set it,
  wait 90 frames, then ~45 A presses skip each intro into gameplay.
- Score: `0x03001D0C`. Player health slots: `0x03001AF6`, 9 slots, stride `0x2C` (cheat targets).
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
- Tests: `pip install -r requirements-dev.txt`, then `python -m pytest`. Tests needing the core or a
  ROM skip when those are missing. `GBA_SLOW_TESTS=1` enables the Naruto campaign test (~1 min).

## Conventions

- Commit messages: a single line, no `Co-Authored-By` or other attribution trailer.
- Changes are made test-first (pytest), and verified on the real games with contact sheets of
  sampled frames before calling them done.
- When editing files through a shell, text containing backslashes (Windows paths, `\n`) has been
  mangled by heredocs before; editing files directly is safer.
