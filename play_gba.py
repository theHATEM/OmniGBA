"""
Real-time GBA viewer for libretro.py + mGBA, using pygame for display and keyboard input.
Frames are upscaled by a neural network running in a separate process (upscale_process.py),
so the emulator always runs at full speed, whatever the upscaler manages.

    pip install pygame numpy libretro.py ncnn opencv-python

Controls
    Arrow keys ... D-pad            Enter ......... Start
    X ............ A                Right Shift ... Select
    Z ............ B                Esc ........... Quit
    S ............ R (shoulder)     P ............. Pause / resume
    A ............ L (shoulder)     R ............. Reset the game
    Tab (hold) ... Fast-forward     F12 ........... Save screenshot
"""

import os
import time

import pygame
from libretro import Session
from libretro.api.input import JoypadState

from fast_video import FastArrayVideoDriver  # keep fast_video.py next to this file
from upscale_process import UpscalerProcess

CORE_PATH = r"core\mgba_libretro.dll"
ROM_PATH = r"roms\rom_d\Dragon Ball GT - Transformation.gba"
SHOT_DIR = os.path.abspath("./captured_frames")

PROFILE = False  # print average ms per stage once a second
TARGET_FPS = 60  # GBA runs at ~59.73 fps
GBA_WIDTH, GBA_HEIGHT = 240, 160
WINDOW_SCALE = 4  # initial window size = GBA screen x this
# Upscaler in models\ (see UpscaleModel in upscale_model.py for measured timings):
# "realesr-animevideov3-x2" (480x320, faster), "realesr-animevideov3-x4" (960x640)
UPSCALE_MODEL = "realesr-animevideov3-x4"
FAST_FORWARD_STEPS = 4  # emulated frames per displayed frame while Tab is held

# keyboard key -> JoypadState constructor argument
KEYMAP = {
    pygame.K_UP: "up",
    pygame.K_DOWN: "down",
    pygame.K_LEFT: "left",
    pygame.K_RIGHT: "right",
    pygame.K_x: "a",
    pygame.K_z: "b",
    pygame.K_s: "r",
    pygame.K_a: "l",
    pygame.K_RETURN: "start",
    pygame.K_RSHIFT: "select",
}

if not os.path.exists(CORE_PATH) or not os.path.exists(ROM_PATH):
    raise FileNotFoundError("Please verify your Windows paths for the DLL and ROM.")


def joypad_poller():
    """
    Infinite generator used as libretro.py's input source.

    The input driver calls next() on it each time the core polls input, so we
    read the keyboard right then and hand back the current button state.
    """
    while True:
        keys = pygame.key.get_pressed()
        yield JoypadState(**{button: bool(keys[key]) for key, button in KEYMAP.items()})


def set_canvas(scale: int) -> pygame.Surface:
    """
    Set the drawing size to the GBA screen x scale.

    SCALED: SDL stretches this canvas to the window on the GPU. Calling it again
    later changes the canvas size but keeps the window as it is.
    """
    return pygame.display.set_mode(
        (GBA_WIDTH * scale, GBA_HEIGHT * scale), pygame.SCALED | pygame.RESIZABLE
    )


def main() -> None:
    pygame.init()
    canvas_scale = WINDOW_SCALE
    screen = set_canvas(canvas_scale)
    pygame.display.set_caption("GBA")
    clock = pygame.time.Clock()

    paused = False
    raw_surface = None  # newest emulator frame
    upscaled_surface = None  # newest upscaled frame (a frame or two behind the emulator)
    frame_surface = None  # what is on screen, kept for redraws while paused and for F12
    error_reported = False

    # Per-stage timing (seconds), printed once per second when PROFILE is True
    timers = {"run": 0.0, "shot": 0.0, "upscale": 0.0, "draw": 0.0, "wait": 0.0}
    frames_in_window = 0
    upscaled_in_window = 0
    window_start = time.perf_counter()

    # The model loads in the worker process while the game boots; until the first
    # upscaled frame arrives (or if the upscaler fails) the raw frames are shown.
    with UpscalerProcess(UPSCALE_MODEL, GBA_WIDTH, GBA_HEIGHT) as upscaler, Session(
        CORE_PATH, ROM_PATH, input=joypad_poller, video=FastArrayVideoDriver
    ) as session:
        running = True
        while running:
            # --- window / hotkey events -------------------------------------
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_p:
                        paused = not paused
                    elif event.key == pygame.K_r:
                        session.reset()
                    elif event.key == pygame.K_F12 and frame_surface is not None:
                        os.makedirs(SHOT_DIR, exist_ok=True)
                        path = os.path.join(
                            SHOT_DIR, f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                        )
                        pygame.image.save(frame_surface, path)
                        print(f"Saved {path}")

            # --- emulate ------------------------------------------------------
            fast = pygame.key.get_pressed()[pygame.K_TAB]
            if not paused:
                t0 = time.perf_counter()
                for _ in range(FAST_FORWARD_STEPS if fast else 1):
                    session.run()
                t1 = time.perf_counter()

                shot = session.video.screenshot()
                t2 = time.perf_counter()
                if shot is not None:
                    size = (shot.width, shot.height)
                    # FastArrayVideoDriver always delivers R,G,B,A bytes
                    raw_surface = pygame.image.frombuffer(shot.data, size, "RGBX")
                    if size == (GBA_WIDTH, GBA_HEIGHT):
                        upscaler.submit(shot.data)  # never waits for the worker
                t3 = time.perf_counter()

                timers["run"] += t1 - t0
                timers["shot"] += t2 - t1
                timers["upscale"] += t3 - t2

            # --- pick up a finished upscaled frame (never waits) ---------------
            t0 = time.perf_counter()
            rgb = upscaler.poll()
            if rgb is not None:
                upscaled_surface = pygame.image.frombuffer(
                    rgb, (rgb.shape[1], rgb.shape[0]), "RGB"
                )
                upscaled_in_window += 1
            if upscaler.error is not None and not error_reported:
                print(f"Upscaler failed, showing the raw picture:\n{upscaler.error}")
                error_reported = True
            if upscaler.ready and upscaler.scale != canvas_scale:
                # Canvas = model output size, so frames are stretched on the GPU
                # instead of resized on the CPU every frame
                canvas_scale = upscaler.scale
                screen = set_canvas(canvas_scale)
            timers["upscale"] += time.perf_counter() - t0

            # --- draw ---------------------------------------------------------
            t0 = time.perf_counter()
            if upscaler.error is None and upscaled_surface is not None:
                frame_surface = upscaled_surface
            else:
                frame_surface = raw_surface
            if frame_surface is not None:
                if frame_surface.get_size() != screen.get_size():
                    # raw frames while the model loads, or if the upscaler failed
                    frame_surface = pygame.transform.scale(frame_surface, screen.get_size())
                screen.blit(frame_surface, (0, 0))
                pygame.display.flip()
            timers["draw"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            clock.tick(0 if fast else TARGET_FPS)
            timers["wait"] += time.perf_counter() - t0

            # --- once per second: update caption + optional profile print ----
            frames_in_window += 1
            now = time.perf_counter()
            if now - window_start >= 1.0:
                fps = frames_in_window / (now - window_start)
                upscaled_fps = upscaled_in_window / (now - window_start)
                if upscaler.error is not None:
                    status = "upscaler failed"
                elif not upscaler.ready:
                    status = "upscaler loading..."
                else:
                    status = f"upscaled {upscaled_fps:.0f} fps"
                pygame.display.set_caption(
                    f"GBA  |  {fps:.0f} fps  |  {status}" + ("  |  PAUSED" if paused else "")
                )
                if PROFILE:
                    parts = "  ".join(
                        f"{name}={1000 * total / frames_in_window:.2f}ms"
                        for name, total in timers.items()
                    )
                    print(f"{fps:5.1f} fps ({status}) | per frame: {parts}")
                timers = dict.fromkeys(timers, 0.0)
                frames_in_window = 0
                upscaled_in_window = 0
                window_start = now

    pygame.quit()


if __name__ == "__main__":
    main()
