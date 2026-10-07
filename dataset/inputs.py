"""
Random controller input for exploration.

Buttons are bit masks (GBA key order). InputPolicy.burst() returns one mask per
frame, mixing four kinds of input so that both movement and special moves happen:

    walk   hold a direction (sometimes with a button) for 12-90 frames
    mash   quick taps of attack buttons, 2-6 frames each
    chord  several buttons held together
    macro  a scripted move (quarter circle + button, dash, charge, combo string),
           randomly mirrored so it works facing either way

Start and Select are pressed rarely and always on their own, so the game cannot
see a soft-reset combination (A+B+Start+Select).
"""

import random
from functools import lru_cache

from libretro.api.input import JoypadState

A, B, SELECT, START, RIGHT, LEFT, UP, DOWN, R, L = (1 << i for i in range(10))

_NAMES = {A: "a", B: "b", SELECT: "select", START: "start", RIGHT: "right",
          LEFT: "left", UP: "up", DOWN: "down", R: "r", L: "l"}

DIRECTIONS = [UP, DOWN, LEFT, RIGHT, UP | LEFT, UP | RIGHT, DOWN | LEFT, DOWN | RIGHT]
ACTIONS = [A, B, L, R]


@lru_cache(maxsize=None)
def mask_to_joypad(mask: int) -> JoypadState:
    return JoypadState(**{name: bool(mask & bit) for bit, name in _NAMES.items()})


def mirror(mask: int) -> int:
    """Swap left and right, so a move written facing right works facing left."""
    swapped = mask & ~(LEFT | RIGHT)
    if mask & RIGHT:
        swapped |= LEFT
    if mask & LEFT:
        swapped |= RIGHT
    return swapped


def macro_frames(steps, mirrored: bool = False) -> list[int]:
    """Expand [(mask, frames), ...] into one mask per frame."""
    out = []
    for mask, frames in steps:
        out.extend([mirror(mask) if mirrored else mask] * frames)
    return out


def _with_button(motion, button):
    *head, (last, _) = motion
    return [*head, (last | button, 4)]


_QCF = [(DOWN, 3), (DOWN | RIGHT, 3), (RIGHT, 3)]
_QCB = [(DOWN, 3), (DOWN | LEFT, 3), (LEFT, 3)]
_DP = [(RIGHT, 3), (DOWN, 3), (DOWN | RIGHT, 3)]
_TAP = 0, 4  # short release between presses

# Common special-move and combo inputs, written facing right.
GENERIC_MACROS = [
    *[_with_button(m, b) for m in (_QCF, _QCB, _DP) for b in (A, B)],
    [(RIGHT, 3), (0, 3), (RIGHT, 20)],  # dash
    [(LEFT, 3), (0, 3), (LEFT, 12)],  # back dash
    [(DOWN, 30), (UP | A, 4)],  # charge down, up + button
    [(DOWN, 30), (UP | B, 4)],
    [(LEFT, 40), (RIGHT | A, 4)],  # charge back, forward + button
    [(A, 45), (0, 2)],  # hold and release
    [(B, 45), (0, 2)],
    [(A, 3), _TAP, (A, 3), _TAP, (A, 3)],  # combo strings
    [(A, 3), _TAP, (A, 3), _TAP, (B, 3)],
    [(B, 3), _TAP, (B, 3), _TAP, (B, 3)],
    [(A, 4), (0, 8), (B, 4)],  # jump, then attack in the air
    [(B, 4), (0, 8), (A, 4)],
    [(UP, 4), (UP | A, 4)],
    [(DOWN | A, 4)], [(DOWN | B, 4)], [(UP | B, 4)], [(RIGHT | B, 4)],
    [(R | A, 4)], [(R | B, 4)], [(L | A, 4)], [(L | R, 6)], [(R, 20)], [(L, 20)],
]


def _drop_opposites(mask: int, rng: random.Random) -> int:
    """A real d-pad cannot press left+right or up+down together."""
    for a, b in ((LEFT, RIGHT), (UP, DOWN)):
        if mask & a and mask & b:
            mask &= ~rng.choice((a, b))
    return mask


class InputPolicy:
    MODES = ("walk", "mash", "chord", "macro")
    MODE_WEIGHTS = (0.35, 0.25, 0.2, 0.2)

    def __init__(self, rng: random.Random, macros=GENERIC_MACROS,
                 start_chance: float = 0.03, select_chance: float = 0.005):
        self._rng = rng
        self.macros = macros
        self.start_chance = start_chance
        self.select_chance = select_chance

    def burst(self, frames: int) -> list[int]:
        out = []
        while len(out) < frames:
            out.extend(self._segment())
        return out[:frames]

    def _segment(self) -> list[int]:
        r = self._rng
        if r.random() < self.start_chance:
            return [START] * 3 + [0] * 20
        if r.random() < self.select_chance:
            return [SELECT] * 3 + [0] * 20

        mode = r.choices(self.MODES, self.MODE_WEIGHTS)[0]
        if mode == "walk":
            mask = r.choice(DIRECTIONS)
            if r.random() < 0.3:
                mask |= r.choice(ACTIONS)
            return [mask] * r.randint(12, 90)

        if mode == "mash":
            out = []
            for _ in range(r.randint(3, 10)):
                mask = 0
                for button in r.sample(ACTIONS, 2 if r.random() < 0.3 else 1):
                    mask |= button
                if r.random() < 0.4:
                    mask |= r.choice(DIRECTIONS)
                out += [mask] * r.randint(2, 6) + [0] * r.randint(1, 4)
            return out

        if mode == "chord":
            mask = 0
            for button in r.sample(ACTIONS + [UP, DOWN, LEFT, RIGHT], r.randint(2, 3)):
                mask |= button
            mask = _drop_opposites(mask, r)
            return [mask] * r.randint(4, 20) + [0] * r.randint(1, 6)

        steps = r.choice(self.macros)
        return macro_frames(steps, mirrored=r.random() < 0.5) + [0] * r.randint(2, 8)
