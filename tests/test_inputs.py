import random

from dataset.inputs import (
    A, B, DOWN, LEFT, RIGHT, SELECT, START, UP,
    InputPolicy, macro_frames, mask_to_joypad, mirror,
)


def _runs(masks):
    """(mask, length) for each run of identical consecutive masks."""
    runs = []
    for m in masks:
        if runs and runs[-1][0] == m:
            runs[-1][1] += 1
        else:
            runs.append([m, 1])
    return runs


def test_burst_has_requested_length():
    policy = InputPolicy(random.Random(1))
    assert len(policy.burst(300)) == 300


def test_burst_is_reproducible_with_same_seed():
    assert InputPolicy(random.Random(5)).burst(500) == InputPolicy(random.Random(5)).burst(500)


def test_burst_only_uses_gba_buttons():
    masks = InputPolicy(random.Random(2)).burst(5000)
    assert all(0 <= m < (1 << 10) for m in masks)


def test_start_and_select_are_only_pressed_alone():
    masks = InputPolicy(random.Random(3), start_chance=0.2).burst(20000)
    assert any(m == START for m in masks)
    for m in masks:
        if m & (START | SELECT):
            assert m in (START, SELECT)


def test_burst_mixes_quick_taps_and_long_holds():
    runs = _runs(InputPolicy(random.Random(4)).burst(20000))
    pressed = [length for mask, length in runs if mask]
    assert any(length <= 6 for length in pressed)
    assert any(length >= 30 for length in pressed)


def test_burst_presses_several_buttons_at_once():
    masks = InputPolicy(random.Random(6)).burst(20000)
    assert any(bin(m & (A | B)).count("1") == 2 for m in masks)


def test_mirror_swaps_left_and_right_only():
    assert mirror(RIGHT | A) == LEFT | A
    assert mirror(LEFT | DOWN) == RIGHT | DOWN
    assert mirror(UP | B) == UP | B


def test_macro_frames_expands_and_mirrors_steps():
    steps = [(DOWN, 2), (DOWN | RIGHT, 1), (RIGHT | A, 3)]
    assert macro_frames(steps) == [DOWN, DOWN, DOWN | RIGHT, RIGHT | A, RIGHT | A, RIGHT | A]
    assert macro_frames(steps, mirrored=True)[2] == DOWN | LEFT


def test_mask_to_joypad_sets_named_buttons():
    pad = mask_to_joypad(A | LEFT)
    assert pad.a and pad.left
    assert not (pad.b or pad.right or pad.start)
