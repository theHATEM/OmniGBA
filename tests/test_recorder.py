import json

import cv2
import numpy as np

from dataset.recorder import FrameRecorder


def _frame(seed):
    return np.random.default_rng(seed).integers(0, 256, (160, 240, 3), dtype=np.uint8)


def test_saves_png_and_metadata(tmp_path):
    rec = FrameRecorder(tmp_path)
    frame = _frame(0)
    assert rec.offer(frame, group=b"g", cell=7, new_graphics=2)
    saved = cv2.cvtColor(cv2.imread(str(tmp_path / "frames" / "000000.png")), cv2.COLOR_BGR2RGB)
    assert np.array_equal(saved, frame)
    meta = json.loads((tmp_path / "frames.jsonl").read_text().splitlines()[0])
    assert meta == {"file": "frames/000000.png", "cell": 7, "new_graphics": 2}
    assert rec.count == 1


def test_rejects_exact_duplicate_even_when_forced(tmp_path):
    rec = FrameRecorder(tmp_path)
    frame = _frame(0)
    rec.offer(frame, group=0)
    rec.offer(_frame(1), group=1)
    assert not rec.offer(frame.copy(), group=2, force=True)


def test_rejects_near_duplicate_of_last_saved_frame(tmp_path):
    rec = FrameRecorder(tmp_path, min_diff=2.0)
    frame = _frame(0)
    rec.offer(frame, group=0)
    nudged = frame.copy()
    nudged[0, 0] ^= 1
    assert not rec.offer(nudged, group=1)
    assert rec.offer(nudged, group=1, force=True)


def test_caps_frames_per_group_unless_forced(tmp_path):
    rec = FrameRecorder(tmp_path, per_group_cap=2)
    assert rec.offer(_frame(0), group=5)
    assert rec.offer(_frame(1), group=5)
    assert not rec.offer(_frame(2), group=5)
    assert rec.offer(_frame(3), group=6)
    assert rec.offer(_frame(4), group=5, force=True)


def test_rejects_flat_frames(tmp_path):
    rec = FrameRecorder(tmp_path)
    black = np.zeros((160, 240, 3), dtype=np.uint8)
    assert not rec.offer(black, group=0, force=True)
