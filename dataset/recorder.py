"""
Saves the frames worth keeping as lossless PNGs, plus one JSON line of metadata each.

A frame is skipped when it is:
  - an exact copy of any frame saved before,
  - almost all one colour (black screens, fades),
  - nearly identical to the previously saved frame,
  - from a group (a fine screen fingerprint) that already has per_group_cap frames.
force=True (used when new graphics appeared) overrides the last two rules.
"""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


class FrameRecorder:
    def __init__(self, out_dir, per_group_cap: int = 3, min_diff: float = 2.0, min_std: float = 4.0):
        self.out_dir = Path(out_dir)
        (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)
        self.per_group_cap = per_group_cap
        self.min_diff = min_diff  # mean absolute difference, 0..255
        self.min_std = min_std
        self.count = 0
        self._hashes = set()
        self._per_group: dict = {}
        self._last = None
        self._meta = open(self.out_dir / "frames.jsonl", "a", encoding="utf-8")

    def offer(self, frame: np.ndarray, group, force: bool = False, **meta) -> bool:
        digest = hashlib.blake2b(frame.tobytes(), digest_size=16).digest()
        if digest in self._hashes or frame.std() < self.min_std:
            return False
        if not force:
            if self._per_group.get(group, 0) >= self.per_group_cap:
                return False
            if self._last is not None:
                diff = np.abs(frame.astype(np.int16) - self._last).mean()
                if diff < self.min_diff:
                    return False

        name = f"frames/{self.count:06d}.png"
        cv2.imwrite(str(self.out_dir / name), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self._meta.write(json.dumps({"file": name, **meta}) + "\n")
        self._meta.flush()

        self._hashes.add(digest)
        self._per_group[group] = self._per_group.get(group, 0) + 1
        self._last = frame.astype(np.int16)
        self.count += 1
        return True

    def close(self) -> None:
        self._meta.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
