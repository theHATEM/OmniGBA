"""
Go-Explore style archive of saved emulator states ("cells") to explore from.

A cell added without a parent starts a new root: the power-on state, or a stage
reached by a scripted route. Every other cell belongs to the root its parent
belongs to. To pick where the next burst starts, the archive first picks a root
(all roots equally often, so a menu with hundreds of credit screens cannot crowd
out a stage), then a cell of that root. Within a root, a cell is picked less often
when it has been picked many times already, and when its recent bursts stopped
finding anything new, so exploration keeps moving to the frontier.
"""

import math
import random
from dataclasses import dataclass


@dataclass(eq=False)
class Cell:
    id: int
    key: bytes
    state: bytes  # compressed emulator save state
    parent: int | None  # cell the explorer started from when it found this one
    root: int  # id of the root cell this one was reached from
    chosen: int = 0
    since_new: int = 0  # bursts from this cell in a row that found nothing new

    @property
    def weight(self) -> float:
        return 1.0 / math.sqrt((1 + self.chosen) * (1 + self.since_new))


class Archive:
    def __init__(self, rng: random.Random):
        self._rng = rng
        self._cells: dict[bytes, Cell] = {}
        self._order: list[Cell] = []
        self._by_root: dict[int, list[Cell]] = {}

    def __len__(self) -> int:
        return len(self._order)

    def __contains__(self, key: bytes) -> bool:
        return key in self._cells

    @property
    def cells(self) -> list[Cell]:
        return self._order

    @property
    def roots(self) -> list[int]:
        return list(self._by_root)

    def get(self, key: bytes) -> Cell:
        return self._cells[key]

    def add(self, key: bytes, state: bytes, parent: int | None = None) -> Cell | None:
        """Store a new cell; returns None (and keeps the old state) if the key exists."""
        if key in self._cells:
            return None
        cell_id = len(self._order)
        root = cell_id if parent is None else self._order[parent].root
        cell = Cell(cell_id, key, state, parent, root)
        self._cells[key] = cell
        self._order.append(cell)
        self._by_root.setdefault(root, []).append(cell)
        return cell

    def choose(self) -> Cell:
        candidates = self._by_root[self._rng.choice(self.roots)]
        cell = self._rng.choices(candidates, [c.weight for c in candidates])[0]
        cell.chosen += 1
        return cell

    def finish_burst(self, cell: Cell, found_new: bool) -> None:
        cell.since_new = 0 if found_new else cell.since_new + 1
