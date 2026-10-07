import random

from dataset.archive import Archive


def test_add_new_key_creates_cell_with_sequential_ids():
    archive = Archive(random.Random(0))
    a = archive.add(b"a", b"state-a")
    b = archive.add(b"b", b"state-b", parent=a.id)
    assert (a.id, b.id) == (0, 1)
    assert b.parent == a.id
    assert len(archive) == 2
    assert b"a" in archive


def test_add_existing_key_keeps_first_state():
    archive = Archive(random.Random(0))
    archive.add(b"a", b"first")
    assert archive.add(b"a", b"second") is None
    assert archive.get(b"a").state == b"first"


def test_choose_counts_selections():
    archive = Archive(random.Random(0))
    cell = archive.add(b"a", b"s")
    assert archive.choose() is cell
    assert cell.chosen == 1


def test_choose_prefers_rarely_chosen_cells():
    archive = Archive(random.Random(0))
    worn = archive.add(b"worn", b"s")
    fresh = archive.add(b"fresh", b"s")
    worn.chosen = 10_000
    picks = [archive.choose() for _ in range(50)]
    assert picks.count(fresh) >= 45


def test_cells_whose_bursts_find_nothing_are_chosen_less():
    archive = Archive(random.Random(0))
    stale = archive.add(b"stale", b"s")
    lively = archive.add(b"lively", b"s")
    for _ in range(20):
        archive.finish_burst(stale, found_new=False)
        archive.finish_burst(lively, found_new=True)
    picks = [archive.choose() for _ in range(50)]
    assert picks.count(lively) > 2 * picks.count(stale)
