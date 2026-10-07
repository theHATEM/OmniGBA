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
    fresh = archive.add(b"fresh", b"s", parent=worn.id)
    worn.chosen = 10_000
    picks = [archive.choose() for _ in range(50)]
    assert picks.count(fresh) >= 45


def test_cells_whose_bursts_find_nothing_are_chosen_less():
    archive = Archive(random.Random(0))
    stale = archive.add(b"stale", b"s")
    lively = archive.add(b"lively", b"s", parent=stale.id)
    for _ in range(20):
        archive.finish_burst(stale, found_new=False)
        archive.finish_burst(lively, found_new=True)
    picks = [archive.choose() for _ in range(50)]
    assert picks.count(lively) > 2 * picks.count(stale)


def test_cells_without_parent_start_new_roots():
    archive = Archive(random.Random(0))
    menu = archive.add(b"menu", b"s")
    stage = archive.add(b"stage", b"s")
    child = archive.add(b"stage-2", b"s", parent=stage.id)
    assert (menu.root, stage.root, child.root) == (menu.id, stage.id, stage.id)
    assert archive.roots == [menu.id, stage.id]


def test_choose_shares_bursts_evenly_between_roots():
    # a menu full of credits text finds hundreds of scenes; a stage only a few
    archive = Archive(random.Random(0))
    menu = archive.add(b"menu", b"s")
    for i in range(300):
        archive.add(b"credits%d" % i, b"s", parent=menu.id)
    stage = archive.add(b"stage", b"s")
    picks = [archive.choose().root for _ in range(400)]
    assert 150 < picks.count(stage.id) < 250
