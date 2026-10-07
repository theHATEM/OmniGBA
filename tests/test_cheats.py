from dataset.cheats import GAMES, cheats_for, rom_game_code


def _fake_rom(path, code: bytes):
    data = bytearray(0xC0)
    data[0xAC:0xB0] = code
    path.write_bytes(bytes(data))
    return path


def test_rom_game_code_reads_header(tmp_path):
    assert rom_game_code(_fake_rom(tmp_path / "x.gba", b"BT4E")) == "BT4E"


def test_known_games_have_names_and_cheats():
    for code in ("BT4E", "BN2E"):
        game = GAMES[code]
        assert game.name
        assert game.cheats


def test_cheat_codes_are_libretro_strings():
    for game in GAMES.values():
        for description, code in game.cheats:
            assert description
            parts = code.split("+")
            assert all(len(p) in (4, 8) and int(p, 16) >= 0 for p in parts)


def test_unknown_game_has_no_cheats():
    assert cheats_for("ZZZZ") == []


def _write_addresses(code):
    """Addresses written by a CodeBreaker code (type 3: 8-bit write, type 8: 16-bit write)."""
    parts = code.split("+")
    return {int(parts[i], 16) & 0x0FFFFFFF for i in range(0, len(parts), 2) if parts[i][0] in "38"}


def test_naruto_cheats_leave_opponents_beatable():
    # 0x0300454A-D hold the four ninjas' energy; mission 1's opponent is Rock Lee,
    # so locking those makes the first mission impossible to win
    written = set().union(*(_write_addresses(code) for _, code in GAMES["BN2E"].cheats))
    assert not written & set(range(0x0300454A, 0x0300454E))
