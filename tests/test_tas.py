import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.recorder import FrameRecorder
from dataset.tas import (MovieError, bizhawk_command, check_movie, convert, dumped_frames, main_version,
                         read_movie, rom_index, select_frames, write_config)

GBA_LOG_KEY = "LogKey:#Tilt X|Tilt Y|Tilt Z|Light Sensor|Up|Down|Left|Right|Start|Select|B|A|L|R|Power|"
MGBA_SYNC = ('{"o":{"$type":"BizHawk.Emulation.Cores.Nintendo.GBA.MGBAHawk+SyncSettings, '
             'BizHawk.Emulation.Cores","SkipBios":true}}')


def bk2_bytes(frames=3, inputs=None, **header) -> bytes:
    """A BizHawk GBA movie; `inputs` are the button columns (Up..Power) of each frame."""
    head = {"MovieVersion": "BizHawk v2.0", "emuVersion": "Version 2.11.1", "Platform": "GBA",
            "GameName": "Test", "SHA1": "AB" * 20, "Core": "mGBA", **header}
    inputs = inputs or ["..........."] * frames
    log = ["[Input]", GBA_LOG_KEY] + [f"|    0,    0,    0,    0,{row}|" for row in inputs] + ["[/Input]"]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Header.txt", "\r\n".join(f"{k} {v}" for k, v in head.items()) + "\r\n")
        z.writestr("Input Log.txt", "\r\n".join(log) + "\r\n")
        z.writestr("SyncSettings.json", MGBA_SYNC)
    return buf.getvalue()


def write_png(path, frame):
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def noise(seed):
    return np.random.default_rng(seed).integers(0, 256, (160, 240, 3), dtype=np.uint8)


def test_reads_header_and_counts_input_frames(tmp_path):
    path = tmp_path / "run.bk2"
    path.write_bytes(bk2_bytes(frames=5, SHA1="ab" * 20, GBA_Firmware_Bios="cd" * 20,
                               emuVersion="Version 2.9.1"))
    movie = read_movie(path)
    assert movie.frames == 5
    assert movie.sha1 == "AB" * 20
    assert movie.bios_sha1 == "CD" * 20
    assert movie.core == "mGBA"
    assert movie.emu_version == "2.9.1"


def test_movie_without_bios_line_has_no_bios_hash(tmp_path):
    path = tmp_path / "run.bk2"
    path.write_bytes(bk2_bytes())
    assert read_movie(path).bios_sha1 is None


def test_reads_movie_from_tasvideos_download_zip(tmp_path):
    path = tmp_path / "7000M.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("author-game.bk2", bk2_bytes(frames=4))
    movie = read_movie(path)
    assert movie.frames == 4
    assert zipfile.ZipFile(io.BytesIO(movie.bk2)).read("Header.txt").startswith(b"MovieVersion")


@pytest.mark.parametrize("name", ["author-game.vbm", "author-game.gbmv"])
def test_rejects_movies_from_other_emulators(tmp_path, name):
    path = tmp_path / "download.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, b"not a bk2")
    with pytest.raises(MovieError, match="bk2"):
        read_movie(path)


def test_rejects_file_that_is_not_a_zip(tmp_path):
    path = tmp_path / "old.vbm"
    path.write_bytes(b"VBM\x1a" + bytes(100))
    with pytest.raises(MovieError, match="bk2"):
        read_movie(path)


def make_rom(path, code=b"BT4E", fill=b"\x00"):
    data = fill * 0xAC + code + fill * 0x100
    path.write_bytes(data)
    return hashlib.sha1(data).hexdigest().upper()


def test_rom_index_maps_sha1_to_rom(tmp_path):
    (tmp_path / "sub").mkdir()
    a = make_rom(tmp_path / "a.gba")
    b = make_rom(tmp_path / "sub" / "b.gba", fill=b"\x01")
    assert rom_index(tmp_path) == {a: tmp_path / "a.gba", b: tmp_path / "sub" / "b.gba"}


def movie_for(tmp_path, **header):
    path = tmp_path / "run.bk2"
    path.write_bytes(bk2_bytes(**header))
    return read_movie(path)


def test_check_movie_finds_rom_and_accepts_matching_bios(tmp_path):
    sha1 = make_rom(tmp_path / "game.gba")
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"bios")
    movie = movie_for(tmp_path, SHA1=sha1, GBA_Firmware_Bios=hashlib.sha1(b"bios").hexdigest())
    assert check_movie(movie, rom_index(tmp_path), bios) == tmp_path / "game.gba"


def test_check_movie_needs_the_rom(tmp_path):
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"bios")
    with pytest.raises(MovieError, match="no ROM"):
        check_movie(movie_for(tmp_path), {}, bios)


def test_check_movie_needs_the_bios_the_movie_was_made_with(tmp_path):
    sha1 = make_rom(tmp_path / "game.gba")
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"other bios")
    movie = movie_for(tmp_path, SHA1=sha1, GBA_Firmware_Bios=hashlib.sha1(b"bios").hexdigest())
    with pytest.raises(MovieError, match="BIOS"):
        check_movie(movie, rom_index(tmp_path), bios)


def test_check_movie_only_takes_mgba_gba_movies(tmp_path):
    sha1 = make_rom(tmp_path / "game.gba")
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"bios")
    with pytest.raises(MovieError, match="mGBA"):
        check_movie(movie_for(tmp_path, SHA1=sha1, Core="VBA-Next"), rom_index(tmp_path), bios)
    with pytest.raises(MovieError, match="GBA"):
        check_movie(movie_for(tmp_path, SHA1=sha1, Platform="SNES"), rom_index(tmp_path), bios)


def test_main_version_drops_build_hash():
    assert main_version("2.11.1+bdddf4a58aa1a022afb11dc73294a81a5aa7bbd5") == "2.11.1"
    assert main_version("2.10") == "2.10"


def test_config_sets_bios_speed_and_version(tmp_path):
    write_config(tmp_path / "config.ini", "2.11.1", Path(r"C:\bios\gba_bios.bin"))
    config = json.loads((tmp_path / "config.ini").read_text())
    assert config["LastWrittenFrom"] == "2.11.1"  # otherwise BizHawk asks about the version first
    assert config["FirmwareUserSpecifications"] == {"GBA+Bios": r"C:\bios\gba_bios.bin"}
    assert config["Unthrottled"] is True
    assert config["VideoWriterAudioSync"] is False  # it would drop or repeat frames to match the sound


def test_bizhawk_command_dumps_png_frames_and_quits():
    cmd = bizhawk_command(Path("EmuHawk.exe"), Path("c.ini"), Path("m.bk2"), Path("game.gba"), Path("d/f.png"))
    assert cmd[0] == "EmuHawk.exe"
    assert cmd[-1] == "game.gba"  # BizHawk wants the ROM last
    flags = dict(zip(cmd, cmd[1:]))
    assert flags["--config"] == "c.ini"
    assert flags["--movie"] == "m.bk2"
    assert flags["--dump-type"] == "imagesequence"
    assert flags["--dump-name"] == str(Path("d/f.png"))
    assert "--dump-close" in cmd


def test_dumped_frames_sorted_by_frame_number(tmp_path):
    for i in (0, 10, 2, 1):
        write_png(tmp_path / f"f_{i}.png", noise(i))
    assert [i for i, _ in dumped_frames(tmp_path)] == [0, 1, 2, 10]


def test_select_frames_skips_repeats_and_records_movie_frame(tmp_path):
    dump = tmp_path / "dump"
    dump.mkdir()
    frames = [noise(0), noise(0), np.zeros((160, 240, 3), np.uint8), noise(1)]
    for i, frame in enumerate(frames):
        write_png(dump / f"f_{i}.png", frame)
    with FrameRecorder(tmp_path / "out", per_group_cap=10) as recorder:
        select_frames(dumped_frames(dump), recorder)
    rows = [json.loads(line) for line in (tmp_path / "out" / "frames.jsonl").read_text().splitlines()]
    assert [row["movie_frame"] for row in rows] == [0, 3]
    saved = cv2.cvtColor(cv2.imread(str(tmp_path / "out" / rows[1]["file"])), cv2.COLOR_BGR2RGB)
    assert np.array_equal(saved, frames[3])


def fake_dump(bizhawk, version, movie, rom, bios, work_dir):
    dump = Path(work_dir) / "dump"
    dump.mkdir()
    for i in range(movie.frames):
        write_png(dump / f"f_{i}.png", noise(i))
    return dumped_frames(dump)


def test_convert_writes_frames_and_run_info(tmp_path):
    roms = tmp_path / "roms"
    roms.mkdir()
    sha1 = make_rom(roms / "game.gba")
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"bios")
    movie = tmp_path / "author-game.bk2"
    movie.write_bytes(bk2_bytes(frames=3, SHA1=sha1, emuVersion="Version 2.9.1"))
    out = tmp_path / "out"

    out_dir = convert(movie, Path("EmuHawk.exe"), "2.11.1", bios, rom_index(roms), out, dump=fake_dump)

    assert out_dir == out / "dbgt" / "tas_author-game"
    assert len(list((out_dir / "frames").glob("*.png"))) == 3
    run = json.loads((out_dir / "run.json").read_text())
    assert run["source"] == "tas"
    assert run["movie"] == str(movie)
    assert run["rom"] == str(roms / "game.gba")
    assert run["game_code"] == "BT4E"
    assert run["bizhawk"] == "2.11.1"
    assert run["movie_bizhawk"] == "2.9.1"
    assert run["frames_dumped"] == 3
    assert run["frames_saved"] == 3
    assert [p.name for p in out.iterdir()] == ["dbgt"]  # the dump folder is gone


def test_convert_skips_finished_movies_and_redoes_interrupted_ones(tmp_path):
    roms = tmp_path / "roms"
    roms.mkdir()
    sha1 = make_rom(roms / "game.gba")
    bios = tmp_path / "bios.bin"
    bios.write_bytes(b"bios")
    movie = tmp_path / "m.bk2"
    movie.write_bytes(bk2_bytes(frames=2, SHA1=sha1))
    out = tmp_path / "out"
    leftover = out / "dbgt" / "tas_m" / "frames"
    leftover.mkdir(parents=True)
    write_png(leftover / "000000.png", noise(9))

    out_dir = convert(movie, Path("EmuHawk.exe"), "2.11.1", bios, rom_index(roms), out, dump=fake_dump)
    assert len((out_dir / "frames.jsonl").read_text().splitlines()) == 2
    assert len(list((out_dir / "frames").glob("*.png"))) == 2

    assert convert(movie, Path("EmuHawk.exe"), "2.11.1", bios, rom_index(roms), out, dump=fake_dump) is None


# With real BizHawk: set GBA_BIZHAWK to EmuHawk.exe and GBA_BIOS to a GBA BIOS file (any BIOS
# works here, the test movie does not name one). BizHawk's window shows for a few seconds.
BIZHAWK = os.environ.get("GBA_BIZHAWK")
BIOS = os.environ.get("GBA_BIOS")


@pytest.mark.skipif(not BIZHAWK or not BIOS, reason="set GBA_BIZHAWK and GBA_BIOS")
def test_real_bizhawk_dumps_every_movie_frame_at_native_size(tmp_path):
    from dataset.cheats import rom_game_code
    from dataset.tas import bizhawk_version, dump_movie

    roms = sorted(Path("roms").rglob("*.gba")) if Path("roms").exists() else []
    rom = next((r for r in roms if rom_game_code(r) == "BT4E"), None)
    if rom is None:
        pytest.skip("needs the BT4E ROM")
    sha1 = hashlib.sha1(rom.read_bytes()).hexdigest().upper()
    # press Start now and then to get past the logos
    inputs = ["....S......" if i % 60 < 3 else "..........." for i in range(600)]
    path = tmp_path / "test.bk2"
    path.write_bytes(bk2_bytes(inputs=inputs, SHA1=sha1))
    work = tmp_path / "work"
    work.mkdir()

    frames = dump_movie(Path(BIZHAWK), bizhawk_version(Path(BIZHAWK)), read_movie(path), rom, Path(BIOS), work)

    assert [i for i, _ in frames] == list(range(600))
    first, last = (cv2.imread(str(frames[i][1])) for i in (0, -1))
    assert first.shape == last.shape == (160, 240, 3)
    assert last.std() > 4  # a picture, not a blank screen
