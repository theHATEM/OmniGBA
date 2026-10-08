import hashlib
import io
import json
import os
import re
import struct
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from dataset.recorder import FrameRecorder
from dataset.tas import (Emulator, MovieError, check_movie, convert, dumped_frames, emulator_command, read_movie,
                         rom_index, select_frames, write_config)

MGBA_LOG_KEY = "LogKey:#Tilt X|Tilt Y|Tilt Z|Light Sensor|Up|Down|Left|Right|Start|Select|B|A|L|R|Power|"
GBAHAWK_LOG_KEY = "LogKey:#P1 Up|P1 Down|P1 Left|P1 Right|P1 Start|P1 Select|P1 B|P1 A|P1 L|P1 R|P1 Power|"
GBA_BUTTONS = ["Up", "Down", "Left", "Right", "Start", "Select", "B", "A", "L", "R", "Power"]
SYNC = {"mGBA": '{"o":{"$type":"BizHawk.Emulation.Cores.Nintendo.GBA.MGBAHawk+SyncSettings, '
                'BizHawk.Emulation.Cores","SkipBios":true}}',
        "GBAHawk": '{"o":{"$type":"BizHawk.Emulation.Cores.Nintendo.GBA.GBAHawk+GBASyncSettings, '
                   'BizHawk.Emulation.Cores"}}'}


def bk2_bytes(frames=3, inputs=None, core="mGBA", **header) -> bytes:
    """A BizHawk (core mGBA) or GBAHawk movie; `inputs` are the Up..Power columns of each frame."""
    head = {"MovieVersion": "BizHawk v2.0", "emuVersion": "Version 2.11.1", "Platform": "GBA",
            "GameName": "Test", "SHA1": "AB" * 20, "Core": core, **header}
    inputs = inputs or ["..........."] * frames
    if core == "mGBA":
        log = [MGBA_LOG_KEY] + [f"|    0,    0,    0,    0,{row}|" for row in inputs]
    else:
        log = [GBAHAWK_LOG_KEY] + [f"|{row}|" for row in inputs]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("Header.txt", "\r\n".join(f"{k} {v}" for k, v in head.items()) + "\r\n")
        z.writestr("Input Log.txt", "\r\n".join(["[Input]"] + log + ["[/Input]"]) + "\r\n")
        z.writestr("SyncSettings.json", SYNC.get(core, "{}"))
    return buf.getvalue()


def vbm_bytes(inputs, title=b"DBGT", code=b"BT4E", start=0, options=0b11, controllers=0b1) -> bytes:
    """A VisualBoyAdvance-rr movie: 2 bytes of buttons per frame and controller."""
    header = struct.pack("<4sIIIIBBBB12s12sBBH4sII", b"VBM\x1a", 1, 0, len(inputs), 7, start, controllers,
                         0b1, options, bytes(12), title.ljust(12, b"\0"), 1, 0, 0, code, 0, 0x100)
    text = b"author".ljust(64, b"\0") + b"description".ljust(128, b"\0")
    others = bin(controllers).count("1") - 1
    return header + text + b"".join(struct.pack("<H", m) + bytes(2 * others) for m in inputs)


def input_rows(movie):
    log = zipfile.ZipFile(io.BytesIO(movie.bk2)).read("Input Log.txt").decode()
    return [line for line in log.splitlines() if line.startswith("|")]


def pressed(row):
    buttons = row.strip("|").split(",")[-1]
    return {name for name, c in zip(GBA_BUTTONS, buttons) if c != "."}


def sync_settings(movie):
    return json.loads(zipfile.ZipFile(io.BytesIO(movie.bk2)).read("SyncSettings.json"))["o"]


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


@pytest.mark.parametrize("name", ["author-game.bk2", "author-game.gbmv"])
def test_reads_movie_from_tasvideos_download_zip(tmp_path, name):
    path = tmp_path / "7000M.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, bk2_bytes(frames=4))
    movie = read_movie(path)
    assert movie.frames == 4
    assert zipfile.ZipFile(io.BytesIO(movie.bk2)).read("Header.txt").startswith(b"MovieVersion")


def test_reads_gbahawk_movie(tmp_path):
    path = tmp_path / "run.gbmv"
    path.write_bytes(bk2_bytes(frames=6, core="GBAHawk"))
    movie = read_movie(path)
    assert movie.core == "GBAHawk"
    assert movie.frames == 6


def test_rejects_unknown_movie_inside_zip(tmp_path):
    path = tmp_path / "download.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("author-game.fm2", b"not a GBA movie")
    with pytest.raises(MovieError, match="bk2"):
        read_movie(path)


def test_rejects_file_that_is_no_known_movie(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_bytes(b"hello")
    with pytest.raises(MovieError, match="bk2"):
        read_movie(path)


def test_vbm_becomes_mgba_movie_with_same_buttons(tmp_path):
    path = tmp_path / "old.vbm"
    path.write_bytes(vbm_bytes([0x0001, 0x0048, 0x0300, 0x0000, 0x0800]))
    movie = read_movie(path)
    assert movie.core == "mGBA"
    assert movie.frames == 5
    assert [pressed(row) for row in input_rows(movie)] == [
        {"A"}, {"Start", "Up"}, {"R", "L"}, set(), {"Power"}]  # VBA's reset becomes a power cycle
    assert movie.header["GameCode"] == "BT4E"
    assert movie.header["GameName"] == "DBGT"
    assert movie.sha1 == ""  # a .vbm names its ROM by game code and title only


def test_vbm_inside_zip_skips_other_controllers(tmp_path):
    path = tmp_path / "download.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("old.vbm", vbm_bytes([0x0002, 0x0010], controllers=0b11))
    assert [pressed(row) for row in input_rows(read_movie(path))] == [{"B"}, {"Right"}]


@pytest.mark.parametrize("options, skip_bios", [(0b01, False), (0b11, True), (0b00, True)])
def test_vbm_keeps_the_bios_intro_only_when_it_was_recorded(tmp_path, options, skip_bios):
    path = tmp_path / "old.vbm"
    path.write_bytes(vbm_bytes([0], options=options))
    assert sync_settings(read_movie(path))["SkipBios"] is skip_bios


def test_vbm_starting_from_savestate_is_rejected(tmp_path):
    path = tmp_path / "old.vbm"
    path.write_bytes(vbm_bytes([0], start=0b01))
    with pytest.raises(MovieError, match="savestate"):
        read_movie(path)


def make_rom(path, code=b"BT4E", title=b"DBGT", fill=b"\x00"):
    data = fill * 0xA0 + title.ljust(12, b"\0") + code + fill * 0x100
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


@pytest.fixture
def bios(tmp_path):
    path = tmp_path / "bios.bin"
    path.write_bytes(b"bios")
    return path


def test_check_movie_finds_rom_and_accepts_matching_bios(tmp_path, bios):
    sha1 = make_rom(tmp_path / "game.gba")
    movie = movie_for(tmp_path, SHA1=sha1, GBA_Firmware_Bios=hashlib.sha1(b"bios").hexdigest())
    assert check_movie(movie, rom_index(tmp_path), bios) == tmp_path / "game.gba"


def test_check_movie_needs_the_rom(tmp_path, bios):
    with pytest.raises(MovieError, match="no ROM"):
        check_movie(movie_for(tmp_path), {}, bios)


def test_check_movie_needs_the_bios_the_movie_was_made_with(tmp_path):
    sha1 = make_rom(tmp_path / "game.gba")
    other = tmp_path / "other.bin"
    other.write_bytes(b"other bios")
    movie = movie_for(tmp_path, SHA1=sha1, GBA_Firmware_Bios=hashlib.sha1(b"bios").hexdigest())
    with pytest.raises(MovieError, match="BIOS"):
        check_movie(movie, rom_index(tmp_path), other)


def test_check_movie_takes_mgba_and_gbahawk_gba_movies_only(tmp_path, bios):
    sha1 = make_rom(tmp_path / "game.gba")
    assert check_movie(movie_for(tmp_path, SHA1=sha1, core="GBAHawk"), rom_index(tmp_path), bios)
    with pytest.raises(MovieError, match="VBA-Next"):
        check_movie(movie_for(tmp_path, SHA1=sha1, Core="VBA-Next"), rom_index(tmp_path), bios)
    with pytest.raises(MovieError, match="GBA"):
        check_movie(movie_for(tmp_path, SHA1=sha1, Platform="SNES"), rom_index(tmp_path), bios)


def test_check_movie_finds_vbm_rom_by_game_code_and_title(tmp_path, bios):
    make_rom(tmp_path / "other.gba", code=b"AXVE", title=b"POKEMON RUBY")
    make_rom(tmp_path / "dbgt.gba")
    path = tmp_path / "old.vbm"
    path.write_bytes(vbm_bytes([0]))
    assert check_movie(read_movie(path), rom_index(tmp_path), bios) == tmp_path / "dbgt.gba"
    path.write_bytes(vbm_bytes([0], code=b"ZZZE"))
    with pytest.raises(MovieError, match="ZZZE"):
        check_movie(read_movie(path), rom_index(tmp_path), bios)


def test_config_sets_bios_speed_version_and_save_folder(tmp_path):
    write_config(tmp_path / "config.ini", "2.11.1", Path(r"C:\bios\gba_bios.bin"), Path(r"C:\work\GBA"))
    config = json.loads((tmp_path / "config.ini").read_text())
    assert config["LastWrittenFrom"] == "2.11.1"  # otherwise the emulator asks about the version first
    assert config["FirmwareUserSpecifications"] == {"GBA+Bios": r"C:\bios\gba_bios.bin"}
    assert config["Unthrottled"] is True
    assert config["VideoWriterAudioSync"] is False  # BizHawk would drop or repeat frames to match the sound
    # save RAM and states go to the work folder, not into the emulator's own folder
    assert config["PathEntries"]["Paths"] == [{"System": "GBA", "Type": "Base", "Path": r"C:\work\GBA"}]


def test_emulator_command_dumps_png_frames_and_quits():
    cmd = emulator_command(Path("EmuHawk.exe"), Path("c.ini"), Path("m.bk2"), Path("game.gba"), Path("d/f.png"), 900)
    assert cmd[0] == "EmuHawk.exe"
    assert cmd[-1] == "game.gba"  # the ROM goes last
    # GBAHawk only understands --flag=value; BizHawk takes that form too. GBAHawk does not
    # work out the movie's length by itself and would dump forever.
    assert cmd[1:-1] == ["--config=c.ini", "--movie=m.bk2", "--dump-type=imagesequence",
                         f"--dump-name={Path('d/f.png')}", "--dump-length=900", "--dump-close"]


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


BIZHAWK = Emulator("BizHawk", Path("EmuHawk.exe"), "2.11.1")
GBAHAWK = Emulator("GBAHawk", Path("GBAHawk.exe"), "3.0.0")


class FakeDump:
    def __init__(self):
        self.emulators = []

    def __call__(self, emulator, movie, rom, bios, work_dir):
        self.emulators.append(emulator)
        dump = Path(work_dir) / "dump"
        dump.mkdir()
        for i in range(movie.frames):
            write_png(dump / f"f_{i}.png", noise(i))
        return dumped_frames(dump)


@pytest.fixture
def roms(tmp_path):
    folder = tmp_path / "roms"
    folder.mkdir()
    make_rom(folder / "game.gba")
    return rom_index(folder)


def test_convert_writes_frames_and_run_info(tmp_path, roms, bios):
    movie = tmp_path / "author-game.bk2"
    movie.write_bytes(bk2_bytes(frames=3, SHA1=next(iter(roms)), emuVersion="Version 2.9.1"))
    out = tmp_path / "out"
    dump = FakeDump()

    out_dir = convert(movie, {"mGBA": BIZHAWK, "GBAHawk": GBAHAWK}, bios, roms, out, dump=dump)

    assert dump.emulators == [BIZHAWK]
    assert out_dir == out / "dbgt" / "tas_author-game_bk2"
    assert len(list((out_dir / "frames").glob("*.png"))) == 3
    run = json.loads((out_dir / "run.json").read_text())
    assert run["source"] == "tas"
    assert run["movie"] == str(movie)
    assert run["rom"] == str(tmp_path / "roms" / "game.gba")
    assert run["game_code"] == "BT4E"
    assert run["emulator"] == "BizHawk 2.11.1"
    assert run["movie_emulator_version"] == "2.9.1"
    assert run["frames_dumped"] == 3
    assert run["frames_saved"] == 3
    assert [p.name for p in out.iterdir()] == ["dbgt"]  # the dump folder is gone


def test_convert_names_folder_after_rom_file_when_rom_has_no_game_code(tmp_path, bios):
    folder = tmp_path / "roms"
    folder.mkdir()
    sha1 = make_rom(folder / "Some Homebrew.gba", code=bytes(4))  # homebrew often leaves it empty
    movie = tmp_path / "m.bk2"
    movie.write_bytes(bk2_bytes(frames=1, SHA1=sha1))
    out_dir = convert(movie, {"mGBA": BIZHAWK}, bios, rom_index(folder), tmp_path / "out", dump=FakeDump())
    assert out_dir == tmp_path / "out" / "Some Homebrew" / "tas_m_bk2"


def test_convert_keeps_movies_with_the_same_name_apart(tmp_path, roms, bios):
    sha1 = next(iter(roms))
    (tmp_path / "run.bk2").write_bytes(bk2_bytes(frames=1, SHA1=sha1))
    (tmp_path / "run.gbmv").write_bytes(bk2_bytes(frames=1, core="GBAHawk", SHA1=sha1))
    emulators = {"mGBA": BIZHAWK, "GBAHawk": GBAHAWK}
    a = convert(tmp_path / "run.bk2", emulators, bios, roms, tmp_path / "out", dump=FakeDump())
    b = convert(tmp_path / "run.gbmv", emulators, bios, roms, tmp_path / "out", dump=FakeDump())
    assert a is not None and b is not None and a != b


def test_convert_plays_gbahawk_movies_in_gbahawk(tmp_path, roms, bios):
    movie = tmp_path / "author-game.gbmv"
    movie.write_bytes(bk2_bytes(frames=2, core="GBAHawk", SHA1=next(iter(roms))))
    dump = FakeDump()
    convert(movie, {"mGBA": BIZHAWK, "GBAHawk": GBAHAWK}, bios, roms, tmp_path / "out", dump=dump)
    assert dump.emulators == [GBAHAWK]


def test_convert_needs_the_emulator_for_the_movie(tmp_path, roms, bios):
    movie = tmp_path / "author-game.gbmv"
    movie.write_bytes(bk2_bytes(core="GBAHawk", SHA1=next(iter(roms))))
    with pytest.raises(MovieError, match="GBAHawk"):
        convert(movie, {"mGBA": BIZHAWK}, bios, roms, tmp_path / "out", dump=FakeDump())


def test_convert_skips_finished_movies_and_redoes_interrupted_ones(tmp_path, roms, bios):
    movie = tmp_path / "m.bk2"
    movie.write_bytes(bk2_bytes(frames=2, SHA1=next(iter(roms))))
    out = tmp_path / "out"
    leftover = out / "dbgt" / "tas_m_bk2" / "frames"
    leftover.mkdir(parents=True)
    write_png(leftover / "000000.png", noise(9))
    emulators = {"mGBA": BIZHAWK}

    out_dir = convert(movie, emulators, bios, roms, out, dump=FakeDump())
    assert len((out_dir / "frames.jsonl").read_text().splitlines()) == 2
    assert len(list((out_dir / "frames").glob("*.png"))) == 2

    assert convert(movie, emulators, bios, roms, out, dump=FakeDump()) is None


# With the real emulators: set GBA_BIZHAWK to EmuHawk.exe, GBA_GBAHAWK to GBAHawk.exe and GBA_BIOS to a
# GBA BIOS file (any BIOS works here, the test movies do not name one). Each emulator's window shows
# for a few seconds.
BIOS = os.environ.get("GBA_BIOS")


def real_dump(tmp_path, exe, core, inputs):
    from dataset.cheats import rom_game_code
    from dataset.tas import dump_movie, emulator_version

    roms = sorted(Path("roms").rglob("*.gba")) if Path("roms").exists() else []
    rom = next((r for r in roms if rom_game_code(r) == "BT4E"), None)
    if rom is None:
        pytest.skip("needs the BT4E ROM")
    version = emulator_version(Path(exe))
    assert re.fullmatch(r"\d+\.\d+(\.\d+)?", version)
    sha1 = hashlib.sha1(rom.read_bytes()).hexdigest().upper()
    path = tmp_path / ("test.gbmv" if core == "GBAHawk" else "test.bk2")
    path.write_bytes(bk2_bytes(inputs=inputs, core=core, SHA1=sha1))
    work = tmp_path / "work"
    work.mkdir()
    emulator = Emulator("BizHawk" if core == "mGBA" else "GBAHawk", Path(exe), version)
    return dump_movie(emulator, read_movie(path), rom, Path(BIOS), work)


# press Start now and then to get past the logos
START_NOW_AND_THEN = ["....S......" if i % 60 < 3 else "..........." for i in range(600)]


@pytest.mark.skipif(not os.environ.get("GBA_BIZHAWK") or not BIOS, reason="set GBA_BIZHAWK and GBA_BIOS")
def test_real_bizhawk_dumps_every_movie_frame_at_native_size(tmp_path):
    frames = real_dump(tmp_path, os.environ["GBA_BIZHAWK"], "mGBA", START_NOW_AND_THEN)
    assert [i for i, _ in frames] == list(range(600))
    first, last = (cv2.imread(str(frames[i][1])) for i in (0, -1))
    assert first.shape == last.shape == (160, 240, 3)
    assert last.std() > 4  # a picture, not a blank screen


@pytest.mark.skipif(not os.environ.get("GBA_GBAHAWK") or not BIOS, reason="set GBA_GBAHAWK and GBA_BIOS")
def test_real_gbahawk_dumps_movie_at_native_size(tmp_path):
    frames = real_dump(tmp_path, os.environ["GBA_GBAHAWK"], "GBAHawk", START_NOW_AND_THEN)
    # GBAHawk matches video to sound when dumping, so a frame can be repeated or dropped now and then
    assert abs(len(frames) - 600) <= 10
    first, last = (cv2.imread(str(frames[i][1])) for i in (0, -1))
    assert first.shape == last.shape == (160, 240, 3)
    assert last.std() > 4
