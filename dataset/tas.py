r"""
Frames from TAS movies (tool-assisted speedruns, for example from tasvideos.org).

A TAS movie holds the buttons pressed on every frame, not video. Replaying it in the
emulator it was made with and saving every frame gives lossless 240x160 frames of a whole
playthrough. They go through the same selection as the explorer's frames (recorder.py),
so the output folder looks the same: frames\*.png, frames.jsonl (with the movie frame
number of each) and run.json.

    python -m dataset.tas C:\tas

Movies can be given one by one or as folders, as movie files or the .zip files
tasvideos.org downloads. GBA movies there come in three kinds:
  - .bk2, made with BizHawk's mGBA core: played in BizHawk (--bizhawk).
  - .gbmv, made with GBAHawk: played in GBAHawk (--gbahawk). GBAHawk is a cut-down
    BizHawk; a .gbmv is a .bk2 under another name.
  - .vbm, made with VisualBoyAdvance-rr: converted to a BizHawk mGBA movie, as BizHawk's
    own importer does. VBA-rr times things differently from mGBA, so these can desync;
    check the frames (contact sheet of the last minutes) before using them.
Each movie also needs:
  - its ROM, found under --roms by the SHA1 in the movie (.vbm: by game code and title);
  - a GBA BIOS (--bios): the emulators do not play GBA movies without one, and most
    movies name the official BIOS by its SHA1, which then has to be that exact file;
  - ideally the emulator version it was made with (its tasvideos.org page says which).
    Newer versions usually play old movies the same way, but can desync.
The emulators run with a config file of their own in a temporary folder, so your own
settings are not touched. Their window shows while they run.
"""

import argparse
import hashlib
import io
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2

from dataset.cheats import GAMES, rom_game_code
from dataset.novelty import screen_key
from dataset.recorder import FrameRecorder

# emulator for each movie core
EMULATOR_FOR_CORE = {"mGBA": "BizHawk", "GBAHawk": "GBAHawk"}
MOVIE_EXTENSIONS = (".bk2", ".gbmv", ".vbm")


class MovieError(ValueError):
    """A movie that cannot be played: wrong format, or its ROM, BIOS or emulator is missing."""


@dataclass(frozen=True)
class Emulator:
    name: str  # "BizHawk" or "GBAHawk"
    exe: Path
    version: str  # as its config files record it, e.g. "2.11.1"


@dataclass(frozen=True)
class Movie:
    path: Path
    bk2: bytes  # the movie as a .bk2 (zip) file, also when it came inside a .zip or as a .vbm
    header: dict[str, str]
    frames: int

    @property
    def sha1(self) -> str:
        return self.header.get("SHA1", "").upper()

    @property
    def bios_sha1(self) -> str | None:
        return self.header.get("GBA_Firmware_Bios", "").upper() or None

    @property
    def core(self) -> str:
        return self.header.get("Core", "")

    @property
    def emu_version(self) -> str:
        """Version of the emulator the movie was last saved with, e.g. "2.9.1"."""
        return self.header.get("emuVersion", "").removeprefix("Version ").strip()


# A .bk2 for BizHawk's mGBA core: 4 analog values (tilt, light sensor), then the buttons
MGBA_LOG_KEY = "LogKey:#Tilt X|Tilt Y|Tilt Z|Light Sensor|Up|Down|Left|Right|Start|Select|B|A|L|R|Power|"
MGBA_SYNC = ('{{"o":{{"$type":"BizHawk.Emulation.Cores.Nintendo.GBA.MGBAHawk+SyncSettings, '
             'BizHawk.Emulation.Cores","SkipBios":{skip}}}}}')
# (button bit in a .vbm frame, mnemonic) in the order of the .bk2 columns Up..R
VBM_BUTTONS = [(6, "U"), (7, "D"), (5, "L"), (4, "R"), (3, "S"), (2, "s"), (1, "B"), (0, "A"), (9, "l"), (8, "r")]
VBM_RESETS = 0x0C00  # "reset (old timing)" and "reset (new timing)"


def _vbm_to_bk2(data: bytes, name: str) -> bytes:
    """A VisualBoyAdvance-rr movie as a BizHawk mGBA movie, like BizHawk's VbmImport."""
    (signature, _, _, frames, rerecords, start, controllers, system, options, _, title, _, _, _, code,
     _, first_frame) = struct.unpack_from("<4sIIIIBBBB12s12sBBH4sII", data)
    if signature != b"VBM\x1a":
        raise MovieError(f"{name}: not a VisualBoyAdvance movie")
    if not system & 1:
        raise MovieError(f"{name}: a Game Boy movie, not GBA")
    if start & 3:
        raise MovieError(f"{name}: starts from a savestate or save data, which cannot be converted")
    author = data[0x40:0x80].split(b"\0")[0].decode("latin-1")
    rows = []
    stride = 2 * bin(controllers & 0xF).count("1")  # other controllers' bytes are skipped
    for i in range(frames):
        (buttons,) = struct.unpack_from("<H", data, first_frame + i * stride)
        row = "".join(c if buttons >> bit & 1 else "." for bit, c in VBM_BUTTONS)
        rows.append(f"|    0,    0,    0,    0,{row}{'P' if buttons & VBM_RESETS else '.'}|")
    # The BIOS intro is in the movie only if it was made with a BIOS file and without skipping it.
    # (BizHawk's importer always skips it.)
    skip_bios = not (options & 1 and not options & 2)
    header = {"MovieVersion": "BizHawk v2.0", "Author": author, "emuVersion": "VBA-rr", "Platform": "GBA",
              "GameName": title.split(b"\0")[0].decode("latin-1").strip(),
              "GameCode": code.decode("latin-1"), "Core": "mGBA", "rerecordCount": str(rerecords)}
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("Header.txt", "\r\n".join(f"{k} {v}" for k, v in header.items()) + "\r\n")
        z.writestr("Input Log.txt", "\r\n".join(["[Input]", MGBA_LOG_KEY, *rows, "[/Input]"]) + "\r\n")
        z.writestr("SyncSettings.json", MGBA_SYNC.format(skip=str(skip_bios).lower()))
    return buf.getvalue()


def _bk2_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    if data[:4] == b"VBM\x1a":
        return _vbm_to_bk2(data, path.name)
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise MovieError(f"{path.name}: not a .bk2, .gbmv or .vbm movie") from None
    names = archive.namelist()
    if "Header.txt" in names:
        return data
    inner = [n for n in names if n.lower().endswith(MOVIE_EXTENSIONS)]
    if len(inner) != 1:
        raise MovieError(f"{path.name}: holds {', '.join(names)}; only .bk2, .gbmv and .vbm movies are supported")
    data = archive.read(inner[0])
    return _vbm_to_bk2(data, inner[0]) if inner[0].lower().endswith(".vbm") else data


def read_movie(path) -> Movie:
    path = Path(path)
    bk2 = _bk2_bytes(path)
    archive = zipfile.ZipFile(io.BytesIO(bk2))
    try:
        header_text = archive.read("Header.txt").decode("utf-8")
        log = archive.read("Input Log.txt").decode("utf-8")
    except KeyError as e:
        raise MovieError(f"{path.name}: not a .bk2 movie ({e})") from None
    header = {}
    for line in header_text.splitlines():
        key, _, value = line.strip().partition(" ")
        if key:
            header[key] = value
    frames = sum(line.startswith("|") for line in log.splitlines())
    return Movie(path, bk2, header, frames)


def _sha1(path) -> str:
    return hashlib.sha1(Path(path).read_bytes()).hexdigest().upper()


def rom_index(roms_dir) -> dict[str, Path]:
    """SHA1 -> path of every .gba file under roms_dir."""
    return {_sha1(p): p for p in sorted(Path(roms_dir).rglob("*.gba"))}


def _rom_title(path) -> str:
    with open(path, "rb") as f:
        f.seek(0xA0)
        return f.read(12).split(b"\0")[0].decode("latin-1").strip()


def check_movie(movie: Movie, roms: dict[str, Path], bios) -> Path:
    """The ROM to play `movie` with; MovieError if it cannot be played as given."""
    name = movie.path.name
    platform = movie.header.get("Platform", "")
    if platform != "GBA":
        raise MovieError(f"{name}: a {platform or 'unknown'} movie, not GBA")
    if movie.core not in EMULATOR_FOR_CORE:
        raise MovieError(f"{name}: made with the {movie.core or 'unknown'} core; "
                         f"only mGBA (BizHawk) and GBAHawk movies are supported")
    if movie.sha1:
        rom = roms.get(movie.sha1)
        wanted = f"SHA1 {movie.sha1}"
    else:  # converted from .vbm
        code, title = movie.header.get("GameCode", ""), movie.header.get("GameName", "")
        rom = next((p for p in roms.values() if rom_game_code(p) == code and _rom_title(p) == title), None)
        wanted = f"game code {code} and title {title}"
    if rom is None:
        raise MovieError(f"{name}: no ROM with {wanted} ({movie.header.get('GameName', '?')})")
    if movie.bios_sha1 and _sha1(bios) != movie.bios_sha1:
        raise MovieError(f"{name}: made with the BIOS with SHA1 {movie.bios_sha1}, but {bios} is a different one")
    return rom


def emulator_version(exe) -> str:
    """
    The version BizHawk or GBAHawk writes into its config file (GBAHawk's .exe does not
    carry it): start it with an empty config and a Lua script that quits, and read the
    config it saves on the way out.
    """
    with tempfile.TemporaryDirectory(prefix="emu_version_") as work:
        config, script = Path(work) / "config.ini", Path(work) / "quit.lua"
        script.write_text("client.exit()\n")
        try:
            subprocess.run([str(Path(exe).resolve()), f"--config={config}", f"--lua={script}"],
                           timeout=120, capture_output=True)
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{exe} did not quit within 120 s") from None
        if not config.exists():
            raise RuntimeError(f"{exe} saved no config file")
        return json.loads(config.read_text(encoding="utf-8-sig"))["LastWrittenFrom"]


def write_config(path, version: str, bios, gba_dir) -> None:
    """A config for dumping; the emulator fills in everything else with its defaults."""
    Path(path).write_text(json.dumps({
        "LastWrittenFrom": version,  # otherwise the emulator first asks about a config from another version
        "FirmwareUserSpecifications": {"GBA+Bios": str(bios)},
        "Unthrottled": True,
        "VideoWriterAudioSync": False,  # BizHawk: otherwise frames are dropped or repeated to match the sound
        "SoundEnabled": False,
        # save RAM and states go here instead of a GBA folder next to the emulator
        "PathEntries": {"Paths": [{"System": "GBA", "Type": "Base", "Path": str(gba_dir)}]},
    }, indent=2))


def emulator_command(exe, config, movie, rom, dump_name, frames: int) -> list[str]:
    """Play `movie` and save its first `frames` frames as <dump_name stem>_<n>.png, then quit."""
    # GBAHawk only understands --flag=value; BizHawk takes that form too. GBAHawk does not
    # work out the movie's length by itself, so without --dump-length it dumps forever.
    return [str(exe), f"--config={config}", f"--movie={movie}", "--dump-type=imagesequence",
            f"--dump-name={dump_name}", f"--dump-length={frames}", "--dump-close",
            str(rom)]  # the ROM goes last


def dumped_frames(dump_dir) -> list[tuple[int, Path]]:
    """(frame number, path) of the dumped f_<n>.png files, in frame order."""
    frames = [(int(p.stem.rsplit("_", 1)[1]), p) for p in Path(dump_dir).glob("f_*.png")]
    return sorted(frames)


def dump_movie(emulator: Emulator, movie: Movie, rom, bios, work_dir, timeout=None) -> list[tuple[int, Path]]:
    """Play `movie` in its emulator and save every frame into work_dir\\dump."""
    work_dir = Path(work_dir).resolve()  # the emulator gets absolute paths only
    dump_dir = work_dir / "dump"
    dump_dir.mkdir()  # the emulator does not make it, and stops with an error dialog
    movie_path = work_dir / ("movie.gbmv" if emulator.name == "GBAHawk" else "movie.bk2")
    movie_path.write_bytes(movie.bk2)
    config = work_dir / "config.ini"
    write_config(config, emulator.version, Path(bios).resolve(), work_dir / "GBA")
    cmd = emulator_command(Path(emulator.exe).resolve(), config, movie_path, Path(rom).resolve(),
                           dump_dir / "f.png", movie.frames)
    if timeout is None:
        timeout = 120 + movie.frames / 30  # BizHawk dumps ~300 frames a second, GBAHawk ~80
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"{emulator.name} did not finish within {timeout:.0f} s; it is probably showing a dialog. "
            f"Run it by hand to see it:\n  {subprocess.list2cmdline(cmd)}") from None
    frames = dumped_frames(dump_dir)
    if not frames:
        raise RuntimeError(f"{emulator.name} saved no frames:\n  {subprocess.list2cmdline(cmd)}")
    if len(frames) < movie.frames * 0.99:
        print(f"warning: {emulator.name} saved {len(frames)} of the movie's {movie.frames} frames")
    return frames


def select_frames(frames, recorder: FrameRecorder, screen_shape=(8, 6, 4)) -> None:
    """Offer every dumped frame to the recorder, which keeps the varied ones."""
    for number, path in frames:
        frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        recorder.offer(frame, group=screen_key(frame, *screen_shape), movie_frame=number)


def convert(path, emulators: dict[str, Emulator], bios, roms: dict[str, Path], out,
            per_group_cap: int = 2, min_diff: float = 4.0, dump=dump_movie) -> Path | None:
    """
    Dump one movie and keep its varied frames in out\\<game>\\tas_<movie name>_<extension>\\.
    `emulators` maps a movie core ("mGBA", "GBAHawk") to the emulator that plays it.
    None if that folder is already complete (it has run.json).
    """
    movie = read_movie(path)
    rom = check_movie(movie, roms, bios)
    emulator = emulators.get(movie.core)
    if emulator is None:
        raise MovieError(f"{movie.path.name}: needs {EMULATOR_FOR_CORE[movie.core]}, which was not given")
    code = rom_game_code(rom)
    if code in GAMES:
        game = GAMES[code].name
    else:  # homebrew often leaves the game code empty
        game = code if code.isascii() and code.isalnum() else Path(rom).stem
    # the extension too, so run.bk2 and run.vbm do not share a folder
    out_dir = Path(out) / game / f"tas_{movie.path.stem}_{movie.path.suffix.lstrip('.').lower()}"
    if (out_dir / "run.json").exists():
        return None
    if out_dir.exists():  # left by an interrupted run
        shutil.rmtree(out_dir)
    if movie.emu_version and movie.emu_version != emulator.version:
        print(f"note: {movie.path.name} was made with {movie.emu_version}, playing it in "
              f"{emulator.name} {emulator.version}; if the frames go wrong partway, try the movie's version")
    Path(out).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tas_dump_", dir=out) as work_dir:
        frames = dump(emulator, movie, rom, bios, work_dir)
        with FrameRecorder(out_dir, per_group_cap=per_group_cap, min_diff=min_diff) as recorder:
            select_frames(frames, recorder)
    (out_dir / "run.json").write_text(json.dumps({
        "source": "tas", "movie": str(movie.path), "rom": str(rom), "game_code": code,
        "movie_game": movie.header.get("GameName"), "author": movie.header.get("Author"),
        "emulator": f"{emulator.name} {emulator.version}", "movie_emulator_version": movie.emu_version,
        "frames_dumped": len(frames), "frames_saved": recorder.count,
    }, indent=2))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect GBA frames from TAS movies (.bk2, .gbmv, .vbm)")
    ap.add_argument("movies", nargs="+", help="movie files or tasvideos.org .zip files, or folders of them")
    ap.add_argument("--bizhawk", default=r"bizhawk\EmuHawk.exe", help="EmuHawk.exe, plays .bk2 and .vbm movies")
    ap.add_argument("--gbahawk", default=r"gbahawk\GBAHawk.exe", help="GBAHawk.exe, plays .gbmv movies")
    ap.add_argument("--bios", default=r"BIOS\gba_bios.bin", help="GBA BIOS file (the official one for most movies)")
    ap.add_argument("--roms", default="roms", help="folder searched for each movie's ROM")
    ap.add_argument("--out", default=r"data\raw")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in a log file as it happens

    if not Path(args.bios).exists():
        ap.error(f"no BIOS at {args.bios}")
    paths = []
    for item in map(Path, args.movies):
        if item.is_dir():
            paths += sorted(p for p in item.iterdir() if p.suffix.lower() in (*MOVIE_EXTENSIONS, ".zip"))
        else:
            paths.append(item)
    emulators = {}
    for core, name, exe in (("mGBA", "BizHawk", args.bizhawk), ("GBAHawk", "GBAHawk", args.gbahawk)):
        if Path(exe).exists():
            emulators[core] = Emulator(name, Path(exe), emulator_version(exe))
            print(f"{name} {emulators[core].version}: {exe}")
        else:
            print(f"no {name} at {exe}; its movies are skipped")
    roms = rom_index(args.roms)
    print(f"{len(roms)} ROMs, {len(paths)} movies")
    for path in paths:
        try:
            out_dir = convert(path, emulators, args.bios, roms, args.out)
        except (MovieError, RuntimeError) as e:
            print(f"skipped: {e}")
            continue
        if out_dir is None:
            print(f"{path.name}: already done")
        else:
            run = json.loads((out_dir / "run.json").read_text())
            print(f"{path.name}: {run['frames_saved']} of {run['frames_dumped']} frames kept in {out_dir}")


if __name__ == "__main__":
    main()
