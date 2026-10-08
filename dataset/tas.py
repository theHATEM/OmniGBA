r"""
Frames from TAS movies (tool-assisted speedruns, for example from tasvideos.org).

A BizHawk movie (.bk2) holds the buttons pressed on every frame, not video. Replaying it
in BizHawk, the emulator it was made with, and saving every frame gives lossless 240x160
frames of a whole playthrough. They go through the same selection as the explorer's
frames (recorder.py), so the output folder looks the same: frames\*.png, frames.jsonl
(with the movie frame number of each) and run.json.

    python -m dataset.tas --bizhawk "C:\BizHawk\EmuHawk.exe" --bios "C:\bios\gba_bios.bin" tas

Movies can be .bk2 files or the .zip files tasvideos.org downloads, given one by one or
as folders. Each movie needs:
  - the ROM it was made for, found under --roms by the SHA1 in the movie;
  - a GBA BIOS: BizHawk does not play GBA movies without one, and most movies name the
    official BIOS (by its SHA1), which then has to be that exact file;
  - ideally the BizHawk version it was made with (its tasvideos.org page says which).
    Newer versions usually play old movies the same way, but can desync.
Only movies made with BizHawk's mGBA core are supported, not VBA-rr (.vbm) or GBAHawk
(.gbmv) ones. BizHawk runs with a config file of its own in a temporary folder, so your
BizHawk settings are not touched. Its window shows while it runs.
"""

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import cv2

from dataset.cheats import GAMES, rom_game_code
from dataset.novelty import screen_key
from dataset.recorder import FrameRecorder


class MovieError(ValueError):
    """A movie that cannot be played: wrong format, or its ROM or BIOS is missing."""


@dataclass(frozen=True)
class Movie:
    path: Path
    bk2: bytes  # the .bk2 file itself, also when it came inside a download .zip
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
        """BizHawk version the movie was last saved with, e.g. "2.9.1"."""
        return self.header.get("emuVersion", "").removeprefix("Version ").strip()


def _bk2_bytes(path: Path) -> bytes:
    data = path.read_bytes()
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        raise MovieError(f"{path.name}: not a BizHawk .bk2 movie") from None
    names = archive.namelist()
    if "Header.txt" in names:
        return data
    inner = [n for n in names if n.lower().endswith(".bk2")]
    if len(inner) != 1:
        raise MovieError(f"{path.name}: holds {', '.join(names)}; only BizHawk .bk2 movies are supported")
    return archive.read(inner[0])


def read_movie(path) -> Movie:
    path = Path(path)
    bk2 = _bk2_bytes(path)
    archive = zipfile.ZipFile(io.BytesIO(bk2))
    try:
        header_text = archive.read("Header.txt").decode("utf-8")
        log = archive.read("Input Log.txt").decode("utf-8")
    except KeyError as e:
        raise MovieError(f"{path.name}: not a BizHawk .bk2 movie ({e})") from None
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


def check_movie(movie: Movie, roms: dict[str, Path], bios) -> Path:
    """The ROM to play `movie` with; MovieError if it cannot be played as given."""
    name = movie.path.name
    platform = movie.header.get("Platform", "")
    if platform != "GBA":
        raise MovieError(f"{name}: a {platform or 'unknown'} movie, not GBA")
    if movie.core != "mGBA":
        raise MovieError(f"{name}: made with BizHawk's {movie.core or 'unknown'} core; only mGBA movies are supported")
    rom = roms.get(movie.sha1)
    if rom is None:
        raise MovieError(f"{name}: no ROM with SHA1 {movie.sha1} ({movie.header.get('GameName', '?')})")
    if movie.bios_sha1 and _sha1(bios) != movie.bios_sha1:
        raise MovieError(f"{name}: made with the BIOS with SHA1 {movie.bios_sha1}, but {bios} is a different one")
    return rom


def main_version(product_version: str) -> str:
    """BizHawk's version as its config files record it: "2.11.1+<commit>" -> "2.11.1"."""
    return product_version.split("+")[0].strip()


def bizhawk_version(exe) -> str:
    """The version of EmuHawk.exe, read from its file properties (Windows only)."""
    import ctypes
    from ctypes import wintypes

    version = ctypes.WinDLL("version")
    size = version.GetFileVersionInfoSizeW(str(exe), None)
    if not size:
        raise OSError(f"no version information in {exe}")
    info = ctypes.create_string_buffer(size)
    version.GetFileVersionInfoW(str(exe), 0, size, info)
    value, length = ctypes.c_void_p(), wintypes.UINT()
    version.VerQueryValueW(info, r"\VarFileInfo\Translation", ctypes.byref(value), ctypes.byref(length))
    language, codepage = ctypes.cast(value, ctypes.POINTER(wintypes.WORD * 2)).contents
    if not version.VerQueryValueW(info, rf"\StringFileInfo\{language:04x}{codepage:04x}\ProductVersion",
                                  ctypes.byref(value), ctypes.byref(length)):
        raise OSError(f"no product version in {exe}")
    return main_version(ctypes.wstring_at(value.value))


def write_config(path, version: str, bios) -> None:
    """A BizHawk config for dumping; BizHawk fills in everything else with its defaults."""
    Path(path).write_text(json.dumps({
        "LastWrittenFrom": version,  # otherwise BizHawk first asks about a config from another version
        "FirmwareUserSpecifications": {"GBA+Bios": str(bios)},
        "Unthrottled": True,
        "VideoWriterAudioSync": False,  # with it on, frames are dropped or repeated to match the sound
        "SoundEnabled": False,
    }, indent=2))


def bizhawk_command(bizhawk, config, movie, rom, dump_name) -> list[str]:
    """Play `movie` and save every frame as <dump_name stem>_<n>.png, then quit."""
    return [str(bizhawk), "--config", str(config), "--movie", str(movie),
            "--dump-type", "imagesequence", "--dump-name", str(dump_name), "--dump-close",
            str(rom)]  # BizHawk wants the ROM last


def dumped_frames(dump_dir) -> list[tuple[int, Path]]:
    """(frame number, path) of BizHawk's f_<n>.png files, in frame order."""
    frames = [(int(p.stem.rsplit("_", 1)[1]), p) for p in Path(dump_dir).glob("f_*.png")]
    return sorted(frames)


def dump_movie(bizhawk, version: str, movie: Movie, rom, bios, work_dir, timeout=None) -> list[tuple[int, Path]]:
    """Play `movie` in BizHawk and save every frame into work_dir\\dump."""
    work_dir = Path(work_dir).resolve()  # BizHawk gets absolute paths only
    dump_dir = work_dir / "dump"
    dump_dir.mkdir()  # BizHawk does not make it, and stops with an error dialog
    movie_path = work_dir / "movie.bk2"
    movie_path.write_bytes(movie.bk2)
    config = work_dir / "config.ini"
    write_config(config, version, Path(bios).resolve())
    cmd = bizhawk_command(bizhawk, config, movie_path, Path(rom).resolve(), dump_dir / "f.png")
    if timeout is None:
        timeout = 60 + movie.frames / 100  # BizHawk dumps about 300 frames a second
    try:
        subprocess.run(cmd, timeout=timeout, capture_output=True)
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"BizHawk did not finish within {timeout:.0f} s; it is probably showing a dialog. "
            f"Run it by hand to see it:\n  {subprocess.list2cmdline(cmd)}") from None
    frames = dumped_frames(dump_dir)
    if not frames:
        raise RuntimeError(f"BizHawk saved no frames:\n  {subprocess.list2cmdline(cmd)}")
    if len(frames) < movie.frames:
        print(f"warning: BizHawk saved {len(frames)} of the movie's {movie.frames} frames")
    return frames


def select_frames(frames, recorder: FrameRecorder, screen_shape=(8, 6, 4)) -> None:
    """Offer every dumped frame to the recorder, which keeps the varied ones."""
    for number, path in frames:
        frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        recorder.offer(frame, group=screen_key(frame, *screen_shape), movie_frame=number)


def convert(path, bizhawk, version: str, bios, roms: dict[str, Path], out,
            per_group_cap: int = 2, min_diff: float = 4.0, dump=dump_movie) -> Path | None:
    """
    Dump one movie and keep its varied frames in out\\<game>\\tas_<movie name>\\.
    None if that folder is already complete (it has run.json).
    """
    movie = read_movie(path)
    rom = check_movie(movie, roms, bios)
    code = rom_game_code(rom)
    game = GAMES[code].name if code in GAMES else code
    out_dir = Path(out) / game / f"tas_{movie.path.stem}"
    if (out_dir / "run.json").exists():
        return None
    if out_dir.exists():  # left by an interrupted run
        shutil.rmtree(out_dir)
    if movie.emu_version and movie.emu_version != version:
        print(f"note: {movie.path.name} was made with BizHawk {movie.emu_version}, this is {version}; "
              f"if the frames go wrong partway, use {movie.emu_version}")
    Path(out).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="tas_dump_", dir=out) as work_dir:
        frames = dump(bizhawk, version, movie, rom, bios, work_dir)
        with FrameRecorder(out_dir, per_group_cap=per_group_cap, min_diff=min_diff) as recorder:
            select_frames(frames, recorder)
    (out_dir / "run.json").write_text(json.dumps({
        "source": "tas", "movie": str(movie.path), "rom": str(rom), "game_code": code,
        "movie_game": movie.header.get("GameName"), "author": movie.header.get("Author"),
        "bizhawk": version, "movie_bizhawk": movie.emu_version,
        "frames_dumped": len(frames), "frames_saved": recorder.count,
    }, indent=2))
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect GBA frames from BizHawk TAS movies")
    ap.add_argument("movies", nargs="+", help=".bk2 or tasvideos.org .zip files, or folders of them")
    ap.add_argument("--bizhawk", required=True, help=r"path of EmuHawk.exe")
    ap.add_argument("--bios", required=True, help="GBA BIOS file (the official one for most movies)")
    ap.add_argument("--roms", default="roms", help="folder searched for each movie's ROM")
    ap.add_argument("--out", default=r"data\raw")
    args = ap.parse_args()

    paths = []
    for item in map(Path, args.movies):
        paths += sorted(p for p in item.iterdir() if p.suffix.lower() in (".bk2", ".zip")) if item.is_dir() else [item]
    version = bizhawk_version(args.bizhawk)
    roms = rom_index(args.roms)
    print(f"BizHawk {version}, {len(roms)} ROMs, {len(paths)} movies")
    for path in paths:
        try:
            out_dir = convert(path, args.bizhawk, version, args.bios, roms, args.out)
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
