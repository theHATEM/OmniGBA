"""
Cheat codes that remove difficulty walls during automated exploration.

Codes are CodeBreaker codes for the US releases, in libretro's "+"-separated form,
and are applied by mGBA's cheat engine every frame. They come from community lists
(libretro-database for Dragon Ball GT, a kh-vids.net thread for Naruto); the core
applying them is tested, but what each one does in game should be checked on the
captured frames.
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Game:
    name: str
    cheats: list[tuple[str, str]] = field(default_factory=list)  # (description, code)


GAMES = {
    "BT4E": Game(
        "dbgt",
        [
            ("All Secrets Unlocked", "43001CF8+FFFF+00000004+0002"),
            ("Infinite Health All", "43001AF6+0079+00000009+002C"),
            ("Max Health All", "43001AF8+0079+00000009+002C"),  # keeps health == max
            ("Infinite Ki All", "43001AFC+0064+00000009+002C"),
        ],
    ),
    "BN2E": Game(
        "naruto",
        [
            ("Invincible", "33000350+0003+33000352+0002"),
            ("Infinite Chakra", "33000356+00FF"),
            ("Infinite Energy (Naruto)", "3300454A+00C8"),
            ("Infinite Energy (Sasuke)", "3300454B+00C8"),
            ("Infinite Energy (Sakura)", "3300454C+00C8"),
            ("Infinite Energy (Rock Lee)", "3300454D+00C8"),
            ("Infinite Double Jump", "33000347+0000"),
        ],
    ),
}


def rom_game_code(path) -> str:
    """The 4-letter game code from the ROM header, e.g. "BT4E" (E = USA)."""
    with open(Path(path), "rb") as f:
        f.seek(0xAC)
        return f.read(4).decode("ascii", errors="replace")


def cheats_for(game_code: str) -> list[str]:
    game = GAMES.get(game_code)
    return [code for _, code in game.cheats] if game else []
