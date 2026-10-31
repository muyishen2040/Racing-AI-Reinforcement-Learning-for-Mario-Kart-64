"""Scripted menu navigation from power-on to the start of a Time Trial.

Timings are in emulated frames. The emulator is deterministic, so a fixed
sequence always lands on the same screen. Start-line savestates are cached on
disk so an environment reset costs one unserialize call instead of a menu walk.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from mk64_env.core import NO_INPUT, ControllerInput, EmulatorCore
from mk64_env.ram import COURSE_IDS, COURSE_TIMER

A = ControllerInput(a=True)
START = ControllerInput(start=True)
UP = ControllerInput(joy_y=80)
DOWN = ControllerInput(joy_y=-80)
LEFT = ControllerInput(joy_x=-80)
RIGHT = ControllerInput(joy_x=80)

CHARACTERS = {
    "mario": (0, 0),
    "luigi": (0, 1),
    "peach": (0, 2),
    "toad": (0, 3),
    "yoshi": (1, 0),
    "dk": (1, 1),
    "wario": (1, 2),
    "bowser": (1, 3),
}
"""(row, column) of each character on the Player Select grid."""

COURSES = {
    "LuigiRaceway": (0, 0),
    "MooMooFarm": (0, 1),
    "KoopaTroopaBeach": (0, 2),
    "KalimariDesert": (0, 3),
    "ToadsTurnpike": (1, 0),
    "FrappeSnowland": (1, 1),
    "ChocoMountain": (1, 2),
    "MarioRaceway": (1, 3),
    "WarioStadium": (2, 0),
    "SherbetLand": (2, 1),
    "RoyalRaceway": (2, 2),
    "BowsersCastle": (2, 3),
    "DKsJungleParkway": (3, 0),
    "YoshiValley": (3, 1),
    "BansheeBoardwalk": (3, 2),
    "RainbowRoad": (3, 3),
}
"""(cup index, position in cup) of each course on the Map Select screen."""

TITLE_FRAMES = 900
"""Frames from power-on until the title screen accepts START."""
GREEN_LIGHT_TIMEOUT = 2000


def press(core: EmulatorCore, button: ControllerInput, times: int = 1, hold: int = 4, gap: int = 8) -> None:
    """Press `button` `times` times, holding `hold` frames and releasing `gap` frames."""
    for _ in range(times):
        core.run_frames(hold, button)
        core.run_frames(gap, NO_INPUT)


def navigate_to_time_trial(core: EmulatorCore, course: str, character: str) -> None:
    """Walk the menus from power-on to the green light of a Time Trial.

    Returns on the first frame where the course timer is running. The core
    must be freshly started; the sequence assumes the intro has not been
    skipped yet.
    """
    row, col = CHARACTERS[character]
    cup, index = COURSES[course]
    core.run_frames(TITLE_FRAMES, NO_INPUT)
    press(core, START)
    core.run_frames(120)
    # Game Select: 1P GAME, then Time Trials, then BEGIN, then OK.
    press(core, A)
    core.run_frames(30)
    press(core, DOWN)
    core.run_frames(30)
    press(core, A)
    core.run_frames(30)
    press(core, A)
    core.run_frames(30)
    press(core, A)
    core.run_frames(90)
    # Player Select: move to the top-left corner, then to the character.
    press(core, UP)
    press(core, LEFT, times=3)
    press(core, DOWN, times=row)
    press(core, RIGHT, times=col)
    press(core, A)
    core.run_frames(30)
    press(core, A)
    core.run_frames(90)
    # Map Select: cup, then course, then confirm OK.
    press(core, LEFT, times=3)
    press(core, RIGHT, times=cup)
    press(core, A)
    core.run_frames(30)
    press(core, UP, times=3)
    press(core, DOWN, times=index)
    press(core, A)
    core.run_frames(40)
    press(core, A)
    core.set_input(NO_INPUT)
    for _ in range(GREEN_LIGHT_TIMEOUT):
        core.run_frame()
        if core.read_f32(COURSE_TIMER) > 0.0:
            break
    else:
        raise RuntimeError("race did not start; the menu sequence probably desynchronized")
    course_id = core.read_s16(0x800DC5A0)
    if course_id != COURSE_IDS[course]:
        raise RuntimeError(f"expected course id {COURSE_IDS[course]} for {course}, got {course_id}")


def start_state_path(cache_dir: str | os.PathLike[str], core_path: str | os.PathLike[str], course: str, character: str) -> Path:
    """Cache file for the start-line savestate of one course and character.

    The core binary's hash is part of the name because savestates are only
    valid for the core build that wrote them.
    """
    digest = hashlib.sha1(Path(core_path).read_bytes()).hexdigest()[:10]
    return Path(cache_dir) / f"{course}_{character}_{digest}.state"


def load_or_create_start_state(core: EmulatorCore, cache_dir: str | os.PathLike[str], course: str, character: str) -> bytes:
    """Return the start-line savestate, generating and caching it on first use.

    The core must be started. On a cache miss this walks the menus, which
    takes about ten seconds of wall time.
    """
    path = start_state_path(cache_dir, core.core_path, course, character)
    if path.is_file():
        return path.read_bytes()
    navigate_to_time_trial(core, course, character)
    state = core.save_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(state)
    return state
