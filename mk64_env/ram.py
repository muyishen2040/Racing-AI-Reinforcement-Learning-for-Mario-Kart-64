"""Mario Kart 64 (US) RAM addresses and a reader for the race state.

Addresses come from the n64decomp/mk64 decompilation and were confirmed by
watching them change during a Time Trial on Luigi Raceway. See
docs/ram_map.md for the evidence behind each one.
"""

from __future__ import annotations

from dataclasses import dataclass

from mk64_env.core import EmulatorCore

PLAYERS_BASE = 0x800F6990
"""Address of gPlayers[0]; player i is at PLAYERS_BASE + i * PLAYER_SIZE."""
PLAYER_SIZE = 0xDD8

RACE_STATE = 0x800DC512
"""u16 gRaceState."""
COURSE_TIMER = 0x800DC598
"""f32 gCourseTimer, seconds since the green light."""
CURRENT_COURSE_ID = 0x800DC5A0
"""s16 gCurrentCourseId."""
LAP_COUNT_BY_PLAYER = 0x80164390
"""s32 gLapCountByPlayerId[10]; -1 before the first start-line crossing."""
RANK_BY_PLAYER = 0x801643B8
"""s32 gGPCurrentRaceRankByPlayerId[10]."""
COURSE_PROGRESS_BY_PLAYER = 0x801644D0
"""f32 gCourseCompletionPercentByPlayerId[10]; lap count plus lap fraction."""

# Offsets inside the Player struct.
PLAYER_TYPE = 0x00
PLAYER_RANK = 0x04
PLAYER_LAP = 0x08
PLAYER_POS = 0x14
PLAYER_ROTATION = 0x2C
PLAYER_SPEED = 0x94
PLAYER_CURRENT_SPEED = 0x9C
PLAYER_TOP_SPEED = 0x214

# gRaceState values from the decompilation's defines.h.
RACE_NONE = 0
RACE_PREP = 1
RACE_STAGING = 2
RACE_IN_PROGRESS = 3
RACE_HUMAN_FINISHED = 4
RACE_DONE = 5

COURSE_IDS = {
    "MarioRaceway": 0,
    "ChocoMountain": 1,
    "BowsersCastle": 2,
    "BansheeBoardwalk": 3,
    "YoshiValley": 4,
    "FrappeSnowland": 5,
    "KoopaTroopaBeach": 6,
    "RoyalRaceway": 7,
    "LuigiRaceway": 8,
    "MooMooFarm": 9,
    "ToadsTurnpike": 10,
    "KalimariDesert": 11,
    "SherbetLand": 12,
    "RainbowRoad": 13,
    "WarioStadium": 14,
    "DKsJungleParkway": 18,
}
"""gCurrentCourseId value for each course name."""


@dataclass(frozen=True)
class RaceState:
    """Snapshot of the human player's race progress read from RDRAM."""

    race_state: int
    course_id: int
    timer: float
    lap: int
    progress: float
    rank: int
    speed: float
    position: tuple[float, float, float]
    yaw: int

    @property
    def finished(self) -> bool:
        """True once the player has completed all three laps."""
        return self.lap >= 3 or self.race_state in (RACE_HUMAN_FINISHED, RACE_DONE)


def read_race_state(core: EmulatorCore, player: int = 0) -> RaceState:
    """Read the race state for one player slot (0 is the human in Time Trials)."""
    base = PLAYERS_BASE + player * PLAYER_SIZE
    return RaceState(
        race_state=core.read_u16(RACE_STATE),
        course_id=core.read_s16(CURRENT_COURSE_ID),
        timer=core.read_f32(COURSE_TIMER),
        lap=core.read_s32(LAP_COUNT_BY_PLAYER + 4 * player),
        progress=core.read_f32(COURSE_PROGRESS_BY_PLAYER + 4 * player),
        rank=core.read_s32(RANK_BY_PLAYER + 4 * player),
        speed=core.read_f32(base + PLAYER_CURRENT_SPEED),
        position=(
            core.read_f32(base + PLAYER_POS),
            core.read_f32(base + PLAYER_POS + 4),
            core.read_f32(base + PLAYER_POS + 8),
        ),
        yaw=core.read_s16(base + PLAYER_ROTATION + 2),
    )
