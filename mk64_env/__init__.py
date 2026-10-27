"""Headless Mario Kart 64 Gymnasium environments.

Importing this package registers two environment ids per course, for example
``MarioKart64/LuigiRaceway-v0`` (hybrid Box actions) and
``MarioKart64/LuigiRaceway-Discrete-v0`` (Discrete(12) actions).
"""

from gymnasium.envs.registration import register

from mk64_env.core import ControllerInput, EmulatorCore
from mk64_env.env import DISCRETE_ACTIONS, MarioKartDiscreteEnv, MarioKartEnv
from mk64_env.menu import CHARACTERS, COURSES
from mk64_env.ram import RaceState, read_race_state

__all__ = [
    "CHARACTERS",
    "COURSES",
    "DISCRETE_ACTIONS",
    "ControllerInput",
    "EmulatorCore",
    "MarioKartDiscreteEnv",
    "MarioKartEnv",
    "RaceState",
    "read_race_state",
]

for _course in COURSES:
    register(id=f"MarioKart64/{_course}-v0", entry_point="mk64_env.env:MarioKartEnv", kwargs={"course": _course})
    register(
        id=f"MarioKart64/{_course}-Discrete-v0",
        entry_point="mk64_env.env:MarioKartDiscreteEnv",
        kwargs={"course": _course},
    )
