"""Tests that need neither the core nor the ROM. CI runs these."""

import struct

import gymnasium as gym
import numpy as np

import mk64_env
from mk64_env.core import JOYSTICK_MAX, ControllerInput, EmulatorCore
from mk64_env.env import DISCRETE_ACTIONS, MarioKartEnv
from mk64_env.menu import CHARACTERS, COURSES
from mk64_env.ram import (
    COURSE_IDS,
    RACE_DONE,
    RACE_HUMAN_FINISHED,
    RACE_IN_PROGRESS,
    RaceState,
)


def test_controller_maps_n64_buttons_to_retropad():
    """N64 A is RetroPad B, N64 B is RetroPad Y, Z is L2, C buttons are the right stick."""
    port = ControllerInput(a=True, b=True, z=True, r=True, start=True, c_up=True, c_left=True).to_port_state()
    assert port.joypad.b and port.joypad.y and port.joypad.l2 and port.joypad.r and port.joypad.start
    assert not port.joypad.a and not port.joypad.x
    assert port.analog.rstick == (-32767, -32767)


def test_controller_stick_scales_and_inverts_y():
    """Full deflection reaches 32767 and libretro's Y axis points down."""
    port = ControllerInput(joy_x=JOYSTICK_MAX, joy_y=JOYSTICK_MAX).to_port_state()
    assert port.analog.lstick == (32767, -32767)
    assert ControllerInput(joy_x=-JOYSTICK_MAX).to_port_state().analog.lstick[0] == -32767
    assert ControllerInput(joy_x=2 * JOYSTICK_MAX).to_port_state().analog.lstick[0] == 32767
    assert ControllerInput().to_port_state().analog.lstick == (0, 0)


def test_action_to_input_thresholds_buttons():
    """Buttons press at 0.5 and stick values round to integers."""
    ctrl = MarioKartEnv.action_to_input(np.array([-39.6, 12.4, 0.5, 0.49, 1.0]))
    assert ctrl == ControllerInput(joy_x=-40, joy_y=12, a=True, b=False, r=True)


def test_discrete_actions():
    """Twelve unique presets; every steering preset holds A; left and right mirror."""
    names = [name for name, _ in DISCRETE_ACTIONS]
    assert len(names) == 12 and len(set(names)) == 12
    by_name = dict(DISCRETE_ACTIONS)
    for side in ("SOFT_", "", "HARD_", "EXTREME_"):
        left, right = by_name[side + "LEFT"], by_name[side + "RIGHT"]
        assert left.a and right.a
        assert left.joy_x == -right.joy_x < 0
    assert by_name["STRAIGHT"] == ControllerInput(a=True)
    assert by_name["NO_OP"] == ControllerInput()


def test_course_and_character_tables_agree():
    """Every menu course has a course id, and the grids are complete."""
    assert set(COURSES) == set(COURSE_IDS)
    assert len(COURSES) == 16 and len(CHARACTERS) == 8
    assert len(set(COURSES.values())) == 16 and len(set(CHARACTERS.values())) == 8


def test_ram_reads_swap_bytes_within_words():
    """RDRAM is host-endian words, so N64 byte A lives at host offset (A & 0x7FFFFF) ^ 3."""
    buf = bytearray(0x800000)
    address = 0x800DC510
    struct.pack_into("<I", buf, address & 0x7FFFFF, 0x11223344)
    struct.pack_into("<f", buf, (address + 4) & 0x7FFFFF, 1.5)
    core = EmulatorCore.__new__(EmulatorCore)
    core._ram = memoryview(buf)
    assert core.read_u8(address) == 0x11
    assert core.read_u8(address + 3) == 0x44
    assert core.read_u16(address) == 0x1122
    assert core.read_u16(address + 2) == 0x3344
    assert core.read_s16(address + 2) == 0x3344
    assert core.read_u32(address) == 0x11223344
    assert core.read_f32(address + 4) == 1.5
    struct.pack_into("<i", buf, address & 0x7FFFFF, -1)
    assert core.read_s32(address) == -1 and core.read_s16(address) == -1


def test_race_state_finished():
    """Three laps or a finished race state ends the episode."""
    def state(lap: int, race_state: int) -> RaceState:
        return RaceState(race_state, 8, 0.0, lap, float(lap), 0, 0.0, (0.0, 0.0, 0.0), 0)

    assert not state(2, RACE_IN_PROGRESS).finished
    assert state(3, RACE_IN_PROGRESS).finished
    assert state(2, RACE_HUMAN_FINISHED).finished
    assert state(2, RACE_DONE).finished


def test_registered_ids_cover_every_course():
    """Each course has a Box and a Discrete id with the right entry point."""
    for course in COURSES:
        box = gym.spec(f"MarioKart64/{course}-v0")
        disc = gym.spec(f"MarioKart64/{course}-Discrete-v0")
        assert box.entry_point == "mk64_env.env:MarioKartEnv" and box.kwargs["course"] == course
        assert disc.entry_point == "mk64_env.env:MarioKartDiscreteEnv" and disc.kwargs["course"] == course
    assert len([s for s in gym.registry if s.startswith("MarioKart64/")]) == 32


def test_env_constructs_without_core():
    """Construction only records paths and spaces; the core starts on reset."""
    env = mk64_env.MarioKartDiscreteEnv(course="RainbowRoad", core_path="/nonexistent.so", rom_path="/nonexistent.n64")
    assert env.action_space.n == 12
    assert env.observation_space.shape == (120, 160, 3)
    assert env._core is None
