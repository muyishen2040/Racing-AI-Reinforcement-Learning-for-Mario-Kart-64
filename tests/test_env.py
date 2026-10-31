"""Integration tests that drive the real core. They need cores/ and roms/."""

import numpy as np
import pytest

import mk64_env
from mk64_env.env import DEFAULT_CORE, DEFAULT_ROM

pytestmark = [
    pytest.mark.emulator,
    pytest.mark.skipif(
        not (DEFAULT_CORE.is_file() and DEFAULT_ROM.is_file()), reason="core or ROM missing"
    ),
]

STRAIGHT = 1


@pytest.fixture(scope="module")
def env():
    """One discrete Luigi Raceway environment shared by the module."""
    env = mk64_env.MarioKartDiscreteEnv(course="LuigiRaceway", frame_skip=4, render_mode="rgb_array")
    yield env
    env.close()


def test_reset_starts_at_green_light(env):
    obs, info = env.reset()
    assert obs.shape == (120, 160, 3) and obs.dtype == np.uint8
    assert info["lap"] == -1
    assert 0.0 < info["timer"] < 1.0
    assert env.core.read_s16(0x800DC5A0) == 8


def test_holding_a_makes_progress(env):
    env.reset()
    total = 0.0
    for _ in range(120):
        obs, reward, terminated, truncated, info = env.step(STRAIGHT)
        total += reward
        assert not terminated and not truncated
    assert info["lap"] == 0, "the kart should have crossed the start line"
    assert info["progress"] > 0.05
    assert info["speed"] > 200
    assert total > 5.0


def test_frame_matches_libretro_screenshot(env):
    env.reset()
    env.step(STRAIGHT)
    shot = env.core._video.screenshot()
    expected = np.frombuffer(shot.data, dtype=np.uint8).reshape(shot.height, shot.width, 4)[..., :3]
    frame = env.render()
    assert frame.shape == expected.shape == (240, 640, 3)
    assert np.array_equal(frame, expected)


def test_reset_is_deterministic(env):
    env.reset()
    first = [env.step(STRAIGHT)[4]["position"] for _ in range(30)]
    env.reset()
    second = [env.step(STRAIGHT)[4]["position"] for _ in range(30)]
    assert first == second


def test_first_reset_matches_reset_after_driving():
    """A fresh core's first reset and a mid-race reset give the same trajectory."""
    fresh = mk64_env.MarioKartDiscreteEnv(course="LuigiRaceway")
    try:
        fresh.reset()
        first = [fresh.step(STRAIGHT)[4]["position"] for _ in range(5)]
        for _ in range(40):
            fresh.step(STRAIGHT)
        fresh.reset()
        second = [fresh.step(STRAIGHT)[4]["position"] for _ in range(5)]
    finally:
        fresh.close()
    assert first == second


def test_two_envs_in_one_process():
    a = mk64_env.MarioKartDiscreteEnv(course="LuigiRaceway")
    b = mk64_env.MarioKartDiscreteEnv(course="LuigiRaceway")
    try:
        a.reset()
        b.reset()
        for _ in range(20):
            a.step(STRAIGHT)
        info_b = b.step(0)[4]
        info_a = a.step(STRAIGHT)[4]
        assert info_a["progress"] > info_b["progress"]
    finally:
        a.close()
        b.close()


def test_registered_ids():
    env = mk64_env.__dict__["MarioKartEnv"]
    import gymnasium as gym

    made = gym.make("MarioKart64/LuigiRaceway-Discrete-v0")
    assert made.action_space.n == 12
    made.close()
