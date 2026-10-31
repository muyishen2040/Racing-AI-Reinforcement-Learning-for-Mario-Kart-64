"""Gymnasium environments for Mario Kart 64 Time Trials."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from PIL import Image

from mk64_env.core import JOYSTICK_MAX, ControllerInput, EmulatorCore
from mk64_env.menu import load_or_create_start_state
from mk64_env.ram import RaceState, read_race_state

DEFAULT_CORE = Path("cores/mupen64plus_next_libretro.so")
DEFAULT_ROM = Path("roms/marioKart.n64")
DEFAULT_CACHE = Path(".mk64_cache")


def _resolve(value: str | os.PathLike[str] | None, env_var: str, default: Path) -> Path:
    if value is not None:
        return Path(value)
    return Path(os.environ.get(env_var, default))


class MarioKartEnv(gym.Env):
    """Time Trial on one course with a hybrid continuous action space.

    Observation: RGB frame resized to `obs_shape` (height, width), uint8.
    Action: Box of 5 floats [joy_x, joy_y, A, B, R]. The stick axes are in
    N64 units [-80, 80]; a button is pressed when its value is >= 0.5.
    Reward: `progress_scale` times the change in course progress (one unit
    per lap) plus `step_penalty` per step.
    Episode end: terminated when the third lap completes, truncated after
    `max_steps` steps.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(
        self,
        course: str = "LuigiRaceway",
        character: str = "mario",
        frame_skip: int = 4,
        obs_shape: tuple[int, int] = (120, 160),
        max_steps: int = 2500,
        progress_scale: float = 100.0,
        step_penalty: float = 0.0,
        core_path: str | os.PathLike[str] | None = None,
        rom_path: str | os.PathLike[str] | None = None,
        cache_dir: str | os.PathLike[str] | None = None,
        render_mode: str | None = None,
    ):
        """Configure the environment. The emulator starts on the first reset().

        core_path, rom_path, and cache_dir fall back to the MK64_CORE,
        MK64_ROM, and MK64_CACHE environment variables, then to cores/,
        roms/, and .mk64_cache/ under the working directory.
        """
        self.course = course
        self.character = character
        self.frame_skip = frame_skip
        self.obs_shape = obs_shape
        self.max_steps = max_steps
        self.progress_scale = progress_scale
        self.step_penalty = step_penalty
        self.render_mode = render_mode
        self.core_path = _resolve(core_path, "MK64_CORE", DEFAULT_CORE)
        self.rom_path = _resolve(rom_path, "MK64_ROM", DEFAULT_ROM)
        self.cache_dir = _resolve(cache_dir, "MK64_CACHE", DEFAULT_CACHE)
        self.observation_space = spaces.Box(0, 255, shape=(obs_shape[0], obs_shape[1], 3), dtype=np.uint8)
        self.action_space = spaces.Box(
            low=np.array([-JOYSTICK_MAX, -JOYSTICK_MAX, 0, 0, 0], dtype=np.float32),
            high=np.array([JOYSTICK_MAX, JOYSTICK_MAX, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self._core: EmulatorCore | None = None
        self._start_state: bytes | None = None
        self._last_progress = 0.0
        self._steps = 0
        self._last_frame: np.ndarray | None = None

    @property
    def core(self) -> EmulatorCore:
        """The running emulator, started on demand."""
        if self._core is None:
            self._core = EmulatorCore(self.core_path, self.rom_path)
            self._core.start()
        return self._core

    def _observe(self) -> np.ndarray:
        frame = self.core.frame()
        self._last_frame = frame
        image = Image.fromarray(frame).resize((self.obs_shape[1], self.obs_shape[0]), Image.BILINEAR)
        return np.asarray(image, dtype=np.uint8)

    def _info(self, state: RaceState) -> dict[str, Any]:
        return {
            "lap": state.lap,
            "progress": state.progress,
            "speed": state.speed,
            "timer": state.timer,
            "rank": state.rank,
            "race_state": state.race_state,
            "position": state.position,
            "yaw": state.yaw,
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Load the start-line savestate and return the first observation."""
        super().reset(seed=seed)
        if self._start_state is None:
            self._start_state = load_or_create_start_state(self.core, self.cache_dir, self.course, self.character)
        # The controller state latched in the frame before unserialize leaks
        # into the first frame after it, so neutralize it before loading.
        self.core.run_frames(1, ControllerInput())
        self.core.load_state(self._start_state)
        self.core.run_frames(1, ControllerInput())
        state = read_race_state(self.core)
        self._last_progress = state.progress
        self._steps = 0
        return self._observe(), self._info(state)

    @staticmethod
    def action_to_input(action: np.ndarray) -> ControllerInput:
        """Convert a 5-vector [joy_x, joy_y, A, B, R] into controller input."""
        action = np.asarray(action, dtype=np.float32).reshape(5)
        return ControllerInput(
            joy_x=int(round(float(action[0]))),
            joy_y=int(round(float(action[1]))),
            a=bool(action[2] >= 0.5),
            b=bool(action[3] >= 0.5),
            r=bool(action[4] >= 0.5),
        )

    def _advance(self, controller: ControllerInput):
        self.core.run_frames(self.frame_skip, controller)
        state = read_race_state(self.core)
        reward = self.progress_scale * (state.progress - self._last_progress) + self.step_penalty
        self._last_progress = state.progress
        self._steps += 1
        terminated = state.finished
        truncated = self._steps >= self.max_steps and not terminated
        return self._observe(), float(reward), terminated, truncated, self._info(state)

    def step(self, action):
        """Hold `action` for frame_skip frames and return the Gymnasium 5-tuple."""
        return self._advance(self.action_to_input(action))

    def render(self):
        """Return the last full-resolution frame as an RGB array."""
        if self.render_mode != "rgb_array":
            return None
        if self._last_frame is None:
            self._last_frame = self.core.frame()
        return self._last_frame

    def close(self):
        """Shut the emulator down."""
        if self._core is not None:
            self._core.close()
            self._core = None


DISCRETE_ACTIONS: list[tuple[str, ControllerInput]] = [
    ("NO_OP", ControllerInput()),
    ("STRAIGHT", ControllerInput(a=True)),
    ("BRAKE", ControllerInput(b=True)),
    ("BACK_UP", ControllerInput(joy_y=-80, b=True)),
    ("SOFT_LEFT", ControllerInput(joy_x=-20, a=True)),
    ("LEFT", ControllerInput(joy_x=-40, a=True)),
    ("HARD_LEFT", ControllerInput(joy_x=-60, a=True)),
    ("EXTREME_LEFT", ControllerInput(joy_x=-80, a=True)),
    ("SOFT_RIGHT", ControllerInput(joy_x=20, a=True)),
    ("RIGHT", ControllerInput(joy_x=40, a=True)),
    ("HARD_RIGHT", ControllerInput(joy_x=60, a=True)),
    ("EXTREME_RIGHT", ControllerInput(joy_x=80, a=True)),
]
"""Named controller presets for the discrete environment."""


class MarioKartDiscreteEnv(MarioKartEnv):
    """MarioKartEnv with a Discrete(12) action space of steering presets."""

    def __init__(self, **kwargs):
        """Accept the same keyword arguments as MarioKartEnv."""
        super().__init__(**kwargs)
        self.action_space = spaces.Discrete(len(DISCRETE_ACTIONS))

    def step(self, action):
        """Apply the preset at index `action` for frame_skip frames."""
        return self._advance(DISCRETE_ACTIONS[int(action)][1])
