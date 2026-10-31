"""Observation pipelines shared by the training scripts."""

from __future__ import annotations

import gymnasium as gym
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation

from mk64_env.env import MarioKartDiscreteEnv, MarioKartEnv


def make_pixel_env(
    course: str = "LuigiRaceway",
    frame_skip: int = 4,
    size: int = 128,
    stack: int = 4,
    discrete: bool = False,
    **kwargs,
) -> gym.Env:
    """Build an env whose observations are `stack` grayscale frames of size x size.

    Observations are uint8 arrays of shape (stack, size, size). Extra keyword
    arguments go to MarioKartEnv. Set discrete=True for the Discrete(12)
    action space.
    """
    cls = MarioKartDiscreteEnv if discrete else MarioKartEnv
    env = cls(course=course, frame_skip=frame_skip, obs_shape=(size, size), **kwargs)
    env = GrayscaleObservation(env, keep_dim=False)
    return FrameStackObservation(env, stack_size=stack)
