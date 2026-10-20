"""Guard checkpoint rewards against isolated game pixels and hidden HUDs."""

import pytest

from gym_mupen64plus.envs.MarioKart64.mario_kart_env import MarioKartEnv


def env_with_hud_values(values):
    env = MarioKartEnv.__new__(MarioKartEnv)
    env.CHECKPOINT_LOCATIONS = list(range(len(values)))
    env._evaluate_checkpoint = lambda index: values[index]
    env.lap = 1
    return env


def test_isolated_matching_pixel_is_not_a_completed_lap():
    values = [-1] * 933
    values[932] = 1
    env = env_with_hud_values(values)

    assert env._get_current_checkpoint() == -1
    assert env._get_lap() == 1


@pytest.mark.parametrize(
    ("values", "checkpoint", "lap"),
    [
        ([1] * 60 + [-1] * 873, 59, 1),
        ([2] * 100 + [1] * 833, 99, 2),
        ([3] * 933, 932, 3),
        ([1] * 4 + [-1] * 929, -1, 1),
    ],
)
def test_contiguous_hud_progress(values, checkpoint, lap):
    env = env_with_hud_values(values)

    assert env._get_current_checkpoint() == checkpoint
    assert env._get_lap() == lap


def test_missing_hud_does_not_apply_backwards_penalty():
    env = env_with_hud_values([-1] * 933)
    env.last_known_ckpt = 25
    env.episode_over = False
    env._get_current_checkpoint = lambda: -1

    assert env._get_reward() == env.DEFAULT_STEP_REWARD
    assert env.last_known_ckpt == 25
