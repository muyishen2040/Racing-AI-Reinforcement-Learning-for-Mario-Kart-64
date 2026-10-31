"""Record a human demonstration with a gamepad and save it as a pickle.

The pickle holds stacked grayscale observations (what the networks see),
the 5-dim actions, rewards, done flags, and the RGB frames for Grad-CAM.
Usage: python GetExpertData.py out.pkl [--course LuigiRaceway]
"""

from __future__ import annotations

import argparse
import pickle

import numpy as np
import pygame

from mk64_env.wrappers import make_pixel_env

WINDOW = (640, 480)


def read_gamepad(joystick: pygame.joystick.JoystickType, buttons: tuple[int, int, int]) -> np.ndarray:
    """Return the 5-dim action [joy_x, joy_y, A, B, R] from the first gamepad.

    Stick axes are clipped to 80% of travel and scaled to N64 units; the
    vertical axis is flipped so pushing up gives a positive joy_y.
    """
    pygame.event.pump()
    clip = lambda value: float(np.clip(value, -0.8, 0.8)) * 100.0
    a, b, r = (int(joystick.get_button(index)) for index in buttons)
    return np.array([clip(joystick.get_axis(0)), -clip(joystick.get_axis(1)), a, b, r], dtype=np.float32)


def main() -> None:
    """Drive one episode in real time and write the recording."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--buttons", type=int, nargs=3, default=(0, 1, 7), metavar=("A", "B", "R"), help="gamepad button indices")
    args = parser.parse_args()

    pygame.init()
    if pygame.joystick.get_count() != 1:
        raise SystemExit(f"expected exactly one gamepad, found {pygame.joystick.get_count()}")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()
    screen = pygame.display.set_mode(WINDOW)
    clock = pygame.time.Clock()

    env = make_pixel_env(args.course, frame_skip=args.frame_skip, render_mode="rgb_array")
    history = {"obs": [], "frames": [], "action": [], "reward": [], "done": []}
    obs, info = env.reset()
    done = False
    total = 0.0
    while not done:
        action = read_gamepad(joystick, tuple(args.buttons))
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        history["obs"].append(np.asarray(obs))
        history["frames"].append(env.render())
        history["action"].append(action)
        history["reward"].append(reward)
        history["done"].append(done)
        obs = next_obs
        total += reward
        frame = pygame.surfarray.make_surface(np.transpose(env.render(), (1, 0, 2)))
        screen.blit(pygame.transform.scale(frame, WINDOW), (0, 0))
        pygame.display.flip()
        clock.tick(60 // args.frame_skip)
        print(f"reward {reward:+.3f} lap {info['lap']} progress {info['progress']:.3f}", end="\r")
    print(f"\ntotal reward {total:.2f} over {len(history['action'])} steps")
    with open(args.output, "wb") as f:
        pickle.dump(history, f)
    env.close()


if __name__ == "__main__":
    main()
