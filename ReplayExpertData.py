"""Replay a recorded demonstration through the environment and report the return.

Usage: python ReplayExpertData.py demo.pkl [--video out.mp4]
"""

from __future__ import annotations

import argparse
import pickle

from mk64_env.video import VideoRecorder
from mk64_env.wrappers import make_pixel_env


def main() -> None:
    """Feed the recorded actions back into a fresh episode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    with open(args.recording, "rb") as f:
        history = pickle.load(f)
    env = make_pixel_env(args.course, frame_skip=args.frame_skip, render_mode="rgb_array")
    recorder = VideoRecorder(args.video, fps=60 // args.frame_skip) if args.video else None
    env.reset()
    total = 0.0
    steps = 0
    for action in history["action"]:
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
        steps += 1
        if recorder:
            recorder.add(env.render())
        if terminated or truncated:
            break
    print(f"total reward {total:.2f} over {steps} steps; final progress {info['progress']:.3f}")
    if recorder:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
