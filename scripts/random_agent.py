"""Run one episode with a random or a hold-A agent and record it to MP4.

Usage: python scripts/random_agent.py [--agent random|straight] [--steps N] [--video out.mp4]
"""

from __future__ import annotations

import argparse
import time

import mk64_env
from mk64_env.video import VideoRecorder


def main() -> None:
    """Parse arguments, run the episode, and print progress statistics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=["random", "straight"], default="random")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--video", default=None)
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()

    env = mk64_env.MarioKartDiscreteEnv(course=args.course, frame_skip=args.frame_skip, max_steps=args.steps, render_mode="rgb_array")
    recorder = VideoRecorder(args.video, fps=60 // args.frame_skip) if args.video else None
    obs, info = env.reset()
    total = 0.0
    t0 = time.time()
    for step in range(args.steps):
        action = env.action_space.sample() if args.agent == "random" else 1
        obs, reward, terminated, truncated, info = env.step(action)
        total += reward
        if recorder:
            recorder.add(env.render())
        if step % 100 == 0:
            print(f"step {step} lap {info['lap']} progress {info['progress']:.3f} speed {info['speed']:.0f} reward {total:.2f}")
        if terminated or truncated:
            break
    elapsed = time.time() - t0
    print(f"{step + 1} steps in {elapsed:.1f}s ({(step + 1) * args.frame_skip / elapsed:.0f} frames/s); total reward {total:.2f}; final {info}")
    if recorder:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
