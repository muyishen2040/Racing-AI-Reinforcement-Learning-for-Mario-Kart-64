"""Evaluate a saved SAC actor for a few episodes and print the returns.

Usage: python test.py checkpoints/sac/actor_best.pth [--episodes 10] [--video out.mp4]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mk64_env.video import VideoRecorder
from mk64_env.wrappers import make_pixel_env
from src.model import ActorNet


def main() -> None:
    """Load the actor and run evaluation episodes with sampled actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", default=None)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ActorNet(80).to(device)
    actor.load_state_dict(torch.load(args.actor, map_location=device))
    actor.eval()
    env = make_pixel_env(args.course, render_mode="rgb_array")
    recorder = VideoRecorder(args.video, fps=15) if args.video else None
    for episode in range(args.episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            state = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255).to(device)
            with torch.no_grad():
                mean, std, binary_logits = actor(state)
            continuous = torch.tanh(torch.distributions.Normal(mean, std).sample()) * 80
            binary = torch.distributions.Bernoulli(logits=binary_logits).sample()
            action = torch.cat([continuous, binary], dim=-1).cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            total += reward
            done = terminated or truncated
            if recorder:
                recorder.add(env.render())
        print(f"episode {episode + 1}: reward {total:.2f} progress {info['progress']:.3f} lap {info['lap']}")
    if recorder:
        recorder.close()
    env.close()


if __name__ == "__main__":
    main()
