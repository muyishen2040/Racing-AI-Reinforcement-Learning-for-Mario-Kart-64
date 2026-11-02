"""Evaluate a saved SAC actor for a few episodes and print the returns.

Usage: python test.py checkpoints/sac/actor_best.pth [--episodes 10] [--video out.mp4]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from mk64_env.video import VideoRecorder
from mk64_env.preview import LivePreview
from mk64_env.wrappers import make_pixel_env
from src.model import ActorNet


def main() -> None:
    """Load the actor and run evaluation episodes with sampled actions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--video", default=None)
    parser.add_argument("--preview", action="store_true", help="show emulator frames in a window")
    parser.add_argument("--preview-every", type=int, default=4, help="update the window every N steps")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of evaluating the deterministic policy")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor = ActorNet(80).to(device)
    actor.load_state_dict(torch.load(args.actor, map_location=device))
    actor.eval()
    env = make_pixel_env(args.course, render_mode="rgb_array")
    recorder = VideoRecorder(args.video, fps=15) if args.video else None
    preview = LivePreview(args.preview_every) if args.preview else None
    step = 0
    episode_returns, episode_progress = [], []
    completed = 0
    for episode in range(args.episodes):
        obs, _ = env.reset()
        total = 0.0
        done = False
        while not done:
            state = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255).to(device)
            with torch.no_grad():
                mean, std, binary_logits = actor(state)
            if args.stochastic:
                continuous = torch.tanh(torch.distributions.Normal(mean, std).sample()) * 80
                binary = torch.distributions.Bernoulli(logits=binary_logits).sample()
            else:
                continuous = torch.tanh(mean) * 80
                binary = (binary_logits >= 0).float()
            action = torch.cat([continuous, binary], dim=-1).cpu().numpy().flatten()
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            if preview:
                preview.show(env.render(), step)
            total += reward
            done = terminated or truncated
            if recorder:
                recorder.add(env.render())
        print(f"episode {episode + 1}: reward {total:.2f} progress {info['progress']:.3f} lap {info['lap']}")
        episode_returns.append(total)
        episode_progress.append(info["progress"])
        completed += int(terminated)
    if episode_returns:
        print(f"mean reward {np.mean(episode_returns):.2f}, mean progress {np.mean(episode_progress):.3f}, completion rate {completed / len(episode_returns):.1%}")
    if recorder:
        recorder.close()
    if preview:
        preview.close()
    env.close()


if __name__ == "__main__":
    main()
