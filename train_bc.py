"""Behavior cloning of the SAC actor and critic from recorded demonstrations.

Usage: python train_bc.py demo1.pkl demo2.pkl ... [--epochs 150]
Checkpoints go to checkpoints/bc/. Every --eval-every epochs the actor drives
one episode in the environment.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import torch

from mk64_env.wrappers import make_pixel_env
from src.model import ActorNet, CriticNet

GAMMA = 0.99
BATCH_SIZE = 256


def discounted_returns(rewards: list[float], dones: list[bool], gamma: float) -> torch.Tensor:
    """Monte Carlo return of each step, bootstrapped with 0.5 at the final step."""
    q = torch.zeros((len(rewards), 1))
    q[-1] = 0.5
    for i in range(len(rewards) - 2, -1, -1):
        q[i] = rewards[i] + gamma * q[i + 1] * (1 - float(dones[i]))
    return q


def load_demos(paths: list[str], device: torch.device):
    """Concatenate observations, actions, and returns from the recordings."""
    obs, actions, returns = [], [], []
    for path in paths:
        with open(path, "rb") as f:
            data = pickle.load(f)
        obs.append(torch.from_numpy(np.stack(data["obs"])).float().div(255))
        actions.append(torch.tensor(np.stack(data["action"]), dtype=torch.float32))
        returns.append(discounted_returns(data["reward"], data["done"], GAMMA))
    return torch.cat(obs).to(device), torch.cat(actions).to(device), torch.cat(returns).to(device)


def evaluate(actor: ActorNet, course: str, device: torch.device) -> float:
    """Drive one episode with the actor's mean action and return the reward sum."""
    env = make_pixel_env(course)
    obs, _ = env.reset()
    done = False
    total = 0.0
    while not done:
        state = torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255).to(device)
        with torch.no_grad():
            mean, _, binary_logits = actor(state)
        action = np.concatenate([(torch.tanh(mean) * 80).squeeze(0).cpu().numpy(), (binary_logits.squeeze(0).cpu().numpy() > 0).astype(np.float32)])
        obs, reward, terminated, truncated, _ = env.step(action)
        total += reward
        done = terminated or truncated
    env.close()
    return total


def main() -> None:
    """Fit the actor to the demonstrated actions and the critic to the returns."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recordings", nargs="+")
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--eval-every", type=int, default=150)
    parser.add_argument("--checkpoint-dir", default="checkpoints/bc")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obs_all, action_all, q_all = load_demos(args.recordings, device)
    order = np.arange(obs_all.shape[0])

    actor = ActorNet(80).to(device)
    critic = CriticNet().to(device)
    actor_lr, critic_lr = 5e-4, 1e-5
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
    actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, 0.99)
    critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, 0.99)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        np.random.shuffle(order)
        for start in range(0, len(order), BATCH_SIZE):
            idx = order[start : start + BATCH_SIZE]
            obs, action, q = obs_all[idx], action_all[idx], q_all[idx]
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            mean, std, binary_logits = actor(obs)
            continuous = torch.tanh(torch.distributions.Normal(mean, std).rsample()) * 80
            loss_c = torch.nn.functional.mse_loss(continuous, action[:, :2])
            loss_d = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, action[:, 2:])
            q1, q2 = critic(obs, action)
            loss_q = torch.nn.functional.mse_loss(q1, q) + torch.nn.functional.mse_loss(q2, q)
            (loss_c + loss_d + loss_q).backward()
            actor_optimizer.step()
            critic_optimizer.step()
        print(f"epoch {epoch:4d} actor loss {loss_c.item() + loss_d.item():.4f} critic loss {loss_q.item():.4f}")
        if actor_scheduler.get_last_lr()[0] > actor_lr * 1e-2:
            actor_scheduler.step()
            critic_scheduler.step()
        if epoch % args.eval_every == 0:
            torch.save(actor.state_dict(), checkpoint_dir / f"actor_{epoch}.pth")
            torch.save(critic.state_dict(), checkpoint_dir / f"critic_{epoch}.pth")
            print(f"epoch {epoch} evaluation reward {evaluate(actor, args.course, device):.2f}")


if __name__ == "__main__":
    main()
