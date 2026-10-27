"""Train Soft Actor-Critic with the hybrid 5-dim action space on one course.

Observations are 4 stacked 128x128 grayscale frames. Checkpoints go to
checkpoints/sac/ and TensorBoard logs to runs/. Run with --help for options.
"""

from __future__ import annotations

import argparse
import collections
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mk64_env.wrappers import make_pixel_env
from src.model import Actor, Critic, Entropy


@dataclass
class Config:
    """Hyperparameters for SAC training."""

    course: str = "LuigiRaceway"
    gamma: float = 0.99
    tau: float = 0.01
    entropy_lr: float = 1e-4
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    batch_size: int = 256
    episodes: int = 10000
    frame_skip: int = 4
    stack_frames: int = 4
    memory_len: int = 100_000
    replay_buffer_warmup: int = 2000
    stuck_window: int = 90
    stuck_penalty: float = -10.0
    checkpoint_dir: str = "checkpoints/sac"


class ReplayBuffer:
    """Fixed-size FIFO of transitions with uint8 observations."""

    def __init__(self, maxlen: int):
        """Create a buffer holding at most `maxlen` transitions."""
        self.buffer: collections.deque = collections.deque(maxlen=maxlen)

    def push(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        """Store one transition."""
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device):
        """Return a batch of float tensors on `device`, observations scaled to [0, 1]."""
        batch = random.sample(self.buffer, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        to_obs = lambda frames: torch.from_numpy(np.stack(frames)).to(device).float() / 255.0
        return (
            to_obs(obs),
            torch.from_numpy(np.stack(actions)).float().to(device),
            torch.tensor(rewards, dtype=torch.float32, device=device).unsqueeze(1),
            to_obs(next_obs),
            torch.tensor(dones, dtype=torch.float32, device=device).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class SAC:
    """Actor, twin critics, and entropy temperature with one update step."""

    def __init__(self, config: Config, device: torch.device):
        """Build the networks on `device`."""
        self.config = config
        self.device = device
        self.memory = ReplayBuffer(config.memory_len)
        min_action = torch.tensor([-80.0, -80.0, 0, 0, 0])
        max_action = torch.tensor([80.0, 80.0, 1, 1, 1])
        self.actor = Actor(device, config.actor_lr, min_action, max_action)
        self.critic = Critic(device, config.critic_lr, config.tau)
        self.entropy = Entropy(device, config.entropy_lr, 2)

    def update(self) -> tuple[float, float, float]:
        """Run one gradient step on a sampled batch. Returns actor, critic, and alpha losses."""
        states, actions, rewards, next_states, dones = self.memory.sample(self.config.batch_size, self.device)
        with torch.no_grad():
            new_action, log_prob_, _, _, _ = self.actor.evaluate(next_states)
            target_q1, target_q2 = self.critic.get_target_q_value(next_states, new_action)
            target_q = rewards + (1 - dones) * self.config.gamma * (torch.min(target_q1, target_q2) - self.entropy.alpha * log_prob_)
        current_q1, current_q2 = self.critic.get_q_value(states, actions)
        q_loss = self.critic.learn(current_q1, current_q2, target_q.detach())
        a_, log_prob, _, _, _ = self.actor.evaluate(states)
        q1, q2 = self.critic.get_q_value(states, a_)
        actor_loss = (self.entropy.alpha * log_prob - torch.min(q1, q2)).mean()
        self.actor.learn(actor_loss)
        alpha_loss = -(self.entropy.log_alpha.exp() * (log_prob + self.entropy.target_entropy).detach()).mean()
        self.entropy.learn(alpha_loss)
        self.entropy.alpha = self.entropy.log_alpha.exp()
        self.critic.update()
        return actor_loss.item(), q_loss, alpha_loss.item()

    def save(self, directory: Path, tag: str) -> None:
        """Write actor, critic, and log-alpha state dicts under `directory`."""
        directory.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.actor_net.state_dict(), directory / f"actor_{tag}.pth")
        torch.save(self.critic.critic_net.state_dict(), directory / f"critic_{tag}.pth")
        torch.save(self.entropy.log_alpha, directory / f"entropy_{tag}.pth")


def main() -> None:
    """Train SAC until the episode budget is spent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--episodes", type=int, default=Config.episodes)
    parser.add_argument("--memory-len", type=int, default=Config.memory_len)
    parser.add_argument("--warmup", type=int, default=Config.replay_buffer_warmup, help="steps before updates start")
    args = parser.parse_args()
    config = Config(course=args.course, episodes=args.episodes, memory_len=args.memory_len, replay_buffer_warmup=args.warmup)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    env = make_pixel_env(config.course, config.frame_skip, 128, config.stack_frames)
    agent = SAC(config, device)
    checkpoint_dir = Path(config.checkpoint_dir)
    best_reward = -np.inf
    cur_step = 0
    for episode in range(config.episodes):
        obs, _ = env.reset()
        reward_window: deque = deque(maxlen=config.stuck_window)
        total_reward = total_q = total_policy = total_alpha = 0.0
        done = False
        step_in_episode = 0
        while not done:
            cur_step += 1
            step_in_episode += 1
            with torch.no_grad():
                action = agent.actor.choose_action(torch.from_numpy(np.asarray(obs)).unsqueeze(0).float().div(255).to(device))
            action = action.flatten()
            next_obs, reward, terminated, truncated, info = env.step(action)
            reward_window.append(reward)
            if step_in_episode > config.stuck_window and sum(reward_window) <= 0.0:
                reward += config.stuck_penalty
                terminated = True
            done = terminated or truncated
            total_reward += reward
            agent.memory.push(np.asarray(obs), action, reward, np.asarray(next_obs), terminated)
            obs = next_obs
            if len(agent.memory) > max(config.batch_size, config.replay_buffer_warmup):
                policy_loss, q_loss, alpha_loss = agent.update()
                total_q += q_loss
                total_policy += policy_loss
                total_alpha += alpha_loss
        writer.add_scalar("Reward/episode", total_reward, episode)
        writer.add_scalar("Progress/episode", info["progress"], episode)
        writer.add_scalar("Loss/q_loss", total_q / step_in_episode, episode)
        writer.add_scalar("Loss/policy_loss", total_policy / step_in_episode, episode)
        writer.add_scalar("Loss/alpha_loss", total_alpha / step_in_episode, episode)
        writer.add_scalar("Alpha/alpha", agent.entropy.alpha.item(), episode)
        print(f"episode {episode + 1} reward {total_reward:.2f} progress {info['progress']:.3f} steps {step_in_episode} total_steps {cur_step}")
        if total_reward > best_reward:
            best_reward = total_reward
            agent.save(checkpoint_dir, "best")
        if episode % 50 == 0 and episode > 0:
            agent.save(checkpoint_dir, str(episode))


if __name__ == "__main__":
    main()
