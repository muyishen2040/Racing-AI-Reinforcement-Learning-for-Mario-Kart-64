"""Train Soft Actor-Critic with the hybrid 5-dim action space on one course.

Observations are 4 stacked 128x128 grayscale frames. Checkpoints go to
checkpoints/sac/ and TensorBoard logs to runs/. Run with --help for options.
"""

from __future__ import annotations

import argparse
import collections
import os
import random
import signal
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from mk64_env.wrappers import make_pixel_env
from mk64_env.preview import LivePreview
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

    def save_training_state(self, path: Path, *, episode: int, step: int, best_reward: float) -> None:
        """Atomically save the state needed to continue at an episode boundary."""
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "version": 1, "config": asdict(self.config), "episode": episode,
            "step": step, "best_reward": best_reward,
            "actor": self.actor.actor_net.state_dict(),
            "critic": self.critic.critic_net.state_dict(),
            "target": self.critic.target_net.state_dict(),
            "actor_optimizer": self.actor.optimizer.state_dict(),
            "critic_optimizer": self.critic.optimizer.state_dict(),
            "log_alpha": self.entropy.log_alpha.detach(),
            "entropy_optimizer": self.entropy.optimizer.state_dict(),
            "replay": list(self.memory.buffer),
            "random_state": random.getstate(),
            "numpy_state": np.random.get_state(),
            "torch_state": torch.get_rng_state(),
            "cuda_state": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else None,
        }
        temporary = path.with_name(path.name + ".tmp")
        try:
            torch.save(state, temporary)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def load_training_state(self, state: dict) -> None:
        """Restore networks, optimizers, replay buffer, and random generators."""
        self.actor.actor_net.load_state_dict(state["actor"])
        self.critic.critic_net.load_state_dict(state["critic"])
        self.critic.target_net.load_state_dict(state["target"])
        self.actor.optimizer.load_state_dict(state["actor_optimizer"])
        self.critic.optimizer.load_state_dict(state["critic_optimizer"])
        with torch.no_grad():
            self.entropy.log_alpha.copy_(state["log_alpha"])
        self.entropy.optimizer.load_state_dict(state["entropy_optimizer"])
        self.entropy.alpha = self.entropy.log_alpha.exp()
        self.memory.buffer.extend(state["replay"])
        random.setstate(state["random_state"])
        np.random.set_state(state["numpy_state"])
        torch.set_rng_state(state["torch_state"])
        if self.device.type == "cuda" and state["cuda_state"] is not None:
            torch.cuda.set_rng_state_all(state["cuda_state"])

    def load_legacy_weights(self, directory: Path) -> None:
        """Start a new run from old best weights; old optimizer/replay is unavailable."""
        actor = torch.load(directory / "actor_best.pth", map_location=self.device, weights_only=True)
        critic = torch.load(directory / "critic_best.pth", map_location=self.device, weights_only=True)
        alpha = torch.load(directory / "entropy_best.pth", map_location=self.device, weights_only=True)
        self.actor.actor_net.load_state_dict(actor)
        self.critic.critic_net.load_state_dict(critic)
        self.critic.target_net.load_state_dict(critic)
        with torch.no_grad():
            self.entropy.log_alpha.copy_(alpha)
        self.entropy.alpha = self.entropy.log_alpha.exp()


def main() -> None:
    """Train SAC until the episode budget is spent."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--course", default=None)
    parser.add_argument("--episodes", type=int, default=None, help="total episode target (including previous episodes)")
    parser.add_argument("--memory-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None, help="steps before updates start")
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--checkpoint-every", type=int, default=1000, help="save recovery state at least every N steps")
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument("--resume", nargs="?", const="", default=None, metavar="PATH", help="resume a trusted recovery checkpoint (default: checkpoint-dir/latest.pt)")
    resume.add_argument("--init-from", metavar="DIR", help="initialize a new run from legacy best weights")
    parser.add_argument("--preview", action="store_true", help="show emulator frames in a window")
    parser.add_argument("--preview-every", type=int, default=4, help="update the window every N steps")
    args = parser.parse_args()
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be at least 1")
    checkpoint_dir = Path(args.checkpoint_dir or Config.checkpoint_dir)
    recovery_path = checkpoint_dir / "latest.pt"
    saved = None
    if args.resume is not None:
        source = Path(args.resume) if args.resume else recovery_path
        if not source.is_file():
            parser.error(f"recovery checkpoint not found: {source}")
        # This checkpoint contains numpy arrays and Python RNG state; never load
        # an untrusted checkpoint because pickle can execute code.
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if saved.get("version") != 1:
            parser.error("unsupported recovery checkpoint version")
        config = Config(**saved["config"])
        for option, saved_value, name in ((args.course, config.course, "course"),
                                          (args.memory_len, config.memory_len, "memory-len"),
                                          (args.warmup, config.replay_buffer_warmup, "warmup")):
            if option is not None and option != saved_value:
                parser.error(f"--{name} conflicts with the checkpoint")
        if args.episodes is not None:
            config.episodes = args.episodes
        if args.checkpoint_dir is None:
            checkpoint_dir = source.parent
            recovery_path = checkpoint_dir / "latest.pt"
        config.checkpoint_dir = str(checkpoint_dir)
    else:
        if recovery_path.exists():
            parser.error(f"{recovery_path} already exists; use --resume or a different --checkpoint-dir")
        config = Config(course=args.course or Config.course, episodes=args.episodes if args.episodes is not None else Config.episodes,
                        memory_len=args.memory_len if args.memory_len is not None else Config.memory_len,
                        replay_buffer_warmup=args.warmup if args.warmup is not None else Config.replay_buffer_warmup,
                        checkpoint_dir=str(checkpoint_dir))
    if config.memory_len < 1 or config.episodes < 1 or config.replay_buffer_warmup < 0:
        parser.error("episodes and memory-len must be positive; warmup cannot be negative")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    writer = SummaryWriter()
    preview = LivePreview(args.preview_every) if args.preview else None
    env = make_pixel_env(config.course, config.frame_skip, 128, config.stack_frames, render_mode="rgb_array" if preview else None)
    agent = SAC(config, device)
    if saved is not None:
        agent.load_training_state(saved)
        start_episode, cur_step, best_reward = saved["episode"], saved["step"], saved["best_reward"]
        print(f"Resumed at episode {start_episode + 1}, step {cur_step}; replay contains {len(agent.memory)} transitions", flush=True)
        del saved
    else:
        if args.init_from:
            agent.load_legacy_weights(Path(args.init_from))
            print("Initialized from legacy weights; optimizer and replay buffer start fresh.", flush=True)
        start_episode, cur_step, best_reward = 0, 0, -np.inf
    stop_requested = False
    def request_stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("Stop requested; saving a recovery checkpoint after the current step.", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    last_saved_step = cur_step
    for episode in range(start_episode, config.episodes):
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
            if preview:
                preview.show(env.render(), cur_step)
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
            if not done and (stop_requested or cur_step - last_saved_step >= args.checkpoint_every):
                agent.save_training_state(recovery_path, episode=episode, step=cur_step, best_reward=best_reward)
                last_saved_step = cur_step
                print(f"Recovery checkpoint saved at step {cur_step} (resumes at a new episode).", flush=True)
                if stop_requested:
                    break
        if stop_requested and not done:
            break
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
        if stop_requested or cur_step - last_saved_step >= args.checkpoint_every or episode + 1 == config.episodes:
            agent.save_training_state(recovery_path, episode=episode + 1, step=cur_step, best_reward=best_reward)
            last_saved_step = cur_step
            print(f"Recovery checkpoint saved after episode {episode + 1} at step {cur_step}.", flush=True)
        if stop_requested:
            break
    if preview:
        preview.close()
    env.close()
    writer.close()


if __name__ == "__main__":
    main()
