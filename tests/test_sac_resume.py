"""Recovery preserves learning state and leaves the previous save intact on failure."""

import random
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from train_sac import Config, ReplayBuffer, SAC


def small_agent():
    agent = SAC.__new__(SAC)
    agent.config = Config(memory_len=4)
    agent.device = torch.device("cpu")
    agent.memory = ReplayBuffer(4)
    actor = torch.nn.Linear(2, 1)
    critic = torch.nn.Linear(2, 1)
    target = torch.nn.Linear(2, 1)
    alpha = torch.zeros(1, requires_grad=True)
    agent.actor = SimpleNamespace(actor_net=actor, optimizer=torch.optim.Adam(actor.parameters()))
    agent.critic = SimpleNamespace(critic_net=critic, target_net=target, optimizer=torch.optim.Adam(critic.parameters()))
    agent.entropy = SimpleNamespace(log_alpha=alpha, optimizer=torch.optim.Adam([alpha]))
    for model, opt in ((actor, agent.actor.optimizer), (critic, agent.critic.optimizer)):
        model(torch.ones(1, 2)).sum().backward()
        opt.step()
    alpha.sum().backward()
    agent.entropy.optimizer.step()
    return agent


def test_training_state_round_trip(tmp_path):
    original = small_agent()
    obs = np.array([1, 2], dtype=np.uint8)
    original.memory.push(obs, np.array([0.5], dtype=np.float32), 1.0, obs + 1, False)
    path = tmp_path / "latest.pt"
    original.save_training_state(path, episode=7, step=42, best_reward=2.5)
    expected_random = (random.random(), np.random.random(), torch.rand(1))
    restored = small_agent()
    state = torch.load(path, map_location="cpu", weights_only=False)
    restored.load_training_state(state)
    assert (state["episode"], state["step"], state["best_reward"]) == (7, 42, 2.5)
    assert len(restored.memory) == 1
    np.testing.assert_array_equal(restored.memory.buffer[0][0], obs)
    for a, b in ((original.actor.actor_net, restored.actor.actor_net),
                 (original.critic.critic_net, restored.critic.critic_net),
                 (original.critic.target_net, restored.critic.target_net)):
        for p, q in zip(a.parameters(), b.parameters()):
            torch.testing.assert_close(p, q)
    assert restored.actor.optimizer.state_dict()["state"]
    assert restored.critic.optimizer.state_dict()["state"]
    assert restored.entropy.optimizer.state_dict()["state"]
    torch.testing.assert_close(restored.entropy.log_alpha, original.entropy.log_alpha)
    assert random.random() == expected_random[0]
    assert np.random.random() == expected_random[1]
    torch.testing.assert_close(torch.rand(1), expected_random[2])


def test_failed_save_does_not_replace_checkpoint(tmp_path, monkeypatch):
    agent = small_agent()
    path = tmp_path / "latest.pt"
    path.write_bytes(b"previous")
    def fail(state, destination):
        destination.write_bytes(b"incomplete")
        raise OSError("disk full")
    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(OSError, match="disk full"):
        agent.save_training_state(path, episode=0, step=0, best_reward=0.0)
    assert path.read_bytes() == b"previous"
    assert not (tmp_path / "latest.pt.tmp").exists()
