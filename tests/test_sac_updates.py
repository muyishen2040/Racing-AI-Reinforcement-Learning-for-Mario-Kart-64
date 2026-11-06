"""Regression tests for hybrid action gradients and exploration."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("tensorboard")

from src.model import Actor, Critic
from train_sac import hybrid_actor_loss


def test_sampled_continuous_actions_use_independent_noise():
    actor = Actor.__new__(Actor)
    actor.actor_net = lambda state: (torch.zeros(128, 2), torch.ones(128, 2), torch.zeros(128, 3))
    torch.manual_seed(123)
    actions, _, _, _, _ = actor.evaluate(torch.zeros(128, 1))
    # Shared scalar noise would make both axes and every batch row identical.
    assert not torch.equal(actions[:, 0], actions[:, 1])
    assert actions[:, 0].std() > 1.0


def test_critic_feedback_changes_probability_of_good_button():
    torch.manual_seed(1)
    logit = torch.tensor(0.0, requires_grad=True)
    distribution = torch.distributions.Bernoulli(logits=logit.expand(10000, 1))
    action = distribution.sample()
    log_probability = distribution.log_prob(action)
    q = action  # The critic prefers pressing the button.
    loss = hybrid_actor_loss(q, torch.zeros_like(q), log_probability, torch.tensor(0.0))
    loss.backward()
    # Gradient descent must increase the logit of the rewarded button.
    assert -0.28 < logit.grad.item() < -0.22


def test_new_target_network_matches_critic():
    critic = Critic(torch.device("cpu"), critic_lr=1e-3, tau=0.01)
    for main, target in zip(critic.critic_net.parameters(), critic.target_net.parameters()):
        torch.testing.assert_close(main, target)
