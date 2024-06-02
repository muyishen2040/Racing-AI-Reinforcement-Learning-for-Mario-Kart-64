from src.utils import *
from src.ReplayBuffer import *
from src.model import *

IDX2ACTIONS = [
    [0, 0, 0], 
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1]
]

ACTIONS2IDX = {tuple(a): i for i, a in enumerate(IDX2ACTIONS)}


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.actor = PolicyNetwork(config.state_dims, config.action_c_dims, config.action_d_dims, config.action_scale, config.action_bias).to(config.device)
        self.critic = QNetwork(config.state_dims, config.action_c_dims, config.action_d_dims).to(config.device)
        self.critic_target = QNetwork(config.state_dims, config.action_c_dims, config.action_d_dims).to(config.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.lr)
        self.replay_buffer = ReplayBuffer(config)
        self.log_alpha_c = torch.nn.Parameter(torch.tensor(0.0).to(config.device))
        self.log_alpha_d = torch.nn.Parameter(torch.tensor(0.0).to(config.device))
        self.target_entropy_c, self.target_entropy_d = -config.action_c_dims, -config.action_d_dims
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha_c, self.log_alpha_d], lr=config.entropy_lr)
        self.alpha_c, self.alpha_d = self.log_alpha_c.exp(), self.log_alpha_d.exp()


    def act(self, state, eval=False, random=False):
        assert not (random and eval), "random and eval cannot be True at the same time"
        if random:
            return [(np.random.rand()-0.5)*2*self.config.action_scale, 0, *IDX2ACTIONS[np.random.randint(8)]]
        action_c, action_d, _, _, deterministic_action_c, deterministic_action_d = self.actor.sample(state)
        if eval:
            return [deterministic_action_c.item(), 0, *IDX2ACTIONS[deterministic_action_d.item()]]
        else:
            return [action_c.item(), 0, *IDX2ACTIONS[action_d.item()]]

    def preprocess_obs(self, obs):
        return obs.to(self.config.device)

    def learn(self):
        state, action_c, action_d, reward, next_state, done = self.replay_buffer.sample()

        with torch.no_grad():
            next_action_c, next_action_d, next_log_prob_c, next_log_prob_d, _, _ = self.actor.sample(next_state)
            next_prob_d = next_log_prob_d.exp()
            next_target_q_values1, next_target_q_values2 = self.critic_target(next_state, next_action_c)
            next_target_q_values = next_prob_d * (torch.min(next_target_q_values1, next_target_q_values2) - self.alpha_c * next_log_prob_c * next_prob_d - self.alpha_d * next_log_prob_d)
            next_target_q_values = reward.unsqueeze(1) + self.config.gamma * next_target_q_values.sum(1, keepdim=True) * ~done.unsqueeze(1)

        q_values1, q_values2 = self.critic(state, action_c)
        q_value1 = q_values1.gather(1, action_d)
        q_value2 = q_values2.gather(1, action_d)
        q_loss1 = torch.nn.functional.mse_loss(q_value1, next_target_q_values)
        q_loss2 = torch.nn.functional.mse_loss(q_value2, next_target_q_values)
        q_loss = (q_loss1 + q_loss2) / 2

        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()

        action_c, action_d, log_prob_c, log_prob_d, _, _ = self.actor.sample(state)
        prob_d = log_prob_d.exp()
        cur_q_values1, cur_q_values2 = self.critic(state, action_c)
        q_values = torch.min(cur_q_values1, cur_q_values2)
        policy_loss_c = (prob_d * (self.alpha_c * prob_d * log_prob_c - q_values)).sum(1).mean()
        policy_loss_d = (prob_d * (self.alpha_d * log_prob_d - q_values)).sum(1).mean()
        policy_loss = policy_loss_c + policy_loss_d

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        alpha_c_loss = (-self.log_alpha_c * prob_d.detach() * (prob_d.detach() * log_prob_c.detach() + self.target_entropy_c)).sum(1).mean()
        alpha_d_loss = (-self.log_alpha_d * prob_d.detach() * (log_prob_d.detach() + self.target_entropy_d)).sum(1).mean()
        alpha_loss = alpha_c_loss + alpha_d_loss
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha_c, self.alpha_d = self.log_alpha_c.exp(), self.log_alpha_d.exp()

        critic_param = self.critic.state_dict()
        target_param = self.critic_target.state_dict()
        for key in target_param:
            target_param[key] = target_param[key] * (1-self.config.tau) + critic_param[key] * self.config.tau
        self.critic_target.load_state_dict(target_param)
        
        return q_loss.item(), policy_loss.item(), alpha_loss.item()
