# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from torch.distributions import Normal
# import pdb


# class ActorNet(nn.Module):
#     def __init__(self, max_action, input_shape=(4, 128, 128), action_dim=5):
#         super(ActorNet, self).__init__()
#         self.max_action = max_action
#         self.conv_layers = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.flattened_size = self._get_conv_output(input_shape)
#         self.fc = nn.Linear(self.flattened_size, 512)
#         self.mean = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim - 3)
#         )
#         self.log_std = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, action_dim - 3)
#         )
#         self.binary_logits = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 3)
#         )

#         self.forward_bias = nn.Parameter(torch.tensor(10.0))

#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             input = torch.zeros(1, *shape)
#             output = self.conv_layers(input)
#             return int(np.prod(output.size()))

#     def forward(self, state):
#         x = self.conv_layers(state)
#         x = F.relu(self.fc(x))
#         mean = self.mean(x)
#         log_std = self.log_std(x)
#         log_std = torch.clamp(log_std, -20, 2)
#         std = log_std.exp()
#         binary_logits = self.binary_logits(x)

#         binary_logits[:, 0] += self.forward_bias
#         return mean, std, binary_logits

# class CriticNet(nn.Module):
#     def __init__(self, input_shape=(4, 128, 128), action_dim=5):
#         super(CriticNet, self).__init__()
#         self.conv_layers_q1 = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.flattened_size = self._get_conv_output(input_shape)
#         self.fc_q1 = nn.Sequential(
#             nn.Linear(self.flattened_size + action_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#         self.conv_layers_q2 = nn.Sequential(
#             nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=4, stride=2),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1),
#             nn.ReLU(),
#             nn.Flatten()
#         )
#         self.fc_q2 = nn.Sequential(
#             nn.Linear(self.flattened_size + action_dim, 512),
#             nn.ReLU(),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.ReLU(),
#             nn.Linear(256, 1)
#         )

#     def _get_conv_output(self, shape):
#         with torch.no_grad():
#             input = torch.zeros(1, *shape)
#             output = self.conv_layers_q1(input)
#             return int(np.prod(output.size()))

#     def forward(self, state, action):
#         state_features_q1 = self.conv_layers_q1(state)
#         q1 = torch.cat([state_features_q1, action], 1)
#         q1 = self.fc_q1(q1)

#         state_features_q2 = self.conv_layers_q2(state)
#         q2 = torch.cat([state_features_q2, action], 1)
#         q2 = self.fc_q2(q2)
#         return q1, q2
    

# class Actor:
#     def __init__(self, device, actor_lr, min_action, max_action):
#         self.device = device
#         self.actor_lr = actor_lr
#         self.min_action = min_action.to(device)
#         self.max_action = max_action.to(device)
#         self.actor_net = ActorNet(self.max_action).to(device)
#         # self.actor_net = torch.load('checkpoint/agent_actor.pth')
#         # self.actor_net.forward_bias = nn.Parameter(torch.tensor(0.0))
#         self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

#     def choose_action(self, state):
#         mean, std, binary_logits = self.actor_net(state)
#         dist = torch.distributions.Normal(mean, std)
#         continuous_action = dist.sample()
#         continuous_action = torch.tanh(continuous_action)
#         continuous_action = continuous_action * 80

#         binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
#         binary_action = binary_dist.sample()
        
#         action = torch.cat([continuous_action, binary_action], dim=-1)
        
#         return action.detach().cpu().numpy()

#     def evaluate(self, state):
#         mean, std, binary_logits = self.actor_net(state)
#         dist = torch.distributions.Normal(mean, std)
        
#         noise = torch.distributions.Normal(0, 1)
#         z = noise.sample()
        
#         continuous_action = mean + std * z
#         scaled_action = torch.tanh(continuous_action) * 80.0
#         continuous_action_log_prob = dist.log_prob(continuous_action).sum(dim=-1, keepdim=True) - torch.log(1 - torch.tanh(continuous_action).pow(2) + 1e-6).sum(dim=-1, keepdim=True)

#         binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
#         binary_action = binary_dist.sample()
#         binary_action_log_prob = binary_dist.log_prob(binary_action).sum(dim=-1, keepdim=True)

#         action_log_prob = continuous_action_log_prob + binary_action_log_prob
#         action = torch.cat([scaled_action, binary_action], dim=-1)

#         return action, action_log_prob, z, mean, std
    

#     def learn(self, loss):
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# class Critic:
#     def __init__(self, device, critic_lr, tau):
#         self.tau = tau
#         self.critic_lr = critic_lr
#         self.device = device
#         self.critic_net = CriticNet().to(device)
#         self.target_net = CriticNet().to(device)
#         # self.critic_net = torch.load('checkpoint/agent_critic.pth')
#         # self.target_net = torch.load('checkpoint/agent_critic.pth')
#         self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr, eps=1e-5)
#         self.loss_func = nn.MSELoss()

#     def update(self):
#         for target_param, param in zip(self.target_net.parameters(), self.critic_net.parameters()):
#             target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

#     def get_q_value(self, state, action):
#         return self.critic_net(state, action)

#     def get_target_q_value(self, state, action):
#         return self.target_net(state, action)

#     def learn(self, current_q1, current_q2, target_q):
#         loss = self.loss_func(current_q1, target_q) + self.loss_func(current_q2, target_q)
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()


# class Entropy:
#     def __init__(self, device, entropy_lr, action_dim=5):
#         self.entropy_lr = entropy_lr
#         self.target_entropy = -action_dim
#         self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
#         # self.log_alpha = torch.load('checkpoint/agent_entropy.pth')
#         self.alpha = self.log_alpha.exp()
#         self.optimizer = torch.optim.Adam([self.log_alpha], lr=entropy_lr)

#     def learn(self, loss):
#         self.optimizer.zero_grad()
#         loss.backward()
#         self.optimizer.step()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions import Normal
import pdb


class ActorNet(nn.Module):
    def __init__(self, max_action, input_shape=(4, 128, 128), action_dim=5):
        super(ActorNet, self).__init__()
        self.max_action = max_action
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.flattened_size = self._get_conv_output(input_shape)
        self.fc = nn.Linear(self.flattened_size, 512)
        self.mean = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim - 3)
        )
        self.log_std = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim - 3)
        )
        self.binary_logits = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)
        )

        self.forward_bias = nn.Parameter(torch.tensor(10.0))

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers(input)
            return int(np.prod(output.size()))

    def forward(self, state):
        x = self.conv_layers(state)
        x = F.relu(self.fc(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        binary_logits = self.binary_logits(x)

        binary_logits[:, 0] += self.forward_bias
        return mean, std, binary_logits

class CriticNet(nn.Module):
    def __init__(self, input_shape=(4, 128, 128), action_dim=5):
        super(CriticNet, self).__init__()
        self.conv_layers_q1 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.flattened_size = self._get_conv_output(input_shape)
        self.fc_q1 = nn.Sequential(
            nn.Linear(self.flattened_size + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.conv_layers_q2 = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_q2 = nn.Sequential(
            nn.Linear(self.flattened_size + action_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def _get_conv_output(self, shape):
        with torch.no_grad():
            input = torch.zeros(1, *shape)
            output = self.conv_layers_q1(input)
            return int(np.prod(output.size()))

    def forward(self, state, action):
        state_features_q1 = self.conv_layers_q1(state)
        q1 = torch.cat([state_features_q1, action], 1)
        q1 = self.fc_q1(q1)

        state_features_q2 = self.conv_layers_q2(state)
        q2 = torch.cat([state_features_q2, action], 1)
        q2 = self.fc_q2(q2)
        return q1, q2
    

class Actor:
    def __init__(self, device, actor_lr, min_action, max_action):
        self.device = device
        self.actor_lr = actor_lr
        self.min_action = min_action.to(device)
        self.max_action = max_action.to(device)
        self.actor_net = ActorNet(self.max_action).to(device)
        # self.actor_net = torch.load('checkpoint2/agent_actor.pth')
        # self.actor_net.forward_bias = nn.Parameter(torch.tensor(0.0))
        self.optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.actor_lr)

    def choose_action(self, state):
        mean, std, binary_logits = self.actor_net(state)
        dist = torch.distributions.Normal(mean, std)
        continuous_action = dist.sample()
        continuous_action = torch.tanh(continuous_action)
        continuous_action = continuous_action * 80

        binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
        binary_action = binary_dist.sample()
        
        action = torch.cat([continuous_action, binary_action], dim=-1)
        
        return action.detach().cpu().numpy()

    def evaluate(self, state):
        mean, std, binary_logits = self.actor_net(state)
        dist = torch.distributions.Normal(mean, std)
        
        noise = torch.distributions.Normal(0, 1)
        z = noise.sample()
        
        continuous_action = mean + std * z
        scaled_action = torch.tanh(continuous_action) * 80.0
        continuous_action_log_prob = dist.log_prob(continuous_action).sum(dim=-1, keepdim=True) - torch.log(1 - torch.tanh(continuous_action).pow(2) + 1e-6).sum(dim=-1, keepdim=True)

        binary_dist = torch.distributions.Bernoulli(logits=binary_logits)
        binary_action = binary_dist.sample()
        binary_action_log_prob = binary_dist.log_prob(binary_action).sum(dim=-1, keepdim=True)

        action_log_prob = continuous_action_log_prob + binary_action_log_prob
        action = torch.cat([scaled_action, binary_action], dim=-1)

        return action, action_log_prob, z, mean, std
    

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Critic:
    def __init__(self, device, critic_lr, tau):
        self.tau = tau
        self.critic_lr = critic_lr
        self.device = device
        self.critic_net = CriticNet().to(device)
        self.target_net = CriticNet().to(device)
        # self.critic_net = torch.load('checkpoint2/agent_critic.pth')
        # self.target_net = torch.load('checkpoint2/agent_critic.pth')
        self.optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=critic_lr, eps=1e-5)
        self.loss_func = nn.MSELoss()

    def update(self):
        for target_param, param in zip(self.target_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_q_value(self, state, action):
        return self.critic_net(state, action)

    def get_target_q_value(self, state, action):
        return self.target_net(state, action)

    def learn(self, current_q1, current_q2, target_q):
        loss = self.loss_func(current_q1, target_q) + self.loss_func(current_q2, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Entropy:
    def __init__(self, device, entropy_lr, action_dim=5):
        self.entropy_lr = entropy_lr
        self.target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        # self.log_alpha = torch.load('checkpoint2/agent_entropy.pth')
        self.alpha = self.log_alpha.exp()
        self.optimizer = torch.optim.Adam([self.log_alpha], lr=entropy_lr)

    def learn(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()