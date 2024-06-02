import torch
import numpy as np

class QNetwork(torch.nn.Module):
    def __init__(self, input_shape, n_actions_c, n_actions_d):
        super(QNetwork, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[1], 256, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[1], 256, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.valueNet1 = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size + n_actions_c, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions_d)
        )
        self.valueNet2 = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size + n_actions_c, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions_d)
        )

    def _get_conv_out(self, shape):
        o = self.conv1(torch.zeros(shape))
        return int(np.prod(o.size()))

    def forward(self, state: torch.Tensor, action: torch.Tensor):
        # assumes input is uint8
        state = state.to(torch.float32) / 255.
        conv_out1 = self.conv1(state).reshape(state.size()[0], -1)
        conv_out2 = self.conv2(state).reshape(state.size()[0], -1)
        q1 = self.valueNet1(torch.cat([conv_out1, action], dim=1))
        q2 = self.valueNet2(torch.cat([conv_out2, action], dim=1))
        return q1, q2
    
class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_shape, n_actions_c, n_actions_d, action_c_scale, action_c_bias):
        super(PolicyNetwork, self).__init__()
        self.action_c_scale = action_c_scale
        self.action_c_bias = action_c_bias
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(input_shape[1], 256, kernel_size=8, stride=4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=4, stride=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 128, kernel_size=3, stride=1),
            torch.nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.meanNet = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions_c)
        )
        self.logNet = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions_c)
        )
        self.dNet = torch.nn.Sequential(
            torch.nn.Linear(conv_out_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, n_actions_d)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(shape))
        return int(np.prod(o.size()))

    def forward(self, state: torch.Tensor):
        # assumes input is uint8
        state = state.to(torch.float32) / 255.
        conv_out = self.conv(state).reshape(state.size()[0], -1)
        mean = self.meanNet(conv_out)
        log_std = self.logNet(conv_out)
        log_std = torch.clamp(log_std, min=-20, max=2)
        action_d = self.dNet(conv_out)
        return mean, log_std, action_d # logits
    
    def sample(self, state: torch.Tensor):
        mean, log_std, logits_d = self.forward(state)
        std = torch.exp(log_std)
        normal = torch.distributions.Normal(mean, std)
        sample_c = normal.rsample()
        squashed_sample_c = torch.tanh(sample_c)
        action_c = squashed_sample_c * self.action_c_scale + self.action_c_bias
        log_prob_c = normal.log_prob(sample_c) - torch.log(self.action_c_scale * (1 - squashed_sample_c.pow(2)) + 1e-6)
        log_prob_c = log_prob_c.sum(1, keepdim=True)
        
        categorical = torch.distributions.Categorical(logits=logits_d)
        action_d = categorical.sample()
        log_prob_d = categorical.log_prob(action_d).unsqueeze(1)
        
        deterministic_action_c = torch.tanh(mean) * self.action_c_scale + self.action_c_bias
        deterministic_action_d = torch.argmax(logits_d, dim=1)

        return action_c, action_d, log_prob_c, log_prob_d, deterministic_action_c, deterministic_action_d
