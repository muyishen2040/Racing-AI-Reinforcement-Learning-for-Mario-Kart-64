from EnvReceiver import EnvReceiver
import numpy as np
from src.model import *
import collections
import random
import torch
import pdb
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import gym
import pickle
from PIL import Image
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FrameStack():
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(self._preprocess_frame(ob))
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(self._preprocess_frame(ob))
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return torch.stack(list(self.frames), dim=0).squeeze()

    def _preprocess_frame(self, frame):
        return self.preprocess(frame)

class Agent():
    def __init__(self):
        self.actor_net = torch.load('checkpoint/agent_actor.pth')

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


def main():
    
    config = {
        'stack_frame': 4,
    }

    orig_env = EnvReceiver()
    env = FrameStack(orig_env, config['stack_frame'])

    agent = Agent()
    # pdb.set_trace()
    # torch.save(agent.actor_net.state_dict(), 'actor_net_dict.pth')
    
    for episode in range(10):
        total_reward = 0
        done = False
        obs = env.reset()
        step_in_episode = 0
        
        while not done:
            step_in_episode += 1
            action = agent.choose_action(obs.unsqueeze(0).to(device))
            
            next_obs, reward, done, info = env.step(action.flatten().tolist())
            total_reward += reward
            
            obs = next_obs

        print(f"Episode {episode + 1}: Total Reward: {total_reward}")


if __name__ == "__main__":
    main()