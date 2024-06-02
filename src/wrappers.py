import torch
import torchvision
import numpy as np
import torchvision.transforms.functional
from collections import deque

class PIL2TorchWrapper():
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        obs = torch.tensor(obs, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
        obs = torchvision.transforms.functional.resize(obs, (128, 128), antialias=True)
        obs = torchvision.transforms.functional.rgb_to_grayscale(obs)
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        obs = torch.tensor(obs, dtype=torch.uint8).permute(2, 0, 1).unsqueeze(0)
        obs = torchvision.transforms.functional.resize(obs, (128, 128), antialias=True)
        obs = torchvision.transforms.functional.rgb_to_grayscale(obs)
        return obs, rew, done, info

class FrameStackWrapper():
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        return torch.cat(list(self.frames), dim=1)