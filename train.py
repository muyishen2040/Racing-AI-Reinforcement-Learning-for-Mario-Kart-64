from EnvReceiver import EnvReceiver
import numpy as np
from collections import deque
import pdb
import torch
from torchvision import transforms

class FrameSkip():
    def __init__(self, env, skip):
        self.env = env
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info
    
    def reset(self):
        ob = self.env.reset()

        return ob

class FrameStack():
    def __init__(self, env, k):
        self.env = env
        self.k = k
        self.frames = deque([], maxlen=k)
        self.preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((84, 84)),
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

orig_env = EnvReceiver()
skip_env = FrameSkip(orig_env, 4)
env = FrameStack(skip_env, 4)

done = False
total_reward = 0
obs = env.reset()


while not done:
    action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
    obs, reward, done, info = env.step(action)
    print(obs.shape, torch.max(obs), torch.min(obs))
    total_reward += reward
print("Total reward:", total_reward)