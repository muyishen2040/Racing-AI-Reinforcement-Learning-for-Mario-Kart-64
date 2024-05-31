import pickle
import collections
import random
import torch
import numpy as np
from torchvision import transforms
import pdb
from collections import deque
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import gym
from PIL import Image, ImageEnhance, ImageOps
import imageio.v2 as imageio
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
            nn.Linear(256, action_dim - 3)
        )
        self.log_std = nn.Sequential(
            nn.Linear(512, 256),
            nn.Linear(256, action_dim - 3)
        )
        self.binary_logits = nn.Sequential(
            nn.Linear(512, 256),
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

class ActorNetVisualize(ActorNet):
    def __init__(self, *args, **kwargs):
        super(ActorNetVisualize, self).__init__(*args, **kwargs)
        self.gradients = None
        self.activations = None

    def activations_hook(self, grad):
        self.gradients = grad

    def forward(self, state):
        x = self.conv_layers(state)
        x.register_hook(self.activations_hook)
        self.activations = x
        x = F.relu(self.fc(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)
        std = log_std.exp()
        binary_logits = self.binary_logits(x)
        binary_logits[:, 0] += self.forward_bias
        return mean, std, binary_logits

class Agent():
    def __init__(self):
        self.max_action = torch.tensor([80.0, 80.0, 1, 1, 1]).cuda()
        self.actor_net = ActorNetVisualize(self.max_action).to(device)
        self.actor_net.load_state_dict(torch.load('actor_net_dict.pth'))

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

    def visualize_features(self, state, original_observation, output_path):
        self.actor_net.eval()
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, std, binary_logits = self.actor_net(state)
        action = mean.mean()
        action.backward()
        gradients = self.actor_net.gradients
        activations = self.actor_net.activations
        grad_cam = compute_grad_cam(gradients, activations)
        visualize_and_save(original_observation, grad_cam, output_path)
        self.actor_net.train()

def compute_grad_cam(gradients, activations):
    b, c, h, w = 1, 128, 12, 12
    gradients = gradients.view(b, c, h, w)
    activations = activations.view(b, c, h, w)
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    grad_cam = torch.sum(weights * activations, dim=1, keepdim=True)
    grad_cam = F.relu(grad_cam)
    grad_cam = F.interpolate(grad_cam, size=(480, 640), mode='bilinear', align_corners=False)
    epsilon = 1e-8
    grad_cam -= grad_cam.min()
    grad_cam /= (grad_cam.max() + epsilon)
    return grad_cam

def visualize_and_save(original_observation, grad_cam, output_path):
    grad_cam = grad_cam.cpu().detach().numpy().squeeze()
    grad_cam -= grad_cam.min()
    grad_cam /= (grad_cam.max() + 1e-8)
    heatmap = Image.fromarray(np.uint8(255 * grad_cam))
    heatmap = ImageOps.colorize(heatmap, black="blue", white="red")
    original_observation = np.array(original_observation)
    original_observation = Image.fromarray(original_observation).convert("L")
    original_observation = ImageOps.colorize(original_observation, black="black", white="white")
    blended = Image.blend(original_observation, heatmap, alpha=0.5)
    blended.save(output_path)

class ReplayEnv():
    def __init__(self, replay_file):
        self.cur_idx = 0
        self.replay_file = replay_file
        with open(self.replay_file, 'rb') as f:
            self.replay_data = pickle.load(f)

    def reset(self):
        self.cur_idx = 0
        ob = self.replay_data['obs'][self.cur_idx]
        reward = self.replay_data['reward'][self.cur_idx]
        done = self.replay_data['done'][self.cur_idx]
        info = {}
        self.cur_idx += 1
        return ob

    def step(self, action):
        ob = self.replay_data['obs'][self.cur_idx]
        reward = self.replay_data['reward'][self.cur_idx]
        done = self.replay_data['done'][self.cur_idx]
        info = {}
        self.cur_idx += 1
        if self.cur_idx == len(self.replay_data['obs']) - 1:
            self.cur_idx = 0
        return ob, reward, done, info

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

def main():
    replay_file = 'Luigi-Raceway-easy1.pkl'
    FRAME_STACK = 4
    replay_env = ReplayEnv(replay_file)
    env = FrameStack(replay_env, FRAME_STACK)

    agent = Agent()

    state = env.reset()

    episode_length = len(replay_env.replay_data['obs'])

    frames = []

    for _ in tqdm(range(episode_length), desc="Generating Grad-CAM Images"):
        action = agent.choose_action(state.unsqueeze(0).to(device))
        next_state, reward, done, info = env.step(action)

        original_observation = replay_env.replay_data['obs'][replay_env.cur_idx]
        output_image_path = 'temp_output_image.png'
        agent.visualize_features(next_state, original_observation, output_image_path)

        frame = imageio.imread(output_image_path)
        frames.append(frame)

        state = next_state

    output_video_path = 'episode_gradcam_video.mp4'
    imageio.mimwrite(output_video_path, frames, fps=30)

    print('Grad-Cam video saved successfully')

if __name__ == "__main__":
    main()
