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

class ReplayBuffer():
    def __init__(self, buffer_maxlen):
        self.buffer = collections.deque(maxlen=buffer_maxlen)
        self.sequence_length = 16

    def push(self, sequence):
        self.buffer.append(sequence)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        observation, actions, rewards, next_observations, dones = zip(*batch)

        
        try:
            observation = torch.stack(observation).to(device)
            actions = torch.from_numpy(np.array(actions)).to(device)
            rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
            next_observations = torch.stack(next_observations).to(device)
            dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        except:
            pdb.set_trace()

        return observation, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, config):
        self.config = config
        
        if 'expert_data' in config.keys():
            with open(config['expert_data'], 'rb') as f:
                self.memory = pickle.load(f)
            print('Expert Data Loaded')
        else:
            self.memory = ReplayBuffer(config['memory_len'])
        
        self.actor = Actor(config['device'], config['actor_lr'], config['min_action'], config['max_action'])
        self.critic = Critic(config['device'], config['critic_lr'], config['tau'])
        self.entropy = Entropy(config['device'], config['entropy_lr'], config['action_dim'])

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        with torch.no_grad():
            new_action, log_prob_, z, mean, std = self.actor.evaluate(next_states)
            target_q1, target_q2 = self.critic.get_target_q_value(next_states, new_action)
            target_q = rewards + (1 - dones) * self.config['gamma'] * (torch.min(target_q1, target_q2) - self.entropy.alpha * log_prob_)

        actions = actions.squeeze()
        current_q1, current_q2 = self.critic.get_q_value(states, actions)
        self.critic.learn(current_q1, current_q2, target_q.detach())

        a_, log_prob, _, _, _ = self.actor.evaluate(states)
        q1, q2 = self.critic.get_q_value(states, a_)
        q = torch.min(q1, q2)
        actor_loss = (self.entropy.alpha * log_prob - q).mean()
        self.actor.learn(actor_loss)

        alpha_loss = -(self.entropy.log_alpha.exp() * (log_prob + self.entropy.target_entropy).detach()).mean()
        self.entropy.learn(alpha_loss)
        self.entropy.alpha = self.entropy.log_alpha.exp()

        self.critic.update()
        

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

def main():

    writer = SummaryWriter()
    
    config = {
        'device': device,
        'gamma': 0.99,
        'tau': 0.01,
        'min_action': torch.tensor([-80.0, -80.0, 0, 0, 0]),
        'max_action': torch.tensor([80.0, 80.0, 1, 1, 1]),
        'action_dim': 5,
        'memory_len': 1000000,
        'entropy_lr': 1e-4,
        'actor_lr': 5e-4,
        'critic_lr': 5e-4,
        'batch_size': 256,
        'episodes': 10000,
        'skip_frame': 4,
        'stack_frame': 4,
        'replay_buffer_warmup': 0
        # 'warmup_steps': 1000,
        # 'expert_data': 'replay_buffer.pkl'
    }

    orig_env = EnvReceiver()
    # skip_env = FrameSkip(orig_env, config['skip_frame'])
    env = FrameStack(orig_env, config['stack_frame'])
    reward_buffer = deque([], maxlen=50)
    
    # preprocess = transforms.Compose([
    #     transforms.ToPILImage(),
    #     transforms.Grayscale(),
    #     transforms.Resize((84, 84)),
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.5], std=[0.5])
    # ])
    
    agent = SAC(config)
    
    best_reward = -10000.0
    cur_step = 0
    
    for episode in range(config['episodes']):
        total_reward = 0
        done = False
        obs = env.reset()
        reward_buffer.clear()
        # obs = preprocess(obs)
        
        # The following code snippet save the observation stack into 4 image files
        # obs_np = obs.numpy()
        # for i in range(obs_np.shape[0]):
        #     frame = obs_np[i]
        #     frame_img = Image.fromarray((frame * 255).astype(np.uint8), mode='L')
        #     frame_img.save(f'frame_{i}.png')
        step_in_episode = 0
        
        while not done:
            cur_step += 1
            step_in_episode += 1
            
            if cur_step % 10000 == 0:
                print('Current Training Step:', cur_step)
            
            # action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
            action = agent.actor.choose_action(obs.unsqueeze(0).to(device))
            # action = np.array([0, 80, 1, 0, 0])
            # print(action)
            
            next_obs, reward, done, info = env.step(action.flatten().tolist())
            reward_buffer.append(reward)
            # image = Image.fromarray(next_obs.astype('uint8'))
            # image.save('observation.png')
            # pdb.set_trace()

            # next_obs = preprocess(next_obs)
            
            # print(obs.shape, np.max(obs), np.min(obs))
            # print(reward)
            total_reward += reward
            
            if not ('expert_data' in config.keys() and cur_step < config['warmup_steps']):
                agent.memory.push((obs, action.flatten(), reward, next_obs, done))
            
            obs = next_obs
            
            if len(agent.memory) > config['batch_size'] and len(agent.memory) > config['replay_buffer_warmup']:
                agent.update()
                # print(agent.actor.actor_net.forward_bias)
                # print('update')
            
            if step_in_episode > 70 and sum(reward_buffer) / len(reward_buffer) <= -0.099:
                break
            
        
        writer.add_scalar("Reward/episode", total_reward, episode)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}")
        
        if total_reward > best_reward:
            print('\n----------------------')
            print('New best episode: {}'.format(total_reward))
            torch.save(agent.actor.actor_net, 'agdent_actor.pth')
            torch.save(agent.critic.critic_net, 'agent_critic.pth') 
            torch.save(agent.entropy.log_alpha, 'agent_entropy.pth') 
            print('----------------------\n')
            best_reward = total_reward

        if episode % 100 == 0 and episode > 0:
            torch.save(agent.actor.actor_net, 'agent_{}.pth'.format(episode))


if __name__ == "__main__":
    main()