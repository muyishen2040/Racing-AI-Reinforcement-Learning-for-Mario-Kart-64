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
        self.entropy = Entropy(config['device'], config['entropy_lr'], 2)

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        
        with torch.no_grad():
            new_action, log_prob_, z, mean, std = self.actor.evaluate(next_states)
            target_q1, target_q2 = self.critic.get_target_q_value(next_states, new_action)
            target_q = rewards + (1 - dones) * self.config['gamma'] * (torch.min(target_q1, target_q2) - self.entropy.alpha * log_prob_)

        actions = actions.squeeze()
        current_q1, current_q2 = self.critic.get_q_value(states, actions)
        q_loss = self.critic.learn(current_q1, current_q2, target_q.detach())

        a_, log_prob, _, _, _ = self.actor.evaluate(states)
        q1, q2 = self.critic.get_q_value(states, a_)
        q = torch.min(q1, q2)
        actor_loss = (self.entropy.alpha * log_prob - q).mean()
        self.actor.learn(actor_loss)

        alpha_loss = -(self.entropy.log_alpha.exp() * (log_prob + self.entropy.target_entropy).detach()).mean()
        self.entropy.learn(alpha_loss)
        self.entropy.alpha = self.entropy.log_alpha.exp()

        self.critic.update()

        return actor_loss.item(), q_loss, alpha_loss.item()
        

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

    writer = SummaryWriter()
    
    config = {
        'device': device,
        'gamma': 0.99,
        'tau': 0.01,
        'min_action': torch.tensor([-80.0, -80.0, 0, 0, 0]),
        'max_action': torch.tensor([80.0, 80.0, 1, 1, 1]),
        'action_dim': 5,
        'memory_len': 1000000,
        
        # Learning rates for the first stage training
        'entropy_lr': 1e-4,
        'actor_lr': 3e-4,
        'critic_lr': 3e-4,
        
        # Learning rates for the second stage training
        # 'entropy_lr': 1e-5,
        # 'actor_lr': 3e-5,
        # 'critic_lr': 3e-5,
        'batch_size': 256,
        'episodes': 10000,
        'skip_frame': 4,
        'stack_frame': 4,
        'replay_buffer_warmup': 2000
        # 'warmup_steps': 1000,
        # 'expert_data': 'replay_buffer.pkl'
    }

    orig_env = EnvReceiver()
    
    # The skip_env is commented since the original env has already employ frame skipping (5) 
    # skip_env = FrameSkip(orig_env, config['skip_frame'])
    
    env = FrameStack(orig_env, config['stack_frame'])
    reward_buffer = deque([], maxlen=90)
    # reward_buffer = deque([], maxlen=180)
    
    agent = SAC(config)
    
    best_reward = -10000.0
    cur_step = 0
    
    for episode in range(config['episodes']):
        total_reward = 0
        total_q_loss = 0
        total_policy_loss = 0
        total_alpha_loss = 0
        done = False
        obs = env.reset()
        reward_buffer.clear()
        
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

            if cur_step == config['replay_buffer_warmup']:
                print('------------------------------')
                print('--------Training Start--------')
                print('------------------------------')
            
            if cur_step % 10000 == 0:
                print('Current Training Step:', cur_step)
            
            action = agent.actor.choose_action(obs.unsqueeze(0).to(device))
            # print(action)
            
            next_obs, reward, done, info = env.step(action.flatten().tolist())
            reward_buffer.append(reward)
            # For the first training stage, uncomment the following code snippet to enable early stop
            if step_in_episode > 90 and sum(reward_buffer) / len(reward_buffer) <= -0.099:
                early_stop_reward = (1250 - step_in_episode) * -0.1
                reward += early_stop_reward
                done = True
            print(reward, "      ", end='\r')
            # print(reward)
            total_reward += reward
            
            if not ('expert_data' in config.keys() and cur_step < config['warmup_steps']):
                agent.memory.push((obs, action.flatten(), reward, next_obs, done))
            
            obs = next_obs
            
            if len(agent.memory) > config['batch_size'] and len(agent.memory) > config['replay_buffer_warmup']:
                policy_loss, q_loss, alpha_loss = agent.update()
                total_q_loss += q_loss
                total_policy_loss += policy_loss
                total_alpha_loss += alpha_loss
                # print(agent.actor.actor_net.forward_bias)
            
        
        writer.add_scalar("Reward/episode", total_reward, episode)
        writer.add_scalar("Loss/q_loss", total_q_loss/step_in_episode, episode)
        writer.add_scalar("Loss/policy_loss", total_policy_loss/step_in_episode, episode)
        writer.add_scalar("Loss/alpha_loss", total_alpha_loss/step_in_episode, episode)
        writer.add_scalar("Alpha/alpha", agent.entropy.alpha, episode)
        print(f"Episode {episode + 1}: Total Reward: {total_reward}, q_loss: {total_q_loss/step_in_episode}, policy_loss: {total_policy_loss/step_in_episode}, alpha_loss: {total_alpha_loss/step_in_episode}, alpha: {agent.entropy.alpha.item()}, num_step: {step_in_episode}")
        
        if total_reward > best_reward:
            print('\n----------------------')
            print('New best episode: {}'.format(total_reward))
            torch.save(agent.actor.actor_net, 'agent_actor.pth')
            torch.save(agent.critic.critic_net, 'agent_critic.pth') 
            torch.save(agent.entropy.log_alpha, 'agent_entropy.pth') 
            print('----------------------\n')
            best_reward = total_reward

        if episode % 50 == 0 and episode > 0:
            torch.save(agent.actor.actor_net, 'actor_{}.pth'.format(episode))
            torch.save(agent.critic.critic_net, 'critic_{}.pth'.format(episode))
            torch.save(agent.entropy.log_alpha, 'entropy_{}.pth'.format(episode))


if __name__ == "__main__":
    main()