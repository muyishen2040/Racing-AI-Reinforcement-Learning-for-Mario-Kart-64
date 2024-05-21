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
        
        observation = torch.stack(observation).to(device)
        actions = torch.from_numpy(np.array(actions)).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        next_observations = torch.stack(next_observations).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        return observation, actions, rewards, next_observations, dones

    def __len__(self):
        return len(self.buffer)


class SAC:
    def __init__(self, config):
        self.config = config
        self.memory = ReplayBuffer(config['memory_len'])
        self.actor = Actor(config['device'], config['actor_lr'], config['min_action'], config['max_action'])
        self.critic = Critic(config['device'], config['critic_lr'], config['tau'])
        self.entropy = Entropy(config['device'], config['entropy_lr'], config['action_dim'])

    def update(self):
        states, actions, rewards, next_states, dones = self.memory.sample(self.config['batch_size'])
        new_action, log_prob_, z, mean, log_std = self.actor.evaluate(next_states)

        target_q1, target_q2 = self.critic.get_target_q_value(next_states, new_action)

        log_prob_ = log_prob_.sum(dim=1, keepdim=True)
        target_q = rewards + (1 - dones) * self.config['gamma'] * (torch.min(target_q1, target_q2) - self.entropy.alpha * log_prob_)

        actions = actions.squeeze()
        current_q1, current_q2 = self.critic.get_q_value(states, actions)
        self.critic.learn(current_q1, current_q2, target_q.detach())

        a_, log_prob, _, _, _ = self.actor.evaluate(states)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        q1, q2 = self.critic.get_q_value(states, a_)
        q = torch.min(q1, q2)
        actor_loss = (self.entropy.alpha * log_prob - q).mean()
        self.actor.learn(actor_loss)

        alpha_loss = -(self.entropy.log_alpha.exp() * (log_prob + self.entropy.target_entropy).detach()).mean()
        self.entropy.learn(alpha_loss)
        self.entropy.alpha = self.entropy.log_alpha.exp()

        self.critic.update()
        

def main():

    # writer = SummaryWriter()
    env = EnvReceiver()
    
    config = {
        'device': device,
        'gamma': 0.99,
        'tau': 0.01,
        'min_action': torch.tensor([-80.0, -80.0, 0, 0, 0]),
        'max_action': torch.tensor([80.0, 80.0, 1, 1, 1]),
        'action_dim': 5,
        'memory_len': 4000000,
        'entropy_lr': 1e-4,
        'actor_lr': 5e-4,
        'critic_lr': 5e-4,
        'batch_size': 256,
        'episodes': 40000
    }
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),
        transforms.Resize((84, 84)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    agent = SAC(config)
    best_reward = -10000.0
    
    for episode in range(config['episodes']):
        total_reward = 0
        done = False
        obs = env.reset()
        obs = preprocess(obs)
        
        while not done:
            # action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
            action = agent.actor.choose_action(obs.unsqueeze(0).to(device))
            # pdb.set_trace()
            
            next_obs, reward, done, info = env.step(action.flatten().tolist())
            next_obs = preprocess(next_obs)
            
            # print(obs.shape, np.max(obs), np.min(obs))
            total_reward += reward
            agent.memory.push((obs, action, reward, next_obs, done))
            
            obs = next_obs
            
            if len(agent.memory) > config['batch_size']:
                agent.update()
            
        
        # writer.add_scalar("Reward/episode", episode_reward, episode)
        print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
        
        if episode_reward > best_reward:
            print('\n----------------------')
            print('New best episode: {}'.format(episode_reward))
            # torch.save(agent.actor.actor_net, 'agent_actor.pth')
            # torch.save(agent.critic.critic_net, 'agent_critic.pth') 
            # torch.save(agent.entropy.log_alpha, 'agent_entropy.pth') 
            print('----------------------\n')
            best_reward = episode_reward

        # if episode % 100 == 0 and episode > 0:
        #     torch.save(agent.actor.actor_net, 'agent_{}.pth'.format(episode))


if __name__ == "__main__":
    main()