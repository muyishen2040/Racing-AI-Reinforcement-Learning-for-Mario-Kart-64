import torch
from src.utils import *
import numpy as np

class ReplayBuffer:
    def __init__(self, config: AgentConfig):
        self.config = config
        self.mem_state = torch.zeros((config.memory_size, *config.state_dims[1:]), dtype=torch.uint8).to(config.device)
        self.mem_action_c = torch.zeros((config.memory_size, config.action_c_dims), dtype=torch.float32).to(config.device)
        self.mem_action_d = torch.zeros((config.memory_size, 1), dtype=torch.long).to(config.device)
        self.mem_reward = torch.zeros(config.memory_size, dtype=torch.float32).to(config.device)
        self.mem_next_state = torch.zeros((config.memory_size, *config.state_dims[1:]), dtype=torch.uint8).to(config.device)
        self.mem_done = torch.zeros(config.memory_size, dtype=torch.bool).to(config.device)
        self.mem_counter = 0
        self.mem_full = False

    def store_transition(self, state, action_c, action_d, reward, next_state, done):
        remove_idx = self.mem_counter
        self.mem_counter = (self.mem_counter + 1) % self.config.memory_size
        self.mem_full = self.mem_full or self.mem_counter == 0
            
        self.mem_state[remove_idx] = state
        self.mem_action_c[remove_idx] = action_c
        self.mem_action_d[remove_idx] = action_d
        self.mem_reward[remove_idx] = reward
        self.mem_next_state[remove_idx] = next_state
        self.mem_done[remove_idx] = done

    def sample(self):
        assert self.mem_counter >= self.config.batch_size or self.mem_full, "Not enough samples in memory"
        sample_size = self.mem_counter if not self.mem_full else self.config.memory_size

        choices = np.random.choice(sample_size, self.config.batch_size, replace=False)
        state = self.mem_state[choices]
        action_c = self.mem_action_c[choices]
        action_d = self.mem_action_d[choices]
        reward = self.mem_reward[choices]
        next_state = self.mem_next_state[choices]
        done = self.mem_done[choices]
        return state, action_c, action_d, reward, next_state, done
    
    def enought_samples(self):
        return self.mem_counter >= self.config.batch_size or self.mem_full