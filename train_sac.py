import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from src.model import *
from src.utils import *
from src.agent import *

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    frame_stack = 4
    env, obs_shape = make_env(frame_stack)
    config = AgentConfig(
        state_dims=obs_shape,
        action_c_dims=1,
        action_d_dims=2**3,
        action_scale=80,
        action_bias=0,
        frame_stack=frame_stack,
        lr=1e-7,
        entropy_lr=1e-3,
        gamma=0.999,
        tau=0.001,
        memory_size=100000,
        batch_size=256,
        device='cuda'
    )
    writer = SummaryWriter(f"runs/{config}")
    print(config)

    agent = Agent(config)
    num_episodes = 10000
    save_ep = 100
    draw_ep = 10
    highest_reward = -9999
    pbar = tqdm(range(num_episodes))

    for ep in pbar:
        state = env.reset()
        state = agent.preprocess_obs(state)
        total_reward = 0
        step = 0
        total_q_loss = 0
        total_policy_loss = 0
        total_alpha_loss = 0
        while True:
            step += 1
            action = agent.act(state, random=False)
            next_state, reward, done, info = env.step(action)
            next_state = agent.preprocess_obs(next_state)
            agent.replay_buffer.store_transition(state, action[0], ACTIONS2IDX[tuple(action[2:])], reward, next_state, done)
            if agent.replay_buffer.enought_samples():
                q_loss, policy_loss, alpha_loss = agent.learn()
                total_q_loss += q_loss
                total_policy_loss += policy_loss
                total_alpha_loss += alpha_loss
            total_reward += reward
            state = next_state
            if done:
                break
        if ep % save_ep == 0:
            torch.save(agent.actor.state_dict(), f'checkpoints/109062102_hw3_data_{ep}_{config}')
        if ep % draw_ep == 0:
            state = env.reset()
            done = False
            eval_reward = 0
            while not done:
                state = agent.preprocess_obs(state)
                action = agent.act(state, eval=True)
                state, reward, done, _ = env.step(action)
                eval_reward += reward
            if eval_reward > highest_reward:
                highest_reward = eval_reward
                torch.save(agent.actor.state_dict(), f"109062102_hw3_data_highest_{config}")
            writer.add_scalar("Reward/eval reward", eval_reward, ep)
        writer.add_scalar("Reward/train reward", total_reward, ep)
        writer.add_scalar("Loss/q_loss", total_q_loss/step, ep)
        writer.add_scalar("Loss/policy_loss", total_policy_loss/step, ep)
        pbar.set_postfix({'total_reward': total_reward, "q_loss": total_q_loss/step, "policy_loss": total_policy_loss/step, "alpha_loss": total_alpha_loss/step, "num_step": step})

    torch.save(agent.actor.state_dict(), "109062102_hw3_data")