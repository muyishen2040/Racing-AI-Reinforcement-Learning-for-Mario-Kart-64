import numpy as np
import pickle
import torch
from src.model import *
from torchvision import transforms
from collections import deque
from EnvReceiver import EnvReceiver

NUM_EPOCHS = 150
GAMMA = 0.99
BATCH_SIZE = 256

def preprocess_obs(obss):
    preprocessor = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((128, 128)),
            transforms.ToTensor()
            ])
    queue = deque([], maxlen=4)
    for i in range(4):
        queue.append(preprocessor(obss[0]))
    ret_obs = []
    for obs in obss:
        queue.append(preprocessor(obs))
        ret_obs.append(torch.stack(list(queue), dim=0).squeeze())
    return torch.stack(ret_obs, dim=0)

def get_q(rewards, dones, gamma):
    q = torch.zeros((len(rewards), 1))
    q[-1] = 0.5 #rewards[-1]
    for i in range(len(rewards) - 2, -1, -1):
        q[i] = rewards[i] + gamma * q[i + 1] * (1-dones[i])
    return q
    
env = EnvReceiver()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

expert_data_path = [f"Luigi-Raceway-easy{i}.pkl" for i in range(1, 4)]
expert_data_obs = torch.empty((0, 4, 128, 128), dtype=torch.float32, device=device)
expert_data_action = torch.empty((0, 5), dtype=torch.float32, device=device)
expert_data_q = torch.empty((0, 1), dtype=torch.float32, device=device)

for path in expert_data_path:
    with open(path, "rb") as f:
        data = pickle.load(f)
    obs = preprocess_obs(data["obs"]).to(device)
    q = get_q(data["reward"], data["done"], GAMMA).to(device)
    expert_data_obs = torch.cat((expert_data_obs, obs), dim=0)
    expert_data_action = torch.cat((expert_data_action, torch.tensor(data["action"], dtype=torch.float32, device=device)), dim=0)
    expert_data_q = torch.cat((expert_data_q, q), dim=0)

total_data = expert_data_obs.shape[0]
train_order = np.arange(total_data)

actor = ActorNet(80).to(device)
critic = CriticNet().to(device)

actor_lr = 5e-4
critic_lr = 1e-5

actor_optimizer = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr=critic_lr)
actor_scheduler = torch.optim.lr_scheduler.ExponentialLR(actor_optimizer, 0.99)
critic_scheduler = torch.optim.lr_scheduler.ExponentialLR(critic_optimizer, 0.99)

for epoch in range(1, NUM_EPOCHS+1):
    np.random.shuffle(train_order)
    for cnt, batch_start in enumerate(range(0, total_data, BATCH_SIZE)):
        idx = train_order[batch_start:batch_start+BATCH_SIZE]
        obs = expert_data_obs[idx]
        action = expert_data_action[idx]
        q = expert_data_q[idx]

        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        mean, std, binary_logits = actor(obs)
        dist = torch.distributions.Normal(mean, std)
        continuous_action = dist.rsample()
        continuous_action = torch.tanh(continuous_action)
        continuous_action = continuous_action * 80

        loss_c = torch.nn.functional.mse_loss(continuous_action, action[:, :2])
        loss_d = torch.nn.functional.binary_cross_entropy_with_logits(binary_logits, action[:, 2:])

        q1, q2 = critic(obs, action)
        loss_q = torch.nn.functional.mse_loss(q1, q) + torch.nn.functional.mse_loss(q2, q)

        loss = loss_c + loss_d + loss_q
        loss.backward()

        actor_optimizer.step()
        critic_optimizer.step()
        print(f'Train Epoch: {epoch:4d} [{cnt}/{len(train_order)//BATCH_SIZE} ({100.*cnt/len(train_order)*BATCH_SIZE:.0f}%)]\tActor Loss: {loss_c.item()+loss_d.item():13.6f}\tCritic Loss: {loss_q.item():13.6f}\tActor LR: {actor_scheduler.get_lr()[0]:.7f}\tCritic LR: {critic_scheduler.get_lr()[0]:.7f}', end='\r')
    print()

    if actor_scheduler.get_lr()[0] > actor_lr * 1e-2:
        actor_scheduler.step()
        critic_scheduler.step()

    if epoch % 150 == 0:
        torch.save(actor, f"checkpoints/BC/actor_{epoch}.pth")
        torch.save(critic, f"checkpoints/BC/critic_{epoch}.pth")
        done = False
        total_reward = 0
        obs = env.reset()
        while not done:
            obs = preprocess_obs([obs]).to(device)
            with torch.no_grad():
                mean, std, binary_logits = actor(obs)
            action_c = torch.tanh(mean) * 80
            action = [*action_c.squeeze().cpu().numpy().tolist(), *(binary_logits.squeeze().cpu().numpy() > 0.5).tolist()]
            print(action, end='\r')
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f'Epoch {epoch} Total Reward: {total_reward}')
