from EnvReceiver import EnvReceiver
import numpy as np

env = EnvReceiver()

done = False
total_reward = 0
obs = env.reset()
while not done:
    action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
    obs, reward, done, info = env.step(action)
    print(obs.shape, np.max(obs), np.min(obs))
    total_reward += reward
print("Total reward:", total_reward)