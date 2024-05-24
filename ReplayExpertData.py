from EnvReceiver import EnvReceiver
import numpy as np
import pickle
import time

with open("Luigi-Raceway-hard1.pkl", "rb") as f:
    history = pickle.load(f)

env = EnvReceiver()

done = False
total_reward = 0
total_steps = 0
obs = env.reset()
while not done:
    # action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
    action = history["action"][total_steps]
    next_obs, reward, done, info = env.step(action)
    obs = next_obs
    total_reward += reward
    total_steps += 1
    time.sleep(0.08)
print("Total reward:", total_reward)
print("Total steps:", total_steps)