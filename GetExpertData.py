from EnvReceiver import EnvReceiver
import numpy as np
import pygame
import pickle
import time

def clip(x):
    return np.clip(x, -0.8, 0.8)

pygame.init()
assert pygame.joystick.get_count() == 1, pygame.joystick.get_count()
joystick = pygame.joystick.Joystick(0)
joystick.init()
history = {"obs": [], "action": [], "reward": [], "done": []}

env = EnvReceiver()

done = False
total_reward = 0
total_steps = 0
obs = env.reset()
while not done:
    pygame.event.pump()
    print(f"Axis 0: {joystick.get_axis(0):.2f}, Axis 1: {joystick.get_axis(1):.2f}, Axis 2: {joystick.get_axis(2):.2f}, Axis 3: {joystick.get_axis(3):.2f}, {joystick.get_button(0)}, {joystick.get_button(1)}, {joystick.get_button(7)}", end="\r")
    # action = [np.random.uniform(-80, 80), np.random.uniform(-80, 80), np.random.choice([0, 1]), np.random.choice([0, 1]), np.random.choice([0, 1])]
    action = [clip(joystick.get_axis(0)) * 100, clip(joystick.get_axis(1)) * 100, int(joystick.get_button(0)), int(joystick.get_button(1)), int(joystick.get_button(7))]
    next_obs, reward, done, info = env.step(action)
    history["obs"].append(obs)
    history["action"].append(action)
    history["reward"].append(reward)
    history["done"].append(done)
    obs = next_obs
    total_reward += reward
    total_steps += 1
    time.sleep(0.1)
print("Total reward:", total_reward)
print("Total steps:", total_steps)
with open("Luigi-Raceway.pkl", "wb") as f:
    pickle.dump(history, f)