"""Verify remote simulator reset, actions, progression, and a PNG screenshot."""
import argparse
import os
import time

import numpy as np
from PIL import Image

from remote_env import RemoteMarioKartEnv

parser = argparse.ArgumentParser()
parser.add_argument("--track", default="Luigi-Raceway")
parser.add_argument("--steps", type=int, default=30)
args = parser.parse_args()

for attempt in range(60):
    try:
        env = RemoteMarioKartEnv(track=args.track, timeout=120)
        break
    except OSError:
        if attempt == 59:
            raise
        time.sleep(2)

try:
    obs, info = env.reset()
    print("RESET", obs.shape, info)
    assert obs.shape == (480, 640, 3)
    os.makedirs("/app/outputs", exist_ok=True)
    Image.fromarray(obs).save("/app/outputs/reset.png")
    for i in range(args.steps):
        obs, reward, terminated, truncated, info = env.step([0, 0, 1, 0, 0])
        print("STEP", i, reward, terminated, truncated, info)
        if terminated or truncated:
            obs, info = env.reset()
    Image.fromarray(obs).save("/app/outputs/last_step.png")
    print("PASS: saved /app/outputs/reset.png and last_step.png")
finally:
    env.close()
