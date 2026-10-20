"""Run one real emulator episode through Gymnasium's terminal or time limit."""

import argparse
import json
import os
import time

import numpy as np

from remote_env import RemoteMarioKartEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="Luigi-Raceway")
    parser.add_argument("--max-steps", type=int, default=1250)
    args = parser.parse_args()

    env = RemoteMarioKartEnv(track=args.track, timeout=300)
    started = time.monotonic()
    rewards = []
    checkpoint_max = -1
    try:
        observation, _ = env.reset()
        assert observation.shape == (480, 640, 3)
        for step in range(1, args.max_steps + 1):
            observation, reward, terminated, truncated, info = env.step([0, 0, 1, 0, 0])
            assert observation.shape == (480, 640, 3)
            assert np.isfinite(reward)
            rewards.append(float(reward))
            checkpoint_max = max(checkpoint_max, int(info["checkpoint"]))
            if step % 100 == 0 or terminated or truncated:
                print("EPISODE", step, "reward", round(sum(rewards), 2),
                      "lap", info["lap"], "checkpoint", info["checkpoint"],
                      "terminated", terminated, "truncated", truncated, flush=True)
            if terminated or truncated:
                result = {
                    "track": args.track,
                    "steps": step,
                    "terminated": terminated,
                    "truncated": truncated,
                    "total_reward": sum(rewards),
                    "highest_checkpoint": checkpoint_max,
                    "elapsed_seconds": round(time.monotonic() - started, 1),
                }
                os.makedirs("/app/outputs", exist_ok=True)
                with open("/app/outputs/episode_summary.json", "w", encoding="utf-8") as output:
                    json.dump(result, output, indent=2)
                print("EPISODE OK", json.dumps(result), flush=True)
                return
    finally:
        env.close()
    raise AssertionError("Episode did not end within %d steps" % args.max_steps)


if __name__ == "__main__":
    main()
