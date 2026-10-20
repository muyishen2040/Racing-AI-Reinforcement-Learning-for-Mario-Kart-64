"""Exercise real emulator frames and controller actions through the remote API."""
import argparse
import json
import os
import time

import numpy as np
from PIL import Image

from remote_env import RemoteMarioKartEnv


def save_frame(name, frame):
    if frame.shape != (480, 640, 3) or frame.dtype != np.uint8:
        raise AssertionError("Expected a 480x640 uint8 RGB game frame")
    Image.fromarray(frame).save(os.path.join("/app/outputs", name + ".png"))


def run_actions(env, count, action, label):
    previous = env.render()
    differences = []
    checkpoints = []
    for i in range(count):
        frame, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(reward):
            raise AssertionError("Nonfinite reward on %s step %d" % (label, i))
        if "lap" not in info or "checkpoint" not in info:
            raise AssertionError("Missing lap or checkpoint on %s step %d" % (label, i))
        differences.append(float(np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()))
        checkpoints.append(int(info["checkpoint"]))
        print("STEP", label, i, "reward", reward, "lap", info["lap"],
              "checkpoint", info["checkpoint"], "frame_difference", round(differences[-1], 3))
        if terminated or truncated:
            raise AssertionError("Episode ended before smoke-test phase %s completed" % label)
        previous = frame
    save_frame(label, previous)
    if not differences or max(differences) < 0.5:
        raise AssertionError("Frames did not visibly change during %s" % label)
    return {"max_frame_difference": max(differences),
            "first_checkpoint": checkpoints[0], "last_checkpoint": checkpoints[-1]}


def connect(track):
    for attempt in range(60):
        try:
            return RemoteMarioKartEnv(track=track, timeout=120)
        except OSError:
            if attempt == 59:
                raise
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", default="Luigi-Raceway")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--steering-steps", type=int, default=15)
    parser.add_argument("--watch-seconds", type=int, default=0,
                        help="Keep sending acceleration actions for live VNC viewing")
    args = parser.parse_args()
    os.makedirs("/app/outputs", exist_ok=True)
    summary = {"track": args.track}
    env = connect(args.track)
    try:
        frame, info = env.reset()
        save_frame("reset", frame)
        if int(frame.max()) - int(frame.min()) < 20:
            raise AssertionError("Reset returned a nearly blank frame")
        summary["accelerate"] = run_actions(env, args.steps, [0, 0, 1, 0, 0], "accelerate")
        summary["left"] = run_actions(env, args.steering_steps, [-80, 0, 1, 0, 0], "left")
        summary["right"] = run_actions(env, args.steering_steps, [80, 0, 1, 0, 0], "right")
        frame, info = env.reset()
        save_frame("second_reset", frame)
        summary["second_reset"] = info
        watch_until = time.monotonic() + args.watch_seconds
        while time.monotonic() < watch_until:
            frame, reward, terminated, truncated, info = env.step([0, 0, 1, 0, 0])
            if terminated or truncated:
                frame, info = env.reset()
    finally:
        env.close()

    env = connect(args.track)
    try:
        frame, info = env.reset()
        save_frame("reconnect_reset", frame)
        summary["reconnect_reset"] = info
        summary["reconnect_step"] = run_actions(env, 5, [0, 0, 1, 0, 0], "reconnect_step")
    finally:
        env.close()
    with open("/app/outputs/smoke_summary.json", "w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
    print("SMOKE OK: RPC, nonblank changing frames, finite rewards, repeated reset, reconnect")
    print("Inspect /app/outputs/*.png to confirm actual gameplay and steering visually")


if __name__ == "__main__":
    main()
