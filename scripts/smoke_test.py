"""Drive the real game for a few seconds and save screenshots a person can check.

Writes reset.png, accelerate.png, left.png, right.png, second_reset.png, and
smoke_summary.json into --out. Fails if a frame is blank, frames stop
changing, a reward is not finite, acceleration makes no progress, left and
right steering turn the kart the same way, or a second reset diverges from
the first. Changing pixels alone cannot prove that the kart is driving, so
look at the PNGs.

Usage: python scripts/smoke_test.py [--course LuigiRaceway] [--steps 30] [--steering-steps 15] [--out outputs/smoke]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

import mk64_env
from mk64_env.video import stretch_frame

ACCELERATE = np.array([0, 0, 1, 0, 0], dtype=np.float32)
LEFT = np.array([-80, 0, 1, 0, 0], dtype=np.float32)
RIGHT = np.array([80, 0, 1, 0, 0], dtype=np.float32)


def save_frame(out: Path, name: str, frame: np.ndarray) -> None:
    """Write `frame` as out/name.png at 640x480."""
    Image.fromarray(stretch_frame(frame)).save(out / f"{name}.png")


def yaw_delta(before: int, after: int) -> int:
    """Signed change of a 16-bit wrapping yaw."""
    return (after - before + 0x8000) % 0x10000 - 0x8000


def run_phase(env: mk64_env.MarioKartEnv, out: Path, name: str, action: np.ndarray, steps: int) -> dict[str, Any]:
    """Hold `action` for `steps` steps, save the last frame, and summarize the phase."""
    previous = env.render()
    start = env.step(np.zeros(5, dtype=np.float32))[4]
    differences = []
    total = 0.0
    info: dict[str, Any] = start
    for _ in range(steps):
        frame_obs, reward, terminated, truncated, info = env.step(action)
        if not np.isfinite(reward):
            raise AssertionError(f"non-finite reward during {name}")
        if terminated or truncated:
            raise AssertionError(f"episode ended during {name}")
        frame = env.render()
        differences.append(float(np.abs(frame.astype(np.int16) - previous.astype(np.int16)).mean()))
        previous = frame
        total += reward
    save_frame(out, name, previous)
    if max(differences) < 0.5:
        raise AssertionError(f"frames did not change during {name}")
    return {
        "steps": steps,
        "max_frame_difference": max(differences),
        "progress_start": start["progress"],
        "progress_end": info["progress"],
        "yaw_delta": yaw_delta(start["yaw"], info["yaw"]),
        "lap_end": info["lap"],
        "speed_end": info["speed"],
        "total_reward": total,
    }


def main() -> None:
    """Run the phases, write the screenshots and summary, and print the verdict."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--course", default="LuigiRaceway")
    parser.add_argument("--character", default="mario")
    parser.add_argument("--steps", type=int, default=30, help="acceleration steps")
    parser.add_argument("--steering-steps", type=int, default=15)
    parser.add_argument("--frame-skip", type=int, default=4)
    parser.add_argument("--out", default="outputs/smoke")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = mk64_env.MarioKartEnv(course=args.course, character=args.character, frame_skip=args.frame_skip, render_mode="rgb_array")
    started = time.perf_counter()
    summary: dict[str, Any] = {
        "course": args.course,
        "character": args.character,
        "frame_skip": args.frame_skip,
        "core_sha256": hashlib.sha256(env.core_path.read_bytes()).hexdigest(),
    }
    try:
        obs, info = env.reset()
        frame = env.render()
        save_frame(out, "reset", frame)
        if int(frame.max()) - int(frame.min()) < 20:
            raise AssertionError("reset returned a nearly blank frame")
        summary["reset"] = info
        first_step = env.step(ACCELERATE)[4]
        summary["accelerate"] = run_phase(env, out, "accelerate", ACCELERATE, args.steps)
        summary["left"] = run_phase(env, out, "left", LEFT, args.steering_steps)
        summary["right"] = run_phase(env, out, "right", RIGHT, args.steering_steps)
        if summary["accelerate"]["progress_end"] <= summary["accelerate"]["progress_start"]:
            raise AssertionError("holding A made no progress")
        if summary["left"]["yaw_delta"] * summary["right"]["yaw_delta"] >= 0:
            raise AssertionError("left and right steering did not turn the kart in opposite directions")
        obs, info = env.reset()
        save_frame(out, "second_reset", env.render())
        summary["second_reset"] = info
        if env.step(ACCELERATE)[4]["position"] != first_step["position"]:
            raise AssertionError("second reset diverged from the first")
        total_steps = 2 + args.steps + 2 * args.steering_steps + 3
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["steps_per_second"] = total_steps / summary["elapsed_seconds"]
    finally:
        env.close()
    (out / "smoke_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"SMOKE OK: wrote {out}/*.png and smoke_summary.json. Inspect the PNGs to confirm the kart is on the track.")


if __name__ == "__main__":
    main()
