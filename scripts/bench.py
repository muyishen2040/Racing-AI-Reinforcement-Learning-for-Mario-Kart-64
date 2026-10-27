"""Measure CPU seconds per environment step, alone and with several processes.

Reports wall-clock steps per second, CPU milliseconds per step (all threads
of the process), and cores busy for one process with the default and the
single-threaded Angrylion renderer, then the aggregate rate for N processes
with one renderer thread each. The results on the development machine are
in docs/validation.md.

Usage: python scripts/bench.py [--steps 300] [--procs 4,8,16]
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import resource
import time

import numpy as np

ACCELERATE = np.array([0, 0, 1, 0, 0], dtype=np.float32)


def cpu_seconds() -> float:
    """User plus system CPU time of this process, all threads."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def run(rdp_threads: int, steps: int) -> tuple[float, float]:
    """Step one env `steps` times and return (cpu_seconds, wall_seconds) for the loop."""
    import mk64_env.core as core

    core.CORE_OPTIONS["mupen64plus-angrylion-multithread"] = str(rdp_threads)
    from mk64_env.env import MarioKartEnv

    env = MarioKartEnv(frame_skip=4)
    env.reset()
    c0, w0 = cpu_seconds(), time.perf_counter()
    for _ in range(steps):
        env.step(ACCELERATE)
    c1, w1 = cpu_seconds(), time.perf_counter()
    env.close()
    return c1 - c0, w1 - w0


def worker(rdp_threads: int, steps: int, queue: mp.Queue) -> None:
    """Process entry point: run() and report through `queue`."""
    queue.put(run(rdp_threads, steps))


def main() -> None:
    """Parse arguments and print one line per configuration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--procs", default="4,8,16", help="comma-separated process counts")
    args = parser.parse_args()
    steps = args.steps

    for threads in (4, 1):
        cpu, wall = run(threads, steps)
        print(f"1 process, rdp threads={threads}: {steps / wall:6.1f} steps/s, {cpu / steps * 1000:6.1f} CPU-ms/step, {cpu / wall:.2f} cores busy")

    ctx = mp.get_context("spawn")
    for count in [int(n) for n in args.procs.split(",") if n]:
        queue: mp.Queue = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(1, steps, queue)) for _ in range(count)]
        w0 = time.perf_counter()
        for proc in procs:
            proc.start()
        results = [queue.get() for _ in procs]
        for proc in procs:
            proc.join()
        wall = time.perf_counter() - w0
        cpu = sum(r[0] for r in results)
        print(f"{count} processes, rdp threads=1: {count * steps / wall:6.1f} aggregate steps/s incl. startup, {cpu / (count * steps) * 1000:6.1f} CPU-ms/step")


if __name__ == "__main__":
    main()
