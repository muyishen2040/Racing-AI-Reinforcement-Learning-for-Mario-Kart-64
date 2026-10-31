# Validation record

What the libretro environment was tested against, when, and how to repeat
each check. Update this file when the pinned core or a test changes.

## 2026-10-27, core nightly 2026-10-27

| Item | Value |
|---|---|
| Machine | WSL2 on an AMD Ryzen AI 9 HX 370 (12 cores, 24 threads), RTX 5090 Laptop GPU |
| Python and libraries | Python 3.12.12, libretro.py 0.12.0, Gymnasium 1.4.0, numpy 2.5.4, torch 2.14.1+cu130 |
| Core | `cores/mupen64plus_next_libretro.so`, sha256 `26f2b4ec722558d77bc9e1582e62ab80c84190bb20dfadc48a148e6210c8775e`, the pin in `scripts/fetch_core.sh` |
| ROM | Mario Kart 64 (U), MD5 `3A67D9986F54EB282924FCA4CD5F6DFF` as the core logs it |
| Course and character | Luigi Raceway, Mario, Time Trial |

### Checks

| Check | Command | Result |
|---|---|---|
| ROM-free unit tests | `pytest -q -m "not emulator"` | 9 passed in 0.3 s |
| Emulator tests | `pytest -q` | 16 passed in 5.7 s with a cached savestate |
| Screenshot smoke test | `python scripts/smoke_test.py` | SMOKE OK at 39 steps/s. Left steering gave yaw delta -14172 and right +7692. The PNGs show Mario on the first straight of Luigi Raceway. |
| Reset determinism | `tests/test_env.py::test_first_reset_matches_reset_after_driving` | Passes. Before `reset()` ran a neutral frame ahead of the savestate load, the first reset after core start differed from later resets by 3e-5 units in z. |
| Hold-A episode with video | `python scripts/random_agent.py --agent straight --steps 1500 --video straight.mp4` | Progress 0.28 then stuck on the first wall, 162 frames/s, 640x480 MP4 at 15 fps |
| Random episode | `python scripts/random_agent.py --agent random --steps 600` | Progress 0.068 |
| CPU cost per step | `python scripts/bench.py` | See the table below |
| Memory per environment | `resource.getrusage` ru_maxrss | 44 MiB baseline, 195 MiB with one env, 329 MiB with two |
| SAC smoke run | `python train_sac.py --episodes 3` on the GPU | Episode rewards -9.05, 8.70, 32.63; best progress 0.42 |
| Behavior cloning | scripted demo, `python train_bc.py demo.pkl --epochs 2` | Eval reward 19.65 |
| Evaluation and Grad-CAM | `python test.py`, `python grad_cam.py` | Reward 10.46; MP4 written |

CPU cost from `scripts/bench.py`, 300 steps of hold-A at frame skip 4:

| Configuration | Wall steps/s | CPU-ms per step | Cores busy |
|---|---|---|---|
| 1 process, 4 renderer threads (default) | 52.7 | 42.4 | 2.24 |
| 1 process, 1 renderer thread | 33.3 | 30.0 | 1.00 |
| 8 processes, 1 renderer thread each | 161 aggregate | 45.6 | |
| 16 processes, 1 renderer thread each | 226 aggregate | 65.1 | |

For parallel training set
`mk64_env.core.CORE_OPTIONS["mupen64plus-angrylion-multithread"] = "1"`
before creating environments. The 4-thread default costs 40 percent more
CPU per step for the single-process wall-clock gain.

### Not validated

- Race completion. No agent finished three laps, so the `terminated` path
  rests on the lap counter and the race state values in `docs/ram_map.md`.
- Courses other than Luigi Raceway and characters other than Mario. The
  menu sequence is parameterized and the course id is checked after
  loading, so a wrong selection raises.
- `GetExpertData.py` with a real gamepad. It was compile-checked only.
- The Docker image in `Dockerfile`. The Docker daemon was unavailable on the
  development machine.
- The GitHub Actions workflow in `.github/workflows/ci.yml`, until the
  branch is pushed.
