# Racing AI: Reinforcement Learning for Mario Kart 64

A headless Gymnasium environment for Mario Kart 64 Time Trials, plus Soft
Actor-Critic, behavior cloning, and Grad-CAM scripts that use it.

The emulator is the Mupen64Plus-Next libretro core loaded in-process through
[libretro.py](https://pypi.org/project/libretro.py/). It renders with the
Angrylion software RDP, so no display server, GPU, or Docker is needed.
Rewards and episode ends come from the game's RAM (see `docs/ram_map.md`),
and each reset loads a savestate taken at the green light.

This project started as a fork of [bzier/gym-mupen64plus](https://github.com/bzier/gym-mupen64plus).

## Setup

Requirements: Linux x86_64, Python 3.11 or newer, and a US Mario Kart 64
ROM. The ROM is not in the repository.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[train,test]"     # drop [train] to skip torch
scripts/fetch_core.sh              # downloads cores/mupen64plus_next_libretro.so
cp /path/to/your/rom roms/marioKart.n64
pytest                             # about 15 s; generates the Luigi Raceway savestate
python scripts/smoke_test.py       # writes outputs/smoke/*.png for a visual check
```

`scripts/fetch_core.sh` checks the core against a pinned SHA-256, because
the libretro buildbot serves only its latest nightly. When upstream rebuilds
the core, the script refuses the download; rerun it with
`MK64_ACCEPT_NEW_CORE=1`, run `pytest`, and update the pin.
`docs/validation.md` records what the pinned build passed.

`pytest -m "not emulator"` runs the tests that need neither the core nor
the ROM. GitHub Actions runs those on every push (`.github/workflows/ci.yml`).

The environment looks for the core at `cores/mupen64plus_next_libretro.so`,
the ROM at `roms/marioKart.n64`, and caches savestates in `.mk64_cache/`.
Override with the `MK64_CORE`, `MK64_ROM`, and `MK64_CACHE` environment
variables or the constructor arguments.

## Environment

```python
import gymnasium as gym
import mk64_env

env = gym.make("MarioKart64/LuigiRaceway-Discrete-v0")   # Discrete(12) steering presets
env = gym.make("MarioKart64/LuigiRaceway-v0")            # Box [joy_x, joy_y, A, B, R]
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

- Observation: RGB frame resized to 120x160 (configurable with `obs_shape`).
- Reward: 100 times the change in course progress (one unit per lap).
- `terminated` when the third lap completes; `truncated` after `max_steps`.
- `info` carries lap, progress, speed, timer, position, and yaw.
- `frame_skip` (default 4) frames per step. One step costs about 30 CPU-ms
  with a single renderer thread; `docs/validation.md` has the measurements
  and the setting for parallel instances.
- Every course in `mk64_env.COURSES` and character in `mk64_env.CHARACTERS`
  is available. The first reset on a new course walks the menus once
  (about 10 s) and caches the savestate.
- Several environments can run in one process; each loads a private copy of
  the core. `gymnasium.vector.AsyncVectorEnv` works too.

`mk64_env.wrappers.make_pixel_env` builds the 4x128x128 grayscale stack that
the training scripts use.

## Scripts

| Script | Purpose |
|---|---|
| `scripts/smoke_test.py` | Drive for a few seconds, save screenshots, and check steering and resets |
| `scripts/random_agent.py` | Run a random or hold-A episode, optionally recording MP4 |
| `scripts/bench.py` | Measure CPU seconds per step, alone and in parallel |
| `train_sac.py` | Soft Actor-Critic with the hybrid action space |
| `test.py` | Evaluate a saved actor |
| `GetExpertData.py` | Record a gamepad demonstration to a pickle |
| `ReplayExpertData.py` | Replay a demonstration |
| `train_bc.py` | Behavior cloning from demonstrations |
| `grad_cam.py` | Grad-CAM video of a demonstration |

Networks live in `src/model.py`.

## Docker

`Dockerfile` builds a CPU-only image that runs the smoke test by default.
`docs/cloud.md` has the commands and notes for running it on a small cloud VM.

## Documentation

- `docs/ram_map.md`: the RAM addresses behind rewards and episode ends, with evidence.
- `docs/validation.md`: what was tested on which core build, and what was not.
- `docs/cloud.md`: Docker image usage and Azure sizing, cost, and network rules.
