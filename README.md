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

SAC uses reparameterized joystick actions and a score-function gradient for
the discrete A/B/R buttons. Its entropy target treats the joystick in
normalized [-1, 1] units and targets half the maximum binary-button entropy.
Every 20 training episodes it evaluates the deterministic policy for three
races (up to 1,000 steps each). `--eval-every`, `--eval-episodes`, and
`--eval-max-steps` adjust this; `--eval-every 0` disables evaluation.
Evaluation logs mean progress, reward, and completion rate to TensorBoard;
the highest mean evaluation progress determines `actor_best.pth`. The
training-episode best is saved separately as `actor_training_best.pth`.
Evaluation adds emulator time and its maximum length may limit completion.

## Resuming SAC training

SAC writes `checkpoints/sac/latest.pt` every 1,000 environment steps and on
a graceful Ctrl+C or SIGTERM. It includes the model and target networks,
optimizers, replay buffer, counters, training configuration, and random
generator states. Checkpoints are written to a temporary file and then
replaced atomically. Use `--checkpoint-every N` to adjust the interval.
For example:

```bash
python train_sac.py --memory-len 10000 --preview
# After Ctrl+C and the recovery checkpoint has finished saving:
python train_sac.py --resume --preview
```

Use `--forward-bias 10` when starting or resuming to set the trainable A-button
bias to 10. This resets the optimizer moments for that parameter only; later
resumes without the flag keep the bias saved in the checkpoint.

`--episodes` is the total episode target including earlier episodes. A
checkpoint taken mid-episode resumes at the beginning of that episode, not
the exact emulator frame; any work since the last checkpoint can be lost on
a crash, forced kill, or sudden shutdown. Always stop the training process
gracefully and wait for it to exit before stopping the instance. A replay
buffer can make checkpoints large, and saving temporarily needs enough disk
space for both the old and the new checkpoint. Recovery files use Python
pickle: only load checkpoints from trusted sources.

Old `actor_best.pth`, `critic_best.pth`, and `entropy_best.pth` files are model
weights, not full recovery checkpoints. To start a **new** run from them,
use `python train_sac.py --init-from checkpoints/sac --memory-len 10000`.
This starts a fresh replay buffer and optimizers; it cannot restore the old
run's in-memory state. By default a new run refuses to overwrite an existing
`latest.pt`; use `--resume` or choose another `--checkpoint-dir`.

## Live preview over VNC

The optional preview shows full-colour emulator frames while the agent still
receives its normal grayscale observations. Install the preview dependencies
with `pip install -e '.[preview]'` (they are also included in `[train]`).
Start a graphical display, then run one of:

```bash
python scripts/random_agent.py --agent straight --steps 1000 --preview
python train_sac.py --preview
python test.py checkpoints/sac/actor_best.pth --preview
```

The window updates every four environment steps by default; use
`--preview-every 1` for more frequent updates or a larger value to reduce
overhead. Closing the window disables the preview without ending training.
Without `--preview` the scripts remain headless. Previewing can slow training.

To evaluate an actor downloaded from a GitHub release, provide its path to
`test.py`. For example, after setting up the emulator and your own ROM:

```bash
python test.py luigi_raceway_sac_step_247898.pt --course LuigiRaceway --episodes 1 --preview --preview-every 1
```

Evaluation uses deterministic actions by default; add `--stochastic` to sample
actions. The released actor contains inference weights only. To resume training,
use a full recovery checkpoint instead. Episodes are capped at 2,500 steps.

For an EC2 instance, run TigerVNC on the instance bound to localhost and
connect from your computer through an SSH tunnel. For example, after
installing `tigervnc-standalone-server` and `openbox` on Ubuntu:

```bash
vncserver :1 -localhost yes -SecurityTypes None -geometry 1024x768 -xstartup /usr/bin/openbox
export DISPLAY=:1
cd ~/mariokart64
python scripts/random_agent.py --agent straight --steps 1000 --preview
```

This example uses SSH authentication rather than a VNC password: other local
users on the EC2 host can access the display, so use a VNC password or separate
user isolation on a shared machine. Do not expose TCP 5901 to the internet.
From Windows PowerShell, keep this SSH session open and connect a VNC viewer to
`localhost:5901`:

```powershell
ssh -i "$env:USERPROFILE\.ssh\id_ed25519" -L 5901:localhost:5901 ubuntu@<EC2-public-IP>
```

For an EC2 instance and Windows AWS CLI profile, the helper
`powershell -ExecutionPolicy Bypass -File scripts/connect_vnc.ps1 -InstanceId <instance-id> -Profile <profile>` starts
the remote display if needed, opens the tunnel and launches the local
TigerVNC viewer. It assumes the instance is already running and the viewer
is installed at `%LOCALAPPDATA%\mk64-tools\vncviewer.exe`; use its parameters
to override paths or the instance ID. Closing the viewer closes its tunnel.

Stop the display with `vncserver -kill :1` when no longer needed. The instance
continues to incur compute charges until stopped.

## Docker

`Dockerfile` builds a CPU-only image that runs the smoke test by default.
`docs/cloud.md` has the commands and notes for running it on a small cloud VM.

## Documentation

- `docs/ram_map.md`: the RAM addresses behind rewards and episode ends, with evidence.
- `docs/validation.md`: what was tested on which core build, and what was not.
- `docs/cloud.md`: Docker image usage and Azure sizing, cost, and network rules.
