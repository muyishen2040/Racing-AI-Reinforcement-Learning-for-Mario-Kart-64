# Python 3 / Gymnasium migration (work in progress)

This branch is a staged modernization of the **Mario Kart 64** environment.
It uses a two-service Docker Compose layout: `simulator` (Mupen64Plus,
Xvfb, read-only VNC and Gymnasium RPC) and `trainer` (a Python 3 client
and environment smoke test). The original files on `master` remain intact.

> **Status:** source migrated and smoke-test tooling included; native
> Mupen64Plus plugin and the end-to-end container launch have **not yet
> been verified** in an environment with Docker and a lawfully obtained ROM.

## Running locally or on a Linux VM

Requirements: an x86-64 Linux host with Docker Engine and Docker Compose v2.
Use an appropriately obtained and authorized Mario Kart 64 ROM in
`roms/marioKart.n64`. Do not commit proprietary game files.

```bash
mkdir -p roms outputs
docker compose -f compose.python3.yaml build
docker compose -f compose.python3.yaml up -d simulator
docker compose -f compose.python3.yaml --profile smoke run --rm trainer
docker compose -f compose.python3.yaml logs --tail=100 simulator
```

A successful smoke run creates `outputs/reset.png` and
`outputs/last_step.png` after reset and 30 control actions. Those artifacts
are more reliable than a simple port-open check of emulator success.
For headless visualization, run a VNC client on your own computer
against `localhost:5901`. On an Azure Linux VM, use an SSH tunnel:

```bash
ssh -L 5901:127.0.0.1:5901 <username>@<vm-public-ip>
```

Docker maps VNC to the VM's loopback address and x11vnc is view-only, so
never open port 5901 publicly in the Azure network security group.
Restrict SSH (port 22) to your own IP address.

### Trainer usage

```python
from remote_env import RemoteMarioKartEnv

env = RemoteMarioKartEnv(host="simulator", port=5555)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0, 0, 1, 0, 0])
env.close()
```

### Python-only tests

```bash
python -m pip install pytest numpy
PYTHONPATH=simulator python -m pytest tests/test_protocol.py -q
```

### Known limitations

- This branch is **not** a demonstrated emulator-compatible build. The original
  input bot plugin may require adjustments for Ubuntu 22.04 package APIs.
- Mario Kart menu navigation, HUD pixel colors, reward values, and frame timing
  must be regression-tested on the actual game. The simulator relies on the
  Glide64 graphics plugin for HUD colors.
- `MarioKartEnv.close()` cleanup and multi-track switches must be exercised.
- The original standalone `train_sac.py` / `train_bc.py` still use
  `EnvReceiver`; they have **not** been converted to this remote client.
- This server supports one client and one emulator. It is not suitable for
  untrusted networks.
- The old Compose file and Dockerfile are retained for comparison.

### Azure free tier considerations

Azure's introductory free offerings can include 750 VM-hours/month on eligible
burstable B-series Linux VM sizes for 12 months, depending on account eligibility.
A B1s is only 1 vCPU and 1 GiB RAM, without a GPU: useful for a **limited
headless smoke test**, but graphics emulation, Docker builds, and VNC may exhaust
its RAM and CPU credits. A larger paid VM may be required. Always check the
Azure portal's actual price and eligibility before provisioning and set a budget
alert. No Azure VM has been provisioned by this branch.
