# Python 3 / Gymnasium migration (work in progress)

This branch is a staged modernization of the **Mario Kart 64** environment.
It uses a two-service Docker Compose layout: `simulator` (Mupen64Plus,
Xvfb, read-only VNC and Gymnasium RPC) and `trainer` (a Python 3 client
and environment smoke test). The original files on `master` remain intact.

> **Status (2026-10-20):** The published images ran on an Azure Ubuntu 22.04
> B2ats_v2 VM with an authorized, privately mounted ROM. Luigi Raceway,
> Moo Moo Farm, and Koopa Troopa Beach passed real-game smoke tests: moving
> frames, steering, finite rewards, rising checkpoints, repeated resets, and
> reconnects. View-only VNC showed live gameplay over an SSH tunnel.
> See [AZURE_DEPLOYMENT.md](AZURE_DEPLOYMENT.md) for deployment and cost controls.

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

A successful smoke run creates `outputs/reset.png`, `accelerate.png`,
`left.png`, `right.png`, `second_reset.png`, `reconnect_reset.png`, and
`smoke_summary.json`. The script checks frame shape, nonblank and changing
images, finite rewards, repeated reset, and reconnect. Inspect the PNGs
yourself: changing pixels alone cannot prove that the kart is driving.
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
PYTHONPATH=simulator python -m pytest tests/test_protocol.py tests/test_controller.py tests/test_mario_kart_hud.py -q
```

### Known limitations

- The input bot plugin loaded and accepted actions on Ubuntu 22.04 with the
  packaged Mupen64Plus emulator. A full race and every registered track have
  not been validated.
- The environment starts a fresh emulator after its controller server is ready.
  It stops the emulator on client disconnect and restarts it on reconnect or
  track change. A missing plugin poll now times out with a pointer to
  `/tmp/emulator.log` instead of hanging indefinitely.
- HUD checkpoint colors and reward values were checked on three actual
  courses. The first ~15 steps may report checkpoint `-1` while the progress
  HUD has not appeared; this is treated as unknown progress.
- Repeated resets and client reconnects worked on all three tested courses.
- The original standalone `train_sac.py` / `train_bc.py` still use
  `EnvReceiver`; they have **not** been converted to this remote client.
- This server supports one client and one emulator. It is not suitable for
  untrusted networks.
- The old Compose file and Dockerfile are retained for comparison.

### Published images and Azure VM

Feature-branch pushes build both images in GitHub Actions and push immutable
`sha-<full commit SHA>` tags to `ghcr.io/muyishen2040/mk64-simulator` and
`ghcr.io/muyishen2040/mk64-trainer`. Pull requests build without publishing.
The Docker build context excludes both the legacy tracked ROM directory and
the private `roms/` directory. Do not add ROM files to the build context.

The ROM-free GHCR packages are public, so the VM can pull them without a
GitHub token. Never put a token in Compose, source control, or chat.

On the VM, use `compose.azure.yaml` to pull the CI-built images without a
local build:

```bash
mkdir -p ~/mariokart64/roms ~/mariokart64/outputs
cd ~/mariokart64
# Copy compose.azure.yaml here, then transfer your authorized ROM to roms/marioKart.n64.
export MK64_IMAGE_SHA=<full-feature-branch-commit-sha>
docker compose -f compose.azure.yaml --profile smoke config -q
docker compose -f compose.azure.yaml pull
docker compose -f compose.azure.yaml up -d --no-build simulator
docker compose -f compose.azure.yaml --profile smoke run --rm --no-build trainer
docker compose -f compose.azure.yaml logs --tail=200 simulator
docker stats --no-stream
free -h
```

To observe active gameplay through VNC, establish the SSH tunnel described
above and start a longer driver session in a second terminal:

```bash
docker compose -f compose.azure.yaml --profile smoke run --rm --no-build \
  trainer python /app/trainer/smoke_test.py --watch-seconds 120
```

The display is blank until the client calls `reset()`. The emulator advances
while a client sends actions and stops when that client disconnects.

If the smoke test fails, read `/tmp/emulator.log`, `/tmp/xvfb.log`, and
`/tmp/vnc.log` inside the simulator container. An RPC health check means only
that the server is listening; it does not show that the game booted.

Install Docker Engine and Compose v2 on Ubuntu using
`scripts/install_docker_ubuntu_22_04.sh`, which follows the
[official Docker instructions](https://docs.docker.com/engine/install/ubuntu/).
Verify the selected Azure subscription, SKU and quota, free usage, disk, and
public IP price in that account before creating resources. Microsoft currently
lists B2ats v2 as an x86-64 2 vCPU, 1 GiB VM included in the Azure free
account's limited 12-month offer, but the offer and regional capacity depend
on the subscription. A Standard public IPv4 address can be charged separately,
as can the disk after any applicable free allowance. Deallocating a VM stops
compute billing, but attached storage and public IP resources can keep costing
money. Use a budget alert and get approval for any nontrivial projected charge.

### Azure free tier considerations

Azure's introductory free offerings can include 750 VM-hours/month on eligible
burstable B-series Linux VM sizes for 12 months, depending on account eligibility.
A B1s is only 1 vCPU and 1 GiB RAM, without a GPU: useful for a **limited
headless smoke test**, but graphics emulation, Docker builds, and VNC may exhaust
its RAM and CPU credits. A larger paid VM may be required. Always check the
Azure portal's actual price and eligibility before provisioning and set a budget
alert. The simulator ran on B2ats_v2 without an observed out-of-memory kill.
