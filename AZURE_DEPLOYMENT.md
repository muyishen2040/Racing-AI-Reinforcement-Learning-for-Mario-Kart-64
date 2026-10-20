# Azure simulator deployment

Validated on 2026-10-20 with the published image tag
`sha-cd23a7c8263da6b641d6fe75d4c39252c69d5b45`, Ubuntu Server 22.04,
and an authorized ROM mounted privately on an Azure `Standard_B2ats_v2` VM.
The VM used 2 burstable x86-64 vCPUs, 1 GiB RAM, a 64 GiB Premium SSD P6
OS disk, and a Standard static IPv4 address in North Central US. The ROM is
not included in either image or this repository's new commits.

## Provisioning and cost controls

The tested deployment used resource group `mk64-simulator-rg`, VM
`mk64-simulator-dev`, and SSH user `azureuser`. The network security group
permits inbound TCP 22 only from the deployment client's current public IP.
It does not expose RPC port 5555 or VNC port 5901. If your client IP changes,
update that SSH rule; do not open VNC to the internet.

Check the target subscription, its current free service allowance, regional
VM size availability, disk and IP prices, and quota before recreating the
deployment. This account had a Free Trial subscription with the spending
limit on. A `$5/month` Azure budget alert named `mk64-simulator-monthly`
was created at 50% and 100% thresholds. The alert sends email; it does not
stop resources or cap spending. The B2ats v2 VM and two P6 disks are in
Azure's published introductory allowance while the account qualifies. The
Standard public IPv4 address is charged separately at about `$0.005/hour`
in North Central US. Exact remaining free hours and trial credit must be
checked in the Azure portal; the CLI did not expose them here.

To recreate the VM from PowerShell, sign in with Azure CLI and set the
subscription explicitly. Replace the public key and client IP with your
own values. These commands reproduce the tested networking and storage
layout; they create billable resources, so inspect the current prices and
free allowances first.

```powershell
az account set --subscription '<your-subscription-id>'
$clientIp = (Invoke-RestMethod 'https://api.ipify.org').Trim()
$publicKey = Join-Path $HOME '.ssh\id_ed25519.pub'
az group create --name mk64-simulator-rg --location northcentralus
az network nsg create --resource-group mk64-simulator-rg --name mk64-simulator-nsg --location northcentralus
az network nsg rule create --resource-group mk64-simulator-rg --nsg-name mk64-simulator-nsg `
  --name AllowSSHFromCurrentIP --priority 1000 --direction Inbound --access Allow `
  --protocol Tcp --source-address-prefixes "$clientIp/32" --destination-port-ranges 22
az network public-ip create --resource-group mk64-simulator-rg --name mk64-simulator-ip `
  --location northcentralus --sku Standard --allocation-method Static
az vm create --resource-group mk64-simulator-rg --name mk64-simulator-dev `
  --location northcentralus --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest `
  --size Standard_B2ats_v2 --admin-username azureuser --authentication-type ssh `
  --ssh-key-values $publicKey --nsg mk64-simulator-nsg `
  --public-ip-address mk64-simulator-ip --public-ip-sku Standard `
  --os-disk-size-gb 64 --storage-sku Premium_LRS
az vm show --resource-group mk64-simulator-rg --name mk64-simulator-dev -d `
  --query '{power:powerState,ip:publicIps}' --output json
```

Allow SSH only from the current public IP. The static public IP may differ
from the one used in the completed validation deployment. Wait for SSH to
accept the key before copying the deployment files.

## Install Docker and deploy images

From a checkout of this branch on your Windows computer, copy the installer,
Compose file, and your own authorized ROM over SSH. Set `$romPath` to that
ROM's local path; it is never placed in the repository:

```powershell
$vmIp = az vm show --resource-group mk64-simulator-rg --name mk64-simulator-dev -d `
  --query publicIps --output tsv
$romPath = '<path-to-your-authorized-ROM>'
ssh "azureuser@$vmIp" 'mkdir -p ~/mariokart64/scripts ~/mariokart64/roms ~/mariokart64/outputs'
scp scripts/install_docker_ubuntu_22_04.sh "azureuser@${vmIp}:mariokart64/scripts/install_docker_ubuntu_22_04.sh"
scp compose.azure.yaml "azureuser@${vmIp}:mariokart64/compose.azure.yaml"
scp $romPath "azureuser@${vmIp}:mariokart64/roms/marioKart.n64"
```

Install Docker with the included Ubuntu 22.04 installer on the VM:

```bash
bash scripts/install_docker_ubuntu_22_04.sh
```

Set the ROM file mode to 600. Do not put the ROM into Git, an image build
context, or a registry. Then:

```bash
cd ~/mariokart64
mkdir -p roms outputs
chmod 600 roms/marioKart.n64
echo MK64_IMAGE_SHA=cd23a7c8263da6b641d6fe75d4c39252c69d5b45 > .env
sudo docker compose -f compose.azure.yaml --profile smoke pull simulator trainer
sudo docker compose -f compose.azure.yaml up -d simulator
sudo docker compose -f compose.azure.yaml ps
```

The simulator service starts the RPC listener, Xvfb, and view-only VNC. The
game itself boots when the trainer calls `reset()`. The images are public,
ROM-free GHCR packages, so no registry credential is needed. Image builds
belong in GitHub Actions; the 1 GiB VM only pulls prebuilt images.

## Run and inspect the real game

```bash
cd ~/mariokart64
sudo docker compose -f compose.azure.yaml --profile smoke run --rm trainer
sudo docker compose -f compose.azure.yaml --profile smoke run --rm \
  trainer python /app/trainer/smoke_test.py --track Moo-Moo-Farm --steps 20 --steering-steps 5
sudo docker compose -f compose.azure.yaml --profile smoke run --rm \
  trainer python /app/trainer/smoke_test.py --track Koopa-Troopa-Beach --steps 20 --steering-steps 5
sudo docker compose -f compose.azure.yaml logs --tail=100 simulator
sudo docker stats --no-stream
free -h
df -h /
```

The smoke test saves game frames and `smoke_summary.json` in `outputs/`.
Its screenshots are overwritten by subsequent runs, so copy them before
running a different track if you need to retain evidence. Inspect the
frames: during validation Mario drove on three distinct courses, left and
right steering changed his path, and checkpoint numbers increased while
the progress HUD appeared. On Luigi Raceway, checkpoint values advanced
from `-1` to `18`; on Moo Moo Farm to `53`; and on Koopa Troopa Beach to
`24`. `-1` means progress is temporarily unknown, for example before the
HUD's progress line appears. Repeated reset and client reconnect worked on
all three courses. The emulator's deliberate shutdown may log exit `-9`
after a client disconnects; no Linux OOM kill was observed.

For a complete episode with the pinned validated image, copy
`trainer/episode_test.py` from this branch to
`~/mariokart64/episode_test.py` on the VM, then mount it into the trainer:

```bash
sudo docker compose -f compose.azure.yaml --profile smoke run --rm \
  -v /home/azureuser/mariokart64/episode_test.py:/app/trainer/episode_test.py:ro \
  trainer python /app/trainer/episode_test.py --track Luigi-Raceway
```

The script prints progress every 100 steps and writes
`outputs/episode_summary.json` on termination or time-limit truncation. The
validated Luigi Raceway episode reached the 1,250-step time limit in 100.4
seconds and truncated cleanly; a straight-acceleration policy eventually
got stuck, so its cumulative reward was negative.

## Watch through SSH

Start an SSH tunnel on your own computer, then connect a VNC viewer to
`localhost:5901`:

```powershell
ssh -N -L 5901:127.0.0.1:5901 azureuser@<vm-public-ip>
```

In another terminal, start a driver session on the VM:

```bash
cd ~/mariokart64
sudo docker compose -f compose.azure.yaml --profile smoke run --rm \
  trainer python /app/trainer/smoke_test.py --watch-seconds 120
```

VNC is view-only; Gymnasium actions drive the kart. A live tunnel returned
the RFB 3.8 handshake and captured changing 640 x 480 Mario Kart frames.
When the VM is deallocated, start it again before using the tunnel.

## Stop resources

From an authenticated Azure CLI session:

```bash
az vm deallocate --resource-group mk64-simulator-rg --name mk64-simulator-dev
az vm show --resource-group mk64-simulator-rg --name mk64-simulator-dev -d \
  --query powerState --output tsv
```

`VM deallocated` stops VM compute usage. The P6 disk and Standard static
public IPv4 address remain allocated and may continue consuming trial
credit. Deleting the resource group removes those resources and the private
ROM on the VM; do that only when the deployment is no longer needed:

```bash
az group delete --name mk64-simulator-rg --yes
az group exists --name mk64-simulator-rg  # must return false
```
