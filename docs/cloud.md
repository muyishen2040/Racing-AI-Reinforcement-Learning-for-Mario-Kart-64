# Docker and cloud runs

The environment needs only a CPU, so a small Linux VM runs it. These notes
come from the Docker image in `Dockerfile` and from Mu-Yi Shen's Azure
deployment of the earlier Xvfb-based stack on 2026-10-20 (branch
`feature/python3-gymnasium-docker`, commit `05a3ac8`, file
`AZURE_DEPLOYMENT.md`). The cost and network rules carry over unchanged.

## Docker image

```bash
docker build -t mk64-env .                                   # test extras only
docker build -t mk64-env --build-arg EXTRAS=train,test .     # adds torch and the training scripts
mkdir -p roms outputs cache
cp /path/to/your/rom roms/marioKart.n64
docker run --rm -v ./roms:/app/roms:ro -v ./outputs:/app/outputs -v ./cache:/app/cache mk64-env
```

The default command runs `scripts/smoke_test.py` and writes screenshots and
`smoke_summary.json` to `outputs/smoke/`. Any other command works the same
way, for example `python train_sac.py --episodes 50`. Mount `cache/` so the
savestates survive container restarts.

The build downloads the pinned core with `scripts/fetch_core.sh`, so it
fails when the libretro nightly has moved on. `.dockerignore` excludes
`roms/`, and the build fails if a ROM reaches the image anyway. Never put the
ROM in the image, a registry, or Git.

There is no display server or VNC. To look at the game, copy
`outputs/smoke/*.png` back, or record an episode with
`python scripts/random_agent.py --video /app/outputs/run.mp4`.

## Sizing

Each environment instance uses one core and about 150 MiB with the renderer
set to one thread (see `docs/validation.md`). A 2-vCPU, 1 GiB VM runs two
instances. Training with SAC on a VM without a GPU is slow; the VM is for
environment checks and data collection.

## Azure

The 2026-10-20 deployment used Ubuntu Server 22.04 on a `Standard_B2ats_v2`
VM (2 burstable vCPUs, 1 GiB RAM, 64 GiB Premium SSD P6 disk) in North
Central US on a Free Trial subscription. It ran without an out-of-memory
kill. Check the current price, free allowance, and regional quota in the
Azure portal before creating anything; the CLI did not expose remaining
trial credit.

Cost rules from that deployment:

- Create a budget alert (the deployment used $5 per month at 50 and 100
  percent). It emails; it does not cap spending.
- A Standard static public IPv4 address is billed separately at about
  $0.005 per hour and is outside the free allowance.
- `az vm deallocate` stops compute billing. The disk and the public IP keep
  billing until the resource group is deleted.
- Delete the resource group when done and confirm with
  `az group exists --name <group>`, which must print `false`.

Network rules:

- The network security group allows inbound TCP 22 only from your current
  public IP. Nothing else is exposed. This image opens no ports.
- Transfer the ROM with `scp` to the VM, set its mode to 600, and keep it
  out of every image build context.

Commands that reproduce the tested layout. Replace the subscription, key
path, and client IP:

```bash
az account set --subscription <subscription-id>
client_ip=$(curl -s https://api.ipify.org)
az group create --name mk64-rg --location northcentralus
az network nsg create --resource-group mk64-rg --name mk64-nsg --location northcentralus
az network nsg rule create --resource-group mk64-rg --nsg-name mk64-nsg \
  --name AllowSSHFromCurrentIP --priority 1000 --direction Inbound --access Allow \
  --protocol Tcp --source-address-prefixes "$client_ip/32" --destination-port-ranges 22
az network public-ip create --resource-group mk64-rg --name mk64-ip \
  --location northcentralus --sku Standard --allocation-method Static
az vm create --resource-group mk64-rg --name mk64-vm --location northcentralus \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --size Standard_B2ats_v2 --admin-username azureuser --authentication-type ssh \
  --ssh-key-values ~/.ssh/id_ed25519.pub --nsg mk64-nsg \
  --public-ip-address mk64-ip --public-ip-sku Standard \
  --os-disk-size-gb 64 --storage-sku Premium_LRS
```

On the VM, install Docker Engine with the
[official Ubuntu instructions](https://docs.docker.com/engine/install/ubuntu/),
clone this repository, copy the ROM into `roms/` with mode 600, and run the
Docker commands above. Building the image on a 1 GiB VM works because the
image has no compiled components; the `train,test` build pulls torch, which
needs several GiB of disk.

Stop and clean up:

```bash
az vm deallocate --resource-group mk64-rg --name mk64-vm
az group delete --name mk64-rg --yes
az group exists --name mk64-rg
```
