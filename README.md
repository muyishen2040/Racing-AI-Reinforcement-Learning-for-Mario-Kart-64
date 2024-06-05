# Racing AI: Reinforcement Learning for Mario Kart 64

## Acknowledgements

This project is a fork of an original repository that provided the base environment for Mario Kart 64. The original repository can be found here: [Original Repository](https://github.com/bzier/gym-mupen64plus).

## Introduction

The environment in the original repository was based on Python 2, which restricts the use of modern reinforcement learning libraries such as the latest versions of PyTorch and TensorFlow. Directly updating the Python version in the Docker container led to execution issues. Therefore, this repository maintains the original Python 2 environment and uses a socket-based approach to interface with a Python 3 environment for executing reinforcement learning programs.

## Setup

The easiest, cleanest, most consistent way to get up and running with this project is via [`Docker`](https://docs.docker.com/). These instructions will focus on that approach.

### Running with docker-compose

**Pre-requisites:**
- Docker & docker-compose
- Ensure you have a copy of the ROMs you wish to use, and make sure it is placed inside the path under `gym_mupen64plus/ROMs`.

**Steps:**

0. Clone the repository and get into the root folder of the project.

1. Build the docker image with the following command:

    ```
    docker build -t bz/gym-mupen64plus:0.0.1 .
    ```

2. Please be noticed that in order to enable multiple instances of the environment, the original docker-compose file is separated into two parts - base file (docker-compose.yml) and override files (e.g. instance1.yml). The following command gives an example of instantiating an environment:

    ```bash
    docker-compose -p agent1 -f docker-compose.yml -f instance1.yml up --build -d
    ```

    This will start the following 4 containers:
    - `xvfbsrv` runs XVFB
    - `vncsrv` runs a VNC server connected to the Xvfb container
    - `agent` runs the example python script
    - `emulator` runs the mupen64plus emulator

    Note:
    - `-p` flag is the name of this environment instance
    - Before creating a new instance, be sure to create a override file to modify the port numbers (see `instance1.yml` for more details).
    - Make sure that the `docker-compose down` command given below also matches the file name of your instance and file names.

3. Under the root of the repository, there is a Python 3 file `SocketWrapper.py`. This file contains the wrapper for our RL training. We can first create a virtual environment for our project by:

    ```bash
    python -m venv RL_env
    ```

    Activate the environment:
    ```bash
    source RL_env/bin/activate
    ```

    Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

    In your training script:
    ```python
    from SocketWrapper import SocketWrapper
    env = SocketWrapper()
    ```

4. Then you can use your favorite VNC client (e.g., [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)) to connect to `localhost` to watch the XVFB display in real-time. Note that running the VNC server and client can cause some performance overhead.

    For VSCode & TightVNC Users:
    - Forward the port 5900 to the desired port on the local host.
    - Open TightVNC and connect to `localhost::desired_port_num`, e.g. `localhost::5901`.

5. To turn off the docker compose container (e.g. suppose we follow the naming criteria above `agent1` as the instance name and use `instance1.yml` for the override file), use the following command:

    ```bash
    docker-compose -p agent1 -f docker-compose.yml -f instance1.yml down
    ```

    Note:
    - To create another instance, you can create another tmux channel to run another with a different instance name and override file.

**Additional Notes:**

1. To view the status (output log) of a single compose, you can use the following command (suppose our instance name is `agent1`):

    ```bash
    docker-compose -p agent1 logs xvfbsrv
    docker-compose -p agent1 logs vncsrv
    docker-compose -p agent1 logs emulator
    docker-compose -p agent1 logs agent
    ```

## Features

- **SAC Training Script**: A script to train a Soft Actor-Critic (SAC) agent.
- **Grad-CAM Visualization**: Tools to visualize the learned features using Grad-CAM.

This repository enhances the Mario Kart 64 Gym Environment with modern reinforcement learning capabilities. Follow the setup instructions to get started with training and visualizing your own AI agents in Mario Kart 64.
