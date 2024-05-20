# DRL Final Project - MarioKart64

## Setup

The easiest, cleanest, most consistent way to get up and running with this project is via [`Docker`](https://docs.docker.com/). These instructions will focus on that approach.

### Running with docker-compose

**Pre-requisites:**
- Docker & docker-compose
- Ensure you have a copy of the ROMs you wish to use, and make sure it is placed inside the path under `gym_mupen64plus/ROMs` 

**Steps:**

0. Clone the repository and get into the root folder of the project.

1. Run the following command to build the project via `docker`.

    ```
    docker build -t gym_mupen64plus .
    ```

2. Under the root of the repository, there is a python3 file `our_client.py`. This file should be the entry point of our RL training. We can first create a virtual environment of our project by:

    ```
    python -m venv RL_env
    ```

    Activate the environment:
    ```
    source RL_env/bin/activate
    ```

    Create a tmux channel and run the script:

    ```
    python our_client.py
    ```

3. Run the docker compose of the game environment using

    ```
    docker-compose up --build -d
    ```

    This will start the following 4 containers:
    - `xvfbsrv` runs XVFB
    - `vncsrv` runs a VNC server connected to the Xvfb container
    - `agent` runs the example python script
    - `emulator` runs the mupen64plus emulator

4. Then you can use your favorite VNC client (e.g., [VNC Viewer](https://www.realvnc.com/en/connect/download/viewer/)) to connect to `localhost` to watch the XVFB display in real-time. Note that running the VNC server and client can cause some performance overhead.

    For VSCode & TightVNC Users
    - Forward the port 5900 to the desired port on the local host
    - Open tightVNC and connect to `localhost::desired_port_num`, e.g. `localhost::5901`

5. To turn off the docker compose container, use the following command:

    ```
    docker-compose down
    ```

**Additional Notes:**

1. To view the status (output log) of a single compose, you can use the following command:

    ```
    docker-compose logs xvfbsrv
    docker-compose logs vncsrv
    docker-compose logs emulator
    docker-compose logs agent
    ```
