"""Single-client simulator RPC service.

The emulator/Xvfb are launched by the container entrypoint, before this process.
"""
import logging
import os
import socket

import gymnasium as gym
import gym_mupen64plus

from protocol import encode_observation, receive_message, send_message

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
TRACKS = {
    "Luigi-Raceway", "Moo-Moo-Farm", "Koopa-Troopa-Beach", "Kalimari-Desert",
    "Toads-Turnpike", "Frappe-Snowland", "Choco-Mountain", "Mario-Raceway",
    "Wario-Stadium", "Sherbet-Land", "Royal-Raceway", "Bowsers-Castle",
    "DKs-Jungle-Parkway", "Yoshi-Valley", "Banshee-Boardwalk", "Rainbow-Road",
}


class SimulatorServer:
    def __init__(self):
        self.env = None
        self.track = None

    def close_environment(self):
        env, self.env = self.env, None
        self.track = None
        if env is not None:
            env.close()

    def reset(self, track):
        if track not in TRACKS:
            raise ValueError("Unknown track: " + str(track))
        if self.env is None or track != self.track:
            self.close_environment()
            self.env = gym.make("Mario-Kart-" + track + "-v0")
            self.track = track
        observation, info = self.env.reset()
        return {"ok": True, "observation": encode_observation(observation), "info": info}

    def step(self, action):
        if self.env is None:
            raise ValueError("Call reset before step")
        observation, reward, terminated, truncated, info = self.env.step(action)
        return {
            "ok": True, "observation": encode_observation(observation),
            "reward": reward, "terminated": terminated,
            "truncated": truncated, "info": info,
        }

    def serve(self, host="0.0.0.0", port=5555):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
            listener.listen(1)
            LOG.info("Simulator listening on %s:%s", host, port)
            while True:
                client, address = listener.accept()
                LOG.info("Client connected: %s", address)
                with client:
                    try:
                        while True:
                            request = receive_message(client)
                            cmd = request.get("command")
                            if cmd == "close":
                                send_message(client, {"ok": True})
                                break
                            try:
                                if cmd == "reset":
                                    answer = self.reset(request.get("track", "Luigi-Raceway"))
                                elif cmd == "step":
                                    answer = self.step(request["action"])
                                elif cmd == "ping":
                                    answer = {"ok": True}
                                else:
                                    raise ValueError("Unknown command")
                            except Exception as error:
                                LOG.exception("Simulator command failed")
                                if cmd in ("reset", "step"):
                                    try:
                                        self.close_environment()
                                    except Exception:
                                        LOG.exception("Could not close failed environment")
                                answer = {"ok": False, "error": str(error)}
                            send_message(client, answer)
                    except (ConnectionError, OSError, ValueError) as error:
                        LOG.warning("Client disconnected: %s", error)
                    finally:
                        self.close_environment()


if __name__ == "__main__":
    SimulatorServer().serve(port=int(os.getenv("SIMULATOR_PORT", "5555")))
