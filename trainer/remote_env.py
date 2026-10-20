"""Gymnasium-compatible client running in the trainer container."""
import os
import socket

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from protocol import decode_observation, receive_message, send_message


class RemoteMarioKartEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, host=None, port=None, track="Luigi-Raceway", timeout=300):
        super().__init__()
        self.host = host or os.getenv("SIMULATOR_HOST", "simulator")
        self.port = int(port or os.getenv("SIMULATOR_PORT", "5555"))
        self.track = track
        self.socket = socket.create_connection((self.host, self.port), timeout=timeout)
        self.socket.settimeout(timeout)
        self.observation_space = spaces.Box(0, 255, (480, 640, 3), np.uint8)
        self.action_space = spaces.Box(
            low=np.array([-80, -80, 0, 0, 0], dtype=np.float32),
            high=np.array([80, 80, 1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )
        self.last_observation = None

    def _request(self, request):
        send_message(self.socket, request)
        response = receive_message(self.socket)
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "Simulator RPC failed"))
        return response

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        track = (options or {}).get("track", self.track)
        response = self._request({"command": "reset", "track": track})
        self.last_observation = decode_observation(response["observation"])
        return self.last_observation, response.get("info", {})

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (5,) or not np.all(np.isfinite(a)):
            raise ValueError("Expected five finite controller values")
        a = np.clip(a, self.action_space.low, self.action_space.high)
        response = self._request({"command": "step", "action": a.tolist()})
        self.last_observation = decode_observation(response["observation"])
        return (
            self.last_observation, float(response["reward"]),
            bool(response["terminated"]), bool(response["truncated"]),
            response.get("info", {}),
        )

    def render(self):
        return self.last_observation

    def close(self):
        if getattr(self, "socket", None):
            try:
                self._request({"command": "close"})
            except (ConnectionError, OSError, RuntimeError):
                pass
            self.socket.close()
            self.socket = None
