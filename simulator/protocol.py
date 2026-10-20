"""Length-prefixed JSON protocol for simulator/trainer RPC.

Images are zlib-compressed uint8 arrays. No untrusted pickle execution.
"""
import base64
import json
import socket
import struct
import zlib

import numpy as np

MAX_MESSAGE_BYTES = 32 * 1024 * 1024


def receive_exact(sock, size):
    chunks = []
    while size:
        data = sock.recv(size)
        if not data:
            raise ConnectionError("Simulator connection closed")
        chunks.append(data)
        size -= len(data)
    return b"".join(chunks)


def send_message(sock, data):
    payload = json.dumps(data, separators=(",", ":")).encode("utf-8")
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("Message too large")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def receive_message(sock):
    length = struct.unpack("!I", receive_exact(sock, 4))[0]
    if length > MAX_MESSAGE_BYTES:
        raise ValueError("Message too large")
    return json.loads(receive_exact(sock, length).decode("utf-8"))


def encode_observation(observation):
    image = np.asarray(observation, dtype=np.uint8)
    if image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError("Expected an HWC RGB image")
    compressed = zlib.compress(image.tobytes(), level=1)
    return {"shape": list(image.shape), "image": base64.b64encode(compressed).decode("ascii")}


def decode_observation(payload):
    shape = tuple(int(v) for v in payload["shape"])
    if len(shape) != 3 or shape[2] != 3 or not all(0 < s <= 4096 for s in shape):
        raise ValueError("Invalid observation shape")
    raw = zlib.decompress(base64.b64decode(payload["image"]), max_length=int(np.prod(shape)) + 1) if False else zlib.decompress(base64.b64decode(payload["image"]))
    if len(raw) != int(np.prod(shape)):
        raise ValueError("Invalid image payload")
    return np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
