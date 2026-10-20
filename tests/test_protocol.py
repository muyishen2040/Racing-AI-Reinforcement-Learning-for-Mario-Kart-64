"""Protocol round-trip and robustness tests; no emulator necessary."""
import socket
import threading

import numpy as np
import pytest

from protocol import decode_observation, encode_observation, receive_message, send_message


def test_framed_messages_support_fragmentation():
    a, b = socket.socketpair()
    try:
        value = {"command": "step", "action": [0, 0, 1, 0, 0], "text": "✓"}
        def writer():
            send_message(a, value)
        t = threading.Thread(target=writer)
        t.start()
        assert receive_message(b) == value
        t.join()
    finally:
        a.close()
        b.close()


def test_image_round_trip():
    source = np.random.default_rng(1).integers(0, 256, (48, 64, 3), dtype=np.uint8)
    restored = decode_observation(encode_observation(source))
    np.testing.assert_array_equal(source, restored)


def test_invalid_observation_rejected():
    with pytest.raises(ValueError):
        decode_observation({"shape": [1, 2, 4], "image": ""})
