"""Exercise the HTTP interface used by the native input bot, without a ROM."""
import json
import threading
from urllib.request import urlopen

import pytest

from gym_mupen64plus.envs.mupen64plus_env import ControllerHTTPServer, ControllerState


def start_server(timeout=1):
    server = ControllerHTTPServer(('127.0.0.1', 0), timeout, 1)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_input_bot_poll_receives_action():
    server, thread = start_server()
    try:
        sender = threading.Thread(target=server.send_controls,
                                  args=(ControllerState([0, 0, 1, 0, 0] + [0] * 11),))
        sender.start()
        with urlopen('http://127.0.0.1:%d/0' % server.server_port, timeout=2) as response:
            controls = json.load(response)
        sender.join(timeout=2)
        assert not sender.is_alive()
        assert controls['A_BUTTON'] == 1
        assert controls['X_AXIS'] == 0
    finally:
        server.shutdown()
        thread.join(timeout=2)


def test_missing_input_bot_poll_times_out():
    server, thread = start_server(timeout=0.01)
    try:
        with pytest.raises(TimeoutError, match='Input plugin did not poll'):
            server.send_controls(ControllerState())
    finally:
        server.shutdown()
        thread.join(timeout=2)
