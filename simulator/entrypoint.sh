#!/usr/bin/env bash
set -euo pipefail

ROM=/app/roms/marioKart.n64
INPUT=/usr/local/lib/mupen64plus/mupen64plus-input-bot.so
if [[ ! -f "$ROM" ]]; then
  echo "Missing private ROM at $ROM; mount ./roms:/app/roms:ro" >&2
  exit 1
fi
if [[ ! -f "$INPUT" ]]; then
  echo "Missing input plugin: $INPUT" >&2
  exit 1
fi

Xvfb :99 -screen 0 640x480x24 -ac -nolisten tcp >/tmp/xvfb.log 2>&1 &
XVFB_PID=$!
trap 'kill "$XVFB_PID" "${VNC_PID:-}" 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
  if DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then break; fi
  sleep 0.25
done
DISPLAY=:99 xdpyinfo >/dev/null
# VNC listens inside the container; Docker only publishes it to the host loopback
# to Docker's host-loopback published port, never directly over the public internet.
x11vnc -display :99 -forever -shared -viewonly -nopw -rfbport 5900 -quiet >/tmp/vnc.log 2>&1 &
VNC_PID=$!

# The environment starts the emulator after its controller HTTP server is ready.
# It also restarts the emulator when a client reconnects or changes tracks.
python3 /app/simulator/server.py
