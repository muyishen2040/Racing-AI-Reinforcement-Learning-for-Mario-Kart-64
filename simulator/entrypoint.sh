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
trap 'kill "$XVFB_PID" "${EMU_PID:-}" "${VNC_PID:-}" 2>/dev/null || true' EXIT

for i in $(seq 1 30); do
  if DISPLAY=:99 xdpyinfo >/dev/null 2>&1; then break; fi
  sleep 0.25
done
DISPLAY=:99 xdpyinfo >/dev/null
# VNC listens inside the container; Docker only publishes it to the host loopback
# to Docker's host-loopback published port, never directly over the public internet.
x11vnc -display :99 -forever -shared -viewonly -nopw -rfbport 5900 -quiet >/tmp/vnc.log 2>&1 &
VNC_PID=$!

# The input bot is configured to request commands from the Gymnasium controller HTTP server.
mupen64plus --nospeedlimit --nosaveoptions --resolution 640x480 \
  --gfx mupen64plus-video-glide64.so --audio dummy \
  --input "$INPUT" \
  --set 'Input-Bot-Control0[plugged]=1' \
  --set 'Input-Bot-Control0[host]=127.0.0.1' \
  --set 'Input-Bot-Control0[port]=8082' \
  "$ROM" >/tmp/emulator.log 2>&1 &
EMU_PID=$!

exec python3 /app/simulator/server.py
