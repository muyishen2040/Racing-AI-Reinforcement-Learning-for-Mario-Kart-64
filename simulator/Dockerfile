FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    DISPLAY=:99 \
    MARIOKART_MANAGED_DISPLAY=1 \
    MARIOKART_ROM_PATH=/app/roms/marioKart.n64 \
    LIBGL_ALWAYS_SOFTWARE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git gcc make pkg-config \
    libmupen64plus-dev libjson-c-dev mupen64plus-ui-console \
    mupen64plus-video-glide64 mupen64plus-audio-sdl mupen64plus-rsp-hle \
    xvfb x11vnc x11-utils libgl1-mesa-dri ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Build the original controller plugin against the system Mupen64Plus API.
# Its runtime compatibility with the packaged emulator still requires testing.
RUN git clone https://github.com/kevinhughes27/mupen64plus-input-bot.git /tmp/input-bot \
    && cd /tmp/input-bot && git checkout 0a1432035e2884576671ef9777a2047dc6c717a2 \
    && make all && make install \
    && rm -rf /tmp/input-bot

RUN python3 -m pip install --no-cache-dir \
    "gymnasium==0.29.1" "numpy<2" "mss==9.0.1" \
    "PyYAML>=6,<7" "termcolor>=2,<3"

WORKDIR /app
COPY gym_mupen64plus /app/gym_mupen64plus
COPY simulator /app/simulator
# Fail the build if the legacy tracked ROM ever slips past .dockerignore.
RUN test ! -e /app/gym_mupen64plus/ROMs/marioKart.n64
COPY simulator/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
EXPOSE 5555 5900
ENTRYPOINT ["/entrypoint.sh"]
