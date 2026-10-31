# Headless Mario Kart 64 environment. No display server, no GPU.
#
#   docker build -t mk64-env .
#   docker run --rm -v ./roms:/app/roms:ro -v ./outputs:/app/outputs mk64-env
#
# The ROM is never copied into the image; mount it at /app/roms/marioKart.n64.
# Build with --build-arg EXTRAS=train,test for torch and the training scripts.
FROM python:3.12-slim

ARG EXTRAS=test
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY mk64_env ./mk64_env
COPY scripts ./scripts
COPY tests ./tests
COPY src ./src
COPY train_sac.py train_bc.py test.py grad_cam.py GetExpertData.py ReplayExpertData.py ./
RUN python -m pip install --no-cache-dir -e ".[${EXTRAS}]"
RUN scripts/fetch_core.sh
RUN test ! -e /app/roms/marioKart.n64

ENV MK64_ROM=/app/roms/marioKart.n64 \
    MK64_CACHE=/app/cache
VOLUME ["/app/roms", "/app/outputs", "/app/cache"]
CMD ["python", "scripts/smoke_test.py", "--out", "/app/outputs/smoke"]
