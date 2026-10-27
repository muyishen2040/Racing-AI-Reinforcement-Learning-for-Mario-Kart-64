#!/usr/bin/env bash
# Download the Mupen64Plus-Next libretro core nightly build into cores/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p cores
url="https://buildbot.libretro.com/nightly/linux/x86_64/latest/mupen64plus_next_libretro.so.zip"
curl -sSL -o cores/core.zip "$url"
unzip -o -q cores/core.zip -d cores
rm cores/core.zip
ls -la cores/mupen64plus_next_libretro.so
