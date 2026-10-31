#!/usr/bin/env bash
# Download the Mupen64Plus-Next libretro core into cores/ and check its hash.
#
# The libretro buildbot only serves the latest nightly, so the pin is the
# SHA-256 of the validated build. A download with a different hash is
# refused unless MK64_ACCEPT_NEW_CORE=1, which installs it and prints the
# new hash for this file. docs/validation.md records what the pinned
# build was tested against. Keep a copy of the validated .so outside the
# buildbot if the pin must stay reproducible after upstream rebuilds.
set -euo pipefail

PINNED_SHA256="26f2b4ec722558d77bc9e1582e62ab80c84190bb20dfadc48a148e6210c8775e"
PINNED_BUILD="nightly 2026-10-27 08:22 UTC"
URL="https://buildbot.libretro.com/nightly/linux/x86_64/latest/mupen64plus_next_libretro.so.zip"

cd "$(dirname "$0")/.."
target="cores/mupen64plus_next_libretro.so"

hash_of() { sha256sum "$1" | cut -d' ' -f1; }

if [[ -f "$target" && "$(hash_of "$target")" == "$PINNED_SHA256" ]]; then
    echo "$target already matches the pinned build ($PINNED_BUILD)"
    exit 0
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
curl -sSL -o "$tmp/core.zip" "$URL"
unzip -q "$tmp/core.zip" -d "$tmp"
actual="$(hash_of "$tmp/mupen64plus_next_libretro.so")"

if [[ "$actual" != "$PINNED_SHA256" ]]; then
    echo "Downloaded core hash $actual" >&2
    echo "Pinned core hash     $PINNED_SHA256 ($PINNED_BUILD)" >&2
    if [[ "${MK64_ACCEPT_NEW_CORE:-0}" != "1" ]]; then
        echo "The buildbot has moved on. Rerun with MK64_ACCEPT_NEW_CORE=1 to install" >&2
        echo "the new build, then rerun pytest and update PINNED_SHA256 in $0." >&2
        exit 1
    fi
    echo "MK64_ACCEPT_NEW_CORE=1: installing the new build" >&2
fi

mkdir -p cores
mv "$tmp/mupen64plus_next_libretro.so" "$target"
echo "installed $target (sha256 $actual)"
