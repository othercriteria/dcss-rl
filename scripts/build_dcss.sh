#!/usr/bin/env bash
set -euo pipefail

readonly checkout="${DCSS_CHECKOUT:-vendor/crawl}"
readonly jobs="${DCSS_BUILD_JOBS:-$(nproc)}"
readonly source_dir="$checkout/crawl-ref/source"
readonly python="${DCSS_BUILD_PYTHON:-$PWD/.venv/bin/python}"

if [[ ! -d "$source_dir" ]]; then
  echo "DCSS source not found; run 'poe dcss-fetch' first." >&2
  exit 1
fi
if [[ ! -x "$python" ]]; then
  echo "Project Python not found; run 'uv sync' first." >&2
  exit 1
fi

# WEBTILES provides the structured local transport. BUILD_ALL avoids lazy data
# generation races when many rollout workers start together.
# FORCE_CC/FORCE_CXX avoid DCSS selecting Nix's unwrapped target-prefixed
# compiler, which does not carry the wrapper's libc search paths.
make -C "$source_dir" -j"$jobs" \
  WEBTILES=y BUILD_ALL=y \
  PYTHON="$python" \
  FORCE_CC="$(command -v gcc)" \
  FORCE_CXX="$(command -v g++)"
