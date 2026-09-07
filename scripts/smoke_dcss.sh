#!/usr/bin/env bash
set -euo pipefail

readonly binary="${DCSS_BINARY:-vendor/crawl/crawl-ref/source/crawl}"

if [[ ! -x "$binary" ]]; then
  echo "DCSS binary not found; run 'poe dcss-build' first." >&2
  exit 1
fi

"$binary" -version
