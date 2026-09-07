#!/usr/bin/env bash
set -euo pipefail

readonly repo_url="${DCSS_REPOSITORY:-https://github.com/crawl/crawl.git}"
readonly ref="${DCSS_REF:-master}"
readonly checkout="${DCSS_CHECKOUT:-vendor/crawl}"

mkdir -p "$(dirname "$checkout")"
if [[ ! -d "$checkout/.git" ]]; then
  git clone --filter=blob:none --no-checkout "$repo_url" "$checkout"
fi

# DCSS uses `git describe` while compiling, so reachable release tags and their
# history are build inputs rather than optional repository metadata.
if [[ "$(git -C "$checkout" rev-parse --is-shallow-repository)" == "true" ]]; then
  git -C "$checkout" fetch --unshallow --tags origin
else
  git -C "$checkout" fetch --tags origin
fi
git -C "$checkout" fetch origin "$ref"
git -C "$checkout" checkout --detach FETCH_HEAD
git -C "$checkout" submodule update --init
git -C "$checkout" rev-parse HEAD
