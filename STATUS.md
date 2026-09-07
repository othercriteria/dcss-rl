# Project status

Last updated: 2026-09-07

## Current state

Public repository: <https://github.com/othercriteria/dcss-rl>

The Nix/Python environment and upstream trunk build are working. The current local
DCSS checkout was validated at commit `96832895d0253f9d7290d370efe32bf0614679a8`
(`0.35-a0-999-g96832895d0`) with a WebTiles build.

Completed:

- Reproducible Nix flake, Python 3.13 `uv` project, Poe commands, and direnv entry.
- Fetch/build/smoke scripts for trunk or a specified `DCSS_REF`.
- Direct Unix-datagram WebTiles transport with fragmented-message reassembly.
- Flush-delimited observation batches separating player-visible and control messages.
- Isolated per-game process lifecycle with private saves, morgues, macros, and logs.
- Live integration test that launches trunk and reaches a real input boundary.
- Stateful semantic reducer for sparse player/map deltas, inventory, messages, menus,
  and UI input mode; tested on a real Minotaur Berserker game.
- Repository hooks: commit-time Ruff auto-fixes and fast tests, plus a pre-push full
  gate that exercises a live DCSS process.

In progress:

- Structured actions and fail-closed syntactic legality masks (`actions.py` exists but
  is not yet integrated with a Gym environment).

Next:

1. Connect structured actions to a Gym environment.
2. Define raw + semantic episode records and explicit ECHO action/observation masks.
3. Implement a deterministic scripted MiBe baseline and fixed train/eval seed policy.
4. Add aggregate evaluation, champion manifest, and regression thresholds.
5. Add `watch-best` via native WebTiles live spectating and durable replay/morgues.
6. Test trunk plus at least releases 0.34.1 and 0.33.1.

## Verified commands

```sh
poe dcss-fetch
poe dcss-build
poe dcss-smoke
poe check
```

The installed hooks and both hook stages pass, including the live DCSS integration.

## Environment notes

The host has an RTX 4090 with 24,564 MiB VRAM and driver 610.57.04. The managed Codex
sandbox may hide `/dev/nvidia*` and prohibit Unix socket binding; approved host-boundary
commands see both correctly. Training dependencies are an optional `uv` group and have
not yet been GPU-smoke-tested.
