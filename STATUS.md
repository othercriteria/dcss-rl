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
- Fixed Gym action catalog with reversible structured-action encoding and per-state
  boolean legality masks.
- Gymnasium environment that starts a real isolated game, completes MiBe hand-axe
  creation, steps structured actions, emits semantic dictionaries, and reports basic
  depth/XL/terminal rewards. Its live trunk test passes.
- Append-only, replayable JSONL trajectories containing revision/character metadata,
  exact emitted keycodes, wire-ordered raw protocol messages, semantic observations,
  and distinct ECHO-compatible policy/environment loss segments.
- A pinned `ty` type-check gate and a project-wide semantic-type discipline, with
  domain dataclasses/enums and TypedDict schemas at Gym/JSON boundaries. The checked
  source and tests carry no inline type-checker suppressions.
- Fast and live checks are split: 39 unit tests complete in about 0.24 seconds, while
  three isolated DCSS integration tests run concurrently in about 6.2 seconds.
- Deterministic scripted MiBe policy, fixed diagnostic/held-out suite manifests,
  concurrent evaluation runner, metric-vector ranking, and champion manifest writer.
  The first diagnostic run exposed and fixed automatic-input and level-up prompt
  boundaries.
- Frozen scripted MiBe v2 completed the untouched five-seed held-out suite: zero wins,
  zero runes, depth sum 5, XL sum 10, 3,516 game turns, and reward 50. All five cases
  reached the 500-decision truncation on D:1, making descent the clearest policy
  regression target. The replay set occupies 164 MiB, confirming compact semantic
  deltas are also a near-term throughput requirement.

Evaluation worker scaling on five 20-step diagnostic cases (2026-09-07):

| workers | wall time | speedup |
| ---: | ---: | ---: |
| 1 | 39.2 s | 1.00x |
| 2 | 22.5 s | 1.74x |
| 5 | 10.1 s | 3.90x |

Next:

1. Add regression thresholds and compact trajectory state deltas.
2. Replace or more tightly validate quiescence-based policy-input readiness.
3. Add `watch-best` via native WebTiles live spectating and durable replay/morgues.
4. Test trunk plus at least releases 0.34.1 and 0.33.1.

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
