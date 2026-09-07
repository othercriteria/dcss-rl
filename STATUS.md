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
- Fast and live checks are split: 46 unit tests complete in about 0.23 seconds, while
  three isolated DCSS integration tests run concurrently in about 6.7 seconds.
- Deterministic scripted MiBe policy, fixed diagnostic/held-out suite manifests,
  concurrent evaluation runner, metric-vector ranking, and champion manifest writer.
  The first diagnostic run exposed and fixed automatic-input and level-up prompt
  boundaries.
- Frozen scripted MiBe v2 completed the untouched five-seed held-out suite: zero wins,
  zero runes, depth sum 5, XL sum 10, 3,516 game turns, and reward 50. All five cases
  reached the 500-decision truncation on D:1, making descent the clearest policy
  regression target. The replay set occupies 164 MiB, confirming compact semantic
  deltas are also a near-term throughput requirement.
- Compact trajectory schema v2 replaces repeated full snapshots with exactly
  reconstructible semantic patches while preserving raw WebTiles messages and distinct
  policy/environment loss segments. On the matched five-case 20-step workload,
  trajectory JSONL fell from 7.81 MiB to 1.48 MiB (81%).
- `poe watch-best` resolves the champion manifest and renders a deterministic,
  player-centered terminal replay. It supports both the existing schema-v1 champion
  and new schema-v2 trajectories and defaults to manifest order, not a cherry-picked
  episode.
- Ordinary silent commands now use upstream's full-state request as an input-loop
  synchronization probe. This fixed 0.33.1 WAIT deadlock without modifying DCSS;
  automatic travel/rest retain their isolated typed quiescence fallback because they
  can consume control messages before completing.
- `poe compatibility-smoke` exercises semantic MiBe reset, a structured action,
  schema-v2 recording, and exact replay reconstruction against any binary. It passes
  on trunk plus 0.34.1 (`1eebc1a2892e1c89776a0d7a10691f8dac8d9796`) and 0.33.1
  (`9cb173b281c11a5177f40b8c0662bacd3aac2717`).
- `configs/heldout-regression-v1.json` locks the scripted-v2 held-out rank
  `(0, 0, 5, 10, 3516, 50.0)`; `poe evaluate-scripted` enforces it before champion
  promotion.

Evaluation worker scaling on five 20-step diagnostic cases (2026-09-07):

| workers | wall time | speedup |
| ---: | ---: | ---: |
| 1 | 39.2 s | 1.00x |
| 2 | 22.5 s | 1.74x |
| 5 | 10.1 s | 3.90x |

Next:

1. Improve the scripted baseline's descent beyond the locked D:1 floor.
2. GPU-smoke the optional training stack and implement the first learned baseline.
3. Recharacterize rollout scaling on a representative training workload once one
   exists; the current short-episode sweep remains the environment baseline.

## Verified commands

```sh
poe dcss-fetch
poe dcss-build
poe dcss-smoke
poe compatibility-smoke
poe check
```

The installed hooks and both hook stages pass, including the live DCSS integration.

## Environment notes

The host has an RTX 4090 with 24,564 MiB VRAM and driver 610.57.04. The managed Codex
sandbox may hide `/dev/nvidia*` and prohibit Unix socket binding; approved host-boundary
commands see both correctly. Training dependencies are an optional `uv` group and have
not yet been GPU-smoke-tested.
