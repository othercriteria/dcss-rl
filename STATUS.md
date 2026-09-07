# Project status

Last updated: 2026-09-07

## Current state

Public repository: <https://github.com/othercriteria/dcss-rl>

The Nix/Python environment and upstream trunk build are working. The current local
DCSS checkout was validated at commit `96832895d0253f9d7290d370efe32bf0614679a8`
(`0.35-a0-999-g96832895d0`) with a WebTiles build.

Canonical evaluation now uses `mibe-heldout-v2`. Heldout-v1 was retired after its
failure traces informed contextual stair-action masking. The v2 scripted-v3 incumbent
rank is `(0, 0, 2448 depth-progress, 2060 decisions, max-depth sum 17, XL sum 11,
76.0 reward)`; the previous v1 depth-11 hybrid remains an explicit learned milestone,
not a reusable heldout track.

Development evaluation now uses `mibe-diagnostic-v2`, extending the same known five
seeds from 200 to 500 decisions. Scripted-v3 calibrates rank
`(0, 0, 2380, 1194, 20, 16, 115.0)` on this track.

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
- `poe watch-heldout-champion` resolves the champion manifest and renders a deterministic,
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
- `configs/heldout-regression-v1.json` locks the scripted-v2 held-out rank-v4 vector
  `(0, 0, 0, 2500, 5, 10, 50.0)`; `poe evaluate-scripted` enforces it before champion
  promotion.
- PyTorch 2.14.0+cu130 sees the RTX 4090 and completes CUDA tensor operations without
  a flake change. A versioned fixed-width semantic actor/value model, ECHO
  next-feature-delta head, legality-masked class-balanced imitation trainer, and
  scripted-relabel DAgger path now produce restorable checkpoints.
- Pure learned diagnostic iterations established the first behavioral/throughput
  evidence: naive cloning collapsed to autoexplore; class balancing diversified
  actions but hit walls; DAgger plus global player-visible navigation features reached
  depth sum 6 but still underperformed the scripted floor on XL. Five concurrent
  200-decision learned episodes take about two minutes; DCSS internal command work,
  not GPU training, is currently the dominant wall-time cost.
- Scripted-v3 fixes sparse monster handling, ignores positively identified stationary
  flora, and permits BFS to enter stair feature cells. Its diagnostic rank is
  Its depth-progress area is 1,019 over 851 decisions, providing a materially stronger
  multi-depth teacher.
- Canonical held-out and diagnostic leaders now have separate monotonic promotion
  tracks. `poe watch-heldout-champion` remains held-out-only;
  `poe watch-diagnostic-leader` exposes the current diagnostic leader without weakening
  champion hygiene. Their corresponding `*-grid` tasks animate all manifest seeds in
  parallel and mark death, horizon truncation, and ascension explicitly.
- The confidence-gated DAgger-v4 agent matches the depth-20 diagnostic expert while
  making 23.7% of diagnostic decisions neurally. It cleared the locked held-out floor
  and became canonical with maximum-depth sum 11 versus scripted-v2's 5. Its
  depth-progress area is 213 over 1,240 decisions, while the D:1-only baseline has
  zero over 2,500. Held-out learned-action coverage was 2.4%, an explicit
  limitation and the next optimization target. The canonical trajectory records
  checkpoint SHA-256 `e8e1f37e4fa28517a38a01c42eb79c0ba034a016366d39c9c9eb22a0f6b5975b`;
  its first manifest episode reaches D:2 at step 34.
- `configs/curriculum-v1.json` freezes the non-held-out expert bootstrap, four DAgger
  rounds, 0.98 diagnostic confidence gate, and locked held-out promotion stages.

Evaluation worker scaling on five 20-step diagnostic cases (2026-09-07):

| workers | wall time | speedup |
| ---: | ---: | ---: |
| 1 | 39.2 s | 1.00x |
| 2 | 22.5 s | 1.74x |
| 5 | 10.1 s | 3.90x |

Learned hybrid scaling on five 50-step diagnostic cases (250 decisions total,
identical deterministic rank and 25.2% neural coverage, 2026-09-07):

| workers | wall time | decisions/s | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 84.50 s | 2.96 | 1.00x |
| 2 | 45.07 s | 5.55 | 1.88x |
| 5 | 20.61 s | 12.13 | 4.10x |

The near-linear process scaling and sub-30-second GPU training iterations show that
broader rollout suites and longer horizons are affordable next steps. GPU capacity is
not the current bottleneck.

- Online legality-masked PPO now supports concurrent seeded rollouts, GAE, clipped
  policy/value losses, optional scripted imitation and ECHO auxiliaries, portable
  checkpoints, and per-update throughput/loss telemetry. A matched 2,560-decision
  ECHO-on/off pair was behaviorally identical; a 10,240-decision, 20-seed strong-ECHO
  run also held the pure diagnostic frontier at depth sum 6 / XL sum 8. These are
  working negative results: the next gain needs a denser learning signal or collector
  change, not merely more synchronous updates.
- Synchronous online rollout scaling at 5/10/20 workers was 4.53/5.63/7.29 decisions/s.
  Independent per-worker chunks remove the per-action straggler barrier: the matched
  20×64 workload reaches 44.66 decisions/s (6.13× faster), and an exact repeat produced
  bit-identical model tensors despite thread scheduling.
- Training environments can add typed, player-visible potential rewards for newly
  explored cells, branch-local depth, fractional XP progress, and HP preservation.
  Defaults remain zero, so headline evaluation rewards and old behavior are unchanged;
  selected weights are stored in PPO checkpoint metadata. Online scripted imitation
  is class-balanced so rare stair and tactical labels survive common explore actions.
- PPO checkpoints are atomically replaced after every completed update, workers rotate
  through training seeds rather than returning to their initial seed, and transient
  DCSS startup timeouts receive three fresh-directory attempts. These were added after
  a 20-seed experiment lost seven otherwise healthy updates at an episode restart.
- Champion rank v4 uses decision-weighted depth progress and bounded policy survival
  after wins and runes, with maximum depth and monotonic XL as later frontier measures.
  Raw game turns were removed because rest/travel can inflate them behind one action;
  XL area was removed because it rewards risky front-loading rather than more XP.
- DCSS can abort when an overlong run directory produces a Unix socket path beyond
  Linux's 107-byte payload limit. Managed games now reject such paths before launch
  with a clear instruction to shorten the output root.
- Evaluation reports each case immediately on completion while retaining manifest
  ordering for summaries and ranks. Champion-track activation validates suite/rank
  versions, archives the prior canonical manifest, and atomically cuts all heldout
  viewers to the selected replacement; `watch-heldout-grid` now resolves heldout-v2.
- Automatic commands now prefer upstream's flushed `input_mode=1` boundary and retain
  the typed 500 ms quiescence only as a compatibility fallback. Delay-free documented
  DCSS RC options remove presentation sleeps from local agent play. The identical
  five-seed, 2,500-decision learned evaluation improved from 311.11 seconds (8.04/s)
  to 16.18 seconds (154.48/s), a 19.23× speed-up with exactly the same rank.
- Post-change asynchronous PPO collection scales from 36.93 decisions/s at five
  workers to 63.94 at ten and 97.93 at twenty, then declines to 93.28 at forty.
  Twenty workers remain the measured knee and are 2.19× faster than the prior matched
  collector. `online-train-v2` expands training from 20 to 64 disjoint seeds and its
  per-episode horizon from 500 to 1,000 decisions. Episode rotation strides by worker
  count, so concurrent workers consume disjoint seed blocks before wrapping rather
  than shifting into nearly complete overlap.
- A matched 40,960-decision broad-seed v6 ablation did not materially improve the
  fallback-free frontier. On diagnostic-v2, v5-off, v6-on, and v6-off reached
  depth-progress 1,385, 1,351, and 1,391 respectively; all survived 2,500 decisions,
  but three of five seeds remained on D:1. ECHO-off was cheaper and slightly stronger,
  but its +0.4% over v5 is noise rather than evidence for more of the same recipe.
- Structured WebTiles `menu` prompts now retain their visible title, item labels, and
  accepted hotkeys alongside widget-style menus. This removed a 934-action cancel
  loop and reduced v6-off diagnostic-v2 replay from 125.12 to 21.00 seconds without
  changing rank. The revealed failure is a repeated rest/confirm cycle; PPO telemetry
  now reports teacher agreement and imitation loss explicitly for targeted tuning.
- A 10,240-decision imitation-weight probe raised online teacher agreement from 42.2%
  at weight 1 to 47.6% at weight 10, but diagnostic-v2 depth-progress regressed from
  1,383 to 920. Agreement is useful optimizer telemetry, not a quality proxy; the
  stronger arm needs a full-budget test before changing objectives again.
- Full-budget strong imitation improved diagnostic-v2 depth-progress to 1,795, but its
  one locked heldout-v2 evaluation remained D:1 on every seed and failed at
  `(0, 0, 0, 2500, 5, 6, 10.0)`. Replay showed the policy choosing rest 769 times where
  its teacher requested rest only 14 times. PPO now exposes a typed teacher-balance
  exponent: 0 is unweighted, 0.5 (the default) is square-root inverse, and 1 restores
  the old full-inverse rule. The short square-root probe reduced rest/prompt collapse
  but shifted some mass to movement oscillation; its full-budget result is pending.
- Full-budget balance exponents 0.5 and 0.75 scored diagnostic-v2 depth-progress 1,263
  and 1,377. Square-root reached aggregate max-depth 13 but died early; the midpoint
  returned to passive survival. Neither beats full-inverse v8's 1,795, and v8 failed
  heldout. The next ablation separates online imitation from PPO/value gradients.
- A controlled current-build check measured 99.17 decisions/s at 20 workers and
  102.58/104.58 at 24. The 3.4–5.5% gain is modest but repeatable, so new training runs
  use 24 workers while 40 remains demonstrably beyond the scaling knee.

Next:

1. Increase autonomous learned-action coverage beyond 23.7% diagnostic / 2.4%
   held-out while retaining or improving the depth-11 held-out champion rank.
2. Remeasure PPO collection after the input-boundary/runtime speed-up, then broaden
   training seed sets and episode horizons under matched ECHO-on/off budgets.
3. Improve survival/XL after depth progress, then expand toward rune curricula.

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

The host has an RTX 4090 with 24,564 MiB VRAM and driver 610.57.04. PyTorch CUDA
training and inference have been exercised successfully. The managed Codex
sandbox may hide `/dev/nvidia*` and prohibit Unix socket binding; approved host-boundary
commands see both correctly. Training dependencies remain an optional `uv` group.
