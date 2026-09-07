# Agent instructions

## Objective

Deliver and validate a reproducible DCSS RL MVP on local, unmodified trunk: typed
WebTiles boundaries, compact ECHO-ready trajectories, deterministic diagnostic and
held-out evaluation, a scripted champion baseline, robust policy-input readiness,
and `watch-best` live/replay observability. Keep the public repository tested and
documented, retain fast development gates and semantic domain types, characterize
parallel rollout scaling, and establish compatibility with recent stable releases.

The MVP finish line is evidence-backed rather than feature-count based:

- A policy can run reproducibly from a fixed suite manifest and produce replayable
  trajectories whose policy-action and next-environment targets remain distinct.
- Champion selection uses only a fixed held-out suite, has explicit regression
  thresholds, and can be observed without choosing a favorable episode by hand.
- Policy-input readiness either uses a reliable upstream signal or has a documented,
  adversarially tested fallback covering automatic multi-turn actions, prompts,
  menus, and blocking CRT states.
- Unit tests stay subsecond at current scale, the bounded full gate stays practical,
  and rollout-worker scaling is measured at representative workloads rather than
  assumed from a single worker count.
- The same integration/trajectory smoke path passes on canonical trunk and at least
  the two most recent selected stable releases, with exact revisions recorded.
- The baseline provides a stable floor and a useful incremental learning signal even
  before rune acquisition or ascension; depth progression is the current first
  behavioral regression target.

Work autonomously on routine architecture, implementation, dependency, experiment,
commit, and publication decisions. Ask only about material scope changes, credentials,
costs, or destructive actions.

## Invariants

- Use unmodified local DCSS. Trunk is canonical; recent stable releases are
  compatibility targets.
- WebTiles is the primary structured transport. Preserve console/TTY as a future
  black-box validation path.
- Training may use fixed seeds and diagnostic/wizard curricula. Headline evaluation
  must use normal, unseeded play without privileged observations.
- Start with Minotaur Berserker and a hand axe. Expand characters only after the MVP
  evaluation loop is sound.
- Treat an action mask as syntactic/UI legality, never as tactical safety.
- Preserve raw player-visible protocol data in trajectories. Derive stable semantic
  observations separately so future reducers can replay old data.
- ECHO is an objective family, not a dependency choice: retain distinct token masks
  for policy actions and next-environment observations.
- Prefer an explicit upstream readiness signal over timing heuristics. Until one is
  found or added without modifying DCSS, isolate and type any quiescence durations,
  test multi-turn commands and blocking CRT/prompt states, and preserve raw evidence.
- Keep the fast unit loop subsecond at current scale. Parallelize isolated live tests
  and evaluation cases; remeasure worker scaling after material workload changes.
- Select the champion only on a fixed held-out suite. Never hand-pick showcase seeds.
- Generated DCSS source/builds, runs, artifacts, and checkpoints remain untracked.
- Prefer semantic types to raw containers. Use dataclasses/enums for domain objects,
  `TypedDict` at required JSON/Gym boundaries, and named JSON aliases only at protocol
  decoding seams. Do not let `Any` or anonymous nested `dict`/`list` types propagate
  through application code.
- Give scalar values semantic types as well: use `NewType` for identifiers, units,
  indices, seeds, and other values that must not be accidentally interchanged; use
  named aliases where distinct static identity would not add safety. Raw primitives
  belong at serialization and third-party API boundaries.

## Workflow

```sh
direnv allow
uv sync
poe hooks
poe test
poe check
```

Use `apply_patch` for edits. Install hooks once per checkout. `poe test` and commit
hooks run the fast unit suite; the push hook runs the full gate, parallelizing isolated
live DCSS processes.
Unix socket tests and NVIDIA access can fail inside the managed Codex sandbox; run
those commands at the approved host boundary. This is an execution-environment
constraint, not evidence of a host failure.

Read `STATUS.md` before starting substantial work and keep it current when milestones,
architecture, commands, or blockers change. Detailed decisions live in
`docs/architecture.md`.
