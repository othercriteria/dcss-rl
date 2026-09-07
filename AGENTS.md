# Agent instructions

## Objective

Build and evaluate the first learning-based DCSS agent that materially exceeds the
scripted Minotaur Berserker baseline on survival and depth progression, using compact
ECHO-style trajectories and the local RTX 4090. Deliver a reproducible training
pipeline, curriculum and checkpoint format, locked diagnostic and held-out regression
suites, automatic champion promotion, and `watch-best` observation of learned play.
Resolve or sharply characterize policy-input readiness and measure rollout/training
throughput well enough to identify the next bottleneck.

The learning milestone is evidence-backed rather than feature-count based:

- A learned policy and its preprocessing can be restored from a versioned checkpoint
  and run reproducibly from a fixed suite manifest.
- Training uses policy/value objectives plus an ECHO-style auxiliary prediction target
  derived from the next player-visible semantic state; ablations can disable it.
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
- The learned candidate materially exceeds the locked scripted baseline on survival
  and depth progression before held-out champion promotion. Diagnostic improvements
  alone do not satisfy the milestone.

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
