# Agent instructions

## Objective

Train and promote a predominantly autonomous DCSS agent that matches or exceeds the
retired v1 depth-11 learned milestone and the current heldout-v2 scripted incumbent
while eliminating dependence on the scripted fallback. Scale training across broader seed sets and longer episode
horizons, add online masked actor-critic/PPO fine-tuning, compare matched ECHO-on and
ECHO-off runs, and characterize rollout throughput and policy quality as learned-action
coverage rises. Preserve locked held-out promotion, checkpointed replay, and
champion replay/grid observability; use survival, XL, depth, and eventually rune acquisition
as the curriculum frontier.

The learning milestone is evidence-backed rather than feature-count based:

- A fallback-free learned policy and its preprocessing can be restored from a
  versioned checkpoint and run reproducibly from a fixed suite manifest.
- Online training uses masked clipped PPO/value objectives; matched runs can enable or
  disable an ECHO auxiliary target derived from the next player-visible semantic state.
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
- The predominantly autonomous candidate matches or exceeds the retired v1
  depth-11 milestone and current heldout-v2 champion before promotion. Diagnostic improvements and a hybrid whose
  gains mainly come from its scripted fallback do not satisfy the milestone.

Work autonomously on routine architecture, implementation, dependency, experiment,
commit, and publication decisions. Ask only about material scope changes, credentials,
costs, or destructive actions.

Keep active goals thin and focused on the immediate research direction. Durable
constraints belong here, in architecture documentation, or in enforced tooling.
Prioritize current-policy progress before recurrent/LM comparisons. Start with bounded
local pilots and scale to hour-scale runs only when their evidence warrants it.
Use Poe targets wherever possible, adding discoverable targets for recurring workflows.
Use sub-agents to improve velocity on independently owned work; keep a performance
agent progressing through measured bottlenecks alongside policy research. The primary
agent owns integration, experiment interpretation, and all promotion-suite access.

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
`docs/architecture.md`. Keep `docs/experiments.md` as a selective timestamped record
of experiment hypothesis, wall-clock cost, outcome, and next decision.
