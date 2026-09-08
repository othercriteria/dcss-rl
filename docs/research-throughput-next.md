# Next rollout-throughput intervention

2026-09-08. Read-only recommendation; no new games, benchmarks, or source changes.

## Recommendation

Prioritize **amortizing verified static-cache preparation across a 48-worker run**,
not GPU optimization, more workers, or shorter readiness waits. Keep caching opt-in.
First measure the existing cache under corrected readiness with per-reset preparation
timings; implement amortization only if those timings support it. The old result
does not establish that validation/copy contention caused its wall-time regression.

## Evidence and limits

- Three cold resets took 5.61–5.86 seconds, with 4.87–5.12 seconds of DCSS child CPU
  (`artifacts/p/startup/summary.json`). Static db/des generation is a large measured
  restart cost, not merely Python startup speculation.
- A verified 424-member cache loaded/validated in 0.156 seconds; a single cached reset
  including validation/copy took 0.980 seconds and matched the cold 20-action trace
  (`artifacts/p/static-cache/helper-s3001/summary.json`).
- The representative old 48×128 cache pair regressed collection wall from 39.02 to
  50.82 seconds, despite descendant CPU falling from 635.89 to 147.05 seconds.
  Optimization was 0.91 seconds in both arms; checkpoint writing was 0.01 seconds.
  Episodes differed 20 versus 22, and model updates diverged. These are operational
  measurements, not a matched-workload cache speedup
  (`artifacts/p/v66cache/report.json`).
- The corrected cold/cold control at `32bbea8` matches all 6,144 paired policy
  inputs, masks, probabilities, RNG states, actions, rewards, and reset boundaries.
  Its 140 stairs actions per arm never return the old bare busy flush. Twelve
  workers differ only in cell colour; strict/raw inequality remains visible
  (`artifacts/e/readiness68-policy-parity.json`). This clears the observed readiness
  confound, but has **not yet established corrected cache-off/on parity**.

## Concrete optimization seam

PPO loads one `StaticDataCache` before creating its worker threads. Nevertheless,
every `ManagedGame._prepare()` calls `populate()`, which repeats executable/data-tree
content hashing, hashes all snapshot members, copies 424 files, and hashes every
destination. All 48 threads can enter this path during startup, and deaths repeat it.
The initial successful load is not reused as a validated materialization contract.

Candidate: a run-scoped immutable, content-verified materialization snapshot shared
by worker threads, with private destination files for each new game. Amortize repeated
source enumeration/reads; retain destination integrity checks. Do not substitute an
mtime-only cache, reuse player saves, hardlink writable files, or silently weaken
binary/data invalidation. Implementation requires an explicit source-immutability
contract (for example, pinned private run inputs), not an unchecked “already validated”
boolean. No measured speedup is claimed for this design.

## Bounded next measurement

Before implementation, propose one corrected **cold/current-cache** pair: 48 workers,
128 decisions each, identical frozen terrain-v66 model, seed 1, online-train-v3,
fixed inference rows, continuing resets, no optimization or anchors, cycle cost zero,
and raw/sampling recording in both arms. Budget: **12,288 decisions total**, no
automatic rerun. This is proposed work, not authorization to launch it.

Record collection and complete wall time, descendant CPU, reset counts, and per-reset
wall/thread-CPU spans for identity validation, member verification, copying, and
launch-to-ready. Record source/checkpoint/config hashes; run without concurrent heavy
jobs. Require `compare-training-rollouts --require-policy-identical` and unchanged
final model tensors before interpreting wall time; retain strict/raw differences.
Inspect distributions and slowest-worker completion, not summed overlapping wall
spans. Separate startup from later resets and include recording overhead equally.

If cached preparation dominates the critical path, implement the single amortization
candidate and predeclare its separate paired budget. If not, do not optimize this
seam on CPU savings alone. Default activation requires a representative wall-time
win with policy parity; this one-pair pilot cannot establish low-variance speedup.
