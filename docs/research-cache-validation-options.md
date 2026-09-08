# Cache validation: smallest safe next experiment

2026-09-08. Initial read-only proposal; opt-in library implementation now added below.

## Decision

Prefer **opt-in, bounded external-process identity validation** as the next small
implementation. Keep one fresh validation per reset. An immutable executable/data
snapshot could remove more repeated work, but the repository does not currently
provide the enforced immutability needed to safely skip those checks.

This refines the candidate in `research-cache72-timing.md`, not a claim of speedup.
c72 measured initial identity validation at median 10.452 seconds wall and 0.376
seconds thread CPU; the slowest later reset spent 22.928 seconds wall in validation.
That supports investigating scheduling/contention, but does not prove the GIL is
responsible. The cache pair still fails policy parity; no matched performance claim
or default activation is justified.

| Option | Correctness requirement | Scope and remaining cost |
| --- | --- | --- |
| Amortized immutable snapshot | Every game must execute the verified binary and read the same pinned data tree; those bytes and source mtimes cannot change during the run | Requires owned executable/data materialization plus enforceable immutability, launch-path/provenance changes, and compatibility checks; eliminates repeated identity hashing |
| Bounded external validation | Every reset independently runs the existing identity function against the actual launch binary/data path, then compares its result with the cache identity | Adds a small validator boundary, process lifetime, IPC and timing; retains all hashing and may add queue/interpreter overhead |

## Why copying once is not sufficient

`static_data_identity()` hashes the executable and the sorted bundled `dat` tree,
including relative names, nanosecond mtimes, and file contents. `_validate()` invokes
it before checking cache members; `populate()` then makes private copies and verifies
each destination. `ManagedGame` launches the original binary path.

An immutable-snapshot implementation must change that launch path to the private
verified executable with its corresponding `dat` directory, preserve freshness
stamps, and establish a lifetime during which neither can mutate. A read-only mode
bit controlled by the same user, a pre-run digest, or a post-run check does not
establish that invariant. Rechecking only mtimes misses same-mtime edits. A real
read-only filesystem snapshot/content-addressed deployment could supply the contract,
but creating that facility is larger than the present optimization seam. Snapshot
changes would require new identity/provenance and a newly validated cache association;
player saves must never enter the snapshot.

## Minimal external-validator protocol

- A run-owned validator object has a bounded concurrency limit, initially **two**;
  all rollout workers share it. Default `None` retains inline validation. Do not
  introduce a system daemon or alter readiness timing.
- The smallest first implementation uses a fresh, short-lived interpreter per
  request, limited by that semaphore. Invoke an argument vector with `shell=False`,
  not a forked Python worker inheriting CUDA/thread state. A reusable pool is a
  separate optimization only if process startup proves material.
- Each request carries the resolved actual binary path. The helper calls the
  **same** `static_data_identity()` implementation, with no memoization. Its typed
  response contains both digests and helper CPU/wall timings. The parent validates
  response shape and compares the digests with `StaticDataCache.identity` before
  proceeding. No success or identity is shared between resets.
- A typed deadline covers semaphore wait and execution. Nonzero exit, malformed
  response, mismatch, or timeout fails closed. Execution timeout kills/reaps the
  helper through `subprocess.run`; every acquired permit is released in `finally`.
  There is no arbitrary thread/future cancellation API: cancelling a waiting caller's
  future does not promise to interrupt a running helper. Never silently reuse a prior
  result or accept a failed check. The request owner retains the response locally,
  avoiding cross-worker association mistakes.
- Keep manifest/member validation, source-member hashing, private copying, and
  destination verification in their existing order and locations. Source changes
  between resets are therefore detected exactly as before. This does not make the
  pre-existing validation-to-launch interval atomic; do not claim it does.

Suggested ownership: `webtiles/cache.py` for an injected semantic validator interface;
a small new helper module for the subprocess protocol; process/env/PPO plumbing for
run ownership; benchmark reporting/tests for queue, helper, and validation timings.
The helper's CPU must be recorded separately: parent `thread_time()` becomes mostly
waiting and cannot represent offloaded CPU work.

## Bounded verification and measurement proposal

Before any DCSS run, test exact inline/helper digest equality on private fixtures;
binary/data edits, same-mtime content changes, additions/removals, malformed output,
timeouts, concurrency limits, cleanup, and unchanged corruption/copy behavior. Then
propose one no-game 48-request validation burst per mode against fixed known inputs,
including interpreter startup and queue wait. Do not select favorable repetitions
or tune concurrency from an unbounded sweep.

Only after the remaining gameplay parity issue is resolved and that diagnostic is
promising, predeclare one cached-inline/cached-external frozen 48×128 pair: 12,288
decisions total, identical seeds/objectives/recording, no anchors, unchanged model
tensors and policy-rollout parity required. Record absolute reset start/end offsets,
validation queue/helper CPU, full collection/process wall and total descendant CPU.
Do not sum overlapping wall spans. This pair tests the intervention against the
current cached path; it does **not** establish a win over cold default startup.
Neither proposed measurement is authorized or launched by this note.

## Implemented library boundary

`ExternalIdentityValidator(max_concurrency=ValidationConcurrency(2),
timeout=Seconds(30))` lives in `webtiles/identity_validation.py`. It launches a fresh
helper interpreter per request and computes the unchanged `static_data_identity()`.
`StaticDataCache.validate()` and `populate()` accept an optional
`identity_validator: StaticIdentityValidator`; `None` keeps the existing inline path.
No process/environment/PPO caller activates this interface yet.

When preparation timing is requested, `external_identity_validation` records queue
and total request wall time, validation-only helper wall/CPU, and helper process CPU
through response construction. Parent thread CPU remains separate; helper process
CPU includes interpreter startup but not final serialization/exit. Default inline
preparation still reads no timing clocks unless collection was requested.

Focused tests exercise real helper equality/private copies, fresh content/mtime and
file-set checks across requests, invalid replies/failures, queue deadlines, bounded
concurrency, and real timeout kill/reap behavior. They use synthetic temporary
binary/data fixtures, not DCSS games or throughput benchmarks. The existing
validation-to-launch race is neither enlarged into memoized trust nor made atomic.
