# c72: static-cache preparation under 48-worker load

2026-09-08. Source `6a217110f25c91594b81ae9af623a590d56faac7`.
Evidence: `artifacts/p/cache72/report.json` and `policy-parity.json`.
This analysis ran no games or benchmarks and changed no application sources.

## Operational result, not matched-workload speedup

Both arms collected 6,144 decisions with all ten final model tensors unchanged from
their initial checkpoint. Nevertheless, worker 14 fails policy-input/sampling parity.
The benchmark correctly remains `completed=false`; eleven workers differ strictly.

| Measure | Cold | Cached |
| --- | ---: | ---: |
| Collection wall, seconds | 46.30 | 58.41 |
| Complete process wall, seconds | 51.686 | 63.474 |
| Awaited descendant CPU, seconds | 650.364 | 155.102 |
| Recorded resets | 69 | 70 |

Caching again reduces aggregate CPU while increasing observed wall time, but the
different policy trajectories/reset counts prevent a causal matched-performance claim.
Do not enable caching by default or infer that the remaining parity failure is caused
by cached contents. Root is investigating that failure separately.

## Cached preparation distributions

Initial means episode 0 for each worker (48); later means episodes 1+ (22). Entries
are **median / p95 / maximum seconds**, with p95 linearly interpolated from ordered
samples. Timings cover cache preparation only, before launching DCSS.

| Stage | Initial 48 wall | Later 22 wall |
| --- | ---: | ---: |
| Executable/data identity validation | 10.452 / 10.867 / 11.053 | 10.240 / 21.725 / 22.928 |
| Snapshot member verification | 2.346 / 2.434 / 2.470 | 0.905 / 1.008 / 1.248 |
| Private copying | 4.667 / 4.768 / 4.831 | 2.208 / 2.432 / 2.484 |
| Destination verification | 1.804 / 1.871 / 1.883 | 0.766 / 1.029 / 1.034 |
| Total preparation | 19.318 / 19.538 / 19.572 | 14.082 / 25.790 / 27.401 |

Stage quantiles are independently computed and must not be added. Total includes
setup, renaming, and instrumentation overhead; unattributed median wall is only
0.029 seconds initially and 0.022 seconds later. Overlapping worker wall durations
must not be summed as runtime.

Initial preparation consumes 37.131 summed thread-CPU seconds; later preparation
consumes 16.178. Across all 70 resets, identity validation accounts for 26.329 of
53.309 thread-CPU seconds (49.4%), private copying 14.967, source verification 6.166,
and destination verification 4.740. Median per-reset total thread CPU is 0.767
seconds initially and 0.762 later. These timings exclude DCSS child CPU.

## Slow-reset diagnosis and next intervention

The slowest initial preparation is worker 16, episode 0: total 19.572 seconds,
identity 10.711, copying 4.678. The slowest later preparation is worker 13, episode 1:
total **27.401 seconds**, identity **22.928**, copying **2.351**, source verification
1.248, destination verification 0.848. Its entire preparation uses only 0.803
thread-CPU seconds, including 0.400 in identity validation.

Thus executable/data validation is the largest measured component of the longest
reset, not private copying. The large wall/thread-CPU gap indicates waiting or
scheduling amplification; these spans cannot distinguish GIL contention, filesystem
waits, or OS scheduling. Durations have no absolute start/end timestamps, so worker
13 is the slowest measured reset, **not a proven final critical-path worker**.

Conditional on resolving policy parity, the highest-value intervention remains a
run-scoped, immutable, content-verified source snapshot that amortizes executable/data
identity validation across resets. PPO already shares a loaded cache object but
`populate()` repeats the complete validation in each worker. Preserve private files,
destination verification, and fail-closed source identity; do not replace content
verification with an unchecked “already validated” flag or mtime-only trust.

The new measurements justify targeting that seam. They do not quantify its eventual
wall-time gain or authorize another rollout. A future approved comparison needs
policy parity first, matching reset workloads, and reset start/end offsets to identify
the actual completion-critical path. Readiness waits should not be shortened as a
performance shortcut.
