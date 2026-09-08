# Post-more readiness: bounded validation

2026-09-08. Readiness implementation revision:
`a87c158dd7d4b3cb2de5935616485e1fdfb9017c`.
This extends the explicit-input boundary to structured more acknowledgments
(`menu_type=more`, `input_mode=5`, menu selection or cancel), not arbitrary space
keys. Fresh mode 0 prevents a quiet rendering flush from certifying completion;
without fresh busy evidence, the existing no-op/full-state-probe behavior remains.

## Adversarial coverage

The focused transport/process/environment suite passed: 134 tests, two integration
tests deselected, 1.02 seconds. Type checking and focused Ruff checks also passed.
Socket fixtures required the approved host boundary.

- The cache72 worker-14 shaft acknowledgment is represented by delayed prefixes
  before generation UI, during progress UI, and after `ui-pop` /
  `close_all_menus`. Each delay is 40 ms, exceeding ordinary 10 ms quiescence.
  Collection must continue through fresh mode 1 and the D:4 player state.
- Fresh supported input modes (including another more), menus, blocking CRT,
  description UI, line input, and terminal exit remain accepted.
- Busy output without fresh input evidence fails closed; nonblocking progress
  UI and closed menus cannot masquerade as input readiness.
- No-op acknowledgments retain full-state probing; an exchange without fresh
  busy evidence can still finish at quiescence. Space/Enter/Escape routing is
  scoped to the structured more state; raw space retains ordinary handling.

## Selected stable-release smokes

Both commands ran at the host boundary, using the default seed 1 and no static
cache. Both local upstream worktrees had no tracked source differences.

| Release and exact upstream revision | Command | Result |
| --- | --- | --- |
| 0.34.1, `1eebc1a2892e1c89776a0d7a10691f8dac8d9796` | `poe compatibility-smoke --binary vendor/crawl-0.34.1/crawl-ref/source/crawl` | Exit 0; Minotaur D:1; 132 visible cells; exact replay |
| 0.33.1, `9cb173b281c11a5177f40b8c0662bacd3aac2717` | `poe compatibility-smoke --binary vendor/crawl-0.33.1/crawl-ref/source/crawl` | Exit 0; Minotaur D:1; 135 visible cells; exact replay |

The smoke performs semantic reset, one WAIT, raw trajectory recording, and exact
semantic replay. Its temporary trajectory is deleted by the existing command;
the table records the returned results, not persistent replay artifacts. Unrelated
cache-validation work was concurrent in the shared application worktree; these
were uncached correctness smokes, not clean-tree throughput measurements.

## Limits

These two games do not deliberately trigger a shaft or more continuation. They
establish baseline stable-release reset/WAIT/replay compatibility after the
extension; the delayed post-more checks are synthetic transport tests derived
from the captured trunk failure. They do not prove every stable-release UI
sequence safe or restore cache72 paired-policy parity. No held-out evaluation,
new matched performance pair, or additional games were run for this check.

## Proposed preregistration: c74 frozen cold/cache pair

This is a plan, not an executed experiment. On 2026-09-08 both
`artifacts/p/cache74` and `artifacts/p/c74` were absent (including symlinks).
Reserve these paths for one fixed cold-then-cache pair after all source edits,
training, and other heavy tests finish. Do not reuse or overwrite them.

```sh
poe benchmark-startup-cache \
  --checkpoint checkpoints/terrain-v66.pt \
  --suite configs/online-train-v3.json \
  --binary vendor/crawl/crawl-ref/source/crawl \
  --static-data-cache artifacts/p/static-cache/verified-v1 \
  --output artifacts/p/cache74 \
  --run-root artifacts/p/c74 \
  --frozen-policy \
  --collect-static-cache-timing
```

The current CLI fixes seed 1, 48 workers, 128 decisions per worker, and one update
per arm: 6,144 decisions each, 12,288 total. Both start independently from
terrain-v66, feature 6, full action catalog, no history or residual head. The
benchmark uses zero objectives and cycle cost, no anchor replay, and empty
selective-warmup ownership to prevent AdamW drift. Preserve its existing CUDA
inference settings and decision cost 0.01. Do not substitute a residual pilot
checkpoint or enable an external identity validator: cache validation must take
the existing inline default (`identity_validator=None`), with any recorded
`external_identity_validation` absent or null. Inspect the launch commands and
reset metadata to confirm this rather than assuming an opt-in flag stayed off.

Before launch and after completion, require these SHA256 digests:

| Input | SHA256 |
| --- | --- |
| `checkpoints/terrain-v66.pt` | `3ab6b36828249d137b93e323bb25c27612b3c7b00978dba7c0db5f668b753924` |
| `configs/online-train-v3.json` | `f28735471788298ce44fb284d5b9b430b72ca74e01e15e47132e32f8f43b53ad` |
| `vendor/crawl/crawl-ref/source/crawl` | `8ace7d53c5c67b9fc9018dbe2b1f65c9b54191decd2aafa8dcd7130922629ce6` |
| `artifacts/p/static-cache/verified-v1/manifest.json` | `c4a88bb9f5adfaae594491852d49c510cd2ae3fcb78547d46e9b45ff3442a745` |

Record the actual launch revision and complete application-source/pyproject
digests; hold them unchanged throughout both arms. The benchmark records these,
checks source and inputs before each arm, and checks source again afterward.
Repeat input checks after the second arm as well. Cache member verification and
binary/data identity validation must remain enabled. The readiness revision is
the intended behavioral change from c72; unrelated unused code may be present,
but must not change the execution path between arms.

Required artifacts are `artifacts/p/cache74/{report.json,policy-parity.json,
off.log,on.log,off.pt,on.pt}`, and raw trajectories plus immutable collector
checkpoints under `artifacts/p/c74/{off,on}`. Require exact equality of all ten
source model tensors with both collectors and both final checkpoints; serialized
checkpoint hashes need not equal because metadata differs. Every transition's
collector hash must match its own stored checkpoint.

The built-in comparison must find exactly 128 first-collection transitions per
worker in numeric episode/attempt order, including resets and any required
terminal bootstrap. Require exact pre-action feature vectors, masks, normalized
sampling probabilities, pre-choice RNG hashes, actions/keys, rewards, termination,
and policy-relevant reset/bootstrap states for all 48 workers. Also retain strict
raw/semantic differences. Independently reconstruct all differing initial,
before/after, reset, and bootstrap semantic states: only removal of `cells[].col`
may make a semantic difference acceptable, and all actor inputs must already be
exact without that removal. Do not normalize messages, inventory, monsters,
terrain, input modes, menus, or raw records to obtain a pass.

Stop on any process failure, retry/missing trajectory, incomplete budget, changed
input/source/model tensor, external-validator use, policy/sampling mismatch, or
non-color semantic mismatch. Preserve the failed evidence; do not extend or rerun
this budget. In particular inspect worker 14's shaft/more boundary, but acceptance
is all-worker parity, not that example alone. If parity passes, compare recorded
whole-arm wall time and collection time, retaining per-reset stage timings without
summing overlapping wall spans. If cache wall time regresses, stop adoption; do not
reinterpret a stage improvement as an overall gain. A single fixed-order pair
cannot establish a reproducible speedup, promotion, or general readiness guarantee.
No held-out access, policy selection, cache-default change, or hour-scale run is
authorized by this plan.
