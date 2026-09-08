# Startup-cache rollout parity investigation

2026-09-08. Performance remains opt-in; no additional games are authorized by this
note. The next UI-return experiment supplies the recorded cold/cold control below.

## Resolved first-collection evidence

The subsequent recorded cold/cold UI67 pair (`a0e25ad`) rules out caching as a
necessary cause. `artifacts/e/ui67-first-collection.json` proves identical initial
model tensors and every pre-choice RNG digest. Thirty workers match semantically;
eight differ solely in cell colour, with identical policy inputs/behavior. Ten first
sampling divergences occur after identical downstairs choices: one exchange ends
at mode0+flush before floor generation completes. Worker:boundary pairs are
3:29,5:3,8:71,16:90,26:12,27:101,28:12,32:50,33:31,42:71 (zero-based).
Worker5's next premature CANCEL consumes delayed D2 output and produces the logged
`Unknown command.`. Worker28's complete exchange ends at a legitimate mode5 more
prompt, demonstrating why command-mode1 alone is insufficient. This establishes
the level-transition readiness race in the recorded pair; it does not retrospectively
prove every difference in the earlier unrecorded cache benchmark has that cause.

The scoped correction awaits explicit/recognized input evidence for structured stair
actions while preserving menu hotkeys, automatic paths, and raw data. Policy-input
comparisons use the checkpoint's feature version and keep strict semantic/raw colour
differences visible. Recheck frozen-policy parity before interpreting future costs or
startup-cache wall time; do not normalize away unexplained actor-affecting changes.

## Observed result

The matched terrain-v66 startup-cache benchmark used source revision
`9f0d08c2717ee98f568826dd6fc61f4d3f382695`, the same 48 workers, initial checkpoint,
training seed 1, online-train-v3 suite, 128 decisions per worker, objectives, and
27,333 prewarmed anchor transitions. The only gameplay-harness change was static
db/des cache prepopulation. Source/input hashes and complete commands are in
`artifacts/p/v66cache/report.json`.

| Result | Cache off | Cache on |
| --- | ---: | ---: |
| Decisions | 6,144 | 6,144 |
| Collection seconds | 39.02 | 50.82 |
| Process wall seconds | 44.556 | 56.152 |
| Awaited child/descendant CPU seconds | 635.891 | 147.051 |
| Completed episodes | 20 | 22 |
| Short cycles | 982 | 926 |
| UI overflows | 19 | 5 |

Eight of ten checkpoint model-state tensors differ. The unchanged tensors are the
ECHO head, whose objective weight was zero. This is operational evidence of lower
CPU use and worse wall time, with different rollouts; it does not establish an
identical-workload speedup or parity. The earlier two-seed, 20-action exact smoke
and deterministic diagnostic parity do not prove stochastic concurrent parity.

## What the retained logs narrow down

The benchmark predates PPO raw recording. Its `crawl.log` files retain terminal
output, but not policy-action indices, semantic batch boundaries, masks, sampled
probabilities, or RNG state. They cannot identify the first policy-state divergence
conclusively.

For the first episode of every corresponding worker, compare log bytes before the
intentional shutdown's `Crash caused by signal` suffix:

- 27 of 48 log pairs are byte-identical.
- Of the 21 differing pairs, 20 share the startup prefix through the game-seed
  message and differ later.
- Several first differences are an extra or missing `Unknown command.` following
  stair/exploration output. Representative locations are below; byte offsets are
  positions in the off-arm log, not decision indices.
- Worker 22 has an earlier difference in cloud/lava glyph drawing. This may be
  cosmetic; terminal bytes alone cannot determine whether actor features changed.
- All recorded episodes contain 83 versus 84 `Unknown command.` strings overall.
  The evidence concerns where these occur, not a general increase in the cached arm.

| Worker | First-episode seed | First differing byte | Context |
| ---: | ---: | ---: | --- |
| 20 | 3021 | 25,696 | Found downstairs; cached arm adds `Unknown command.` |
| 15 | 3016 | 61,353 | Upstair message; cached arm adds `Unknown command.` |
| 34 | 3035 | 66,701 | Upstair message; cached arm adds `Unknown command.` |
| 2 | 3003 | 118,539 | Upstair message; cached arm adds `Unknown command.` |
| 22 | 3023 | 1,797 | Initial cloud/lava glyphs before game-seed message |

Evidence roots: `artifacts/p/v66c/{off,on}/worker-N/episode-0-attempt-0/crawl.log`.
These are known training seeds, not heldout cases.

## Ranked hypotheses and limits

1. **Timing-sensitive observation/input boundaries.** Ordinary commands use a
   10ms output/settling path; automatic commands use input-ready-or-quiescence.
   The action mask offers only CANCEL when no menu exists and input mode is not 1.
   A transient/incomplete batch could therefore change the next action. Changed
   startup/reset timing and concurrent Python validation/copy work could expose an
   existing scheduling dependence. The log locations support investigating this
   first, but do not prove an incorrect boundary or a CANCEL action.
2. **Other scheduling or inference sensitivity.** `_Worker` creates an independent
   NumPy generator with seed `training_seed + worker_index`, and consumes one
   `choice` per policy decision. `_fixed_inference_inputs` assigns stable worker
   rows in a fixed 48-row matrix. The model contains linear/GELU layers, without
   dropout or batch normalization, and weights remain fixed throughout collection.
   These reduce obvious sources of divergence; they do not prove every probability
   vector or RNG state matched in the failed pair.
3. **Cached-data or rendering differences.** The initial worker-22 discrepancy
   warrants inspecting raw initial observations and derived features. Initial
   agreement elsewhere and prior deterministic parity weigh against a blanket
   map-generation change, but do not exclude a seed-specific cache issue.

The upstream `spectator_joined` handler flushes and sends full state while returning
zero rather than a key. Do not attribute `Unknown command.` directly to the probe
without raw evidence tying it to an emitted key and actual input boundary.

## Next falsifiable control, without extra game budget

Use the scheduled UI-return pair with raw recording enabled and static caches off
in both arms. Its first collections start from the same model and differ only in
training-cost metadata/reward accounting before optimization. Therefore compare
first collections, not the post-update checkpoints or later collections, whose
objectives legitimately differ.

Have the replay comparator locate the earliest difference for each corresponding
worker/episode: initial observation, emitted action, previous and next semantic
state, raw message ordering, input mode, menu, and terminal/reset boundary. Derive
features and legality masks under the recorded source/checkpoint contracts. Preserve
the actual message sequence around the first mismatch; do not infer an action from
the terminal renderer alone.

- If cold/cold first collections differ, cache activation is unnecessary to trigger
  the problem. Investigate the first raw/semantic boundary before changing cache
  behavior or readiness durations.
- If cold/cold is exact, it strengthens—but does not prove—the cache/timing
  hypothesis. A subsequent small cache-off/on probe should retain the original
  worker IDs, seed schedule, and inference matrix shape while shortening the chunk.
  Select its horizon from the now-recorded first-mismatch decision, rather than
  guessing from terminal byte offsets.
- If actions differ despite identical actor inputs and masks, raw protocol records
  alone are insufficient: capture the actual probability vector and a digest of
  the generator state before `choice`, without advancing or replacing the RNG.

No cache optimization, readiness adjustment, promotion, or additional rollout was
performed during this read-only investigation.
