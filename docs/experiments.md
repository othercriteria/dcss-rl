# Experiment journal

This is a deliberately selective, timestamped record of hypotheses, wall-clock cost,
results, and resulting decisions. Git history and generated artifacts hold the detail;
this journal exists to keep research velocity and direction visible.

## 2026-09-07

- **Environment and integration (first working session).** Established reproducible
  Nix/direnv tooling, built unmodified trunk, and reached stable semantic WebTiles
  input boundaries. CUDA validation on the RTX 4090 passed without a flake change.
- **Scripted baseline through first learned champion.** Iterated from a D:1 scripted
  floor through semantic imitation and DAgger. Pure cloning failure modes included
  autoexplore collapse and wall collisions. A stronger scripted-v3 teacher plus global
  navigation features produced the confidence-gated v4 champion: held-out depth sum
  11 versus the previous 5, but only 2.4% neural decisions (23.7% diagnostic). Decision:
  treat autonomous action coverage, rather than hybrid rank alone, as the next frontier.
- **Rollout scaling (5 seeds × 50 decisions).** One, two, and five workers took 84.50,
  45.07, and 20.61 seconds: 2.96, 5.55, and 12.13 decisions/s. Five workers gave 4.10×
  speedup. Decision: broaden seeds and horizons concurrently; DCSS command execution,
  not GPU compute, is the present wall-clock bottleneck.
- **Next experiment.** Build masked online PPO on the restored semantic checkpoint,
  then compare matched ECHO-on/off runs before spending a larger rollout budget.
- **PPO integration smoke (16 decisions, 2 workers).** One CUDA update completed in
  25.8 seconds, saved a restorable checkpoint, and reported finite losses (policy
  -0.021, value 0.402, ECHO 0.034). No episode completed in eight decisions per worker,
  as expected. Decision: amortize process startup with longer matched rollouts.
- **2026-09-07 15:32–15:44 EDT — matched PPO v1 (2 × 2,560 decisions).** Four
  128-step updates with five workers took about 4.5 minutes per arm. ECHO-on (weight
  0.1) and ECHO-off (0.0) both completed five episodes with mean training return 11.8.
  Their fallback-free diagnostic ranks were identical at `(0, 0, 6, 8, 1000, 33.0)`
  over 1,000 decisions, with every episode horizon-truncated. PPO materially changed
  v4 (68% raw argmax agreement; KL 0.051 on the recorded states), but the two arms had
  100% argmax agreement and only `2.6e-5` mean absolute logit difference. Decision:
  add per-update throughput/loss telemetry; a weight-0.1 ECHO ablation is behaviorally
  underpowered at this budget, and longer training must rotate through more seeds.
- **2026-09-07 15:47–15:54 EDT — synchronous PPO worker sweep (64 steps).** Five,
  ten, and twenty workers processed 320, 640, and 1,280 decisions at 4.53, 5.63, and
  7.29 decisions/s. Relative throughput was 1.00×, 1.24×, and 1.61× for 1×, 2×, and
  4× processes; update latency rose from 71 to 114 to 176 seconds because each step
  waits for the slowest DCSS automatic command. Decision: use 20 workers once for
  full-seed breadth, but prefer an asynchronous actor queue over further synchronous
  worker scaling.
- **2026-09-07 15:55–16:10 EDT — broad strong-ECHO PPO v2 (10,240 decisions).**
  Twenty workers, eight 64-step updates, 16 optimizer epochs, and ECHO weight 1.0
  completed in about 11 minutes. Twenty-two episodes averaged return 10.96; throughput
  ranged from 7.27 to 19.47 decisions/s as expensive automatic commands varied. The
  fallback-free diagnostic frontier remained depth sum 6, XL sum 8, reward 33. Raw
  game turns rose from 11,252 to 21,545 without progression, exposing an exploitable
  rank tie-breaker. Decision: rank bounded policy survival instead of automatic game
  turns; do not spend on the matched strong-ECHO-off arm until the learning signal or
  collection architecture changes.
- **2026-09-07 16:18 EDT — ranking follow-up.** Replayed historical trajectories to
  compare decision-weighted progression. The D:1 scripted heldout floor has raw depth
  area 2,500 but zero `depth - 1` progress; the hybrid champion has 213 progress over
  1,240 decisions; scripted-v3 diagnostic has 1,019 over 851. Decision: rank v3 uses
  depth/XL progress area ahead of bounded survival, retaining maximum depth only as a
  later frontier tie-breaker. Branch curricula must replace the Dungeon-local depth
  coordinate with a versioned topology-aware mapping before they become eligible.
- **2026-09-07 16:55 EDT — XL-area correction.** XL is monotonic in normal play, unlike
  reversible dungeon depth. Integrating it over decisions rewards early/risky XP gain
  rather than more eventual XP. Decision: rank v4 retains depth-progress area but uses
  final/maximum XL only, after bounded survival and maximum-depth frontier.
- **2026-09-07 16:25 EDT — asynchronous collector and dense-reward smoke.** The
  matched 20-worker × 64-step workload rose from 7.29 to 44.66 decisions/s (6.13×)
  after removing per-action barriers. A full repeat reached 44.63 decisions/s and
  produced bit-identical model tensors using worker-local seeded samplers. Exploration
  0.01, XP-progress 1.0, and HP-potential 1.0 yielded finite losses and one completed
  episode. Decision: use this foundation for controlled 20-seed dense-reward runs;
  further synchronous worker tuning is no longer the immediate bottleneck.
- **2026-09-07 16:31 EDT — dense PPO v3 interrupted after update 7.** Early updates
  reached 47–64 decisions/s and several completed episodes had shaped returns above
  25, but a fresh-game handshake timed out. Because checkpointing occurred only after
  all 16 updates, no candidate survived. Audit also found that the worker seed formula
  accidentally returned every worker to its initial seed. Decision: atomically save
  each completed update, rotate seeds correctly, and retry transient starts up to
  three times in distinct directories before restarting the controlled experiment.
- **2026-09-07 16:44 EDT — corrected dense ECHO-on v4 (20,480 decisions).** The
  recoverable 16-update run completed 41 episodes at mean shaped return 30.77, but its
  deterministic diagnostic rank regressed to zero depth-progress area: 573 of 1,000
  actions were `explore`, and no game left D:1. Decision: exploration coverage is a
  misaligned dominant reward here. Disable it, add a strong explicit depth potential,
  and class-balance online teacher imitation so rare descent/tactical actions are not
  overwhelmed by the common explore label. Do not run the ECHO-off mate for a reward
  specification already falsified independently of ECHO.
- **2026-09-07 17:10 EDT — v5 ablation and heldout rejection.** With depth-aligned
  shaping and class-balanced teacher loss, ECHO-on reached diagnostic depth-progress
  461; matched ECHO-off reached 482 (4.6% higher), with both surviving all 1,000
  decisions. ECHO-off was selected, but heldout-v1 produced zero progress and 953
  ineffective `stairs_down` actions. The visible under-player stair message cleanly
  distinguishes all recorded legitimate descents. Decision: mask stairs to that UI
  affordance. Because heldout-v1 directly informed this adapter change, retire it from
  promotion and establish disjoint heldout-v2 before evaluating the corrected policy.
- **2026-09-07 17:22 EDT — heldout-v2 calibration.** Scripted-v3 established rank
  `(0, 0, 2448, 2060, 17, 11, 76.0)` on five new seeds. The pre-existing confidence
  hybrid tied every metric except surviving 2,059 rather than 2,060 decisions, so the
  scripted policy remained incumbent. A validated atomic activation archived v1 and
  cut the canonical viewer over to v2. Evaluation now streams completion-order case
  telemetry while persisting summaries in manifest order.
- **2026-09-07 17:44–17:59 EDT — protocol and presentation latency.** The identical
  five-seed, 2,500-decision fallback-free v5 evaluation previously took 311.11 seconds
  (8.04 decisions/s). Ending automatic commands on upstream `input_mode=1` plus flush,
  with the existing 500 ms quiescence as fallback, reduced it to 61.60 seconds while
  reproducing rank `(0, 0, 0, 2500, 5, 5, 0.0)` exactly. Raw traces then explained the
  remaining two-worker tail: repeated monster warnings invoked DCSS's presentation-only
  100 ms animation sleep. Documented delay-free/accessibility RC options, including an
  empty `use_animations` list, reduced the same workload to 16.18 seconds and 154.48
  decisions/s: 19.23× the original throughput with the same rank. Decision: retain the
  explicit readiness/fallback contract and delay-free local runtime, then remeasure PPO
  collection before increasing seed breadth and horizon.
- **2026-09-07 18:09 EDT — post-latency asynchronous worker sweep.** Matched 64-step
  chunks at 5, 10, 20, and 40 workers collected 320, 640, 1,280, and 2,560 decisions at
  36.93, 63.94, 97.93, and 93.28 decisions/s. The 20-worker workload improved 2.19×
  over its prior 44.66/s result; 40 workers regressed slightly. Decision: keep 20 as
  the collector knee and spend the recovered wall clock on a 64-seed, 1,000-decision
  training-only v2 suite rather than further process oversubscription.
- **2026-09-07 18:14 EDT — seed-breadth audit.** Before launching v2, inspection found
  that each worker advanced by one suite entry per episode. Twenty workers therefore
  overlapped almost completely after their first episodes instead of covering the next
  seed block. Decision: stride episode rotations by worker count, with semantic worker,
  episode, case-count, and case-index types. A 20-worker/64-case schedule now covers
  0–19, 20–39, 40–59, then 60–63 plus 0–15 before repeating.
- **2026-09-07 18:24–18:40 EDT — broad matched v6 ablation.** Starting from the same
  fallback-free v5-off checkpoint, each arm consumed 40,960 decisions on the 64-seed,
  1,000-horizon suite with identical depth/XP/HP shaping and teacher weight 1. ECHO-on
  completed 72 episodes at mean shaped return 46.31; ECHO-off completed 73 at 56.20.
  On the 200-step diagnostic, off appeared slightly better (depth-progress 491 versus
  451, and v5-off's 482). Because every run still truncated, diagnostic-v2 extended
  the same known seeds to 500 decisions. There, scripted-v3 calibrated at rank
  `(0, 0, 2380, 1194, 20, 16, 115.0)`; v5-off scored 1,385 depth-progress, v6-on 1,351,
  and v6-off 1,391, with every learned case surviving but three remaining on D:1.
  Decision: treat v6-off's +0.4% as noise, reject v6 without heldout access, retain
  ECHO-off as the cheaper arm, and change the learning/navigation recipe rather than
  merely extending this one.
- **2026-09-07 18:47 EDT — structured-prompt root cause.** V6-off spent 937/2,500
  decisions on cancel, but replay reconstruction showed 934 occurred in input mode 8
  at a visible structured Yes/No menu (“Really rest while Zot is near?”). The reducer
  only understood `ui-push` widgets and discarded upstream `menu` payloads, so the mask
  failed closed to cancel without a way to answer. Promoting titles/items/hotkeys made
  the exact deterministic v6-off rank unchanged but cut diagnostic-v2 from 125.12 to
  21.00 seconds. It also exposed the deeper behavior: 1,867 selections answered yes,
  then rest immediately stopped and prompted again. Decision: add online teacher
  agreement and imitation-loss telemetry, then tune against the rest/prompt loop.
- **2026-09-07 18:55 EDT — imitation-strength probe.** Two ECHO-off arms started from
  the pre-PPO DAgger checkpoint and consumed 10,240 decisions with teacher weights 1
  and 10. Weight 1 ended at 42.2% sampled-action agreement and diagnostic-v2 depth
  progress 1,383. Weight 10 lowered last-minibatch imitation loss from 1.18 to 0.89 and
  raised agreement to 47.6%, but depth progress regressed to 920 and lost one descent.
  Both remained dominated by rest/prompt cycles. Decision: extend only weight 10 to
  the matched 40,960-decision budget to distinguish slow distillation from saturation;
  do not infer policy quality from agreement alone.
- **2026-09-07 19:08 EDT — full strong-imitation v8 and heldout rejection.** At 40,960
  decisions, weight 10 ended at 52.3% teacher agreement and improved diagnostic-v2
  depth-progress to 1,795 (29.0% above v6-off), with aggregate max-depth sum 9. This
  earned one locked heldout-v2 evaluation, where it remained D:1 on all five seeds and
  failed the floor at `(0, 0, 0, 2500, 5, 6, 10.0)`. On those policy states the teacher
  requested rest only 14 times while the policy chose it 769 times. Full inverse class
  balancing gave each rare rest label roughly 44× the weight of common labels.
  Decision: do not promote; make teacher balancing strength explicit and reduce the
  default from full inverse to square-root inverse.
- **2026-09-07 19:16 EDT — square-root balancing probe.** A matched 10,240-decision
  weight-10 run with balance exponent 0.5 reached 45.6% teacher agreement and cut
  rest/prompt actions roughly in half, but directional oscillation replaced part of
  the collapse. Diagnostic-v2 depth-progress was 1,339. Decision: because the prior
  exponent-1 arm rose from 920 at this budget to 1,795 at full budget, run the same
  40,960-decision test before judging the new weighting rule.
- **2026-09-07 19:27–19:43 EDT — full balance-exponent sweep.** At 40,960 decisions,
  exponent 0.5 reached 64.6% teacher agreement and made substantially more progress
  per wall-clock second, but diagnostic-v2 included two early deaths and ranked
  `(0, 0, 1263, 1662, 13, 10, 54.0)`: more aggregate max depth, worse sustained depth
  area. Exponent 0.75 ended at 63.6% agreement but reverted to the passive frontier at
  `(0, 0, 1377, 2500, 8, 8, 39.0)`. Decision: interpolation does not beat exponent-1
  v8's 1,795 depth area, which itself failed heldout. Separate online imitation from
  PPO/value gradients for the next ablation.
- **2026-09-07 19:39 EDT — 24-worker check.** On the fixed initial policy and 64-step
  chunks, the current build measured 99.17 decisions/s at 20 workers and 102.58 then
  104.58/s at 24. The reproducible gain is 3.4–5.5%, not the expected 20%, but is cheap
  enough to adopt. Decision: use 24 workers and therefore 3,072 decisions/update for
  the next fresh experiment.
