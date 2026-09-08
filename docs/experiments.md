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
- **2026-09-07 20:04 EDT — online-imitation isolation.** A 24-worker, 49,152-decision
  run disabled policy, value, entropy, and ECHO gradients while retaining autonomous
  collection and square-root-balanced online teacher imitation. Teacher agreement
  reached 74.7%, but diagnostic-v2 ranked `(0, 0, 150, 2101, 9, 8, 32.0)`: four cases
  remained on D:1 and one rushed to D:5 before dying. The policy selected rest 1,068
  times and answered menus 867 times. Decision: reject without heldout access; online
  imitation alone does not cure the rest-confirm attractor.
- **2026-09-07 20:11 EDT — idle DCSS CPU root cause.** Every paused Crawl worker used
  about one full CPU because its headless WebTiles `pselect` includes stdin and the
  launcher supplied permanently-readable `/dev/null`. Supplying an unwritten pipe
  instead lets the existing socket wait block. In an isolated post-fix measurement,
  Crawl CPU time stayed unchanged at four seconds over a further 12 wall-clock
  seconds. A matched 128-step resweep measured 137.12, 145.24, 150.57, 154.03, and
  149.70 decisions/s at 24, 32, 40, 48, and 64 workers. Decision: retain the pipe and
  use 48 workers for representative training; the new knee is 65% faster than the old
  pre-fix 40-worker result.
- **2026-09-07 20:19 EDT — asynchronous inference batching.** A dedicated inference
  thread now coalesces otherwise serialized worker requests without imposing a game
  step barrier. At 48 workers and 128 steps, one-millisecond/64-request batching
  realized mean batches of 18.0 and raised throughput from 154.03 to 166.18
  decisions/s. Collection took 36.15 seconds, optimization 0.82, and checkpointing
  under 0.01. A three-millisecond arm produced a smaller 17.2 mean batch and 165.13/s.
  Model tensors were bit-identical between unbatched, batched, and repeated batched
  runs. Decision: retain one millisecond and expose timing/batch occupancy; collection
  and environment work, not optimizer/checkpoint work, remain dominant.
- **2026-09-07 20:27–20:34 EDT — unweighted online imitation.** An exponent-zero arm
  consumed the same 49,152 decisions with 48 workers and ended at 71.9% teacher
  agreement. On diagnostic-v2 it ranked `(0, 0, 973, 1600, 11, 9, 38.0)`: two early
  deaths, two D:1 truncations, and one D:3 truncation. Rest fell from 1,068 weighted-arm
  decisions to two, but explore rose to 1,009/1,600 and opposing NW/SE moves accounted
  for another 362. Decision: reject without heldout access; weighting brackets two
  different attractors rather than solving autonomous control.
- **2026-09-07 20:36 EDT — legality-mask consistency audit.** Offline trajectory
  training independently reconstructed a permissive command mask that exposed both
  stair actions everywhere, unlike live execution's player-visible underfoot stair
  affordance. Decision: restore the semantic domain type from trajectory dictionaries
  and route offline training through the same action-mask implementation as live play.
- **2026-09-07 20:40–20:51 EDT — broad scripted curriculum and menu-loop repair.** A
  64-seed/1,000-horizon scripted-v3 collection yielded 32,753 decisions, but 7,242
  selected key `a`; several apparent survivors spent over 900 decisions repeatedly
  selecting the first unaffordable shop item. Scripted-v4 now handles known menu
  semantics explicitly and cancels shops/unknown menus. Recollection produced 28,224
  clean decisions, 42 deaths, 22 truncations, aggregate max-depth 226, and only 190
  menu selections. Early deaths remain useful deeper-state curriculum rather than a
  reason to reward inert horizon survival.
- **2026-09-07 20:55–21:02 EDT — clean broad behavior cloning.** Policy-only clones of
  scripted-v4 reached 94.6% validation accuracy unweighted and 94.4% at balance
  exponent 0.25. Diagnostic ranks were `(0, 0, 670, 1660, 11, 9, 38.0)` and
  `(0, 0, 223, 1659, 10, 9, 35.0)`. Decision: teacher-state validation does not survive
  rollout covariate shift; take only the stronger unweighted clone into DAgger.
- **2026-09-07 21:10–21:24 EDT — true aggregated DAgger.** The existing online
  imitation path discarded prior learner-state labels each update, contrary to DAgger's
  aggregation contract. Retaining them and sampling cumulative imitation minibatches
  eliminated rest/menu loops and made all five diagnostic cases survive, ranking
  `(0, 0, 1420, 2500, 8, 8, 39.0)`. Increasing replay fitting from 4 to 16 epochs cost
  only about 1.5 seconds/update and raised final current-state agreement from 60.8% to
  74.3%, but shifted behavior toward one D:5 death and one D:3 survivor at rank
  `(0, 0, 1167, 2152, 11, 8, 38.0)`. Decision: aggregation fixes forgetting, while
  fitting strength exposes the survival/depth frontier; repeated opposing actions now
  motivate explicit policy memory rather than more large scalar sweeps.
- **2026-09-07 21:28–21:48 EDT — expert cleanup and learned promotion.** Scripted-v6
  fixed unseen-monster, escape-hatch, poison, and fire stalls. Its 64-seed curriculum
  reached aggregate max-depth 257 with 63 deaths and only one truncation in 9,123
  decisions. An unweighted clone plus eight-update aggregate DAgger produced v21,
  which survived all diagnostic-v2 cases at rank `(0, 0, 3122, 2500, 12, 10, 71.0)`.
  Its one earned heldout-v2 evaluation scored `(0, 0, 2744, 2091, 12, 7, 31.0)`,
  exceeding the scripted incumbent and becoming the first fallback-free canonical
  champion beyond the retired depth-11 milestone. Decision: promote v21; retain deaths
  as curriculum rather than optimize inert survival.
- **2026-09-07 21:49–22:01 EDT — feature v3 and heldout rollover.** Heldout-v2 traces
  revealed shop and confirmation repetition, so the track was retired. Before testing
  the resulting feature change, v21 locked disjoint heldout-v3 at
  `(0, 0, 2416, 2120, 13, 12, 84.0)`. Feature-v3 explicitly represents prompt/shop and
  blocking message state while old checkpoints retain v2 preprocessing. A plain v3
  clone led diagnostic depth area at 3,472 but failed heldout-v3 at
  `(0, 0, 528, 1687, 8, 8, 19.0)`. A subsequent aggregate run was interrupted after
  four of eight updates during session compaction and is not treated as a completed
  experiment. Decision: rerun it from the same initial checkpoint; do not tune from
  heldout-v3 failure details.
- **2026-09-07 22:05 EDT — early-death objective audit.** Sparse environment reward
  charged death -10, equal to one XL gain. Under discounting, delaying an unavoidable
  death reduces that cost's present value; reaching a finite horizon can erase it.
  Evaluation's decision-weighted depth area independently forfeits remaining horizon
  opportunity and can prefer safe lingering. Decision: remove the explicit death cost;
  termination and foregone future progress still teach the value model, while terminal
  trajectories remain in the curriculum. Design a versioned discovered-coverage/depth
  rank that rewards progress without stalling or making fast failure globally ruinous.
- **2026-09-07 22:06–22:12 EDT — complete feature-v3 DAgger rerun.** The clean
  eight-update rerun consumed 49,152 decisions and ended at 56.3% current-state teacher
  agreement. Diagnostic-v2 reached D:5 twice and aggregate XL 14, but two early deaths
  and three passive survivors yielded rank `(0, 0, 1193, 1692, 15, 14, 100.0)`, below
  v21's depth-area lead. Decision: reject without heldout-v3 access; the result further
  demonstrates why depth-area and useful risky exploration need separating.
- **2026-09-07 22:14–22:23 EDT — rank v5 recalibration and v23 promotion.** Rank v5
  replaces horizon-sensitive depth area and policy steps with depth-weighted newly
  discovered cells and distinct branch levels. Repeated actions and already-seen
  backtracking add nothing; useful exploration after retreat still counts. Unchanged
  v21 calibrated at 11,483 diagnostic and 15,677 heldout. Under the metric selected
  without heldout-v3 feedback, v23 scored 17,121 diagnostic and then 16,202 on its one
  heldout attempt, clearing the locked floor and becoming canonical despite three
  deaths. Decision: promote; optimize the demonstrated exploration while improving
  survival as a separate frontier rather than conflating it with unused horizon.
- **2026-09-07 22:25–22:38 EDT — matched PPO/ECHO ablation.** From identical v23
  initialization, both arms consumed 49,152 autonomous decisions over the same
  64-seed/1,000-step curriculum with policy/value/entropy/imitation weights
  `1/.5/.01/.1`; only ECHO differed (`0` versus `.1`). ECHO-on reduced its auxiliary
  loss to .026 but diagnostic rank was 14,413 discovered cells, below v23. ECHO-off
  reached 36,377 cells across 21 levels and earned heldout access, but generalized to
  only 13,371 cells across 10 levels and failed the 15,677 floor. Its diagnostic action
  mix was movement-heavy; heldout shifted to 578 explores and 549 opposing NW/SE
  moves. ECHO-on instead selected wait 443 times diagnostically. Decision: neither
  promotes; ECHO is not helping at this weight/budget, and one-step action/history
  memory is a better-targeted next intervention than another scalar sweep.
- **2026-09-07 22:41 EDT — post-change compatibility smoke.** The same semantic MiBe
  reset/action/schema-v2 exact-replay path passed on trunk `96832895d0`, stable 0.34.1
  `1eebc1a2892e`, and stable 0.33.1 `9cb173b281c1`. Decision: the reward, return, and
  rank changes preserve the three-version integration contract; proceed to explicit
  action/history state.
- **2026-09-07 22:44–23:09 EDT — action-history candidate and PPO repeat.** Two
  newest-first policy-action slots were added outside semantic observations, isolated
  per episode/worker and excluded from ECHO targets. An eight-update history-aware
  DAgger run scored 26,857 diagnostic. From it, matched 49,152-decision PPO arms scored
  18,302 ECHO-off and 21,009 ECHO-on; neither earned heldout access. Decision: retain
  the general history capability, but PPO at these weights regresses the initializer
  and ECHO's direction is not stable across starting policies.
- **2026-09-07 23:10–23:25 EDT — causal history control and heldout repair.** The
  matched stateless DAgger control scored 34,118 diagnostic, above history v26's
  26,857; one 500-step rest/prompt-heavy case took 76.8 seconds. Its first clean
  heldout-v3 attempt scored only 10,907 versus history v26's 21,981, suggesting history
  traded known-seed fit for generalization. However, v24 heldout-v3 traces had motivated
  the history design, invalidating v26 promotion on that track. Heldout-v3 was retired.
  Untouched v4 was calibrated with pre-intervention v23 at 8,822, then the already
  frozen v26 and v29 candidates scored 12,910 and 16,853. Decision: promote stateless
  v29 on v4; additional aggregate DAgger is the strongest supported cause, while
  history infrastructure remains available for more targeted outcome-aware context.
- **2026-09-07 23:28–23:48 EDT — doubled broad DAgger curve.** Sixteen updates over a
  disjoint 128-seed/2,000-step curriculum consumed 98,304 decisions and retained every
  checkpoint. Snapshotting cost about .01 seconds/update. Diagnostic rank varied
  non-monotonically from 15,974 to 45,783; update 16 led, reached D:7 twice, and summed
  XL 24. Its one heldout-v4 attempt scored 8,704, narrowly below the locked 8,822 floor.
  Decision: reject; retain intermediate checkpoints permanently for generated runs,
  because agreement and final-update selection do not identify policy quality.
- **2026-09-07 23:50–23:57 EDT — Berserk action frontier.** The fixed vocabulary made
  the MiBe fight without Trog abilities. An append-only tail action now opens the
  player-visible ability menu while preserving all 270 legacy indices; Berserk remains
  a normal masked menu choice. The first teacher revision exposed targetless Berserk
  autoexplore as a no-time loop. Waiting until the status expires fixed it, raising the
  scripted diagnostic from 7,970 to 23,509 and reaching D:7. Decision: train the new
  action online from the current learned champion; do not use a hidden two-key macro.
- **2026-09-08 00:00–00:14 EDT — Berserk catastrophic forgetting.** A matched
  98,304-decision DAgger run against scripted-v7 shifted initial agreement from 64.7%
  to 54.1% and every retained diagnostic checkpoint collapsed below 8,107. Reducing
  learning rate 10× produced a best update at 35,249, but it invoked abilities zero
  times and its one heldout attempt scored 13,348 below v29's 16,853. Decision: the
  appended class needs a head-only acquisition stage; preserve all old parameters
  until the new action has a useful logit, then unfreeze jointly.
- **2026-09-08 00:15–08:45 EDT — staged Berserk acquisition and anchored correction.**
  Appended-row warmup preserved v29 exactly through four updates, then opened the
  ability menu but chose Renounce Religion and rejected its prompt. Training companion
  menu-`a` learned the genuine two-policy-decision Berserk flow but retried it while
  already Berserk. Feature v4 appended visible Berserk/exhaustion bits with lossless
  checkpoint migration. A boolean-indexing test also caught frozen-gradient writes to
  a temporary tensor: weights had been restored, but Adam moments could accumulate;
  indexed assignment now zeros them at the source. Online DAgger can preload and
  relabel replay anchors; v38 used 25,733 transitions from 128 non-heldout v19/v20
  trajectories. Selective v39 trained only the two flow rows and new status columns,
  retained v29 exactly through update 4, and by update 8 invoked Berserk once without
  retrying, but did so without a visible threat and ranked only 7,035. Anchored,
  unweighted joint v40 at `1e-6` recovered one D:7 episode and peaked at 11,115, still
  below v29's 34,118. Decision: pretraining machinery is reusable and the UI flow is
  learnable, but do not promote or tune further on these five seeds. Next make online
  reward/returns capable of opposing recurrent no-progress behavior. All conclusions
  here are conditional on this checkpoint, representation, curriculum, weighting, and
  budget; changed surrounding conditions are explicit revisit triggers.
- **2026-09-08 09:07 EDT — loop-resistant return/cost contract.** Online PPO can now
  treat death as a boundary within a continuing reset process: it bootstraps from the
  freshly scheduled reset value but cuts GAE before the next episode's sampled rewards.
  Wins remain terminal. Independent typed decision and exact semantic short-cycle costs
  leave reported environment return and evaluation unchanged; cycle telemetry is
  explicit. The host gate passed 92 fast tests in 1.95 seconds and three live tests in
  8.72 seconds. Decision: pilot matched no-cost, decision-cost, and cycle-cost arms from
  frozen v29 before committing the full training budget or consulting heldout-v4.
- **2026-09-08 09:18 EDT — matched-rollout reproducibility.** The initial three
  6,144-decision arms differed before their first update despite identical checkpoint,
  seed, and per-worker RNG: asynchronous GPU batches changed shape and produced 890–910
  detected cycles. Batch size one made matched 1,536-decision collection exact but
  serialized inference. Fixed-size padding reduced the discrepancy to one transition;
  additionally assigning each worker a stable matrix row made two matched 768-decision
  collections identical at 104 cycles and 68.0% teacher agreement. Decision: retain
  stable rows and re-establish representative throughput while running the full arms;
  treat the earlier v41–v43 ranks as safety pilots, not a clean causal comparison.
- **2026-09-08 09:23–10:02 EDT — continuing-cost ablation and v51 promotion.** Three
  arms restored v29, preloaded the same 25,733 v19/v20 anchors, disabled ECHO, and each
  consumed 49,152 decisions (eight 48×128 updates) on online-train-v3; only cost differed.
  Exhaustive diagnostic curves peaked at update 1/no-cost 35,140, update 2/decision
  `0.01` 39,956, and update 2/cycle `0.1` 31,991, then regressed to 12,362–15,894 in
  several late snapshots. v51 update 2 reduced rest from v29's 281/1,431 actions to
  4/1,150 and exact 8-decision semantic recurrence from 803 (56.1%) to 329 (28.6%);
  the no-cost best still rested 278 times. Its single locked heldout-v4 evaluation
  scored `(0, 0, 17613, 19, 19, 12, 112.0)` versus v29's
  `(0, 0, 16853, 15, 15, 14, 120.0)`, promoting v51 update 2. Four cases died on D:3–5
  and one survived 500 decisions on D:2. Decision: the general decision cost is locally
  better than exact cycle punishment and advances the canonical metric/level frontier,
  but survival and the true D:11 milestone remain open; retain every early snapshot and
  revisit the cycle arm when state/memory/objective context changes.
- **2026-09-08 10:03 EDT — isolated harness-performance handoff.** A first subagent in
  `/tmp/dcss-rl-harness-performance` committed `143b2af7` without merge authority. CPU
  evidence found duplicate online/offline encodes cost about 1.07 seconds per 2,472
  real observations; representative directional samples improved 2.4–3.8%. Decision:
  manually port the changes after the ablation, preserving distinct terminal-ECHO and
  reset-policy features, then remeasure on current main. The ported 48-worker sanity
  run collected 6,144 decisions at 149.45 decisions/s. Because its asynchronous policy
  path diverged from prior runs, retain the controlled samples as the comparative
  evidence and treat the end-to-end result only as a regression check.
- **2026-09-08 10:17 EDT — heldout-v5 firewall.** Detailed v51 outcomes made v4
  historical evidence before survival-objective design began. A disjoint five-seed v5
  suite was therefore locked and calibrated exactly once with the promoted checkpoint:
  rank `(0, 0, 16234, 18, 18, 14, 129.0)`, four deaths, and one 500-decision survivor.
  Decision: install that result as the v5 floor/canonical champion, archive v4, and do
  not inspect v5 again until a candidate passes diagnostic selection.
- **2026-09-08 10:20–10:40 EDT — matched ECHO continuation.** From immutable v51
  update 2, two anchored 49,152-decision arms retained continuing-reset returns and
  decision cost `0.01`; only ECHO differed (`0` versus `0.1`). Full diagnostic curves
  peaked at update 1/off 28,677 and update 1/on 32,439; on update 6 visited 23 aggregate
  levels but ranked 30,652. All remain below v51's 39,956, so heldout-v5 stays untouched.
  ECHO-on reduced its auxiliary loss and was directionally stronger here, but did not
  prevent late PPO regression. Decision: retain ECHO as a contextual objective and
  next revisit anchored Berserk acquisition from its stable selective checkpoint.
- **2026-09-08 10:41 EDT — adversarial startup reliability.** Twenty concurrent
  diagnostic games caused one transient startup timeout, while a verbose training root
  failed only after four completed updates when rotation reached a longer case label.
  Decision: remove case labels from bounded worker socket paths, type startup counts and
  indices, and retry evaluation starts in isolated attempt directories. The hardened
  evaluator recovered the missing snapshot on its first run.
- **2026-09-08 10:45–11:10 EDT — anchored Berserk under the stronger objective.**
  Continuing-reset decision-cost PPO reused two v39 selective-pretraining snapshots.
  Conservative update 4 never invoked abilities after joint training and peaked at
  28,398 diagnostic. Invocation-capable update 8 produced 17,997 training-cycle hits
  and deterministic `abilities → menu-a → wait` repetition (411/410/766 actions in one
  snapshot), peaking at 11,700. A matched two-action-history arm still recorded 17,338
  cycle hits and peaked at 29,115; its best snapshot invoked abilities/menu-a 321/319
  times. All are below v51's 39,956 and received no heldout-v5 access. Decision: retain
  the checkpoints as evidence that short requested-action history is insufficient;
  encode player-visible action outcome/availability before further joint training.
- **2026-09-08 11:10–12:10 EDT — parallel-agent seam trial.** Two isolated worktree
  agents ran while the primary agent completed the v57 train/evaluate curve. The broad
  performance agent found a byte-identical semantic-cell cache that reduced median
  replay preprocessing from 2.514 to 1.953 seconds (22.3%). The watch agent verified
  diagnostic-v2 resolves to v51, added manifest/suite/policy/checkpoint identity plus
  pause/step/quit controls, and removed two deprecated ambiguous Poe aliases. Main
  reviewed and integrated both commits; one documentation conflict was trivial, and
  the combined host gate passed. Neither agent needed heldout-v5 or an added flake
  dependency. Decision: the isolated-analysis/owned-component seam paid for itself;
  retain primary merge authority and expand parallelism where files and evidence can
  remain similarly independent.
- **2026-09-08 12:10–12:45 EDT — affordance evidence and training renunciation.** A
  read-only parallel audit of all v56/v57 diagnostic snapshots partitioned 4,804
  Berserk selections exactly into 76 starts, nine stochastic failures, 4,271 active
  rejections, and 448 cooldown rejections. All 4,719 state rejections consumed no game
  turn. Enabled menus had no colour field; state-disabled Berserk was visibly colour 8.
  Feature v4 also aliased structured `-Berserk` cooldown with active rage. Separately,
  retained training logs rendered the Renounce Religion confirmation 1,356 times;
  this is a redraw-sensitive proxy rather than a direct action count. Six renunciations
  completed, and all six games died to monsters created by Trog's wrath on D:1–6.
  Decision: add feature-v5 exact phase/applicability/outcome semantics, teach scripted-v8
  to cancel visibly inapplicable Berserk without changing the syntactic mask, and retain
  a typed `poe audit-affordances` utility for future cheap probes. A root-cause state fix
  resets the compute prior: start below the prior 49,152-decision budget and scale only
  if the failure mode changes and the diagnostic curve earns it.
- **2026-09-08 12:45–13:00 EDT — reduced-budget outcome probes.** Two feature-v5 arms
  each used 12,288 decisions from invocation-capable v39 update 8, one quarter of the
  previous budget. Ordinary continuing PPO v58 ranked only 9,532/6,459 at updates 1/2;
  update 2 still made 619 zero-turn rejected Berserk selections out of 624. Selective
  imitation-only v59 trained only appended semantic columns plus abilities/menu-`a`/
  cancel rows. Its ranks were 4,270/4,732. Update 1 still made 459 rejected selections;
  update 2 cut rejection to one but shifted into 697 ability-menu opens and 689 cancels,
  including 241 of 246 applicable Berserk menus. Neither approached v51's 39,956 and
  heldout remained untouched. Decision: do not scale either arm. Applicability is
  learnable, but square-root teacher balancing collapses the coupled choice to cancel;
  use a matched full-inverse selective probe before changing representation again.
- **2026-09-08 13:00–13:31 EDT — full-inverse affordance repair and the latent
  Renounce competitor.** The matched v60 probe changed only teacher balancing from
  square-root to full inverse while selectively training feature-v5 columns and the
  abilities/menu-`a`/cancel rows from v39 update 8. Its diagnostic snapshots ranked
  20,739 and 20,869. They opened seven ability menus and selected Berserk seven times:
  five starts plus two zero-turn active-state rejections. This is evidence that the
  representation and balanced target can express the flow, but not that misuse or
  survival is solved. Two 12,288-decision probes then applied the selective repair
  directly to canonical v51. v61 left the frozen menu-`X` row outside the update: its
  first snapshot exactly retained v51's 39,956 diagnostic rank, but its second opened
  338 applicable menus and selected Renounce 336 times, ranking 31,507. v62 added the
  menu-`X` row to the trainable set. Its first snapshot again retained rank 39,956; its
  second reduced Renounce to 225 selections and ranked 36,530, but selected Berserk
  zero times. A two-update v63 continuation crossed the inherited action margin but
  transferred rather than solved the loop. Its completed update-2 diagnostic ranked
  13,465: 864 menu opens led to 861 Berserk selections, six starts, two stochastic
  failures, and 853 zero-turn active-state rejections. An earlier partial update-1
  diagnostic ran while the host applied a CPU-heavy system configuration; its wall
  time and startup timeout carry no performance meaning. Decision: no heldout-v5
  access and no brute-force continuation.
- **2026-09-08 13:31–13:45 EDT — menu-logit and anchor-coverage audit.** Across the
  25,733 frozen v19/v20 anchor transitions, scripted-v8 relabeling supplies 4,308
  `abilities` targets but zero menu-`a` targets, zero menu-`X` targets, and zero
  ability-menu observations. Menu-`a` is legal as a non-target in 16 unrelated menus,
  so cross-entropy can suppress it; full-inverse balancing assigns weight only to
  present target classes and cannot create the missing positive class. On the same
  five applicable menu states, v51's mean `a-X` logit margin was -15.795, moving through
  -13.969 at v61 update 2, -10.866 at v62 update 2, -5.304 at v63 update 1, and +1.131
  at v63 update 2. Direct v5-feature contributions were only about +0.28–0.42 for `a`
  and -0.02–+0.17 for `X`, so shared-feature drift was secondary to the inherited
  margin and missing data. The audit also caught AdamW decay leaking into nominally
  frozen parameters because boolean-indexed `.copy_()` mutated a temporary. Decision:
  restore frozen rows/columns with indexed assignment and a bit-exact optimizer-step
  test. Before more training, seed non-heldout replay with both visible
  applicable→menu-`a` and inapplicable→cancel states, record per-class target and
  legal-exposure histograms in checkpoints, and assert that a one-minibatch update
  improves the `a-X` margin without moving unowned parameters.
- **2026-09-08 13:11–13:31 EDT — burst-aware UI interaction objective.** An isolated
  component agent implemented a per-worker token bucket that gives two free UI
  interactions and refills 0.25 token per actual game turn, then optionally charges
  zero-turn interactions after exhaustion. Abilities, menu selection, and cancel are
  explicit members of an extensible interaction set. All units and mutable balances
  are semantic types; checkpoint and update telemetry record the knob and overflow
  incidence. The overflow cost defaults to zero, and evaluation, masks, and raw
  environment returns are unchanged. Decision: integrate the mechanism now, but delay
  its matched cost comparison until the direct Renounce-logit failure and current host
  load are resolved. Keep overflow detection active at zero cost so an unperturbed run
  provides counterfactual incidence for sizing the later experiment.
- **2026-09-08 13:15–13:47 EDT — anchor-preload performance handoff.** A structural
  profile attributed 25.03 of 31.96 sampled seconds to recursive deep copies while
  reconstructing 2,208 anchor states. A dedicated transient replay reducer cut a
  matched ten-trajectory sample from 12.58 to 2.90 seconds while preserving feature,
  legality-mask, and teacher-label hashes; its content-addressed warm cache
  loaded in 0.31 seconds. Absolute timings overlap the host rebuild, so the profile and
  paired equivalence are stronger evidence than wall-clock totals. Candidate commit
  `c6a9057` is published on `agent/performance-round2` but deliberately unmerged because
  it overlaps newer PPO work on main. Decision: rebase/review it in the fresh session,
  retaining explicit cache-contract invalidation and preparation telemetry.
- **2026-09-08 — audited conditional ability acquisition.** Eight existing training
  seeds contributed 1,600 deliberate-exposure decisions (`artifacts/c/ac1`): 33
  applicable and 154 inapplicable ability menus, 14 missing-choice menus, 33 Berserk
  selections (32 starts/one stochastic failure), 168 cancels, and no active/cooldown
  rejection or Renounce selection. Balanced masked CE from immutable v51 update 2
  fitted only menu-a/menu-X/cancel rows and appended feature columns, with six training
  and two validation episodes. The initial gate compared cancel with a even though X
  dominated both contexts; it incorrectly stopped an update improving the true choice.
  The corrected gate measures target minus strongest legal competitor while retaining
  pairwise telemetry. An intervening margin-auxiliary experiment was abandoned and not
  used for selection. Canonical unchanged-CE attempt `artifacts/c/ability-probe-v4`
  completed its predeclared 1/16/64/256 steps in about 4.5 seconds. Step 64 was first
  to validate perfectly (9/9 applicable, 38/38 inapplicable). All unowned parameters
  remained exact, but 58/1,399 non-ability-menu actions changed in the collected data;
  preserving parameters is insufficient to claim behavioral preservation.
  The one selected diagnostic check (`artifacts/e/menu64-diagnostic`) matched v51's
  complete `(0, 0, 39956, 22, 22, 21, 211.0)` rank. Four sequences were identical;
  case 202 added one cancel. No abilities opened. Decision: conditional menu choice
  is learned with no aggregate diagnostic regression; next test useful opening in a
  bounded arm. Do not promote or scale from menu accuracy alone.
- **2026-09-08 — broader v51 development baseline.** Locked sixteen new disjoint
  development seeds with 1,000-decision horizons before candidate comparison. V51's
  `artifacts/e/v51-dev16` result ranks `(0, 0, 70665, 59, 59, 44, 409.0)` in 49.78
  seconds (6,981 decisions): ten deaths, six truncations, maximum actual D:6 and XL5.
  A manifest-wide replay audit found every truncation in a loop; repeating suffixes
  consume 5,195 decisions (74.4%). Two involve terrain obscured/misclassified by glyphs,
  two weapon-warning denial cycles under transformation, and two movement oscillations.
  All relevant terrain/form/unarmed protocol fields already exist. Decision: retain
  this fixed broader comparison and investigate terrain-aware navigation separately
  from ability training; detailed evidence is in `docs/research-v51-frontier.md`.
- **2026-09-08 — replay preparation integration and encoder profile.** Reviewed the
  pending performance candidate against current selective-update/UI-budget semantics.
  Corrected cache invalidation, corruption handling, and reconstruction ordering before
  integration. On the same 1,600 curriculum rows, preparation took 1.560 seconds in the
  reference loader, 0.497 transient, 0.518 cold cache, and 0.0102 warm cache. All feature,
  mask, and teacher-label arrays matched exactly. Seven alternating feature benchmarks
  separately measured redundant-presence-scan removal at 0.8427→0.7628 seconds (9.5%),
  with exact equality over 6,400 version-2–5 vectors. A 768-decision, 48-worker PPO smoke
  verified checkpointed anchor/replay counts (1,600/2,368; 33 positive anchor menu-a
  targets). Smoke timing overlapped diagnostic evaluation and is not a scaling result.
  Decision: use cached preparation and measure the current operational worker knee;
  stop spending time on the non-dominant fast-test startup cost per user direction.
- **2026-09-08 14:45 EDT — predeclared opening-only pilot.** Starting from the first
  perfectly validating conditional-choice snapshot (ability-probe-v4 step 64), train
  only the existing abilities action row; keep the repaired menu rows and all feature
  columns frozen. Use the same v19/v20 anchors plus ac1, full-inverse teacher balancing,
  masked imitation weight 1, learning rate 1e-4, four epochs per update, 48×128
  decisions per update, at most two updates. PPO/value/ECHO/entropy weights are zero
  for this isolated DAgger test. Retain continuing-reset/decision-cost metadata but
  do not attribute an imitation-only result to those unused return objectives.
  Evaluate the two scheduled snapshots only on existing diagnostics. Scale or test
  broader development only if a snapshot starts Berserk, makes no Renounce or
  active/cooldown-rejected selections, and retains at least the incumbent full rank
  and its one surviving horizon. Otherwise stop this arm and prioritize independently
  measured navigation failures. No heldout access. Launch follows the uncontended
  worker sweep; code and outputs will identify the exact attempt.
- **2026-09-08 14:41–14:44 EDT — current worker knee.** An uncontended sequential
  24/48/64-worker sweep at integrated revision `86464b1` used immutable v51 update 2,
  online-train-v3, fixed 128-decision chunks, and the same 27,333 v19/v20/ac1 anchors.
  It completed 17,408 decisions in 163.49 seconds overall. Collection rates were
  137.51, 161.73, and 138.36 decisions/s; collection took 22.34/37.99/59.21 seconds
  versus optimization 0.68/0.92/1.06. Cold preparation on the first arm took 26.965
  seconds; warm preparation on later arms took 0.302/0.307. Worker count changes seed
  coverage and inference shape, so these are operational rates, not identical-workload
  speedups. Commands, source/input hashes, losses and caveats are recorded in
  `artifacts/p/astra-scaling.json`. Decision: retain 48 workers; the clear operating
  choice does not warrant another sweep now.
- **2026-09-08 — opening-only pilot outcome.** V65 completed 12,288 decisions in
  about 73 seconds of update time with warm anchor preparation in 0.286 seconds.
  `poe checkpoint-audit` verifies only the abilities row changed; the repaired menu
  rows and encoder remained exact. Update 1 retained v51's diagnostic rank and opened
  no abilities. Update 2 opened 331 menus: five applicable selections produced five
  Berserk starts, while 326 inapplicable menus produced 325 cancels and one horizon
  boundary. No Renounce or state-rejected selections occurred. Its diagnostic rank
  fell to `(0, 0, 26016, 19, 19, 16, 152.0)`. Decision: conditional acquisition worked,
  but opening timing moved the loop to open/cancel and failed the predeclared quality
  gate. Stop this arm; retain it for a later matched UI-burst-cost hypothesis and move
  to the independently evidenced terrain failures. No broader or heldout evaluation.
- **2026-09-08 — predeclared terrain-only zero-update pilot.** Feature spec 6 keeps
  v5's width but explicitly reinterprets traversal hints using verified minimap wall
  categories and grounded lava. It updates local passage, adjacent navigation, and
  BFS traversal only; masks, rewards, policy weights, flora handling, prompts, and
  history do not change. Flight removes the grounded-lava objection without asserting
  universal passage; uncertain categories retain the old glyph fallback. Specs 2–5
  remain reproducible. Migrate immutable v51 directly (not the menu-repaired branch),
  evaluate the fixed five-case diagnostic once, and run the sixteen-case development
  comparison only if rank does not regress. Primary evidence is useful discovery,
  XL, deaths, and the known blocked-state suffixes, not fewer loops by itself. A
  failed zero-update arm does not justify automatic long fine-tuning.
- **2026-09-08 — terrain-only outcome and diagnostic promotion.** V66's fixed
  diagnostic ranks `(0, 0, 42853, 24, 24, 21, 217.0)` in 14.64 seconds. The predeclared
  broader comparison ranks `(0, 0, 97476, 66, 66, 46, 450.0)` in 34.38 seconds:
  discovery rises 37.9%, both terrain loops escape, but deaths rise 10→13 and maximum
  depth remains D:6 (maximum XL5→6). Three horizon loops remain. Decision: promote
  only the diagnostic leader from its retained summary with `poe promote-diagnostic`;
  prioritize survival and remaining prompt/oscillation failures, not a held-out claim.
  `poe watch-diagnostic-grid --no-animate --frame-limit 1` verifies the selected
  terrain-v66 replay. Heldout-v5 was not evaluated or changed.
- **2026-09-08 — static game-data startup cache.** Matched two-seed probes measured
  cold resets at 5.802/5.928 seconds versus 0.726/0.732 seconds after private static
  cache copies; complete validated-helper startup was 0.980 seconds. Initial state
  and twenty transitions matched exactly. Cached v51 diagnostic action sequences
  and full rank also matched; its 12.35-second wall time was not an uncontended
  speedup benchmark. Cold/cached compatibility passed on 0.34.1
  (`1eebc1a2892e1c89776a0d7a10691f8dac8d9796`) and 0.33.1
  (`9cb173b281c11a5177f40b8c0662bacd3aac2717`), with separate content-validated snapshots
  and verification reports under `artifacts/p/static-cache/`. Decision: retain
  explicit opt-in caching and remeasure representative rollout scaling before
  changing the worker default; no player saves or shared writable caches are reused.
- **2026-09-08 15:16 EDT — predeclared matched UI-return pilot.** Start both arms
  from immutable v65 update 2 migrated explicitly to feature6 as
  `checkpoints/ui-cost-v67-start.pt`; evaluate this zero-update diagnostic control
  separately from terrain-v66. Train only the abilities row and value head; encoder,
  conditional menu rows, and ECHO head remain exact. The new critic opt-in defaults
  off and has actual optimizer preservation tests. Each arm uses two 48×128 updates,
  seed1, online-train-v3, four epochs/update, learning rate1e-4, clipped PPO1/value0.5,
  entropy0.01, imitation0.1 with full-inverse class balancing and the same v19/v20/ac1
  anchors, ECHO0, continuing-reset returns, and decision cost0.01. Only UI overflow
  cost differs:0 versus0.1, with capacity2/refill0.25. Both use validated static data
  caching. Total budget24,576 decisions, no automatic extension.
  Check exact selective ownership and inspect the two scheduled diagnostic snapshots
  per arm. Require real activations, no Renounce/state-rejected selections, and lower
  inapplicable open/cancel incidence per ordinary decision in the cost arm. Broader
  development is allowed only for a snapshot retaining v66 discovery (42,853) and
  improving XL or combat survival without replacing activity with UI loops. Otherwise
  stop; fewer menus alone does not meet the goal. First-update collection should match
  across arms; subsequent divergence is expected from the different objectives.
  Heldout-v5 stays untouched. Representative cache timing runs finish before this pair.
- **2026-09-08 — UI-pilot control and retreat coverage prerequisites.** The migrated
  opening-capable control retains rank `(0, 0, 26016, 19, 19, 16, 152.0)` in 12.49
  seconds:331 opens,5 successful activations,326 inapplicable menus,325 cancels,
  no Renounce or state-rejected selections. Its ac1 validation remains9/9 applicable
  and38/38 inapplicable correct; training remains20/24 and109/116, not perfect.
  These are qualifications to the repaired starting point, not new live success.
  Separately, all136 training trajectories contain447 upstairs-legal pre-action
  observations with valid HP, but minimum HP fraction is0.79365. No threatened
  low-HP examples exist at the predeclared25/50/75% bands. Decision: do not fit an
  upstairs escape row from these anchors; deliberate training exposure is necessary.
  Details: `docs/research-retreat-coverage.md`.
- **2026-09-08 — representative static-cache pair and prelaunch amendment.**
  `poe benchmark-startup-cache` ran matched48×128 single-update configurations from
  terrain-v66, with anchor priming31.313 seconds separately and warm per-arm
  preparation0.249/0.243 seconds. Off/on collection took39.02/50.82 seconds;
  wall44.556/56.152 and awaited child CPU635.891/147.05 seconds. Source/input hashes
  remained unchanged. Despite the CPU reduction, outcomes diverged:20/22 completed
  episodes,982/926 cycles,19/5 UI overflows, and8/10 model-state tensors changed.
  This is confounded operational evidence, not a matched-workload speedup or proof
  of cache-induced behavior change. Earlier deterministic parity probes do not
  establish stochastic48-worker parity. Report: `artifacts/p/v66cache/report.json`.
  Decision: do not enable cache by default or spend more throughput budget yet.
  Before launching the UI-return pair, amend both arms to cold startup and enable
  optional raw rollout recording with actual per-update collector checkpoints.
  The audit found existing PPO retained game logs but not replayable trajectories;
  aggregate counts alone cannot verify matched first-update collection. Budgets,
  objectives, success/stop gates, and held-out exclusion remain unchanged.
