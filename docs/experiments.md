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
- **2026-09-07 16:25 EDT — asynchronous collector and dense-reward smoke.** The
  matched 20-worker × 64-step workload rose from 7.29 to 44.66 decisions/s (6.13×)
  after removing per-action barriers. A full repeat reached 44.63 decisions/s and
  produced bit-identical model tensors using worker-local seeded samplers. Exploration
  0.01, XP-progress 1.0, and HP-potential 1.0 yielded finite losses and one completed
  episode. Decision: use this foundation for controlled 20-seed dense-reward runs;
  further synchronous worker tuning is no longer the immediate bottleneck.
