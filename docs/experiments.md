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
