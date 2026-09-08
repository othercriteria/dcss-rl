# Project status

Last updated: 2026-09-08

## Current state

Public repository: <https://github.com/othercriteria/dcss-rl>

The Nix/Python environment and upstream trunk build are working. The current local
DCSS checkout was validated at commit `96832895d0253f9d7290d370efe32bf0614679a8`
(`0.35-a0-999-g96832895d0`) with a WebTiles build.

Canonical evaluation uses `mibe-heldout-v5` and rank v5. The fallback-free
`continuing-decision-v51` update-2 champion ranks
`(0 wins, 0 runes, 16,234 depth-weighted discovered cells, 18 levels,
max-depth sum 18, XL sum 14, 129.0 reward)`. This is the freshly calibrated v5 floor;
on now-retired v4 it scored 17,613, above v29's 16,853 and the locked pre-intervention
v23 floor of `(0, 0, 8,822, 13, 13, 9, 64.0)`. v21 first earned promotion on heldout-v2 at
`(0, 0, 2744, 2091, 12, 7, 31.0)`, exceeding that track's scripted-v3 incumbent and
the retired depth-11 learned milestone. Heldout-v2 was then retired because its traces
informed the next feature design; v3 was locked before that design was evaluated.

Development evaluation uses `mibe-diagnostic-v2`, extending the same known five seeds
from 200 to 500 decisions. Under rank v5, v23 scores 17,121, history-aware v26 scores
26,857, stateless v29 scores 34,118, and continuing-decision v51 update 2 scores
39,956 depth-weighted discovered cells. The current diagnostic leader is `terrain-v66`
at 42,853, using explicitly versioned terrain preprocessing with unchanged v51 weights.
The held-out champion remains v51; no new held-out evaluation was used for this change.

Completed:

- Reproducible Nix flake, Python 3.13 `uv` project, Poe commands, and direnv entry.
- Fetch/build/smoke scripts for trunk or a specified `DCSS_REF`.
- Direct Unix-datagram WebTiles transport with fragmented-message reassembly.
- Flush-delimited observation batches separating player-visible and control messages.
- Isolated per-game process lifecycle with private saves, morgues, macros, and logs.
- Live integration test that launches trunk and reaches a real input boundary.
- Stateful semantic reducer for sparse player/map deltas, inventory, messages, menus,
  and UI input mode; tested on a real Minotaur Berserker game.
- Repository hooks: commit-time Ruff auto-fixes and fast tests, plus a pre-push full
  gate that exercises a live DCSS process.
- Fixed Gym action catalog with reversible structured-action encoding and per-state
  boolean legality masks.
- Gymnasium environment that starts a real isolated game, completes MiBe hand-axe
  creation, steps structured actions, emits semantic dictionaries, and reports basic
  depth/XL/terminal rewards. Its live trunk test passes.
- Append-only, replayable JSONL trajectories containing revision/character metadata,
  exact emitted keycodes, wire-ordered raw protocol messages, semantic observations,
  and distinct ECHO-compatible policy/environment loss segments.
- A pinned `ty` type-check gate and a project-wide semantic-type discipline, with
  domain dataclasses/enums and TypedDict schemas at Gym/JSON boundaries. The checked
  source and tests carry no inline type-checker suppressions.
- Fast and live checks are split: 46 unit tests complete in about 0.23 seconds, while
  three isolated DCSS integration tests run concurrently in about 6.7 seconds.
- Deterministic scripted MiBe policy, fixed diagnostic/held-out suite manifests,
  concurrent evaluation runner, metric-vector ranking, and champion manifest writer.
  The first diagnostic run exposed and fixed automatic-input and level-up prompt
  boundaries.
- Frozen scripted MiBe v2 completed the untouched five-seed held-out suite: zero wins,
  zero runes, depth sum 5, XL sum 10, 3,516 game turns, and reward 50. All five cases
  reached the 500-decision truncation on D:1, making descent the clearest policy
  regression target. The replay set occupies 164 MiB, confirming compact semantic
  deltas are also a near-term throughput requirement.
- Compact trajectory schema v2 replaces repeated full snapshots with exactly
  reconstructible semantic patches while preserving raw WebTiles messages and distinct
  policy/environment loss segments. On the matched five-case 20-step workload,
  trajectory JSONL fell from 7.81 MiB to 1.48 MiB (81%).
- `poe watch-heldout-champion` resolves the champion manifest and renders a deterministic,
  player-centered terminal replay. It supports both the existing schema-v1 champion
  and new schema-v2 trajectories and defaults to manifest order, not a cherry-picked
  episode.
- Ordinary silent commands now use upstream's full-state request as an input-loop
  synchronization probe. This fixed 0.33.1 WAIT deadlock without modifying DCSS;
  automatic travel/rest retain their isolated typed quiescence fallback because they
  can consume control messages before completing.
- `poe compatibility-smoke` exercises semantic MiBe reset, a structured action,
  schema-v2 recording, and exact replay reconstruction against any binary. It passes
  on trunk plus 0.34.1 (`1eebc1a2892e1c89776a0d7a10691f8dac8d9796`) and 0.33.1
  (`9cb173b281c11a5177f40b8c0662bacd3aac2717`).
- `configs/heldout-regression-v1.json` locks the scripted-v2 held-out rank-v4 vector
  `(0, 0, 0, 2500, 5, 10, 50.0)`; `poe evaluate-scripted` enforces it before champion
  promotion.
- PyTorch 2.14.0+cu130 sees the RTX 4090 and completes CUDA tensor operations without
  a flake change. A versioned fixed-width semantic actor/value model, ECHO
  next-feature-delta head, legality-masked class-balanced imitation trainer, and
  scripted-relabel DAgger path now produce restorable checkpoints.
- Pure learned diagnostic iterations established the first behavioral/throughput
  evidence: naive cloning collapsed to autoexplore; class balancing diversified
  actions but hit walls; DAgger plus global player-visible navigation features reached
  depth sum 6 but still underperformed the scripted floor on XL. Five concurrent
  200-decision learned episodes take about two minutes; DCSS internal command work,
  not GPU training, is currently the dominant wall-time cost.
- Scripted-v3 fixes sparse monster handling, ignores positively identified stationary
  flora, and permits BFS to enter stair feature cells. Its diagnostic rank is
  Its depth-progress area is 1,019 over 851 decisions, providing a materially stronger
  multi-depth teacher.
- Canonical held-out and diagnostic leaders now have separate monotonic promotion
  tracks. `poe watch-heldout-champion` remains held-out-only;
  `poe watch-diagnostic-leader` exposes the current diagnostic leader without weakening
  champion hygiene. Their corresponding `*-grid` tasks animate all manifest seeds in
  parallel and mark death, horizon truncation, and ascension explicitly.
- The confidence-gated DAgger-v4 agent matches the depth-20 diagnostic expert while
  making 23.7% of diagnostic decisions neurally. It cleared the locked held-out floor
  and became canonical with maximum-depth sum 11 versus scripted-v2's 5. Its
  depth-progress area is 213 over 1,240 decisions, while the D:1-only baseline has
  zero over 2,500. Held-out learned-action coverage was 2.4%, an explicit
  limitation and the next optimization target. The canonical trajectory records
  checkpoint SHA-256 `e8e1f37e4fa28517a38a01c42eb79c0ba034a016366d39c9c9eb22a0f6b5975b`;
  its first manifest episode reaches D:2 at step 34.
- `configs/curriculum-v1.json` freezes the non-held-out expert bootstrap, four DAgger
  rounds, 0.98 diagnostic confidence gate, and locked held-out promotion stages.

Evaluation worker scaling on five 20-step diagnostic cases (2026-09-07):

| workers | wall time | speedup |
| ---: | ---: | ---: |
| 1 | 39.2 s | 1.00x |
| 2 | 22.5 s | 1.74x |
| 5 | 10.1 s | 3.90x |

Learned hybrid scaling on five 50-step diagnostic cases (250 decisions total,
identical deterministic rank and 25.2% neural coverage, 2026-09-07):

| workers | wall time | decisions/s | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 84.50 s | 2.96 | 1.00x |
| 2 | 45.07 s | 5.55 | 1.88x |
| 5 | 20.61 s | 12.13 | 4.10x |

The near-linear process scaling and sub-30-second GPU training iterations show that
broader rollout suites and longer horizons are affordable next steps. GPU capacity is
not the current bottleneck.

- Online legality-masked PPO now supports concurrent seeded rollouts, GAE, clipped
  policy/value losses, optional scripted imitation and ECHO auxiliaries, portable
  checkpoints, and per-update throughput/loss telemetry. A matched 2,560-decision
  ECHO-on/off pair was behaviorally identical; a 10,240-decision, 20-seed strong-ECHO
  run also held the pure diagnostic frontier at depth sum 6 / XL sum 8. These are
  working negative results: the next gain needs a denser learning signal or collector
  change, not merely more synchronous updates.
- Synchronous online rollout scaling at 5/10/20 workers was 4.53/5.63/7.29 decisions/s.
  Independent per-worker chunks remove the per-action straggler barrier: the matched
  20×64 workload reaches 44.66 decisions/s (6.13× faster), and an exact repeat produced
  bit-identical model tensors despite thread scheduling.
- Training environments can add typed, player-visible potential rewards for newly
  explored cells, branch-local depth, fractional XP progress, and HP preservation.
  Defaults remain zero, so headline evaluation rewards and old behavior are unchanged;
  selected weights are stored in PPO checkpoint metadata. Online scripted imitation
  is class-balanced so rare stair and tactical labels survive common explore actions.
- PPO checkpoints are atomically replaced after every completed update, workers rotate
  through training seeds rather than returning to their initial seed, and transient
  DCSS startup timeouts receive three fresh-directory attempts. These were added after
  a 20-seed experiment lost seven otherwise healthy updates at an episode restart.
- Champion rank v5 uses depth-weighted newly discovered cells and distinct levels after
  wins and runes, with maximum depth and monotonic XL as later frontier measures.
  Lingering, repeated commands, already-seen backtracking, policy steps, and raw game
  turns add no credit; useful exploration after retreat does.
- DCSS can abort when an overlong run directory produces a Unix socket path beyond
  Linux's 107-byte payload limit. Managed games now reject such paths before launch
  with a clear instruction to shorten the output root.
- Evaluation reports each case immediately on completion while retaining manifest
  ordering for summaries and ranks. Champion-track activation validates suite/rank
  versions, archives the prior canonical manifest, and atomically cuts all heldout
  viewers to the selected replacement; `watch-heldout-grid` now resolves heldout-v5.
- Track-specific completed-replay viewers now print resolved suite, policy, checkpoint,
  manifest, case, and outcome identity and offer quit, pause/resume, and frame-step
  controls on interactive terminals. Ambiguous transitional Poe watch aliases are gone.
- Automatic commands now prefer upstream's flushed `input_mode=1` boundary and retain
  the typed 500 ms quiescence only as a compatibility fallback. Delay-free documented
  DCSS RC options remove presentation sleeps from local agent play. The identical
  five-seed, 2,500-decision learned evaluation improved from 311.11 seconds (8.04/s)
  to 16.18 seconds (154.48/s), a 19.23× speed-up with exactly the same rank.
- Post-change asynchronous PPO collection scales from 36.93 decisions/s at five
  workers to 63.94 at ten and 97.93 at twenty, then declines to 93.28 at forty.
  Before the idle-CPU fix, twenty workers were the measured knee and 2.19× faster than
  the prior matched collector. `online-train-v2` expands training from 20 to 64
  disjoint seeds and its
  per-episode horizon from 500 to 1,000 decisions. Episode rotation strides by worker
  count, so concurrent workers consume disjoint seed blocks before wrapping rather
  than shifting into nearly complete overlap.
- A matched 40,960-decision broad-seed v6 ablation did not materially improve the
  fallback-free frontier. On diagnostic-v2, v5-off, v6-on, and v6-off reached
  depth-progress 1,385, 1,351, and 1,391 respectively; all survived 2,500 decisions,
  but three of five seeds remained on D:1. ECHO-off was cheaper and slightly stronger,
  but its +0.4% over v5 is noise rather than evidence for more of the same recipe.
- Structured WebTiles `menu` prompts now retain their visible title, item labels, and
  accepted hotkeys alongside widget-style menus. This removed a 934-action cancel
  loop and reduced v6-off diagnostic-v2 replay from 125.12 to 21.00 seconds without
  changing rank. The revealed failure is a repeated rest/confirm cycle; PPO telemetry
  now reports teacher agreement and imitation loss explicitly for targeted tuning.
- A 10,240-decision imitation-weight probe raised online teacher agreement from 42.2%
  at weight 1 to 47.6% at weight 10, but diagnostic-v2 depth-progress regressed from
  1,383 to 920. Agreement is useful optimizer telemetry, not a quality proxy; the
  stronger arm needs a full-budget test before changing objectives again.
- Full-budget strong imitation improved diagnostic-v2 depth-progress to 1,795, but its
  one locked heldout-v2 evaluation remained D:1 on every seed and failed at
  `(0, 0, 0, 2500, 5, 6, 10.0)`. Replay showed the policy choosing rest 769 times where
  its teacher requested rest only 14 times. PPO now exposes a typed teacher-balance
  exponent: 0 is unweighted, 0.5 (the default) is square-root inverse, and 1 restores
  the old full-inverse rule. The short square-root probe reduced rest/prompt collapse
  but shifted some mass to movement oscillation; its full-budget result is pending.
- Full-budget balance exponents 0.5 and 0.75 scored diagnostic-v2 depth-progress 1,263
  and 1,377. Square-root reached aggregate max-depth 13 but died early; the midpoint
  returned to passive survival. Neither beats full-inverse v8's 1,795, and v8 failed
  heldout. The next ablation separates online imitation from PPO/value gradients.
- A controlled current-build check measured 99.17 decisions/s at 20 workers and
  102.58/104.58 at 24. This pre-idle-fix check motivated using 24 workers for the next
  run; the later corrected sweep supersedes that operating point.
- A 49,152-decision online-imitation-only run reached 74.7% teacher agreement but
  failed diagnostic-v2 at `(0, 0, 150, 2101, 9, 8, 32.0)`, with four D:1 truncations
  and one D:5 death. Its 1,068 rests and 867 menu answers show that removing PPO/value
  gradients does not remove the rest-confirm attractor.
- Paused headless Crawl workers previously consumed a full core because WebTiles waits
  on stdin as well as its socket and `/dev/null` is permanently readable. The launcher
  now keeps an unwritten stdin pipe open; an isolated worker accumulated no additional
  CPU time over a 12-second paused interval after startup. A matched post-fix sweep at
  24/32/40/48/64 workers reached 137.12/145.24/150.57/154.03/149.70 decisions/s, moving
  the measured collector knee to 48 workers.
- A bounded asynchronous inference service combines worker requests without a game
  step barrier. At 48 workers its one-millisecond window realized mean batches of 18
  and improved matched throughput to 166.18 decisions/s; a three-millisecond window
  was slightly worse at 165.13/s. Batched and unbatched model tensors were bit-identical.
  Per-update telemetry now separates collection, optimization, checkpointing, batch
  count, and mean batch size; collection remains over 97% of measured update time.
- Unweighted online imitation removed the rest-confirm collapse but shifted to 1,009
  explore actions and 362 opposing NW/SE moves. Its diagnostic-v2 rank
  `(0, 0, 973, 1600, 11, 9, 38.0)` is below the learned frontier and included two early
  deaths. The weighted and unweighted endpoints bracket distinct failure modes.
- Offline imitation previously reconstructed a broader legality mask than live play,
  including stair commands without a visible under-player stair affordance. Semantic
  observations now round-trip from trajectory dictionaries and both paths use the same
  action-mask implementation.
- Scripted-v4 handles level-up, more, confirmation, shop, and unknown menus explicitly.
  This removed 7,242 repeated first-item shop selections from a 64-seed expert
  curriculum; the clean recollection contains 28,224 decisions, 42 deaths, 22
  truncations, and aggregate max-depth 226. Death trajectories are retained as useful
  deeper-state feedback rather than teaching horizon survival as a dominant objective.
- Online teacher imitation now implements actual DAgger aggregation: PPO/value/ECHO
  remain on-policy, while imitation samples all learner-visited labels accumulated in
  the run. At four fit epochs this removed rest/menu collapse, survived all diagnostic
  horizons, and ranked `(0, 0, 1420, 2500, 8, 8, 39.0)`. Sixteen epochs raised agreement
  to 74.3% and reached D:5/D:3 in two cases but incurred one death, ranking
  `(0, 0, 1167, 2152, 11, 8, 38.0)`. Explicit action/history memory is now a stronger
  hypothesis than further coarse imitation-weight sweeps.
- Scripted-v6 recognizes floor/ceiling escape hatches as stair affordances, advances
  blocking poison/fire states, and handles broader nearby-monster wording. Its clean
  64-seed/1,000-step curriculum produced 9,123 decisions, 63 deaths, one truncation,
  and aggregate max-depth 257. Death/reset trajectories remain useful feedback; the
  terminal trajectories remain useful feedback rather than grounds to optimize inert
  horizon survival.
- Unweighted behavior cloning followed by eight-update aggregate DAgger produced the
  fallback-free `scripted-v6-dagger-v21` policy. It led diagnostic-v2 at
  `(0, 0, 3122, 2500, 12, 10, 71.0)`, survived every case deterministically, then
  cleared heldout-v2 and became the canonical learned champion.
- Feature specification v3 adds explicit shop/more/prompt, completed-wait/explore,
  lethal-poison, on-fire, and nearby-monster inputs. Checkpoint loading and online
  collection preserve v2 widths, so promoted v21 remains reproducible. A v3 clone
  improved diagnostic depth area to 3,472 but failed the freshly locked heldout-v3
  floor at `(0, 0, 528, 1687, 8, 8, 19.0)` and was not promoted.
- Canonical track activation archived heldout-v2 and installed heldout-v3 without
  candidate-informed seed selection. Diagnostic evaluation is non-promoting unless a
  champion path is explicit, preventing cross-suite manifest comparisons.
- Explicit death reward was removed before PPO/value optimization. With discounting,
  a negative terminal reward pays the agent to delay an unavoidable death and a finite
  horizon can erase the charge entirely. Death still ends the episode at zero terminal
  value and forfeits future progress; terminal trajectories remain in the curriculum.
- PPO now bootstraps value at administrative time limits while cutting GAE recursion
  across the reset boundary. Rank-v5 recalibration scored v21 at 11,483 diagnostic and
  15,677 heldout depth-weighted discovered cells. Feature-v3 DAgger v23 scored 17,121
  diagnostic and cleared the locked heldout-v3 floor at 16,202, becoming canonical;
  its three heldout deaths retain their prior progress rather than erasing it.
- Policy-side action history is now checkpointed separately from semantic observations.
  Evaluation and rollout workers maintain isolated per-episode histories; ECHO targets
  remain environment-only. `--action-history-length` expands a stateless checkpoint
  with zero history columns so online DAgger/PPO can learn loop-sensitive behavior.
- Online training can retain immutable `update-NNNN.pt` snapshots alongside its rolling
  checkpoint, enabling diagnostic learning curves and recovery from non-monotonic PPO.
  The disjoint `online-train-v3` curriculum expands to 128 seeds and 2,000-step games.
- A doubled 98,304-decision DAgger run on that curriculum demonstrated strongly
  non-monotonic policy quality: its 16 diagnostic snapshots ranged from 15,974 to
  45,783 depth-weighted cells. Update 16 reached D:7 twice and aggregate XL 24, but
  failed heldout-v4 narrowly at 8,704 versus the 8,822 floor. Snapshot writes cost only
  about 0.01 seconds/update; late collection remained the dominant cost at 127–135/s.
- The append-only action catalog now exposes the ordinary ability-menu command at tail
  index 270 without shifting any legacy command or menu-key index. Old policy heads
  and action-history slots migrate automatically. Scripted-v7 selects visible Berserk
  and waits out targetless Berserk instead of no-op autoexploring; its corrected
  diagnostic rank is 23,509 and it reaches D:7.
- Direct end-to-end Berserk DAgger catastrophically forgot the incumbent: all 16
  diagnostic snapshots scored at most 8,107. A 10× lower learning rate preserved
  more behavior (best 35,249) but selected Berserk zero times and failed heldout at
  13,348 versus champion v29's 16,853. New-action warmup can now train only appended
  head rows while preserving the established encoder and logits.
- Berserk acquisition experiments isolate two recurrent failures. Appended-row-only
  warmup learned to open abilities but selected Renounce Religion; adding the dependent
  menu-`a` row learned the real two-decision Berserk flow, then retried while Berserk.
  Feature v4 now exposes visible Berserk/exhaustion, and selective warmup can train only
  those new inputs plus declared action rows. Replay preloading relabels 25,733
  transitions from 128 non-heldout v19/v20 trajectories and records their paths.
- Selective v39 learned a non-repeating Berserk invocation while leaving v29 behavior
  exactly unchanged through update 4. Its update-8 invocation was tactically premature
  and ranked 7,035 diagnostic. Anchored `1e-6` joint v40 recovered one D:7 episode but
  peaked at 11,115, below v29's 34,118; neither received heldout access. These results
  are local to the current representation/data/objective regime, not permanent verdicts
  on pretraining, class balance, status features, or joint DAgger.
- Heldout-v3 was retired after its v24 traces motivated action history; v26's apparent
  promotion there is not treated as headline evidence. Untouched heldout-v4 was first
  calibrated with pre-intervention v23 at 8,822. The already-frozen history v26 and
  stateless control v29 then scored 12,910 and 16,853 respectively; v29 became the
  canonical champion. This preserves the locked-suite boundary and shows additional
  DAgger, not history alone, drove the strongest general improvement.
- The history-aware matched PPO repeat again promoted neither arm. ECHO-on learned its
  target and scored 21,009 diagnostic versus ECHO-off's 18,302, but both regressed from
  their v26 initialization at 26,857 and received no heldout access.
- PPO now supports continuing-reset death returns: death bootstraps a fresh training
  reset while cutting GAE across the episode boundary, administrative limits bootstrap
  their final visible state, and wins remain terminal. Independent typed per-decision
  and exact semantic short-cycle costs affect training returns only. Checkpoints record
  the boundary/cost contract and update logs report detected cycle incidence. The host
  gate passes 92 fast tests in 1.95 seconds and three live tests in 8.72 seconds.
- Matched cost pilots exposed rollout nondeterminism from variable GPU inference matrix
  shapes: tiny batch-dependent rounding differences could cross stochastic sampling
  thresholds. Padding alone left one divergent transition; assigning each worker a
  stable row made matched 768-decision collections identical (104 cycles and 68.0%
  teacher agreement) while retaining asynchronous coalescing.
- Three anchored, continuing-reset PPO arms each consumed 49,152 decisions on the
  128-seed/2,000-horizon training curriculum. Exhaustive diagnostic snapshot selection
  found best ranks 35,140 with no cost, 39,956 with decision cost `0.01`, and 31,991
  with semantic cycle cost `0.1`; all curves regressed sharply after early updates.
  The selected decision-cost snapshot cut exact semantic recurrence from v29's 56.1%
  to 28.6% and rest from 281 to 4 diagnostic actions. Its single heldout-v4 attempt
  scored 17,613 across 19 levels versus v29's 16,853 across 15, so v51 update 2 is the
  new canonical fallback-free champion. Four heldout games died on D:3–5 and one
  survived the horizon on D:2; it has not reproduced the retired true D:11 event.
- The isolated harness-performance subagent produced commit `143b2af7` without merging.
  Its evidence-backed changes were manually ported onto current return semantics:
  disabled shaping avoids full-map scans, Gym snapshots are materialized once, and
  online/offline training carry already encoded next features. The PPO path keeps
  terminal ECHO targets distinct from freshly reset continuing-task features. A
  current-main 48-worker sanity run collected 6,144 decisions at 149.45 decisions/s;
  unlike the controlled 2.4–3.8% samples, asynchronous trajectory divergence makes
  that single end-to-end rate unsuitable as a precise before/after estimate. The host
  gate passes 94 fast tests in 1.73 seconds and three live tests in 8.63 seconds.
- Before survival work could use v51's heldout-v4 failure pattern, disjoint heldout-v5
  was locked and calibrated exactly once. v51 scored `(0, 0, 16234, 18, 18, 14,
  129.0)` with one 500-decision survivor and four deaths; this became the checked-in
  floor and canonical track, while the v4 champion manifest was archived.
- Matched 49,152-decision continuations from v51 compared ECHO weight `0` and `0.1`
  under the promoted continuing-reset decision-cost objective. Exhaustive diagnostic
  curves peaked at 28,677 off and 32,439 on, both below unchanged v51's 39,956; neither
  received heldout-v5 access. ECHO-on again learned its auxiliary target and was
  directionally stronger than off in this context, but did not prevent PPO regression.
- A four-way diagnostic sweep exposed two harness edge cases: a training case label
  could exceed the Unix socket limit only after later episode rotation, and evaluation
  had no startup retry under transient 20-game contention. Worker paths now exclude
  unbounded labels, startup counts/indices are semantic types, and evaluation preserves
  three isolated attempts. The retry recovered the sole missing ECHO snapshot.
- The Gym-required `dict[str, Any]` reset/step signatures are isolated wrappers;
  application code uses typed reset/step methods carrying `EnvironmentInfo`,
  `ResetOptions`, `ActionIndex`, and `GameSeed`.
- Two stronger-context Berserk continuations reused immutable selective-pretraining
  snapshots. Starting from conservative v39 update 4 never selected abilities and
  peaked at 28,398 diagnostic. Starting from invocation-capable v39 update 8 retained
  the bad `abilities → a → wait` cycle and peaked at only 11,700, with 17,997 training
  cycle hits. Adding two action-history slots reduced neither the attractor nor its
  incidence (17,338 hits) and peaked at 29,115. None received heldout-v5 access. The
  next representation must expose action outcome/availability, not merely requested
  action history.
- The semantic reducer now normalizes changed map cells once and caches their typed,
  deterministic ordering while continuing to return detached snapshots. Replaying the
  501-boundary diagnostic-v2-404 trace five times fell from a 2.514-second median to
  1.953 seconds (22.3%); the complete observation-stream digest remained identical.
- Feature specification v5 preserves visible ability applicability, separates active
  Berserk from `-Berserk` cooldown, and encodes explicit success/rejection/failure/loss
  feedback without turning tactical applicability into an action mask. Scripted-v8
  cancels visibly inapplicable Berserk. A typed `poe audit-affordances` command audits
  raw trajectories plus retained Crawl logs/morgues; it confirmed 4,719 zero-turn
  state rejections in v56/v57 diagnostics and six of six completed training
  renunciations ending in deaths to Trog-wrath monsters.
- Stable promotion-suite/threshold, diagnostic-suite, and training-suite aliases now
  resolve to immutable versioned JSON artifacts. Routine CLI/Poe defaults no longer
  hard-code v5/v2/v3, while summaries retain the exact resolved suite identity.
- Two quarter-budget feature-v5 probes did not qualify. Ordinary PPO v58 retained 619
  zero-turn rejections in its better-trained snapshot. Selective imitation v59 reduced
  rejection to one but replaced it with 689 ability-menu cancels, mostly collapsing
  across applicable/inapplicable choices; its best diagnostic rank was 4,732 versus
  v51's 39,956. Selective warmup can now explicitly own existing structured-action rows
  as well as appended actions/menu keys, enabling a matched balance probe without
  unfreezing the incumbent network.
- Full-inverse teacher balancing made the feature-v5 affordance learnable from the weak
  v39 frontier: v60 selected Berserk on every opened menu, starting it five times across
  both diagnostic snapshots, but still made two visibly inapplicable selections and
  peaked at only 20,869. Applying the same selective repair directly to v51 exposed a
  different frozen-policy competitor. v61 update 1 exactly retained v51's 39,956 rank,
  while update 2 opened 338 applicable menus and chose Renounce Religion 336 times.
  Explicitly training the menu-`X` row in v62 reduced that to 225 selections and raised
  update 2 from 31,507 to 36,530, but learned no Berserk selections. A two-update v63
  continuation finally reversed the inherited menu margin but transferred the loop:
  update 2 ranked 13,465 with 864 menu opens, 861 Berserk selections, six starts, two
  stochastic failures, and 853 zero-turn active-state rejections. No arm qualified for
  heldout-v5.
- A logit/label audit identified missing curriculum coverage as the primary cause.
  The 25,733 frozen v19/v20 anchor transitions contain 4,308 `abilities` targets but
  zero menu-`a` targets and zero ability-menu observations; full-inverse weighting
  cannot balance an absent class. On five applicable menu states, v51 already preferred
  Renounce by a mean 15.795-logit margin. Selective training moved the `a-X` margin to
  +1.131 at v63 update 2 without learning applicability, explaining the transferred
  loop. The next replay must include player-visible applicable→`a` and
  inapplicable→cancel examples and record teacher-label/legal-exposure histograms.
- The same audit found that boolean-indexed `.copy_()` restored a temporary after
  AdamW, leaking weight decay into supposedly frozen action rows and feature columns.
  Selective warmup now restores through indexed assignment, with a test requiring
  unowned parameters to remain bit-exact after an optimizer step.
- A training-only token bucket can now cost bursty zero-turn UI interactions without
  changing evaluation, action masks, or environment reward. Capacity, token balance,
  refill per game-turn, overflow cost, and overflow count have semantic types; state is
  independent per worker episode and telemetry/checkpoints record the configuration.
  Its cost defaults to zero while overflow detection remains active, so baseline runs
  expose counterfactual incidence without silently changing the current objective.
- Performance candidate `c6a9057` is preserved and published on
  `agent/performance-round2`, not merged. It adds preparation timing, a transient
  anchor-replay reducer, and a content-addressed tensor cache; its paired sample fell
  from 12.58 to 2.90 seconds cold and 0.31 seconds warm with identical feature, mask,
  and teacher-label hashes. The fresh session should rebase and review it against the
  newer UI-budget and selective-freeze changes before integration.

Next:

1. Use the verified stair-boundary correction for the next bounded research step:
   investigate conditional-menu representation/exposure after the bounded row-only
   calibration failed, and collect deliberate combat/escape exposure before another
   opening-timing continuation. The v67
   UI-return pair failed safety/quality gates and is stopped. Terrain-v66 remains
   the diagnostic leader; actual D:11 progress and stronger survival remain open.
2. Address the independently measured blocked-action, prompt, and navigation loops
   that dominate the broader development baseline. Prioritize useful discovery and XL
   alongside fewer loops; do not count merely converting stalls into deaths as success.
3. Broaden validation before scaling successful pilots toward actual D:11 progress,
   then branch/rune acquisition. Heldout-v5 remains the promotion gate, separate from
   development design and eventual normal unseeded headline evidence.

## Astra handoff progress (2026-09-08)

- User direction: current-policy progress first, bounded local pilots before justified
  hour-scale runs, thin immediate research goals, Poe workflows, and continuing
  independently owned performance work. These durable preferences are in `AGENTS.md`.
- `poe collect-ability-curriculum --output artifacts/c/ac1` collected 1,600 decisions
  from eight existing training seeds. Raw audit: 33 applicable, 154 inapplicable, and
  14 missing-Berserk menus; 33 selections/32 starts/one stochastic failure, 168 cancels,
  no state-rejected selections, and no Renounce selections. These deliberate exposure
  games are training data, not a candidate benchmark.
- `poe ability-probe` isolates conditional menu choice with six training/two validation
  episodes, balanced masked cross-entropy, and predeclared steps 1/16/64/256. The
  canonical `artifacts/c/ability-probe-v4` run took about 4.5 seconds. Step 64 first
  reached 9/9 applicable and 38/38 inapplicable validation choices. Unowned parameters
  remained bit-exact, but 58/1,399 collected non-ability-menu actions changed (58/1,050
  training and 0/349 validation); parameter freezing is not behavioral preservation.
  Checkpoints include coverage counts, input hashes, code hashes, and measured drift.
  Its one diagnostic check ranks `(0, 0, 39956, 22, 22, 21, 211.0)`, exactly matching
  v51. Four action sequences are identical; case 202 inserts one cancel at step 104.
  No ability menus were opened, so this establishes conditional acquisition and
  diagnostic non-regression, not an autonomous gameplay improvement.
- A fixed sixteen-seed, 1,000-decision `development-validation-v1` baseline uses seeds
  disjoint from all earlier training/diagnostic/promotion suites. V51 ranks
  `(0, 0, 70665, 59, 59, 44, 409.0)` in 49.78 seconds, with ten deaths, six truncations,
  maximum actual D:6 and XL5. Every truncation ends in a loop; the six final repeating
  suffixes occupy 5,195/6,981 decisions (74.4%). No ability was opened in this suite.
- Reviewed and adapted the pending replay optimization rather than merging its stale
  PPO implementation. Transient replay, cold cache, and warm cache take 0.497/0.518/
  0.0102 seconds versus 1.560 seconds for 1,600 reference rows, with exact feature,
  mask, and teacher-label equality. Cache invalidation includes preprocessing sources;
  malformed/unavailable caches rebuild. History and continuing-return/ECHO boundaries
  remain outside the cache. PPO records anchor and actual replay coverage separately.
  A 48-worker, 768-decision plumbing smoke restored its saved checkpoint with 1,600
  anchor and 2,368 actual replay samples, including 33 positive menu-a anchor targets.
- A separate redundant-scan removal reduced paired feature-encoding time by 9.5%
  (0.8427 to 0.7628 seconds), preserving all 6,400 real vectors across feature versions
  2–5 exactly. These are controlled preprocessing measurements, not an end-to-end
  rollout speedup claim.
  The subsequent uncontended operational 24/48/64 sweep at fixed 128-step chunks
  measured 137.51/161.73/138.36 collection decisions/s over 17,408 decisions. Retain
  48 workers. Worker-dependent seed prefixes and batch shapes prevent interpreting
  these as identical-workload speedups. Full evidence: `artifacts/p/astra-scaling.json`.
- The fast suite currently takes about 2.5 seconds: torch import alone costs about
  1.2 seconds and first AdamW construction about 0.8–1.0 seconds. The original
  subsecond fresh-process target is unmet; tests have not been removed to hide this.
  Per user direction, further test-loop optimization is deprioritized in favor of
  rollout throughput and policy research. Integrated commit `86464b1` passes 147 fast
  tests, three live tests, lint, formatting, typing, and commit hooks.

- Terrain v66's zero-update diagnostic ranks `(0, 0, 42853, 24, 24, 21, 217.0)`;
  the fixed sixteen-case development comparison ranks
  `(0, 0, 97476, 66, 66, 46, 450.0)` in 34.38 seconds. Both known terrain-blocked
  loops and one oscillation escape, but thirteen deaths/three truncations and maximum
  D:6/XL6 do not establish the survival or D:11 milestone. `poe promote-diagnostic`
  promoted the retained diagnostic summary without rerunning games; a one-frame
  `poe watch-diagnostic-grid` check resolves terrain-v66 directly from dev-champion.
  Held-out and diagnostic viewers retain separate direct manifests, with no extra
  suite-redirect layer on the viewing path.
- Opt-in static DCSS data caching copies only validated upstream database/description
  caches into private saves. Paired startup probes fell from about 5.8 seconds to
  about 1.0 second including validation/copy, preserving initial and twenty-step
  semantic behavior. The cached v51 diagnostic reproduces all five action sequences
  and the full rank. `poe prepare-game-cache` and `--static-data-cache` expose the
  opt-in path; defaults are unchanged. Cold and cached compatibility smoke both pass
  on 0.34.1 and 0.33.1 at the exact revisions recorded below. Latest full gate:
  186 fast tests, three live tests, formatting, lint, and typing pass.
- Integrated terrain/cache changes are committed at `916408b`; optional selective
  critic learning and the predeclared UI-return experiment are committed at `9f0d08c`.
  A later matched 48-worker startup-cache pair does **not** justify default activation:
  collection regressed 39.02→50.82 seconds despite child CPU 635.89→147.05 seconds,
  and stochastic rollouts/model tensors diverged. Keep caching opt-in and investigate
  parity before further optimization. Exact report: `artifacts/p/v66cache/report.json`.
- The next bounded UI-cost comparison uses the same opening-capable v65→feature 6
  start, abilities-row plus critic learning, and cost 0 versus 0.1. Both arms use cold
  startup following the performance result. Optional raw PPO recording closes an
  evidence gap in earlier training logs: per-transition provenance references exact
  collector checkpoints, including across optimizer boundaries. Its 2×8 smoke passes
  and exact ownership allows only abilities/value-head changes. The completed pair
  is stopped: diagnostic curves 21,518→17,988 (off) and 24,271→28,479 (on) remain below
  terrain-v66. Every horizon survivor loops; stochastic training still samples
  Renounce/rejected choices despite exact 47/47 greedy ac1 validation preservation.
  No broader or held-out evaluation was run.
- A training-only retreat audit finds no low-HP upstairs examples among 447 legal
  states in 136 anchor episodes (minimum HP 79.365%). Do not relabel these as escape
  training; deliberate exposure is needed. `docs/research-retreat-coverage.md` records
  the negative coverage evidence.
- Recording/comparison tools and the full 217-unit/three-live gate are committed at
  `a0e25ad`. First-collection evidence confirms identical initial tensors and all
  RNG-state streams, but ten workers act on a premature downstairs boundary containing
  only mode 0+flush, before the next floor arrives. Eight other workers differ solely
  in cell colour with identical policy behavior. A scoped level-transition readiness
  correction and bounded frozen-policy control are next; static caching stays opt-in.
- The scoped stair-boundary correction is committed at `32bbea8` and passes 273 unit
  tests, three live tests, and both stable-release compatibility smokes. Two frozen
  cold 48×128 controls verify all 6,144 paired policy inputs/masks/probabilities/RNG
  states/actions/rewards/terminal boundaries identical, with every model tensor
  unchanged. All 140 actual stair actions per arm avoid the premature mode 0+flush
  return. Twelve workers still differ strictly in cell colour; those raw differences
  remain visible and explain short-cycle counts 1279/1273 (cycle cost was zero).
  Evidence: `artifacts/e/readiness68-policy-parity.json` and both frozen-model audits.
  Diagnostic rechecks retain terrain-v66's 42,853 and v51's 39,956 full ranks exactly.
  Neither champion manifest nor heldout-v5 was changed. Do not interpret this bounded
  proof as universal byte-identical DCSS output or as a new policy-quality milestone.
- A bounded offline row-only calibration v70 finishes 256 steps in 5.17 seconds but
  fails its predeclared probability/accuracy gates. Training targets become 140/140
  correct, while validation is 9/9 applicable and 36/38 cancel; mean validation target
  probabilities are 0.9466/0.8965, below 0.995. Only menu-a, menu-X, and cancel rows
  change; encoder, opening row, value/ECHO heads, and all unowned parameters remain
  exact. No games or further optimizer steps follow this failed pilot. V69 remains
  separately recorded as stopped after the older first-step margin gate. The new
  `poe ability-calibration` audit measures probability tails and preservation rather
  than treating greedy accuracy as sampled reliability.
- Static-cache preparation now offers opt-in wall/thread-CPU stage timing, without
  relaxing validation or enabling caching by default. Process/benchmark wiring and
  a corrected cache-off/on comparison remain pending; no speedup is claimed. The
  bounded next measurement is specified in `docs/research-throughput-next.md`.
- The integrated calibration/timing changes pass the full Poe gate: 293 unit tests
  in 3.63 seconds, three parallel live tests in 8.75 seconds, lint, formatting, and
  type checking. The audit and row-only protocol reject recorded incompatible
  action catalogs even though the general policy loader can expand them.

## Verified commands

```sh
poe dcss-fetch
poe dcss-build
poe dcss-smoke
poe compatibility-smoke
poe check
```

The installed hooks and both hook stages pass, including the live DCSS integration.
The compatibility trajectory smoke also passes on trunk `96832895d0`, 0.34.1
`1eebc1a2892e`, and 0.33.1 `9cb173b281c1` after the rank-v5 and return changes.

## Environment notes

The host has an RTX 4090 with 24,564 MiB VRAM and driver 610.57.04. PyTorch CUDA
training and inference have been exercised successfully. The managed Codex
sandbox may hide `/dev/nvidia*` and prohibit Unix socket binding; approved host-boundary
commands see both correctly. Training dependencies remain an optional `uv` group.
