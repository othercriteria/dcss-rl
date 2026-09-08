# dcss-rl

Research infrastructure and agents for learning to play
[Dungeon Crawl Stone Soup](https://github.com/crawl/crawl), with ascension as the
ultimate evaluation target and survival, experience, branch depth, and runes as
curriculum milestones.

The project controls an unmodified local DCSS build through public player-facing
interfaces. WebTiles is the primary structured transport; console play is retained as
a black-box compatibility check. Training may use seeded games and diagnostic
curricula, while headline evaluation uses ordinary unseeded games without privileged
state.

## Environment

Authorize the checked-in direnv configuration once; subsequent directory entries
activate the pinned Nix shell and project virtual environment automatically:

```sh
direnv allow
uv sync
poe hooks
poe check
```

The installed commit hook auto-fixes Ruff issues and runs fast tests. The push hook
runs the complete gate, including live DCSS process integration tests. `poe test` is
the subsecond unit-test loop; `poe check` additionally runs isolated live games in
parallel.

For non-interactive use and CI, the equivalent explicit shell entry is
`nix develop`.

Training dependencies are deliberately optional:

```sh
uv sync --group train
```

PyTorch wheels supply their own CUDA user-space libraries and use the host NVIDIA
driver. This avoids pinning the development shell to a second CUDA toolkit.

## Local DCSS

Fetch and build the canonical trunk checkout:

```sh
poe dcss-fetch
poe dcss-build
poe dcss-smoke
```

Set `DCSS_REF` to test another revision, for example
`DCSS_REF=0.34.1 poe dcss-fetch`. Upstream source and build products live under
`vendor/` and are not committed.

Run the same semantic reset, structured WAIT, compact-trajectory, and exact-replay
smoke against any built binary:

```sh
poe compatibility-smoke --binary vendor/crawl/crawl-ref/source/crawl
```

## Research direction

Initial experiments will compare a conventional recurrent policy with language-model
policies trained using policy gradients. ECHO-style environment prediction is a
first-class auxiliary objective: trajectories preserve action and observation token
masks so the same rollout can supervise both control and prediction of the next
semantic game-state delta.

Every evaluation record should identify the DCSS commit, configuration, seed policy,
agent checkpoint, reward specification, and terminal morgue/replay artifacts.

Run the transparent scripted baseline over the fixed held-out suite with:

```sh
poe evaluate-scripted
```

The evaluator defaults to five concurrent isolated games. Override `--workers` when
profiling another machine. Checked-in diagnostic seeds are separate from the untouched
held-out manifest; generated trajectories, game directories, summaries, and
`artifacts/champion.json` remain untracked.

Watch the first manifest-ordered episode from the held-out champion, or animate every
held-out seed together as a compact terminal grid:

```sh
poe watch-heldout-champion
poe watch-heldout-grid
```

Held-out and development leaders are deliberately separate. The scripted diagnostic
task promotes monotonically to `artifacts/dev-champion.json`; observe that track with:

```sh
poe evaluate-scripted-diagnostic
poe watch-diagnostic-leader
poe watch-diagnostic-grid
```

The grid tasks read `artifacts/champion.json` (held-out) and
`artifacts/dev-champion.json` (diagnostic) directly. They replay the selected evaluation;
they do not run the latest checkpoint or follow suite-config redirects. Generic
learned evaluations do not promote unless explicitly requested. To promote an already
completed evaluation of the complete current diagnostic suite without rerunning games:

```sh
poe promote-diagnostic --summary artifacts/e/terrain66-diagnostic/summary.json
```

This command requires retained replay files and a strictly better rank, and cannot
write the held-out champion. Suite redirects select evaluation inputs, not viewer
destinations; keeping the two champion manifests preserves the separate tracks.

For selective PPO experiments, `poe train-ppo --new-action-warmup-updates ...
--warmup-action-kind abilities --warmup-train-value ...` can train the named policy
row and critic while freezing the encoder and other action rows. Critic warmup is
opt-in. Verify the resulting ownership with `poe checkpoint-audit --checkpoint ...
--reference ... --allow-action-kind abilities --allow-value-head`; the audit rejects
unrelated tensor changes and incompatible model configurations.

Audit sampled ability-choice confidence on training-only replay with
`poe ability-calibration --checkpoint CHECKPOINT --trajectories artifacts/c/ac1
--output REPORT.json`. Repeat `--checkpoint` to compare candidates against the first
checkpoint. The audit uses each checkpoint's feature version and reports probability
tails as well as greedy accuracy; a correct argmax alone does not establish reliable
sampling. Its seed-disjoint split is development evidence, not held-out promotion.
Reports include worst-example trajectory hashes and zero-based pre-action frame
indices. For broader deliberate exposure, `poe collect-ability-curriculum --suite
configs/ability-curriculum-v2.json --workers 48 --output NEW_RUN` records 48 fixed
training-only seeds with a 200-decision cap; custom suites are checked against the
current training seed set and must have unique cases/seeds. This collector is not
a candidate policy or a survival evaluation.
`poe ability-probe --row-only-calibration ...` explicitly selects a fixed, bounded
offline calibration protocol with the encoder frozen. Finishing that protocol does
not authorize live scaling or certify its probability gates.

The terminal viewer identifies the resolved suite, policy, checkpoint, manifest, case,
and recorded outcome before reconstructing either legacy full-snapshot trajectories or
compact schema-v2 deltas. Pass `--case CASE_ID` to the track-specific Poe task to inspect
a labeled suite case; the default deliberately follows manifest order instead of
selecting a flattering showcase seed. During an animated replay, press `q` to quit,
space to pause or resume, and `n` to advance one frame while paused. Grid panels remain
on their final frame and visibly distinguish death (`☠`), horizon truncation (`◇`), and
eventual ascension (`★`). The obsolete ambiguous `watch-best` and `watch-dev-best` Poe
aliases have been removed; use the four track-specific names above.

Champion promotion is gated by the checked-in heldout-v5 regression floor. The
canonical champion is a fallback-free learned policy; Native WebTiles spectating can
later complement the deterministic completed replay.

Routine tooling names durable singleton roles rather than their current versions:
`configs/promotion-suite.json`, `promotion-threshold.json`,
`diagnostic-suite.json`, and `training-suite.json` are checked-in redirects to immutable
versioned artifacts. A track retirement changes the redirect while summaries and
trajectories continue recording the resolved suite ID. `artifacts/champion.json` and
`artifacts/dev-champion.json` remain the promotion- and diagnostic-track policy
manifests.

Heldout-v1 was retired after its failure traces directly informed contextual stair
masking. Its manifest remains archived locally for audit, but it is never reused for
promotion. A validated track-activation command atomically archives the old canonical
manifest and cuts every heldout viewer over to a calibrated replacement track.

The optional training stack provides a versioned semantic actor/value model with an
ECHO next-state-delta head. Feature-spec-v5 adds exact active/cooldown Berserk state,
visible ability-menu applicability, and player-visible outcome feedback while retaining
older checkpoint compatibility. Inapplicable choices remain syntactically legal in the
action mask; the policy must learn from their visible state and outcome. A
behavior-cloning or scripted-relabel DAgger run is
reproducible from explicit non-held-out trajectories:

```sh
uv sync --group train
dcss-rl train-imitation artifacts/eval/TRAIN/*/trajectory.jsonl \
  --checkpoint checkpoints/candidate.pt --relabel-scripted
dcss-rl evaluate-learned --checkpoint checkpoints/candidate.pt
```

Offline and online teacher balancing share `--teacher-balance-exponent`; zero is
ordinary cross-entropy and one is full inverse-frequency weighting. Online imitation
aggregates learner-visited labels across updates by default, as required by DAgger;
`--no-aggregate-imitation-replay` exists for matched ablations. PPO, value, and ECHO
objectives always use only the current on-policy rollout.
`--action-history-length N` expands a stateless initial checkpoint with `N`
newest-first action slots. History remains policy-side context: it is reset per game,
kept separate across concurrent workers, and excluded from ECHO environment targets.
Pass `--update-checkpoint-directory DIR` to retain immutable, naturally sorted
`update-NNNN.pt` snapshots in addition to the rolling final checkpoint. The broader
`configs/online-train-v3.json` curriculum uses 128 training-only seeds and 2,000-step
episodes.

Checkpoints and generated rollouts remain untracked. Learned candidates must first
improve the diagnostic track and then clear the checked-in held-out floor before
promotion to the held-out champion track. Evaluation is non-promoting unless an
explicit `--champion` manifest is supplied, preventing a diagnostic suite from being
accidentally compared with the held-out track.

Collect deliberate ability-menu training exposure and run the bounded conditional-choice
probe with the discoverable Poe targets:

```sh
poe collect-ability-curriculum --output artifacts/c/ability-data
poe audit-affordances -- artifacts/c/ability-data
poe ability-probe \
  --checkpoint checkpoints/continuing-decision-v51-updates/update-0002.pt \
  --trajectories artifacts/c/ability-data --output artifacts/c/ability-probe
```

The collector uses eight existing training seeds and retains raw trajectories. The
probe splits episodes six/two, verifies target margins and exact parameter ownership,
and records non-ability-menu action drift separately. Its validation is a conditional
menu-choice check, not evidence of stronger gameplay. Output attempts are immutable.

`poe train-ppo` and `poe evaluate-learned` expose the corresponding CLI workflows.

Use `poe train-ppo --record-rollout-trajectories ...` for experiments that need raw
training replay and matched-collection audits. Each worker episode retains its
trajectory, including raw exchanges; transitions name the collection update and SHA
of an immutable checkpoint under `RUN_ROOT/collector-checkpoints/`. Episode headers
do not claim one checkpoint because an episode can span multiple updates. Recording
is opt-in; old training game logs alone cannot establish exact rollout equality.
Recorded sampling evidence retains the actual probability vector and pre-choice RNG
fingerprint without changing either. Compare complete first collections with:

```sh
poe compare-training-rollouts --reference RUN_A --candidate RUN_B --workers 48 --steps 128
```

The report separates semantic/action agreement, raw-message equality, and sampling
equality, validates collector checkpoints, and checks terminal bootstrap resets.
Missing sampling evidence is unknown, not a successful equality check.
For stateless checkpoints, `--require-policy-identical` requires matching versioned
policy inputs, masks, complete sampling evidence, actions, rewards, and bootstrap
states; cosmetic semantic/raw differences are still reported independently.
For a cache-preparation timing experiment, `--collect-static-cache-timing` on
`poe train-ppo` requires `--record-rollout-trajectories` and records per-reset stage
wall/thread-CPU times in episode metadata. The same timing flag on
`poe benchmark-startup-cache` records both arms and retains individual reset timing
records. Overlapping worker wall times must not be summed into an elapsed runtime;
timing remains opt-in and does not relax cache validation.
PPO prepares anchor replay through a transient reducer and a disposable tensor cache
(`--imitation-cache-directory`, default `.cache/imitation-replay`). Content and
preprocessing-source hashes invalidate the cache; only environment features, legality
masks, and teacher targets are stored. Checkpoints record separate anchor and actual
imitation-replay target/legal-exposure counts, including zero target counts.

Inspect full trajectory evidence and the lighter `crawl.log`/morgue records retained by
online training with one discoverable command:

```sh
poe audit-affordances -- artifacts/t/bt56 artifacts/t/bh57
poe audit-affordances -- /tmp/eval-v56-u3 /tmp/eval-v57-u3
```

The audit distinguishes applicable, inapplicable, and absent Berserk menu entries;
success, active/cooldown rejection, stochastic failure, and ability loss; direct
Renounce selections where trajectories exist; and renunciation prompts/completions and
wrath deaths where only Crawl logs remain.

Rank v5 orders wins, runes, depth-weighted newly discovered cells, levels visited,
maximum depth, maximum XL, and reward. Repeated actions and already-seen backtracking
cannot inflate it, while exploration after a useful retreat still receives credit.
Deaths retain progress earned before reset instead of losing an artificial remainder
of the evaluation horizon.

The structured catalog includes the ordinary ability-menu command while menu choices
remain player-visible masked key actions. Older checkpoints are expanded append-only,
preserving all established command and menu-key indices. The scripted MiBe teacher can
therefore invoke Berserk through the same two policy decisions available to learners.
When appending actions, `--new-action-warmup-updates N` restricts the first `N`
optimizer phases to appended policy rows. Repeatable
`--new-action-warmup-menu-key KEY` arguments include dependent player-visible menu
decisions, while newly appended semantic input columns may learn without changing
established inputs or unrelated outputs. Repeatable `--imitation-trajectory-root`
arguments preload and current-teacher-relabel non-heldout replay anchors.

The current diagnostic track reuses five explicitly known development seeds for 500
decisions, long enough to distinguish sustained navigation and survival from an early
descent. It is not a headline or promotion suite.

Online fine-tuning restores the same checkpoint and samples only legality-masked
actions. `configs/online-train-v2.json` supplies 64 training-only seeds with
1,000-decision horizons. Workers collect independent chunks without per-action barriers;
worker-local seeded samplers make repeated runs deterministic across thread schedules.
Asynchronous requests are combined into bounded GPU inference batches; tune
`--inference-batch-size` and the typed
`--inference-batch-wait-seconds` only from measured worker workloads. Per-update logs
separate collection, optimization, and checkpoint time and report realized mean batch
size.
Optional depth, exploration, XP-progress, and HP-potential rewards use only
player-visible state and affect training environments only. Set `--echo-weight 0` for
the matched ECHO-off ablation. Teacher class balancing uses square-root inverse
frequency by default; `--teacher-balance-exponent 0` disables it and `1` restores full
inverse balancing. Setting `--policy-weight 0 --value-weight 0 --echo-weight 0
--entropy-weight 0` isolates online DAgger imitation while retaining autonomous,
legality-masked policy collection:

`--return-boundary continuing-reset` values death as a reset into the ongoing training
task while keeping wins terminal. Typed `--decision-cost` and `--short-cycle-cost`
settings are training-only; the latter detects recurrence of the same stable semantic
state within `--short-cycle-window` decisions. Reported episode returns and held-out
ranking retain the unmodified environment rewards.

```sh
dcss-rl train-ppo \
  --initial-checkpoint checkpoints/semantic-dagger-echo-v4.pt \
  --checkpoint checkpoints/ppo-echo-on.pt --policy-id ppo-echo-on \
  --suite configs/online-train-v2.json --updates 4 --rollout-length 128 \
  --depth-progress-reward 20 --experience-progress-reward 1 \
  --hp-fraction-reward 1
```

`configs/curriculum-v1.json` records the exact bootstrap, DAgger, confidence gate,
and held-out promotion stages used by the first learned champion. It references only
diagnostic seeds before the final locked evaluation; held-out trajectories are never
training inputs.

## Licensing

Project code is Apache-2.0. DCSS is an external GPLv2+ dependency fetched from its
official repository and is not redistributed here.
