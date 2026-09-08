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

Held-out and development leaders are deliberately separate. Diagnostic evaluations
promote monotonically to `artifacts/dev-champion.json`; observe that track with:

```sh
poe evaluate-scripted-diagnostic
poe watch-diagnostic-leader
poe watch-diagnostic-grid
```

The terminal viewer reconstructs both legacy full-snapshot trajectories and compact
schema-v2 deltas. Pass `--case CASE_ID` to inspect a specifically labeled suite case;
the default deliberately follows manifest order instead of selecting a flattering
showcase seed. Grid panels remain on their final frame and visibly distinguish death
(`☠`), horizon truncation (`◇`), and eventual ascension (`★`). The older `watch-best`
and `watch-dev-best` Poe names remain temporary aliases.

Champion promotion is gated by the checked-in heldout-v4 regression floor. The
canonical champion is a fallback-free learned policy; Native WebTiles spectating can
later complement the deterministic completed replay.

Heldout-v1 was retired after its failure traces directly informed contextual stair
masking. Its manifest remains archived locally for audit, but it is never reused for
promotion. A validated track-activation command atomically archives the old canonical
manifest and cuts every heldout viewer over to a calibrated replacement track.

The optional training stack provides a versioned semantic actor/value model with an
ECHO next-state-delta head. Feature-spec-v3 adds explicit prompt/shop and blocking
status signals while retaining feature-spec-v2 checkpoint compatibility. A
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
