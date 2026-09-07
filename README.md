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

Champion promotion is gated by the checked-in held-out regression floor. Until a
learned policy exceeds it, the champion may be the transparent scripted baseline.
Native WebTiles spectating can later complement the deterministic completed replay.

The optional training stack provides a versioned semantic actor/value model with an
ECHO next-state-delta head. A behavior-cloning or scripted-relabel DAgger run is
reproducible from explicit non-held-out trajectories:

```sh
uv sync --group train
dcss-rl train-imitation artifacts/eval/TRAIN/*/trajectory.jsonl \
  --checkpoint checkpoints/candidate.pt --relabel-scripted
dcss-rl evaluate-learned --checkpoint checkpoints/candidate.pt
```

Checkpoints and generated rollouts remain untracked. Learned candidates must first
improve the diagnostic track and then clear the checked-in held-out floor before
promotion to the held-out champion track.

Online fine-tuning restores the same checkpoint and samples only legality-masked
actions. `configs/online-train-v1.json` supplies 20 training-only seeds with longer
500-decision horizons. Set `--echo-weight 0` for the matched ECHO-off ablation:

```sh
dcss-rl train-ppo \
  --initial-checkpoint checkpoints/semantic-dagger-echo-v4.pt \
  --checkpoint checkpoints/ppo-echo-on.pt --policy-id ppo-echo-on \
  --suite configs/online-train-v1.json --updates 4 --rollout-length 128
```

`configs/curriculum-v1.json` records the exact bootstrap, DAgger, confidence gate,
and held-out promotion stages used by the first learned champion. It references only
diagnostic seeds before the final locked evaluation; held-out trajectories are never
training inputs.

## Licensing

Project code is Apache-2.0. DCSS is an external GPLv2+ dependency fetched from its
official repository and is not redistributed here.
