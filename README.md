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
runs the complete gate, including the live DCSS process integration test.

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

## Research direction

Initial experiments will compare a conventional recurrent policy with language-model
policies trained using policy gradients. ECHO-style environment prediction is a
first-class auxiliary objective: trajectories preserve action and observation token
masks so the same rollout can supervise both control and prediction of the next
semantic game-state delta.

Every evaluation record should identify the DCSS commit, configuration, seed policy,
agent checkpoint, reward specification, and terminal morgue/replay artifacts.

The current champion will be selected by a fixed held-out evaluation suite rather than
by a hand-picked game. A future `poe watch-best` task will run that checkpoint locally
and expose native WebTiles spectating, while preserving completed showcase replays and
morgues. Until a learned policy wins the benchmark, the champion may be a transparent
scripted baseline.

## Licensing

Project code is Apache-2.0. DCSS is an external GPLv2+ dependency fetched from its
official repository and is not redistributed here.
