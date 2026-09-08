# Training replay retreat coverage audit

Date: 2026-09-08. Read-only audit; no new games, relabeling, training, development
labels, or held-out replay access.

## Decision

The existing v19/v20/ac1 anchors cannot support a low-HP upstairs-escape pilot:
they contain no upstairs-legal state at or below 75% HP. Deliberate new curriculum
is required before fitting this behavior. Do not interpret the existence of 447
syntactically legal upstairs exposures as positive survival-target coverage.
The next immediate experiment is the separately specified UI-return pair.

## Inputs and method

The audit read all 136 training trajectories:

- 64 matching `artifacts/t/v19/*/trajectory.jsonl`;
- 64 matching `artifacts/t/v20/*/trajectory.jsonl`;
- 8 matching `artifacts/c/ac1/*/attempt-0/trajectory.jsonl`.

The first pass used `replay_frames` to reconstruct observations, examining exactly
the observation preceding each recorded action. It excluded the final observation
because no action was taken from it. `_training_action_mask` supplied actual
syntactic upstairs exposure. Candidate escape states additionally required player
depth greater than one. HP bands were inclusive comparisons of `hp / hp_max` with
0.25, 0.50, and 0.75. Visible threats used the current teacher's
`_is_tactical_monster` filter; adjacent threats had Chebyshev distance one from
the player. Unique episodes were trajectory paths; unique seeds came from the
trajectory metadata. These are teacher threat semantics, not a validated estimate
of tactical danger.

A second independent reconstruction used `_TransientObservation`, checked the
equivalent command-mode/no-menu/`stairs_affordance_visible` conditions directly,
and counted valid HP fields and the range of observed HP fractions. It reproduced
the first pass's 447 legal exposures and ruled out missing HP data as the reason
for zero low-HP coverage. Both queries were exploratory read-only commands; no
new recurring CLI workflow was introduced.

## Results

All 447 upstairs-legal observations were at depth greater than one. All had integer
`hp` and `hp_max` fields with positive maximum HP; zero had missing or invalid HP.
Their minimum HP fraction was **0.7936507936507936**, and their maximum was 1.0.

| HP fraction | Upstairs-legal states | With visible threat | With adjacent threat | Unique eligible episodes | Unique eligible seeds |
| --- | ---: | ---: | ---: | ---: | ---: |
| ≤ 0.25 | 0 | 0 | 0 | 0 | 0 |
| ≤ 0.50 | 0 | 0 | 0 | 0 | 0 |
| ≤ 0.75 | 0 | 0 | 0 | 0 | 0 |

The existing checkpoint audit is discoverable and reproducible through:

```sh
poe checkpoint-audit --checkpoint checkpoints/ability-open-v65-updates/update-0002.pt
```

Its checkpointed anchor and actual imitation-replay coverage confirms the missing
positive class:

| Action | Anchor targets / legal exposures | Actual replay targets / legal exposures |
| --- | ---: | ---: |
| Upstairs | 0 / 447 | 0 / 632 |
| Downstairs | 1,966 / 2,026 | 2,135 / 2,290 |
| Rest | 786 / 26,421 | 1,480 / 38,338 |
| Abilities menu open | 4,311 / 26,421 | 6,139 / 38,338 |
| Menu `a` | 33 / 203 | 70 / 248 |

Anchor coverage contains 27,333 rows from v19/v20/ac1. Actual replay coverage in
this v65 checkpoint contains 39,621 rows, including its online collection. The
low-HP context audit above concerns only the 27,333 anchor rows; it does not infer
the context distribution of v65's additional online observations.

## Representation and preservation implications

The teacher has no explicit low-HP retreat or upstairs branch. It attacks adjacent
monsters, otherwise approaches visible monsters, and rests when injured with no
visible threat. Its ability-opening heuristic uses adjacency plus monster count,
depth, or HP, with status-string exclusions. More same-teacher training does not
create the missing upstairs target class.

Feature versions 5 and 6 have a local map channel for the `<` glyph, but all 447
upstairs-legal observations showed `@` at the player coordinate. There is an explicit
downstairs-under-player message scalar and a downstairs navigation hint; no analogous
upstairs-under-player scalar or upstairs navigation hint exists.

A future deliberately narrow upstairs-row-only probe could nevertheless use the
existing HP and threat features: its syntactic action mask already restricts where
upstairs can be selected. With encoder and all other policy rows frozen, changing
only this row leaves deterministic actions unchanged wherever upstairs is masked.
This is a narrower behavioral preservation claim than freezing weights while also
learning encoder columns. It does not establish that ascending is tactically safe,
nor does it teach routing toward stairs or general movement-based retreat.

Before choosing that pilot, collect positive low-HP threatened upstairs contexts
and matched negative upstairs contexts on training seeds, audit episode/seed
coverage, and reserve an episode-disjoint validation split. Do not relax the HP
threshold merely to obtain positive examples from the current anchors.

## Training-only setup option (source-verified, not yet live-tested)

Unmodified trunk `96832895d0253f9d7290d370efe32bf0614679a8` supports wizard level-down
(`&d`), placing upstairs underfoot (`&,`), dismissing monsters (`&G`), and named monster
creation (`&m`). The latter places monsters within distance 2 through visibility range;
condition checks must use actual visible state rather than assuming adjacent placement.
No direct HP setter was found. Ordinary bounded waiting against a controlled monster
could provide low-HP threatened states, with healthy-threatened and injured-unthreatened
controls. Record failed setup attempts rather than silently substituting easier seeds.

No setup API currently exists. A future collector must retain raw setup exchanges
separately from ordinary policy-action examples, keep wizard keys out of the learned
catalog/deployment, and assert requested HP/threat/stair conditions before acceptance.
Crucially, even suppressed wizard mode asks `Die?` at lethal damage (`ouch.cc:1548–1563`).
The current teacher rejects generic prompts, which would resurrect the character.
Any wizard collector must accept that death or terminate at the lethal boundary;
otherwise its survival evidence is invalid. This route needs a bounded live smoke
before any curriculum scale-up and cannot supply normal-play headline evidence.
