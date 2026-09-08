# v51 development frontier audit

2026-09-08. Read-only audit of all 16 cases, in manifest order, from
`artifacts/e/v51-dev16/summary.json` and its referenced `attempt-0/trajectory.jsonl`
files. No heldout access or new games. Generated artifacts remain untracked.
Upstream source inspected at revision
`96832895d0253f9d7290d370efe32bf0614679a8` (local `vendor/crawl`).

## Aggregate evidence

The fallback-free baseline completes 6,981 decisions: ten deaths after 52–185
decisions (median 87.5), and six horizon truncations. Maximum actual depth is D:6;
maximum XL is 5. Aggregate maximum depth/XL/depth-weighted discovery is
59/44/70,665. Of 43 observed descents, 37 enter a depth above current XL; only two
occur below 70% HP. This is evidence of early descent relative to character growth,
not proof of a universal safe depth/XL rule.

Eight deaths have decision opportunities below 30% HP, spent almost entirely
continuing melee. The first dead case, 81002, continues fighting gnolls at 2 HP;
81004 attacks while burning; 81006 attacks a hobgoblin while Sigmund attacks from
range. Two other deaths escalate from higher HP. There are no staircase-up actions.
`poe audit-affordances artifacts/e/v51-dev16` reports zero ability opens or Berserk
activations in all 16 cases. Missing ability use is established; its counterfactual
benefit is not.

All six truncations end in repeating action suffixes, totaling 5,195 decisions
(74.4% of all decisions). Steps below are zero-based; suffixes count actions and
can include the first legitimate movement of an alternating sequence.

| Case | First suffix step | Decisions | Evidence |
| --- | ---: | ---: | --- |
| 81001 | 118 | 882 | Bat form attempts westward unarmed attack at a plant, then denies the weapon warning |
| 81003 | 90 | 910 | North into lava under cloud glyph; repeated rejection, no turn advance |
| 81009 | 75 | 925 | NE/SW two-position oscillation, D:1 XL1 |
| 81010 | 219 | 781 | Fungus form attempts westward unarmed attack, then denies warning |
| 81011 | 181 | 819 | South/north two-position oscillation, D:2 XL4 |
| 81012 | 122 | 878 | SW into solid statue; no turn or semantic change |

## What the existing protocol already reveals

At the inspected revision, `source/tag-version.h` defaults to major version 34.
Resolving `source/dungeon-feature-type.h` with those conditionals gives terrain
30 = `DNGN_LAVA`, and 219 = `DNGN_ZOT_STATUE`.
`source/feature-data.h` marks the statue `FFT_SOLID | FFT_NOTABLE`, `MF_WALL`;
lava is `MF_LAVA`. `source/map-feature.h` gives `MF_WALL=2`, `MF_LAVA=17`.
`source/terrain.cc:400` defines solidity from `FFT_SOLID`.

In 81003, the north target at (9,42) has `f=30,mf=17,g=§` while the player is
at (9,43), untransformed and not flying. In 81012, the SW target (-4,47) has
`f=219,mf=2,g=ß` while the player is at (-3,46). These are representation failures
with existing observations: `features._walkable` excludes a few glyphs, so a
cloud can obscure lava and a statue glyph can evade wall classification.
`source/tileweb.cc:1597` sends remembered terrain `f` separately from display
glyph `g` and map category `mf`; semantic cells already preserve these fields.
There is no universal upstream passability boolean. Terrain classification must
respect visible movement state, doors and flight, and revision compatibility;
`mf` can also describe occupants/items rather than underlying terrain.

Both transformed cases already contain persistent structured status entries:
`text=bat-form` with flight in 81001, and `text=fungus-form` in 81010. Status
descriptions explicitly describe melded equipment; fungus describes inability
to move near hostiles. `weapon_index=-1` is also preserved. Upstream
`source/tileweb.cc:1275` sends status text/descriptions and line 1313 sends the
weapon index. `source/form-data.h` contains the matching form strings.
No new transport is required. The feature encoder largely collapses status
detail to a count plus Berserk/exhaustion flags, and does not encode unarmed
state or the weapon-warning prompt meaning. Monster targets also include plants.

The raw menu title is `Really attack while wielding nothing?`.
`source/fight.cc:1023` implements this check: a No answer produces `Okay, then.`;
a Yes answer sets `received_weapon_warning`, suppressing subsequent warnings.
This explains the zero-turn cycle without asserting that Yes is always tactically
appropriate (especially the plant target).

Messages are batch-local, not persistent rejection memory:
`SemanticReducer.apply` starts an empty message list on each batch and snapshots
only that batch's messages. In 81003 every attempt emits the rejection anew.
In 81012 every repeated movement emits no text. In the prompt cycles an attempt
has empty messages plus the structured menu, while denial has `Okay, then.` and
no menu. Thus text-only rejection features cannot identify all four loops, and
the next silent action can erase a previous rejection. Raw trajectories retain
the evidence needed for deterministic action/outcome history.

## Smallest falsifiable next experiment

First isolate **terrain-aware navigation representation**, independently of the
ability pilot. Keep the action mask syntactic. Version the existing adjacent
walkability/BFS feature semantics so known solid terrain and non-traversable lava
are recognized from structured visible data, including flight exceptions. Do not
silently reinterpret older checkpoint preprocessing. Before any game, compare
old/new feature outputs at the two exact blocked states and matched ordinary
floor/door/flying states; verify ordinary control states remain unchanged.

Use a bounded disjoint training curriculum containing blocked and applicable
movement examples, ordinary navigation anchors, and no development-validation
labels. Initialize from v51; compare its unchanged baseline with a short
fine-tuned representation arm, recording target exposure and parameter changes.
A zero-update representation-only diagnostic can distinguish changed navigation
hints from newly learned behavior. Predeclare a fixed decision budget and inspect
all development cases in order. Primary pilot outcomes: blocked-movement suffix
decisions, useful discovery, death rate and XL. Reject scaling if stalls merely
become earlier deaths without better exploration/character growth. This test
targets the two terrain loops; it does not claim to solve all six.

If terrain evidence is negative or insufficient, separately test minimal
action/outcome history: last attempted direction plus whether it changed
position/time or was denied in a prompt. This requires checkpointed deterministic
history reconstruction; silent blockers and two-cell revisitation need different
signals. Do not combine terrain, prompt, recurrence and reward changes in one
first pilot. Prompt-specific training should preserve the distinction between
an unarmed attack against a threat and pointless attacks at plants.

The parallel survival hypothesis is early floor progression plus weak response
to threat, rather than simply missing Berserk. HP/depth/XL are already features;
enemy threat and form detail are poorly represented. Compare ability training
with a later bounded combat/retreat curriculum, measuring intervention before
HP collapses. The action catalog cannot directly drink, read, zap or equip, so
consumables/equipment are future action-space gaps. Postmortem identification of
inventory must not become privileged training input.
