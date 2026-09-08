# Terrain-v66 development frontier

2026-09-08. Audit of all sixteen manifest-ordered cases in
`artifacts/e/terrain66-dev16/summary.json`, using each referenced trajectory and
the corresponding v51 cases in `artifacts/e/v51-dev16/summary.json`. No heldout
access, games or implementation changes. This is seeded development evidence.

## Outcome and remaining failures

V66 reaches aggregate maximum depth/XL/discovery 66/46/97,476 over 4,704
decisions, versus 59/44/70,665 over 6,981 for v51. Thirteen cases die (median 96
decisions; range 74–333), versus ten before. Maximum actual depth remains D:6;
maximum XL rises from 5 to 6. The three horizon survivors are looping, so horizon
survival alone remains a misleading quality measure.

| Case | Outcome | Decisions | Max D / XL | Discovery |
| --- | --- | ---: | --- | ---: |
| 81001 | prompt loop | 1000 | 6 / 3 | 6928 |
| 81002 | death | 74 | 3 / 3 | 3534 |
| 81003 | death | 164 | 5 / 5 | 10414 |
| 81004 | death | 92 | 3 / 3 | 3220 |
| 81005 | death | 81 | 4 / 1 | 3125 |
| 81006 | death | 79 | 4 / 1 | 2038 |
| 81007 | death | 110 | 4 / 4 | 5626 |
| 81008 | death | 83 | 4 / 2 | 4248 |
| 81009 | movement loop | 1000 | 1 / 1 | 1214 |
| 81010 | prompt loop | 1000 | 4 / 4 | 4696 |
| 81011 | death | 278 | 6 / 5 | 17424 |
| 81012 | death | 333 | 5 / 6 | 17472 |
| 81013 | death | 82 | 4 / 2 | 3180 |
| 81014 | death | 96 | 3 / 3 | 3638 |
| 81015 | death | 106 | 4 / 1 | 5172 |
| 81016 | death | 126 | 6 / 2 | 5547 |

Both previously diagnosed terrain stalls escape. Case 81003 reaches D:5 XL5
before dying to a jelly/scorpion fight with poison; 81012 reaches D:5 XL6 before
dying surrounded by Eustachio and summons. Previously oscillating 81011 now reaches
D:6 XL5 before dying to a drude/ogre fight. Terrain changes also alter 81006 and
81015: the latter regresses from D:6 XL5 to D:4 XL1. Gains are not uniform.

The unchanged suffixes are 81001 step118 onward (882 decisions), 81009 step75
onward (925), and 81010 step219 onward (781). Together these consume 2,588/4,704
decisions, 55.0%. The two prompt cycles still deny unarmed-attack warnings in bat
and fungus forms. The movement cycle remains NE/SW on D:1. These need distinct
representation/history work; they are not evidence that Berserk is the universal
fix. Detail and upstream source verification are in `research-v51-frontier.md`.

Eleven of thirteen deaths contain opportunities below 30% HP; those actions
mostly continue melee. The other two deaths escalate rapidly from higher HP.
The first manifest death, 81002, still attacks at 2 HP; 81004 still attacks while
burning. New 81011/81012 deaths show additional combat threats after productive
exploration. `poe audit-affordances artifacts/e/terrain66-dev16` reports no ability
opens or Berserk starts in all sixteen cases. No staircase-up actions occur.

## Recommended bounded pilot: learn opening timing with an actual return signal

Hypothesis: the repaired conditional ability chooser can improve combat survival
if opening is trained against its downstream costs. The prior opening-only v65
imitation experiment actually started Berserk five times, but spent 325 decisions
cancelling inapplicable menus; its PPO/value coefficients were zero. Consequently
that result does not test whether a UI-burst return cost can discourage those
open/cancel attractors. This is the most direct small survival experiment using
capabilities and audited data already available. Benefit remains conjectural;
no counterfactual rescue claim follows from the deaths above.

Use the same opening-capable, conditional-choice-repaired starting checkpoint
for both arms, explicitly migrated to terrain spec6. Preserve the repaired menu
rows and encoder and audit those tensors after updates. Permit the abilities row
and value head to learn. Compare matched clipped PPO/value arms with UI burst
overflow cost off/on; keep all other objectives, seeds, horizons and sampling
settings equal. Retain the same bounded anchor imitation in both arms. Freeze
ECHO at zero for this isolated question. A common migrated zero-update baseline
separates changes due to terrain from changes due to optimization; v66 is the
external current-policy reference, not the starting control for this ablation.

Bound each arm to at most two 48x128 updates (24,576 decisions total across the
pair), with telemetry at both scheduled snapshots. This is a proposal, not an
executed experiment or tuned hyperparameter claim. Use existing online training
seeds plus v19/v20/ac1 training anchors; never relabel development-validation
trajectories for training. Existing ac1 intentionally includes applicable and
inapplicable menus. Its limitation is deliberate periodic opening, so preserve
real online combat/no-threat/cooldown exposures rather than treating ac1's
collection actions as optimal opening targets. A new training-only matched
encounter curriculum is an alternative if online rollouts contain too few valid
combat activations; it should precede expanding compute, and record exposure.

First require actual PPO/value learning, at least some successful activation,
unchanged conditional-choice correctness, and fewer inapplicable opens/cancels
per ordinary game decision in the cost arm. Stop if both arms remain dormant,
or loop changes produce no combat/frontier benefit. Evaluate only the scheduled
diagnostic snapshots, then permit one broader development comparison if the
selected result retains v66 discovery and improves XL or combat survival.
Record active game turns and discovery alongside death/truncation: replacing
combat with a prompt loop must not count as survival improvement. No heldout
access is warranted by a telemetry-only improvement.

If the matched return-cost test is negative, prefer a combat/retreat curriculum
with earlier danger response over repeating opening-only imitation. Existing
HP/XL/depth features permit some learning, but enemy threat/status representation
and unavailable consumable/equipment actions bound what this policy can do.
