# Cache72 policy-parity failure

Read-only audit, 2026-09-08. Application revision:
`6a217110f25c91594b81ae9af623a590d56faac7`. Canonical upstream:
`96832895d0253f9d7290d370efe32bf0614679a8`.
Evidence: `artifacts/p/cache72/policy-parity.json` and numeric worker/episode
trajectories under `artifacts/p/c72/off` and `artifacts/p/c72/on`.
No new games or held-out data were used.

## Result

Of 48 workers × 128 decisions, only worker 14 has policy-affecting divergence.
The initial model tensors match. Worker 14 has 19 different pre-action feature
vectors/probability distributions, 14 different actions, and no differing
pre-choice RNG hashes. Its first differing next state is decision 108; its first
different policy input/action is 109. Terminal status differs at 121, followed
by episode identity at 122. This is a readiness failure, not evidence of a
learned-weight change or direct cache-induced actor computation change.

Ten additional workers have semantic differences: 1, 9, 18, 22, 25, 27, 28, 32,
36, and 46. Reconstructing all their first 128 before/after states and removing
only `cells[].col` makes every state equal. Their feature vectors, masks,
probabilities, actions, and rewards already match without that removal.
Strict raw equality remains a separate, stronger property and does not hold.

## First actor-affecting boundary

Both arms: `worker-14/episode-0-attempt-0/trajectory.jsonl`, seed 3015.
Decision indices below are zero-based, matching the recorded `step` field.

| Decision | Cache off | Cache on |
| --- | --- | --- |
| 107 | Space acknowledges more; shaft-drop message; still D:2, mode 5 | Same |
| 108 | Space, probability 1; receives generation UI and eventual D:4/mode 1; reward 6 | Same action/probability; returns after generation UI closes, still D:2/mode 0; reward 0 |
| 109 | Explores, probability about .80431; advances into D:4 | Forced cancel, probability 1; receives delayed D:4 state and “Unknown command.”; reward 6 |

The cache-on raw exchange at 108 contains `input_mode:0`, a `ui-push` of type
`progress-bar` with generation ID 2 and title “Generating dungeon...”, several
progress updates/flushes, then `ui-pop`, `close_all_menus`, and a flush. It ends
there. Cache off continues beyond that same prefix through a milestone/flush
to `input_mode:1`, the new player/map state, the shaft-collapse message, and
another flush. The identical pre-choice RNG hash at 109 is
`8568a5a15b3181c29b4b2deade3f61ccd8d13e212eaaf8b9a07906abc1ef1aa7`.

This is a shaft transition released by a more acknowledgment, not an explicit
stairs action. `DcssEnv.step_typed` selects `LEVEL_TRANSITION` only for
`STAIRS_UP`/`STAIRS_DOWN`; menu space uses `QUIESCENCE`. The latter accepts a
10 ms quiet flush even after mode 0. Existing `_LevelBoundaryEvidence` rejects
this nonblocking generation/closure prefix, but is consulted only for the
explicit level-transition boundary.

## Smallest next check

Use the captured acknowledgment exchange as an adversarial transport fixture:
force silence longer than 10 ms after the initial busy prefix, intermediate
progress flushes, and the final menu-close flush; then deliver the fresh D:4
input boundary. None of those prefixes should expose a policy decision.

A proposed narrow extension is a UI-continuation boundary that reuses
`_LevelBoundaryEvidence`: after current-exchange mode 0, bare quiescence must
not certify completion; require fresh supported input-mode evidence, genuine
blocking UI/line input, or terminal exit. Do not key the rule to shaft text or
wait until a progress-bar push is observed, since an earlier prefix can split.
Retain the no-op/full-state-probe path when no fresh busy evidence appears.
Required negative controls include a new more prompt, menu, blocking CRT,
line input, silent/no-op acknowledgment, and terminal exit. This is a proposed
scope, not a cross-version readiness guarantee. No implementation or rerun was
performed in this audit.
