# V73: bounded margin fit stops short of feasibility

2026-09-08. Read-only inspection after the fixed 256-step Adam run; no further
optimizer steps, games, or held-out access. Runtime was 21.7886 seconds.

## Evidence and independent checks

- Report: `artifacts/c/ability-residual-v73/report.json`, SHA256
  `0de23948de2df031209aa180aba6f440f23e4e3f8b7c5ce1489378a0fd74ec6b`.
- Source: `checkpoints/terrain-v66.pt`, SHA256
  `3ab6b36828249d137b93e323bb25c27612b3c7b00978dba7c0db5f668b753924`.
- Zero-enabled `start.pt`, SHA256
  `4f37b0412c8b8fa28f8e9c2db08b6ff9c9a3b09b27c8255f2ee179dcf0efa467`.
- Final `step-0256.pt`, SHA256
  `958dc7f9dd928d2253b581d6711aacf86bca4d2da9c9b1b3a5c1488f7213c3f7`.

All six source/start/snapshot file hashes match the report. Each saved snapshot's
metrics, objective, and training coverage agree with its checkpoint metadata.
Independent checkpoint audits pass residual-only ownership for steps 1/16/64/256;
source base tensors are bit-exact in the zero-enabled checkpoint. The automated
`gate.json` finds no qualifying checkpoint. All 8,388 non-menu and four other-menu
states preserve logits, masked probabilities, and argmax exactly. The other-menu
examples are training-only; validation has none.

Final greedy accuracy is 100% in all three contexts on both splits. Every final
confidence gate passes except training/missing Renounce probability:
`1.028255883e-5 > 1e-5` (2.826% relative excess). Validation missing maximum is
`7.46009e-8`, but its seven examples cover cooldown only, unlike training's nine
inactive missing examples.

Directly recomputed training logit gaps show two residual objective violations:

| Target versus competitor | Minimum gap | Required gap | Deficit |
| --- | ---: | ---: | ---: |
| Missing Cancel versus X | 11.48505116 | 11.51292546 | 0.02787431 |
| Applicable a versus Cancel | 5.29146194 | 5.29529681 | 0.00383486 |

Each occurs on one example. Final worst-margin loss is `0.0002638882` in float32.
The second violation does not fail the actual mean/minimum probability gates:
applicable training mean is 0.999556 and minimum 0.994989. The first does fail the
Renounce cap. This is unfinished optimization of known-feasible training constraints,
not evidence of insufficient residual capacity. Do not relax the threshold or
continue this run.

## Proposed decisive next solve, not yet executed

Prefer a bounded linear feasibility/optimization solve over repeated Adam tuning.
With frozen base logits, every required target gap is an affine inequality in the
same 18 residual parameters. The 1,783 training competitor inequalities reduce
exactly to nine distinct constraint directions by grouping identical five-feature
residual inputs, target, and competitor, retaining the strongest training RHS.
Validation must not contribute constraints or coefficients.

A concrete LP is: minimize `t`, subject to all training margin inequalities and
`-t <= theta_k <= t`, `t >= 0`. This chooses a minimum-infinity-norm learned residual
without hand-setting its coefficients. It has 19 variables and nine margin
constraints plus norm bounds. Use float64, a predeclared solver iteration/time cap,
and an explicit small numerical margin buffer (for example 1e-4 logit units) to
separate solver tolerance from the unchanged probability gate. Save only a solver-
certified feasible result; cast to deployment float32 and independently recheck
actual model outputs, ownership, training gates, then validation gates. Failure at
any stage is terminal for that attempt. A tolerance buffer is stricter training,
not permission to weaken the audit threshold.

SciPy/HiGHS is not installed in this environment; adopting this option needs an
explicit tooling dependency. PyTorch L-BFGS is an available dependency-free fallback,
but the current max-over-examples objective is nonsmooth at active-example switches.
L-BFGS with a fixed evaluation cap may converge faster than Adam, yet offers no
finite-budget feasibility certificate by itself. A direct small LP is the stronger
test of these finite constraints. Neither approach establishes unseen-state safety
or authorizes live continuation automatically.
