# V71: confidence is not a capacity failure

2026-09-08. Read-only analysis of `artifacts/c/ability-residual-v71/report.json`
and its retained source/start/snapshot checkpoints. No additional training or games.

The 18-parameter residual achieves perfect greedy accuracy but fails probability
tails. The worst missing-Berserk training examples are nine inactive states, where
all five residual inputs are zero and only its learned biases contribute. Their
maximum Renounce probability is 0.07924. Twenty-three missing examples are in
cooldown and one is active; all seven validation missing examples are in cooldown.
The apparently better missing-context validation tail therefore does not cover the
hardest observed subtype. The three context losses already receive equal weight;
the issue is not simply that the missing context has fewer total samples.

## Feasibility, not a hand-set policy

Let q be the existing visible applicable flag. A residual direction that adds Kq
to a and K(1−q) to Cancel, leaving X unchanged, lies in the existing parameter
space. It increases every labeled target-versus-legal-competitor gap by K for the
three training contexts. The existing ability-menu gate leaves other UI untouched.
This is a capacity certificate, not a proposal to deploy hand-set coefficients.
The next experiment must still learn its weights exclusively from the training split.

The largest training deficit against the probability-derived margins below requires
K >=27.15128; the binding example has an original applicable a−X gap of −15.63835.
Thus no base-weight unfreezing or larger representation is needed merely to make
the finite training constraints feasible. This does not prove unseen-state safety.

## One bounded alternative objective

Retain zero initialization from terrain-v66, the same fixed ac2 36/12 split, CPU
Adam learning rate 0.1, and 256 steps with snapshots 1/16/64/256. Replace average
cross-entropy with the average over the three contexts of the **worst example's
sum of squared positive margin violations** across syntactically legal competitors.
Validation remains measurement-only. This is a proposed next experiment, not an
extension or rerun of v71.

For logit gap g_j between target and competitor j, its relative odds are exp(−g_j).
Set a Renounce odds cap of 1e−5 everywhere, an inapplicable Berserk cap of 1e−3,
and an applicable Cancel cap of (1−0.995)/0.995−1e−5. Required gaps are approximately:

| Competitor | Required target gap |
| --- | ---: |
| Renounce X | 11.51293 |
| Cancel when Berserk is applicable | 5.29530 |
| Berserk a when it is inapplicable | 6.90776 |

Since target probability is 1/(1+sum of competitor odds), zero training loss implies
the stated training confidence gates for these observed menus. The loss is convex
in the residual parameters with the base frozen, but convergence within the fixed
budget and validation performance remain empirical. Preserve exact base tensors,
nonability outputs, checkpointed coverage, and all existing stop criteria. Additional
or unknown choices must be handled explicitly rather than silently omitted.
