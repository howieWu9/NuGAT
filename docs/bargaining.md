# Cooperative-solution comparisons

This module implements the revised comparison definitions and a matched Nash
control. It does not contain
the supplied experimental scores, simulate a judge, or certify that the original
experiments use this newly supplied numerical implementation.

All solutions retain the complete set of configured modalities, QAT, the utility
definition in Eq. (15), and NUF's subspace retention. Only the training objective
and, for KSS/EGS, the stated equality constraints change:

| Variant | Objective minimized | Constraint |
| --- | --- | --- |
| UTS | negative sum of modality utilities | none |
| EGS | negative minimum modality utility | equal modality utilities |
| KSS | negative minimum utility divided by its estimated ideal utility | equal normalized modality utilities |

These definitions use the positive utilities in the manuscript, with zero
disagreement utility. Introducing nonzero disagreement values would change the
comparison and is not done here.

## Ideal utilities

KSS first starts one separate optimization for each modality from the same
initial model state. Each run maximizes that modality's utility, while retaining
all modalities and all model constraints. The default budget is 30 epochs per
modality. For each training query, the largest utility attained at initialization
or an epoch checkpoint in the corresponding run is recorded. These values are
then fixed throughout KSS training.

The runs optimize a mean modality utility over training minibatches. Consequently,
the recorded per-query maxima are **attained estimates**, not globally maximized
ideal utilities or proven upper bounds. A later candidate may exceed an estimated
ideal utility; ratios are not silently clipped to one. This nonconvex numerical
approximation must be disclosed when presenting the comparison.

## Equality-constrained optimization

EGS/KSS use an augmented Lagrangian. Every training query has its own vector of
dual variables. The primal objective includes a linear dual term and a quadratic
equality penalty; multipliers are updated every three epochs, and the penalty
increases when feasibility improves insufficiently. This is not an unconstrained
max-min substitute or a fixed-penalty-only implementation.

Equality residuals use log utilities: equal positive utilities are equivalent to
equal log utilities. For KSS the fixed log ideal is subtracted first. This avoids
declaring tiny but very unequal utilities equivalent just because their absolute
difference is small. The default tolerance is a maximum absolute log-ratio
residual of `1e-3`, approximately a 0.1% multiplicative mismatch. Constraints are
per training query, not merely equal average utilities across a dataset.

Training evaluates fixed minibatches in a stable order, with a fixed Gaussian
noise realization per minibatch. These common random numbers allow repeated
constraint and multiplier evaluations. The retained model is also checked using
the deterministic evaluation gate rule. A candidate is marked feasible only if
both checks meet the tolerance. This certification concerns the finite training
queries and these gate evaluations; it is not a global-optimality or unseen-query
guarantee.

This fixed-noise, fixed-order protocol differs from the standard training
runner's fresh gate samples and shuffled minibatches. Use `--solution nash` in
this same comparison runner for the Nash control. Do not compare the constrained
runs against a differently trained Nash checkpoint while claiming all other
training settings are unchanged. All four solutions in this runner share the
same initialization, example order, fixed gate-noise rule and base epoch budget;
KSS additionally incurs its separately reported ideal-estimation runs.

The normal training budget is 30 epochs. Equality constraints may not converge
within that budget. Such a run exits unsuccessfully by default and saves
`candidate_infeasible.pt`, not a successful `checkpoint.pt`. `--allow-infeasible`
permits inspection but preserves all failure labels. Do not report a failed run
as the constrained EGS/KSS result. Increase and report the optimization budget or
change solver settings explicitly if further optimization is needed.

## Commands

Use the same actual feature manifest and dataset configuration as the Nash run:

```bash
python scripts/train_bargaining.py --config configs/audiocaps_qa.yaml --manifest data/manifest.jsonl --output runs/nash_control --solution nash
python scripts/train_bargaining.py --config configs/audiocaps_qa.yaml --manifest data/manifest.jsonl --output runs/egs --solution egs
python scripts/train_bargaining.py --config configs/audiocaps_qa.yaml --manifest data/manifest.jsonl --output runs/kss --solution kss --ideal-epochs 30
python scripts/train_bargaining.py --config configs/audiocaps_qa.yaml --manifest data/manifest.jsonl --output runs/uts --solution uts
```

By default all configured independent seeds are run. `--seed` selects one seed.
Each KSS run stores fixed per-query ideals, their query IDs and modality order.
Each constrained run stores objective history, equality residuals, penalty
history, dual variables, solver settings, and a feasibility flag. No test data
are used for either ideal estimation or optimization. Downstream answerer/judge
evaluation remains a separate genuine experiment.
