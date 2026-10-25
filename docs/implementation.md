# Implementation and manuscript correspondence

This revision implements the NuGAT equations in manuscript PR-D-26-09510 and
the agreed revision details: learnable Xavier-initialized modality evidence,
the shared compatibility MLP, 30 training epochs, three refinement iterations,
and five independent runs. It is a new implementation of that specification,
not a recovered copy of the code that produced the manuscript's tables.

## Equations

| Manuscript | Implementation |
| --- | --- |
| Eqs. (3)–(4) | `nugat/qat.py`: shared evidence-query projection, modality-specific key/value projections, scaled dot-product scores |
| Eq. (5) | Shared MLP: concatenated projected query, sample key and previous evidence; `3d → d → 1`, GELU; source-axis log-softmax |
| Eq. (8) | `softmax(S / tau_QAT + B, dim=source)`; only evidence columns are normalized |
| Grounded evidence | `P.transpose(-2, -1) @ values`; shape `[batch, n, d]` |
| Eqs. (11)–(12) | `nugat/fusion.py`: thin SVD, Gaussian latent gates, hard binary masking of singular directions |
| Eq. (13) | Uniform mean of retained modalities plus the previous shared representation with residual coefficient lambda |
| Eq. (14) | Cosine similarities normalized across modalities at the same evidence index, then averaged across evidence indices |
| Eqs. (15)–(16) | Exponential utility and normalized utilities for the next iteration's retention strength |
| Eq. (17) | Negative sum of log utilities with epsilon, evaluated using `logaddexp`; batch mean |
| Final evidence | `nugat/model.py`: exactly L QAT/NUF updates, returning the final shared representation |

At the reported dimension d=256 the compatibility MLP is 768→256→1. Input
dimensions are inferred from the actual feature files. The output dimension is
256, not the sum of modality dimensions. Query and modality encoders are frozen;
only the NuGAT parameters are optimized by the training runner.

## Explicit implementation conventions

The supplied manuscript does not uniquely specify the following choices. They
are made explicit here rather than represented as recovered experimental facts.

* Each modality has one learnable n×d initial evidence matrix, Xavier uniform
  initialized once. It is reused across examples and learned during training.
* Shared Y^0 is the mean of these initial modality matrices. The original
  shared-initialization prose calls its inputs retained, but the formula uses
  the grounded-evidence symbol and does not define a zeroth retention pass.
  This implementation does not add an unreported extra QAT pass.
* The previous grounded modality matrix is the next QAT input. The shared
  fused matrix is the residual state in Eq. (13).
* The first retention strengths are uniform. Thresholds are sigmoid-parameterized
  per modality, singular direction and iteration. Their initial values equal
  the initial uniform strength (0.5 for a single modality).
* The forward gate is the specified hard indicator. Its training derivative
  uses a straight-through Gaussian-CDF surrogate because a literal indicator
  has zero derivative almost everywhere. Evaluation uses the sign of the
  Gaussian mean; `stochastic_eval=True` explicitly enables sampled evaluation.
* Ordinary differentiable thin SVD is used for distinct nonzero singular values.
  At degenerate spectra, the computed basis is held fixed for the backward
  projection. Forward reconstruction is preserved; this is a numerical gradient
  convention, not an exact derivative at an undefined singular basis.
* The loss is taken at the final iteration; gradients flow through earlier
  iterations. Epsilon is 1e-8. Seeds 42–46 are new reproducible defaults, not
  asserted historical seeds. They are saved with every run.

## Ablations

`qat_variant: without_prior` removes B and is exactly dot-product cross-attention
with the specified temperature. `eot` uses squared Euclidean cost; `cot` uses
negative attention similarity. EOT/COT use two-sided uniform Sinkhorn marginals.
Their unit-total plans are multiplied by n for normalized evidence aggregation.
Sinkhorn checks its marginal residual and fails if its iteration budget does
not reach the declared tolerance. Its default budget is 1000, with early
stopping after convergence; this is an implementation setting separate from
the three NuGAT refinement iterations. It is not a reported paper setting.

`fusion_variant: uniform` implements UWF by directly averaging grounded
modalities. `subspace_weighted` implements SWF using the top-k right singular
vectors, reconstruction errors of the mean grounded evidence, and softmax
weights on negative errors. SWF requires an explicit `subspace_rank` because
the manuscript does not report k. UWF/SWF return their direct aggregation;
they do not apply NUF's binary retention or its residual update.

The native-sample w/o-QAT baseline is not replaced with an invented pooling
operation: the supplied manuscript does not specify how variable-length native
samples enter its downstream answerer. Cooperative-solution comparisons and
their constraint tolerances are documented in `bargaining.md`.

## Objective scope

Equation (14) normalizes over modalities, not over other evidence positions or
batch examples. The implementation preserves that equation. With negligible
epsilon, the Nash objective encourages equal relative cosine probabilities;
it does not by itself guarantee high absolute alignment or exclude collapsed
representations. With only one modality its loss is constant. The standard
training runner therefore rejects one-modality Nash training instead of
silently adding an unreported objective.

## Reproduction boundary

The archive contains no original QA/MCV data, released modality features,
historical checkpoints, five-run logs, exact encoder checkpoint versions,
answerer interface, generation prompts, or Llama-3 judge rubric. Code tests
establish numerical behavior and executable training mechanics; they do not
establish reproduction of any reported M.J. or concept-verification score.
No manuscript score is hard-coded into training or evaluation.

AudioCaps-QA and VALOR configurations use the values confirmed in the revision.
VCGPT/MCV hyperparameters, historical grid values and the original selection
metric are not supplied, so no dataset-specific values are invented for them.
Exact benchmark evaluation additionally requires the missing downstream
protocol described in `evaluation.md`.
