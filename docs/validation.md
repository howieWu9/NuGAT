# Validation of this implementation

Checked on 25 October 2026 with Python 3.12 and PyTorch 2.14.1, using CPU
execution. This records software validation, not benchmark reproduction.

* The complete automated suite passes: **70 tests**. Tests cover QAT equations,
  masking, balanced-OT convergence, NUF utility/gate equations and gradients,
  constrained bargaining, dataset splits, training, checkpoints and metrics.
* A forward/backward check uses the reported n=16, d=256 and L=3 with three
  modalities. Outputs have shape `[2, 16, 256]`, and the loss and gradients are
  finite, including when source counts are below the evidence count.
* The ordinary training CLI completes two epochs for two independent seeds
  on explicitly synthetic test fixtures. Model parameters, including the
  compatibility MLP, update; saved checkpoints reproduce identical inference
  after reloading.
* Nash/EGS/KSS/UTS comparison CLI tests exercise actual model training and
  serialization on small fixtures. Solver tests check equality feasibility and
  rejection/marking of failed candidates; they do not certify global optima.
* Inference exports fused evidence and source transport arrays from a trained
  test checkpoint, with correct column normalization and separately identified
  current/next retention strengths.
* Encoder freezing, feature-file interoperability and preprocessing validation
  are tested. Actual pretrained weights are not downloaded or evaluated here.

To run the checks:

```bash
OMP_NUM_THREADS=1 python -m pytest -q
python -m compileall -q nugat scripts
```

No H100 benchmark training, real VideoLLaMA2/Qwen2.5-Omni inference, Llama-3
judging, or reproduction of the manuscript's numeric results is claimed. The
required original inputs and unresolved protocol details are listed in
`implementation.md` and `evaluation.md`.
