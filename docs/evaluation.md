# Evaluation and grounding exports

The supplied manuscript identifies VideoLLaMA2/Qwen2.5-Omni as frozen answerers
and Llama-3 as the judge. It does not identify exact checkpoint revisions, the
latent-evidence input adapter, the complete prompting protocol, or the judge
score scale. Those details cannot be recovered from the original ZIP. This
implementation exports actual NuGAT representations and accepts explicit model
adapters; it does not replace the answerers with string rules or hard-code paper
scores. Using an adapter does not establish that it reproduces the original
experimental protocol.

## Export a trained model's evidence

```bash
python scripts/evaluate.py infer --manifest data/manifest.jsonl \
  --checkpoint runs/audiocaps/seed_42/checkpoint.pt --split test \
  --output runs/audiocaps/seed_42/evaluation
```

Use the actual checkpoint filename emitted by training. Each example receives a
numerical NPZ file containing the fused evidence, final transport matrices, and
both modality retention vectors: `retention` is alpha^L used by the final gate;
`next_retention` is alpha^(L+1) computed from final utilities for a possible next
iteration. `predictions.jsonl` also records the source indices
ranked by mean transport mass across evidence units. These are grounding traces,
not question-answer predictions. Files are associated with the original sample
ID; filenames use SHA-256 to prevent sample IDs from becoming unsafe paths.

`--predictor my_adapter:make_answerer --predictor-options answerer.json` loads a
trusted local factory. Its options should specify the exact frozen checkpoint,
revision, evidence-to-input conversion, prompt and decoding settings. The factory
receives an options dictionary and returns a callable with this interface:

```python
def answerer(*, example, grounding) -> str:
    # Run the real frozen VideoLLaMA2 or Qwen2.5-Omni model here.
    # example contains the query, native evidence metadata, and file paths.
    # grounding contains fused [n,d], per-modality transport [samples,n],
    # retention/next_retention [modalities], rankings and the manifest directory.
    ...
```

The callback has access to numerical grounded evidence and native source paths,
so the chosen model-specific bridge can use the actual multimodal inputs. No
implicit text serialization or untrained latent projection is substituted.
Only declared input fields are passed to the predictor: ID, query, cached feature
paths, evidence text, raw modality sources and optional plain-string candidate
choices. Arbitrary metadata and annotations are excluded. Native audio objects
expose only their path and sampling rate, and candidate objects carrying labels
are rejected. Reference answers remain available only to the judge.
`--judge my_adapter:make_judge --judge-options judge.json` returns a callable
`judge(query=..., reference=..., prediction=...) -> float` that runs the actual
frozen Llama-3 judge. Missing reference answers raise an error. Direct PyTorch
module adapters are frozen automatically and all calls run under inference mode;
factory implementations wrapping multiple models must freeze those models too.
Adapters can execute trusted local code and should not be imported from an
untrusted source. No model downloads or remote inference occur without an
explicitly supplied adapter.

For MCV, pass `--task mcv` and a real verifier through `--predictor`. The callable
returns a probability of label 1 and receives no gold label. The manifest must
include a binary `label`. The paper does not specify a trainable verification
head or its supervision, so this release does not invent one.

## Score recorded predictions

MQA JSONL records require `id` and numeric `judge_score`; answer text may also be
included. The reporting conversion below explicitly uses
`100 * mean(judge_score) / score_max`. Choose `score_max` and `judge_protocol` to
match the actual judge, not a guessed historical configuration.

```bash
python scripts/evaluate.py score --task mqa --dataset AudioCaps-QA \
  --predictions runs/seed_42/evaluation/predictions.jsonl \
  --seed 42 --experiment-id audiocaps-config-and-protocol-v1 \
  --answerer YOUR_EXACT_ANSWERER_ID --judge-protocol YOUR_JUDGE_AND_PROMPT_ID \
  --score-max YOUR_JUDGE_SCALE_MAXIMUM --output runs/seed_42/metrics.json
```

MCV records require `id`, integer `label` in `{0,1}`, and numerical `score` in
`[0,1]`. Precision, recall and F1 use the unweighted average over both classes;
accuracy and MCC use the binary confusion matrix. AUC uses continuous scores,
not thresholded predictions, and is null when only one class is present.
Undefined per-class precision/recall/F1 is zero. The threshold defaults to 0.5;
if it is tuned, select it on validation data only and record it.

```bash
python scripts/evaluate.py score --task mcv --dataset science-MCV \
  --predictions verifier_predictions.jsonl --seed 42 \
  --experiment-id mcv-config-and-protocol-v1 --threshold 0.5 \
  --output runs/seed_42/metrics.json
```

Use `--expected-ids evaluation_ids.jsonl` to verify complete coverage; that file
contains one `{"id": "..."}` per example. Duplicate IDs, missing scores, NaN/Inf,
or out-of-range scores are rejected rather than silently removed.

## Five runs

```bash
python scripts/summarize_runs.py runs/seed_{42,43,44,45,46}/metrics.json \
  --output runs/summary.json
```

The summary requires five distinct seeds and matching dataset, answerer,
experiment ID, example IDs and judge/metric protocol. It reports mean and sample
standard deviation (`ddof=1`) over complete run scores. For a cross-dataset
average, first compute that average independently for each matched seed and then
calculate its standard deviation; averaging dataset standard deviations is not
the standard deviation of the aggregate score. No paper result is claimed until
real data and the specified frozen answerer/judge are evaluated.
