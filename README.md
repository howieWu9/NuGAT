Multimodal reasoning integrates query-relevant information distributed across heterogeneous modalities to support the subsequent inference task. However, existing reasoning methods often fail to establish fine-grained correspondences between modality-specific samples and shared evidence units, causing query-irrelevant or semantically mismatched samples to be aggregated into the same evidence representation and thereby weakening subsequent multimodal fusion. Moreover, grounded modality representations can also differ substantially in reliability and informativeness, leading to modality dominance and suppressing weaker yet complementary evidence. To address these two issues, this paper proposes Nash utility-Guided Attention Transport (NuGAT), which integrates query-conditioned evidence grounding with balanced multimodal fusion. Specifically, the Query-Conditioned Attention Transport (QAT) component is introduced to learn sample–evidence correspondences by combining attention-based matching with a query-conditioned compatibility prior, while Nash Utility Fusion (NUF) component is proposed to then estimate modality utility from evidence-unit grounding quality and perform utility-guided subspace retention before iterative fusion. The two components are jointly optimized in an end-to-end manner to connect evidence grounding and multimodal fusion. The ablation study results further demonstrate that QAT and NUF provide complementary benefits by improving fine-grained evidence grounding and balancing modality contributions, respectively.

## Implementation alignment

Transport plans have shape `[batch, samples, evidence_units]`. All four variants enforce **P >= 0**, with unit column sums over valid samples. Padding receives zero mass. QAT uses `ColSoftmax(S / tau + B)`; no additional clipping is required.

| `qat_variant` | Matching | Marginal constraints | Regularization |
|---|---|---|---|
| `eot` | Squared Euclidean cost | Two-sided | Entropy, no query prior |
| `cot` | Attention-derived scores | Two-sided | Entropy, no query prior |
| `without_prior` (w/o B) | Attention-derived scores | One-sided | Entropy, no query prior |
| `qat` | Attention-derived scores | One-sided | KL(P \|\| exp(B)) |

EOT/COT fix each valid source row sum to `evidence_units / valid_samples`, keeping the column mass identical across comparisons. With identical projections and fixed marginals, squared Euclidean cost differs from a scaled dot-product cost by row/column constants; temperature must therefore be considered when interpreting EOT versus COT.

Initial evidence follows `Y_hat_m^0 = W_hat_m @ X_m^V`. `NuGAT` requires `modality_sizes`, the fixed source capacity for each modality, to initialize `W_hat_m` before optimizer creation. Training infers these capacities from training features unless `model.modality_sizes` is supplied. Shorter inputs are masked; larger inputs require capacities configured before training. Version 1 checkpoints used independent evidence parameters and require retraining; version 2 stores source capacities with the weights.

NUF retains the Gaussian hard masks and utility feedback in Eqs. (11)–(17). Hard-mask training uses a Gaussian-CDF straight-through gradient, and degenerate SVD cases use a fixed-basis gradient. These backward-pass choices are not specified by the manuscript. Existing deterministic evaluation remains the default; set `model.stochastic_eval: true` to retain Gaussian sampling during evaluation.

`scripts/evaluate.py` supports `--task mvu` for multiple-choice accuracy. MVU predictions contain `candidates` and zero-based integer `label`/`prediction` fields; reported accuracy is a fraction (multiply by 100 for the paper's percentage tables). The paper uses LongVideoBench validation and full Video-MME without subtitles, with Qwen2.5-VL-7B. Dataset preparation and the frozen answerer adapter must implement that protocol; the generic adapter interface alone does not reproduce benchmark results.

Core runtime dependencies are PyTorch, NumPy, and PyYAML. Run the regression suite with `python -m pytest -q` after installing pytest. Feature extraction additionally requires the encoder dependencies listed in `nugat/encoders.py`.
