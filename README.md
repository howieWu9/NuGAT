# NuGAT

Multimodal reasoning requires models to ground task-relevant evidence from heterogeneous modalities, such as text, images, audio, video, and structural. Despite recent progress, existing methods often yield fragmented evidence grounding and rely on fixed alignment costs, predefined shared spaces, or static fusion weights. As a result, they struggle to explicitly model evidence-anchor correspondences and remain vulnerable to modality dominance, where strong modalities overwhelm weaker but complementary ones. We propose NuGAT, a general evidence-grounded multimodal alignment and fusion framework based on Attention Transport (AT) and Nash Utility Fusion (NUF). AT formulates evidence grounding as a structured evidence-anchor correspondence learning problem, in which modality-specific evidence units are adaptively transported to evidence anchors through attention-guided and structure-aware transport. This design produces explicit and interpretable evidence-anchor correspondences beyond fixed alignment costs. NUF treats each modality as a utility-bearing modality and estimates its utility gain by comparing its alignment quality with a modality-specific disagreement state. Rather than using utility gain as a scalar fusion weight, NuGAT uses it to control modality-specific subspace retention, preserving reliable evidence directions while suppressing noisy or weakly grounded signals. Experiments on multimodal QA and concept verification demonstrate the effectiveness of NuGAT. Compared with the recent strongest baseline, NuGAT achieves an average improvement of 3.37 across audio-visual QA under fixed answerers.

The default pipeline downloads only dataset split files. Hugging Face model ids are recorded
for reproducibility, but model weights are not downloaded by default. When released modality
features are unavailable, the code materializes deterministic modality features from the real
dataset entity ids so that training and evaluation can run end to end.

## Verified Model Registry

The model ids below are real Hugging Face repositories and are used as encoder specs:

| Modality | Model id | Role |
| --- | --- | --- |
| Text | `sentence-transformers/all-MiniLM-L6-v2` | sentence/text feature encoder |
| Image | `openai/clip-vit-base-patch32` | image feature encoder |
| Audio | `facebook/wav2vec2-base-960h` | audio feature encoder |
| Video | `MCG-NJU/videomae-base-finetuned-kinetics` | video feature encoder |
| QA MLLM | `Qwen/Qwen2.5-VL-7B-Instruct` | fixed downstream answerer reference |
| Audio QA MLLM | `Qwen/Qwen2-Audio-7B-Instruct` | fixed audio answerer reference |

## Quick Start

```bash
python scripts/download_data.py --dataset TIVA
python scripts/smoke.py --dataset TIVA --epochs 1 --limit-train 256 --eval-max-triples 8
```

Train a small run:

```bash
python scripts/train.py --dataset TIVA --epochs 5 --batch-size 128 --limit-train 2048
python scripts/evaluate.py --dataset TIVA --checkpoint runs/TIVA/latest.pt --eval-max-triples 64
```

Run a paper-style configuration by increasing `--epochs`, removing `--limit-train`, and
using the defaults from `configs/default.yaml`.

## Data

The downloader fetches the public NativE benchmark split files for:

- `TIVA`
- `Kuai16K`, used as the KVC16K benchmark in the paper

Files are stored under `data/raw/<dataset>`. Generated deterministic features are stored under
`data/processed/<dataset>/features.pt`. These generated features are for runnable experiments
and smoke tests. If released modality features are available, place them under the processed
directory and set `feature_source: released` in the config.

## Paper Hyperparameters Covered

The default implementation follows the paper's reported configuration for evidence grounding and utility-guided fusion:

- learning rate: `1e-4`
- batch size: `1024`
- maximum epochs: `1000`
- scoring hidden dimension: `256`
- attention transport temperature `tau_at`: `1e-1`
- utility temperature `tau_u`: `1.5e-1`
- residual fusion coefficient `lambda`: `2e-1`
- subspace gate temperature `tau_g`: `5e-2`
- Nash utility loss weight `lambda_nash`: `5e-2`
- parameter regularization `lambda_reg`: `1e-5`
- early stopping by validation MRR
- uniform transport marginals
- Sinkhorn iterations and Nash fusion iterations

## No Implicit Model Downloads

The code never downloads Hugging Face model weights unless `allow_hf_model_load` is set to
`true`. Even then, the loader uses `local_files_only=True`, so the default behavior remains
offline with respect to model weights.

