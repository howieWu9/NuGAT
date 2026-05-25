# FoTNuF

FoTNuF is a runnable implementation of Functional Optimal Transport with Nash Utility
Fusion for multimodal knowledge graph completion. The code follows the paper modules:
heterogeneous modality encoding, Functional OT alignment, Nash utility fusion, KGC
training, filtered ranking evaluation, QA evidence reranking, and concept verification.

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

The default config includes the paper values:

- learning rate: `1e-4`
- batch size: `1024`
- maximum epochs: `1000`
- scoring hidden dimension: `256`
- Functional OT transport cost strength `eta`: `5e-1`
- OT entropy regularization `gamma`: `1e-1`
- Nash utility loss weight `lambda_nash`: `5e-2`
- parameter regularization `lambda_reg`: `1e-5`
- early stopping by validation MRR
- uniform OT marginals
- Sinkhorn iterations and Nash fusion iterations

## No Implicit Model Downloads

The code never downloads Hugging Face model weights unless `allow_hf_model_load` is set to
`true`. Even then, the loader uses `local_files_only=True`, so the default behavior remains
offline with respect to model weights.

