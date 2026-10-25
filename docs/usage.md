# Running NuGAT

The README contains only the paper abstract. Operational details are here.

## Installation and checks

Use Python 3.10 or newer and install the appropriate PyTorch build for the
machine, then install this package:

```bash
python -m pip install -e '.[test]'
python -m pytest
```

For feature extraction, additionally install `.[encoders]`. Large pretrained
weights and benchmark data are not bundled or downloaded implicitly.

## Cached real features

Supply one JSONL manifest with unique IDs and explicit train, validation and
test splits. Each query is a floating-point `.npy` vector; each modality is a
floating-point `.npy` matrix with one row per native sample. Paths are relative
to the manifest. This is a schema example, not a supplied benchmark record:

```json
{"id":"example-001","split":"train","query":"What happens?","query_features":"features/query_001.npy","modalities":{"text":"features/text_001.npy","audio":"features/audio_001.npy"},"answer":"Reference answer"}
```

All records in an experiment must have the same modality names and feature
dimensions; sample counts may differ. Batches are padded with source masks.
Original evidence text, media paths and candidate metadata can be included for
the downstream adapter. Labels and reference answers are not training inputs.
Structural features, if used, must be supplied as actual cached features with
their extraction provenance; no entity-ID pseudo-feature fallback exists.

## Frozen encoder extraction

`scripts/extract_features.py` supports SBERT text/query features, CLIP image
features, VideoMAE clips and wav2vec2 audio segments. Supply exact checkpoint IDs
or local model paths and revisions in a JSON configuration under `query` and
each used modality name. Each entry has `model_id` and optionally `revision`.
No exact historical checkpoint is guessed.

Raw manifest modalities are lists: text strings, image paths, video clip paths,
or audio objects with `path` and `sampling_rate`. Video files are predecoded
uint8 RGB arrays `[frames, 3, height, width]` with the checkpoint's frame count.
Audio is mono, at the declared encoder sampling rate. Sampling, resampling and
channel mixing must be done explicitly. These preprocessing conventions are
recorded as implementation choices.

```bash
python scripts/extract_features.py --manifest data/raw.jsonl --output data/features.jsonl --encoder-config data/encoders.json
```

Use `--allow-download` only when the specified weights should be fetched.

## Training

```bash
python scripts/train.py --config configs/audiocaps_qa.yaml --manifest data/audiocaps_qa.jsonl --output runs/audiocaps_qa
python scripts/train.py --config configs/valor.yaml --manifest data/valor.jsonl --output runs/valor
```

The supplied configurations run Adam at 1e-4, batch size 1024, 30 epochs and
L=3, with n=16 and d=256. Each configuration runs five independent seeds.
Each seed directory contains the actual checkpoint, resolved settings and
training log. Existing nonempty run directories are rejected to avoid mixing
runs. `--validate` additionally evaluates the validation split.

| Dataset | tau_QAT | tau_u | lambda | tau_g |
| --- | ---: | ---: | ---: | ---: |
| AudioCaps-QA | 0.10 | 0.15 | 0.20 | 0.05 |
| VALOR | 0.20 | 0.10 | 0.40 | 0.10 |

Create an explicit configuration for other datasets from their verified run
settings; the two configurations above are not claimed to apply to VCGPT/MCV.

## Grid search

Pass a YAML mapping of model parameter names to candidate lists via `--grid`.
The actual historical grids are not supplied. The script evaluates candidates
on validation data only and saves the criterion, seed results and chosen values.

```bash
python scripts/grid_search.py --config configs/audiocaps_qa.yaml --manifest data/audiocaps_qa.jsonl --grid data/search_grid.yaml --output runs/search --metric-callback my_validation:evaluate_mj --metric-name M.J. --maximize
```

The callback has signature `(model, validation_loader, device) -> float` and
must run genuine validation evaluation. Without a callback the runner uses
validation Nash loss and records that choice. Searching utility temperature
with that default is rejected because its loss scale changes with temperature.
Test-set M.J. is never a hyperparameter-selection criterion in this runner.

## Inference and evaluation

```bash
python scripts/evaluate.py infer --checkpoint runs/valor/seed_42/checkpoint.pt --manifest data/valor.jsonl --split test --output runs/valor_inference
```

This exports actual fused evidence and transport traces. It does not generate
answers unless an actual frozen answerer adapter is supplied. See
`evaluation.md` for answerer/judge contracts, scoring recorded predictions and
summarizing independent runs. See `bargaining.md` for KSS/EGS/UTS experiments.
