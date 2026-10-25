"""Extract actual frozen encoder features into the NuGAT cached-feature schema.

Raw JSONL records contain id, split, query, and modalities. text is a list of
strings; image a list of image paths; video a list of uint8 RGB TCHW .npy clips;
audio a list of {path, sampling_rate} objects. Paths are relative to the raw
manifest. A JSON encoder config supplies query and modality objects containing
model_id and optional revision; no historical checkpoint is guessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nugat.encoders import ENCODER_TYPES, FrozenSentenceEncoder


def extract_manifest(raw_manifest: Path, output_manifest: Path, encoder_config: dict, *, device: str = "cpu", allow_download: bool = False, batch_size: int = 32) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    raw_manifest, output_manifest = raw_manifest.resolve(), output_manifest.resolve()
    if output_manifest.exists():
        raise FileExistsError(f"Output manifest already exists: {output_manifest}; use a new output path")
    records = [json.loads(line) for line in raw_manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise ValueError("Raw manifest is empty")
    expected_modalities = set(records[0].get("modalities", {}))
    if not expected_modalities or expected_modalities - set(ENCODER_TYPES):
        raise ValueError("Supported modality names are text, image, video, and audio")
    ids = set()
    for record in records:
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id or record_id in ids:
            raise ValueError("Record IDs must be unique nonempty strings")
        ids.add(record_id)
        if record.get("split") not in {"train", "validation", "test"}:
            raise ValueError("Every record requires split: train, validation, or test")
        if not isinstance(record.get("query"), str) or not record["query"].strip():
            raise ValueError("Every record requires a nonempty query")
        if set(record.get("modalities", {})) != expected_modalities:
            raise ValueError("All records must expose the same configured modalities")
        if any(not isinstance(samples, list) or not samples for samples in record["modalities"].values()):
            raise ValueError("Every modality requires a nonempty list of raw samples")
    required = expected_modalities | {"query"}
    if required - set(encoder_config):
        raise ValueError(f"Encoder config requires entries for {sorted(required)}")
    common = {"device": device, "allow_download": allow_download}
    for key in required:
        spec = encoder_config[key]
        if not isinstance(spec, dict) or not isinstance(spec.get("model_id"), str) or not spec["model_id"].strip():
            raise ValueError(f"Encoder {key} requires an explicit model_id")
        if set(spec) - {"model_id", "revision"}:
            raise ValueError(f"Unknown options for encoder {key}; only model_id and revision are supported")
    query_encoder = FrozenSentenceEncoder(**encoder_config["query"], **common)
    encoders = {}
    for name in sorted(expected_modalities):
        if name == "text" and encoder_config[name] == encoder_config["query"]:
            encoders[name] = query_encoder
        else:
            encoders[name] = ENCODER_TYPES[name](**encoder_config[name], **common)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    feature_dir = output_manifest.parent / (output_manifest.stem + "_features")
    feature_dir.mkdir(exist_ok=False)
    output_records = []
    for index, record in enumerate(records):
        result = dict(record)
        query_features = query_encoder.encode([record["query"]], batch_size=batch_size)[0].numpy()
        if query_features.ndim != 1 or query_features.size < 1 or not np.isfinite(query_features).all():
            raise ValueError("The query encoder returns an invalid feature vector")
        query_path = feature_dir / f"{index:08d}_query.npy"
        np.save(query_path, query_features.astype(np.float32, copy=False), allow_pickle=False)
        result["query_features"] = query_path.relative_to(output_manifest.parent).as_posix()
        result["modalities"] = {}
        result["raw_modalities"] = {}
        for name, samples in record["modalities"].items():
            if name == "text":
                prepared = samples
            elif name == "audio":
                if any(not isinstance(item, dict) or "path" not in item or "sampling_rate" not in item for item in samples):
                    raise ValueError("Audio samples require path and sampling_rate")
                prepared = [{"path": str((raw_manifest.parent / item["path"]).resolve()), "sampling_rate": item["sampling_rate"]} for item in samples]
            else:
                prepared = [(raw_manifest.parent / path).resolve() for path in samples]
            # Keep only actual inputs for downstream answerers, never arbitrary
            # annotations from raw sample dictionaries.
            result["raw_modalities"][name] = [str(value) if isinstance(value, Path) else value for value in prepared]
            features = encoders[name].encode(prepared, batch_size=batch_size).numpy()
            if features.ndim != 2 or features.shape[0] != len(samples) or not np.isfinite(features).all():
                raise ValueError(f"Encoder {name} returns an invalid feature matrix")
            path = feature_dir / f"{index:08d}_{name}.npy"
            np.save(path, features.astype(np.float32, copy=False), allow_pickle=False)
            result["modalities"][name] = path.relative_to(output_manifest.parent).as_posix()
        result["feature_provenance"] = {"encoders": encoder_config, "frozen": True, "extraction": "SBERT sentences; CLIP projected images; mean VideoMAE tokens per clip; mean wav2vec2 frames per unpadded segment"}
        output_records.append(result)
    temporary = output_manifest.with_suffix(output_manifest.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in output_records), encoding="utf-8")
    temporary.replace(output_manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="Raw JSONL manifest")
    parser.add_argument("--output", type=Path, required=True, help="New cached-feature JSONL manifest")
    parser.add_argument("--encoder-config", type=Path, required=True, help="JSON with query/text/image/audio/video checkpoint specifications")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--allow-download", action="store_true", help="Allow loading requested pretrained weights from Hugging Face")
    args = parser.parse_args()
    config = json.loads(args.encoder_config.read_text(encoding="utf-8"))
    extract_manifest(args.manifest, args.output, config, device=args.device, allow_download=args.allow_download, batch_size=args.batch_size)
    print(f"Cached features written to {args.output}")


if __name__ == "__main__":
    main()
