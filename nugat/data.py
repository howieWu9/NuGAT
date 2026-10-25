from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class FeatureSchema:
    query_dim: int
    modality_dims: dict[str, int]


@dataclass
class EvidenceBatch:
    features: dict[str, torch.Tensor]
    query: torch.Tensor
    masks: dict[str, torch.Tensor]
    ids: list[str]
    metadata: list[dict[str, Any]]

    def to(self, device: torch.device | str) -> "EvidenceBatch":
        return EvidenceBatch(
            {name: value.to(device) for name, value in self.features.items()},
            self.query.to(device),
            {name: value.to(device) for name, value in self.masks.items()},
            self.ids,
            self.metadata,
        )


def read_manifest(path: str | Path) -> list[dict[str, Any]]:

    path = Path(path)
    records: list[dict[str, Any]] = []
    seen: dict[str, str] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Record {line_number} must be a JSON object")
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError(f"Record {line_number} requires a nonempty string id")
            split = record.get("split")
            if split not in SPLITS:
                raise ValueError(f"Record {identifier}: split must be one of {sorted(SPLITS)}")
            if identifier in seen:
                raise ValueError(
                    f"Duplicate sample id {identifier!r} in {seen[identifier]} and {split}; "
                    "sample ids must be disjoint across splits"
                )
            seen[identifier] = split
            if not isinstance(record.get("query"), str) or not record["query"].strip():
                raise ValueError(f"Record {identifier} requires a nonempty query")
            if not isinstance(record.get("query_features"), str):
                raise ValueError(f"Record {identifier} requires query_features as a .npy path")
            modalities = record.get("modalities")
            if not isinstance(modalities, dict) or not modalities:
                raise ValueError(f"Record {identifier} requires nonempty modalities")
            if any(not isinstance(k, str) or not k or not isinstance(v, str)
                   for k, v in modalities.items()):
                raise ValueError(f"Record {identifier}: modalities must map names to .npy paths")
            records.append(record)
    if not records:
        raise ValueError(f"Manifest is empty: {path}")
    return records


def _feature_path(base: Path, value: str) -> Path:
    path = Path(value)
    path = path if path.is_absolute() else base / path
    if path.suffix.lower() != ".npy":
        raise ValueError(f"Cached features must be .npy arrays: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Cached feature file is missing: {path}")
    return path


def _read_array(path: Path, dimensions: int, *, metadata_only: bool = False) -> np.ndarray:

    try:
        array = np.load(path, allow_pickle=False, mmap_mode="r" if metadata_only else None)
    except (ValueError, OSError) as exc:
        raise ValueError(f"Cannot read numerical features from {path}: {exc}") from exc
    if not isinstance(array, np.ndarray) or array.ndim != dimensions:
        raise ValueError(f"{path} must contain a {dimensions}-dimensional array")
    if not all(array.shape):
        raise ValueError(f"Empty feature axis in {path}")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"Features must have a floating dtype: {path}")
    if not metadata_only and not np.isfinite(array).all():
        raise ValueError(f"Non-finite feature value in {path}")
    return array


class ManifestDataset(Dataset):


    def __init__(self, manifest: str | Path, split: str,
                 schema: FeatureSchema | None = None) -> None:
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split}")
        self.manifest = Path(manifest).resolve()
        self.split = split
        self.records = [r for r in read_manifest(self.manifest) if r["split"] == split]
        if not self.records:
            raise ValueError(f"No {split!r} records in {self.manifest}")
        base = self.manifest.parent
        self._paths: list[tuple[Path, dict[str, Path]]] = []
        for record in self.records:
            query_path = _feature_path(base, record["query_features"])
            modality_paths = {m: _feature_path(base, p) for m, p in record["modalities"].items()}
            query_shape = _read_array(query_path, 1, metadata_only=True).shape
            dims = {m: _read_array(p, 2, metadata_only=True).shape[1]
                    for m, p in modality_paths.items()}
            current = FeatureSchema(query_shape[0], dims)
            if schema is None:
                schema = current
            if current != schema:
                raise ValueError(
                    f"Feature schema mismatch for {record['id']}: {current}; expected {schema}. "
                    "Every record must have the same modalities and encoder dimensions."
                )
            self._paths.append((query_path, modality_paths))
        assert schema is not None
        self.schema = schema

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        query_path, modality_paths = self._paths[index]
        return {
            "id": record["id"],
            "query": torch.from_numpy(_read_array(query_path, 1).astype(np.float32, copy=True)),
            "features": {
                name: torch.from_numpy(_read_array(path, 2).astype(np.float32, copy=True))
                for name, path in modality_paths.items()
            },
            "metadata": dict(record),
        }


def collate_evidence(samples: Sequence[Mapping[str, Any]]) -> EvidenceBatch:

    if not samples:
        raise ValueError("Cannot collate an empty batch")
    names = list(samples[0]["features"])
    if any(set(sample["features"]) != set(names) for sample in samples):
        raise ValueError("Modalities differ within a batch")
    features: dict[str, torch.Tensor] = {}
    masks: dict[str, torch.Tensor] = {}
    for name in names:
        tensors = [sample["features"][name] for sample in samples]
        width = tensors[0].shape[1]
        if any(t.ndim != 2 or t.shape[1] != width or t.shape[0] == 0 for t in tensors):
            raise ValueError(f"Invalid feature shape for modality {name}")
        length = max(t.shape[0] for t in tensors)
        padded = tensors[0].new_zeros((len(tensors), length, width))
        valid = torch.zeros((len(tensors), length), dtype=torch.bool)
        for i, tensor in enumerate(tensors):
            padded[i, :tensor.shape[0]] = tensor
            valid[i, :tensor.shape[0]] = True
        features[name], masks[name] = padded, valid
    return EvidenceBatch(
        features=features,
        query=torch.stack([sample["query"] for sample in samples]),
        masks=masks,
        ids=[str(sample["id"]) for sample in samples],
        metadata=[dict(sample["metadata"]) for sample in samples],
    )
