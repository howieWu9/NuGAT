"""Small numerical fixtures test mechanics only; they are not benchmark data."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nugat.data import ManifestDataset, collate_evidence
from nugat.training import (TrainingSettings, fit, grid_search, load_checkpoint,
                            make_loader, seed_everything, train_runs, validation_nash_loss)


def write_fixture(tmp_path: Path, duplicate: bool = False) -> Path:
    records = []
    rng = np.random.default_rng(0)
    for i, split in enumerate(["train", "train", "validation", "test"]):
        np.save(tmp_path / f"q{i}.npy", rng.normal(size=5).astype("float32"))
        np.save(tmp_path / f"text{i}.npy", rng.normal(size=(i + 1, 4)).astype("float32"))
        np.save(tmp_path / f"audio{i}.npy", rng.normal(size=(2, 3)).astype("float32"))
        records.append({"id": "sample0" if duplicate and split == "test" else f"sample{i}",
                        "split": split, "query": f"Test query {i}",
                        "query_features": f"q{i}.npy",
                        "modalities": {"text": f"text{i}.npy", "audio": f"audio{i}.npy"}})
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    return manifest


def test_source_padding_and_feature_schema(tmp_path):
    dataset = ManifestDataset(write_fixture(tmp_path), "train")
    batch = collate_evidence([dataset[0], dataset[1]])
    assert dataset.schema.query_dim == 5
    assert dataset.schema.modality_dims == {"text": 4, "audio": 3}
    assert batch.features["text"].shape == (2, 2, 4)
    assert torch.equal(batch.masks["text"], torch.tensor([[True, False], [True, True]]))
    assert torch.count_nonzero(batch.features["text"][0, 1]) == 0
    assert batch.ids == ["sample0", "sample1"]


def test_duplicate_ids_across_splits_rejected(tmp_path):
    with pytest.raises(ValueError, match="disjoint"):
        ManifestDataset(write_fixture(tmp_path, duplicate=True), "train")


def test_feature_dimensions_checked_before_training(tmp_path):
    manifest = write_fixture(tmp_path)
    np.save(tmp_path / "text1.npy", np.zeros((2, 9), dtype=np.float32))
    with pytest.raises(ValueError, match="schema mismatch"):
        ManifestDataset(manifest, "train")


def test_nonfinite_and_object_features_rejected(tmp_path):
    manifest = write_fixture(tmp_path)
    np.save(tmp_path / "q0.npy", np.full(5, np.nan, dtype=np.float32))
    dataset = ManifestDataset(manifest, "train")
    with pytest.raises(ValueError, match="Non-finite"):
        dataset[0]
    np.save(tmp_path / "q0.npy", np.array([{"unsafe": "object"}], dtype=object))
    with pytest.raises(ValueError, match="Cannot read numerical"):
        ManifestDataset(manifest, "train")


class TinyLossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, features, query, masks=None):
        return SimpleNamespace(loss=(self.scale * query).square().mean())


def test_training_count_and_test_split_protection(tmp_path):
    manifest = write_fixture(tmp_path)
    training = ManifestDataset(manifest, "train")
    settings = TrainingSettings(epochs=2, batch_size=1, learning_rate=0.1, seeds=(1,), device="cpu")
    model = TinyLossModel()
    loader = make_loader(training, 1, shuffle=True, seed=1)
    history, _ = fit(model, loader, settings, torch.device("cpu"))
    assert len(history) == 2 and [r["examples"] for r in history] == [2, 2]
    assert model.scale.item() < 1.0
    test_loader = make_loader(ManifestDataset(manifest, "test"), 1, shuffle=False, seed=1)
    with pytest.raises(ValueError, match="only accepts the train"):
        fit(model, test_loader, settings, torch.device("cpu"))
    with pytest.raises(ValueError, match="validation split"):
        validation_nash_loss(model, test_loader, torch.device("cpu"))


def test_seed_configuration_requires_independent_runs():
    assert TrainingSettings.from_mapping({}).epochs == 30
    assert len(TrainingSettings.from_mapping({}).seeds) == 5
    with pytest.raises(ValueError, match="distinct"):
        TrainingSettings.from_mapping({"seeds": [42, 42]})


def test_temperature_search_needs_a_comparable_validation_metric(tmp_path):
    with pytest.raises(ValueError, match="loss scale"):
        grid_search({}, tmp_path / "unused.jsonl", {"utility_temperature": [0.1, 0.2]},
                    tmp_path / "run")


def test_real_model_training_checkpoint_round_trip(tmp_path):
    from nugat.model import NuGAT

    manifest = write_fixture(tmp_path)
    # Avoid rank-deficient fixtures when checking gradients of the SVD model.
    rng = np.random.default_rng(9)
    for i in range(4):
        np.save(tmp_path / f"text{i}.npy", rng.normal(size=(4, 4)).astype("float32"))
        np.save(tmp_path / f"audio{i}.npy", rng.normal(size=(4, 3)).astype("float32"))
    config = {"dataset": "synthetic_mechanics_test_only",
              "model": {"num_evidence_units": 2, "evidence_dim": 4, "iterations": 3},
              "training": {"epochs": 1, "batch_size": 2, "learning_rate": 1e-3,
                           "seeds": [7, 8], "device": "cpu"}}
    summary = train_runs(config, manifest, tmp_path / "runs", validation=True)
    assert summary["run_count"] == 2
    assert all(run["epochs"] == 1 for run in summary["runs"])
    checkpoint_path = tmp_path / "runs" / "seed_7" / "checkpoint.pt"
    model, checkpoint = load_checkpoint(checkpoint_path)
    seed_everything(7)
    initial = NuGAT(**checkpoint["model_kwargs"])
    initial_parameters = dict(initial.named_parameters())
    changes = {name: not torch.equal(parameter, initial_parameters[name])
               for name, parameter in model.named_parameters()}
    assert any(changes.values())
    assert any(changed and "compatibility" in name for name, changed in changes.items())
    dataset = ManifestDataset(manifest, "validation")
    batch = collate_evidence([dataset[0]])
    again, _ = load_checkpoint(checkpoint_path)
    with torch.no_grad():
        first = model(batch.features, batch.query, masks=batch.masks).fused
        second = again(batch.features, batch.query, masks=batch.masks).fused
    assert torch.equal(first, second)
