import numpy as np
import pytest
import torch
from torch import nn
import json
import importlib.util
from pathlib import Path

from nugat.encoders import FrozenModule, _load_pretrained, _loading_options, freeze_module, load_audio_waveform, load_video_frames


def test_frozen_wrapper_disables_gradients_and_stays_in_eval_mode():
    model = nn.Sequential(nn.Linear(3, 4), nn.Dropout(0.8), nn.Linear(4, 2))
    wrapped = FrozenModule(model)
    wrapped.train(True)
    assert not wrapped.training and not model.training
    assert all(not p.requires_grad for p in model.parameters())
    inputs = torch.ones(2, 3, requires_grad=True)
    first, second = wrapped(inputs), wrapped(inputs)
    torch.testing.assert_close(first, second)
    assert not first.requires_grad


def test_freeze_helper_returns_the_original_module():
    model = nn.Linear(2, 2)
    assert freeze_module(model) is model
    assert not model.training
    assert all(not p.requires_grad for p in model.parameters())


def test_loading_requires_explicit_checkpoint_and_is_offline_by_default():
    with pytest.raises(ValueError, match="explicit pretrained"):
        _loading_options("", None, False)
    assert _loading_options("a/local-checkpoint", "fixed-revision", False) == {"local_files_only": True, "revision": "fixed-revision"}


def test_checkpoint_missing_encoder_parameters_is_rejected():
    class IncompleteCheckpoint:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return nn.Linear(2, 2), {"missing_keys": ["weight"], "unexpected_keys": []}

    with pytest.raises(ValueError, match="does not fully initialize"):
        _load_pretrained(IncompleteCheckpoint, "test-double", {})


def test_video_loader_preserves_explicit_clip_frames(tmp_path):
    path = tmp_path / "clip.npy"
    frames = np.zeros((4, 3, 8, 8), dtype=np.uint8)
    np.save(path, frames)
    np.testing.assert_array_equal(load_video_frames(path, 4), frames)
    with pytest.raises(ValueError, match="sample clips explicitly"):
        load_video_frames(path, 16)
    np.save(path, frames.astype(np.float32))
    with pytest.raises(ValueError, match="uint8 RGB"):
        load_video_frames(path, 4)


def test_audio_loader_does_not_implicitly_mix_channels(tmp_path):
    path = tmp_path / "segment.npy"
    waveform = np.linspace(-1, 1, 400, dtype=np.float32)
    np.save(path, waveform)
    np.testing.assert_array_equal(load_audio_waveform(path, 16000), waveform)
    np.save(path, np.stack([waveform, waveform], axis=-1))
    with pytest.raises(ValueError, match="mono waveform"):
        load_audio_waveform(path, 16000)


def test_extraction_manifest_is_accepted_by_training_dataset(tmp_path, monkeypatch):


    spec = importlib.util.spec_from_file_location("extract_features_test", Path(__file__).parents[1] / "scripts" / "extract_features.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class EncoderDouble:
        def __init__(self, **kwargs):
            pass

        def encode(self, samples, batch_size=32):
            return torch.tensor([[float(len(sample)), 1., 2.] for sample in samples])

    monkeypatch.setattr(module, "FrozenSentenceEncoder", EncoderDouble)
    monkeypatch.setitem(module.ENCODER_TYPES, "text", EncoderDouble)
    raw_path = tmp_path / "raw.jsonl"
    record = {"id": "sample-1", "split": "train", "query": "Which sound?", "modalities": {"text": ["A dog barks.", "A bell rings."]}, "answer": "A dog"}
    raw_path.write_text(json.dumps(record) + "\n")
    output = tmp_path / "cached.jsonl"
    module.extract_manifest(raw_path, output, {"query": {"model_id": "test-double-only"}, "text": {"model_id": "test-double-only"}})
    from nugat.data import ManifestDataset
    dataset = ManifestDataset(output, "train")
    assert dataset.schema.query_dim == 3
    assert dataset.schema.modality_dims == {"text": 3}
    result = json.loads(output.read_text())
    assert result["answer"] == "A dog"
    assert result["feature_provenance"]["frozen"] is True
    assert result["raw_modalities"] == record["modalities"]
    assert not Path(result["query_features"]).is_absolute()
