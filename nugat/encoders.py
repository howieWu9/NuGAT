from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor, nn


def freeze_module(model: nn.Module) -> nn.Module:

    return model.requires_grad_(False).eval()


class FrozenModule(nn.Module):


    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = freeze_module(model)
        self.train(False)

    def train(self, mode: bool = True) -> "FrozenModule":
        super().train(False)
        return self

    @torch.inference_mode()
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.model(*args, **kwargs)


def _loading_options(model_id: str, revision: str | None, allow_download: bool) -> dict[str, Any]:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("An explicit pretrained checkpoint ID or local model path is required")
    result: dict[str, Any] = {"local_files_only": not allow_download}
    if revision is not None:
        result["revision"] = revision
    return result


def _transformers() -> Any:
    try:
        import transformers
    except ImportError as exc:
        raise ImportError("Feature extraction requires transformers>=4.45,<5, sentence-transformers>=3,<6, Pillow, and soundfile") from exc
    return transformers


def _load_pretrained(model_class: Any, model_id: str, options: dict[str, Any]) -> nn.Module:

    model, loading = model_class.from_pretrained(model_id, output_loading_info=True, **options)
    missing = loading.get("missing_keys", [])
    mismatched = loading.get("mismatched_keys", [])
    errors = loading.get("error_msgs", [])
    if missing or mismatched or errors:
        raise ValueError(f"Checkpoint does not fully initialize the requested encoder: missing={missing}, mismatched={mismatched}, errors={errors}")


    return model


def _check_samples(samples: Sequence[Any]) -> None:
    if isinstance(samples, (str, bytes)) or not samples:
        raise ValueError("Supply a nonempty list of modality-specific samples")


class FrozenSentenceEncoder(FrozenModule):


    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu", allow_download: bool = False) -> None:
        options = _loading_options(model_id, revision, allow_download)
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("SBERT extraction requires sentence-transformers>=3,<6") from exc
        super().__init__(SentenceTransformer(model_id, device=device, **options))

    @torch.inference_mode()
    def encode(self, samples: Sequence[str], batch_size: int = 32) -> Tensor:
        _check_samples(samples)
        if any(not isinstance(text, str) or not text.strip() for text in samples):
            raise ValueError("Text samples must be nonempty strings")
        return self.model.encode(list(samples), batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=False, show_progress_bar=False).detach().cpu().float()


class FrozenImageEncoder(FrozenModule):


    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu", allow_download: bool = False) -> None:
        options = _loading_options(model_id, revision, allow_download)
        transformers = _transformers()
        model = _load_pretrained(transformers.CLIPVisionModelWithProjection, model_id, options).to(device)
        super().__init__(model)
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id, **options)
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode(self, samples: Sequence[str | Path], batch_size: int = 32) -> Tensor:
        _check_samples(samples)
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Image feature extraction requires Pillow") from exc
        outputs = []
        for start in range(0, len(samples), batch_size):
            images = []
            for path in samples[start:start + batch_size]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            inputs = self.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            outputs.append(self.model(pixel_values=pixels).image_embeds.cpu().float())
        return torch.cat(outputs, dim=0)


def load_video_frames(path: str | Path, expected_frames: int) -> np.ndarray:

    frames = np.load(path, allow_pickle=False)
    if frames.dtype != np.uint8 or frames.ndim != 4 or frames.shape[1] != 3:
        raise ValueError("Video input must be a uint8 RGB .npy array with shape [frames, 3, height, width]")
    if frames.shape[0] != expected_frames:
        raise ValueError(f"Expected {expected_frames} frames per clip; sample clips explicitly before extraction")
    if frames.shape[2] < 1 or frames.shape[3] < 1:
        raise ValueError("Video frames must have nonzero spatial dimensions")
    return frames


class FrozenVideoEncoder(FrozenModule):


    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu", allow_download: bool = False) -> None:
        options = _loading_options(model_id, revision, allow_download)
        transformers = _transformers()
        model = _load_pretrained(transformers.VideoMAEModel, model_id, options).to(device)
        super().__init__(model)
        self.processor = transformers.VideoMAEImageProcessor.from_pretrained(model_id, **options)
        self.device = torch.device(device)
        self.num_frames = int(self.model.config.num_frames)

    @torch.inference_mode()
    def encode(self, samples: Sequence[str | Path], batch_size: int = 1) -> Tensor:
        _check_samples(samples)
        outputs = []
        for path in samples:
            frames = load_video_frames(path, self.num_frames)

            images = [np.transpose(frame, (1, 2, 0)) for frame in frames]
            inputs = self.processor(images=images, return_tensors="pt")
            pixels = inputs["pixel_values"].to(self.device)
            outputs.append(self.model(pixel_values=pixels).last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(outputs, dim=0)


def load_audio_waveform(path: str | Path, sampling_rate: int) -> np.ndarray:

    path = Path(path)
    if not isinstance(sampling_rate, int) or sampling_rate < 1:
        raise ValueError("An explicit positive integer audio sampling_rate is required")
    if path.suffix.lower() == ".npy":
        waveform = np.load(path, allow_pickle=False)
        if not np.issubdtype(waveform.dtype, np.floating):
            raise ValueError("Audio .npy arrays must contain floating-point waveform values")
    else:
        try:
            import soundfile
        except ImportError as exc:
            raise ImportError("Reading audio files requires soundfile; .npy waveforms are also supported") from exc
        waveform, file_rate = soundfile.read(path, dtype="float32", always_2d=False)
        if file_rate != sampling_rate:
            raise ValueError(f"Audio file rate {file_rate} differs from declared sampling_rate {sampling_rate}")
    if waveform.ndim != 1 or waveform.size < 1 or not np.isfinite(waveform).all():
        raise ValueError("Audio must be a finite nonempty mono waveform; prepare channel mixing explicitly")
    return np.asarray(waveform, dtype=np.float32)


class FrozenAudioEncoder(FrozenModule):


    def __init__(self, model_id: str, *, revision: str | None = None, device: str = "cpu", allow_download: bool = False) -> None:
        options = _loading_options(model_id, revision, allow_download)
        transformers = _transformers()
        model = _load_pretrained(transformers.Wav2Vec2Model, model_id, options).to(device)
        super().__init__(model)
        self.processor = transformers.AutoFeatureExtractor.from_pretrained(model_id, **options)
        self.device = torch.device(device)

    @torch.inference_mode()
    def encode(self, samples: Sequence[dict[str, Any]], batch_size: int = 1) -> Tensor:
        _check_samples(samples)
        outputs = []
        for item in samples:
            if not isinstance(item, dict) or "path" not in item or "sampling_rate" not in item:
                raise ValueError("Each audio sample requires path and sampling_rate")
            rate = item["sampling_rate"]
            if rate != int(self.processor.sampling_rate):
                raise ValueError(f"Encoder expects {self.processor.sampling_rate} Hz, got {rate}; resample explicitly before extraction")
            waveform = load_audio_waveform(item["path"], rate)
            inputs = self.processor(waveform, sampling_rate=rate, return_tensors="pt", padding=False)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}

            outputs.append(self.model(**inputs).last_hidden_state.mean(dim=1).cpu().float())
        return torch.cat(outputs, dim=0)


ENCODER_TYPES = {"text": FrozenSentenceEncoder, "image": FrozenImageEncoder, "video": FrozenVideoEncoder, "audio": FrozenAudioEncoder}
