from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import yaml


@dataclass(frozen=True)
class EncoderSpec:
    modality: str
    backend: str
    model_id: str
    output_dim: int
    units: int
    huggingface_url: str | None = None
    downloads_weights_by_default: bool = False


def load_encoder_specs(path: str | Path = "configs/models.yaml") -> Dict[str, EncoderSpec]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    specs: Dict[str, EncoderSpec] = {}
    for modality, values in raw.get("encoders", {}).items():
        specs[modality] = EncoderSpec(
            modality=modality,
            backend=values["backend"],
            model_id=values["model_id"],
            output_dim=int(values["output_dim"]),
            units=int(values.get("units", 1)),
            huggingface_url=values.get("huggingface_url"),
            downloads_weights_by_default=bool(values.get("downloads_weights_by_default", False)),
        )
    return specs


class OfflineHFEncoder:
    """Optional adapter for local-only Hugging Face encoders.

    The adapter intentionally uses local_files_only=True. This prevents accidental model
    downloads while still allowing users to run released local checkpoints.
    """

    def __init__(self, spec: EncoderSpec, allow_load: bool = False) -> None:
        self.spec = spec
        self.allow_load = allow_load
        self.model: Any | None = None
        self.processor: Any | None = None

    def load(self) -> "OfflineHFEncoder":
        if not self.allow_load:
            raise RuntimeError(
                f"Loading model weights is disabled for {self.spec.model_id}. "
                "Set allow_hf_model_load=true and provide a local cache to use this adapter."
            )
        if self.spec.backend == "sentence_transformers":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.spec.model_id, local_files_only=True)
            return self
        if self.spec.backend == "transformers":
            from transformers import AutoModel, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.spec.model_id,
                local_files_only=True,
            )
            self.model = AutoModel.from_pretrained(
                self.spec.model_id,
                local_files_only=True,
            )
            return self
        raise ValueError(f"Unsupported encoder backend: {self.spec.backend}")

    def describe(self) -> Mapping[str, Any]:
        return {
            "modality": self.spec.modality,
            "backend": self.spec.backend,
            "model_id": self.spec.model_id,
            "output_dim": self.spec.output_dim,
            "units": self.spec.units,
            "loaded": self.model is not None,
        }

