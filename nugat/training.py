"""Fixed-epoch Nash training and validation-only hyperparameter selection.

This runner does not manufacture paper scores. Its validation Nash loss is a
declared operational grid-search criterion; the supplied paper does not specify
the original grid or validation selection metric.
"""
from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EvidenceBatch, ManifestDataset, collate_evidence


@dataclass(frozen=True)
class TrainingSettings:
    epochs: int = 30
    batch_size: int = 1024
    learning_rate: float = 1e-4
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46)
    device: str = "auto"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "TrainingSettings":
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unknown training options: {sorted(unknown)}")
        data = dict(values)
        if "seeds" in data:
            data["seeds"] = tuple(data["seeds"])
        result = cls(**data)
        if not isinstance(result.epochs, int) or result.epochs < 1:
            raise ValueError("epochs must be a positive integer")
        if not isinstance(result.batch_size, int) or result.batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if not math.isfinite(result.learning_rate) or result.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not result.seeds or any(not isinstance(s, int) or s < 0 or s >= 2**32 for s in result.seeds):
            raise ValueError("seeds must contain integers in [0, 2**32)")
        if len(set(result.seeds)) != len(result.seeds):
            raise ValueError("Repeated runs require distinct seeds")
        return result


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is requested but is not available")
    return device


def make_loader(dataset: ManifestDataset, batch_size: int, *, shuffle: bool,
                seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, collate_fn=collate_evidence, generator=generator)


@torch.no_grad()
def validation_nash_loss(model: torch.nn.Module, loader: DataLoader,
                         device: torch.device) -> float:
    if getattr(loader.dataset, "split", None) != "validation":
        raise ValueError("Hyperparameter selection only accepts the validation split")
    model.eval()
    weighted_loss, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        loss = model(batch.features, batch.query, masks=batch.masks).loss
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise FloatingPointError("Validation loss must be a finite scalar")
        size = len(batch.ids)
        weighted_loss += float(loss) * size
        count += size
    if count == 0:
        raise ValueError("Validation split is empty")
    return weighted_loss / count


def fit(model: torch.nn.Module, train_loader: DataLoader, settings: TrainingSettings,
        device: torch.device, *, log_path: Path | None = None) -> tuple[list[dict[str, Any]], torch.optim.Adam]:
    """Optimize only the model's Nash loss for the stated number of epochs."""
    if getattr(train_loader.dataset, "split", None) != "train":
        raise ValueError("Training only accepts the train split")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate)
    history: list[dict[str, Any]] = []
    for epoch in range(1, settings.epochs + 1):
        model.train()
        loss_sum, examples = 0.0, 0
        for batch in train_loader:
            batch: EvidenceBatch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model(batch.features, batch.query, masks=batch.masks).loss
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite or nonscalar training loss at epoch {epoch}")
            loss.backward()
            bad = [name for name, parameter in model.named_parameters()
                   if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())]
            if bad:
                raise FloatingPointError(f"Non-finite gradients at epoch {epoch}: {bad}")
            optimizer.step()
            count = len(batch.ids)
            loss_sum += float(loss.detach()) * count
            examples += count
        if examples == 0:
            raise ValueError("Training split is empty")
        record = {"epoch": epoch, "training_nash_loss": loss_sum / examples,
                  "examples": examples}
        history.append(record)
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
    return history, optimizer


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _new_run_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Run directory is not empty; choose a new output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _model_kwargs(config: Mapping[str, Any], dataset: ManifestDataset, *,
                  allow_alternative_objective: bool = False) -> dict[str, Any]:
    kwargs = dict(config.get("model", {}))
    # These dimensions come from the actual frozen encoder features, not defaults.
    for key, inferred in (("modality_dims", dataset.schema.modality_dims),
                          ("query_dim", dataset.schema.query_dim)):
        if key in kwargs and kwargs[key] != inferred:
            raise ValueError(f"Configured {key} differs from the feature files")
        kwargs[key] = inferred
    if not allow_alternative_objective and kwargs.get("objective", "nash") != "nash":
        raise ValueError("The standard training runner uses objective='nash'; use explicit ablation code for other objectives")
    return kwargs


def build_model(config: Mapping[str, Any], dataset: ManifestDataset, *,
                allow_alternative_objective: bool = False):
    """Construct from actual encoder dimensions and return checkpointable kwargs.

    Alternative objectives are opt-in for dedicated ablation runners. This does
    not change the standard runner's use of the Nash loss.
    """
    from .model import NuGAT

    kwargs = _model_kwargs(config, dataset,
                           allow_alternative_objective=allow_alternative_objective)
    return NuGAT(**kwargs), kwargs


def train_runs(config: Mapping[str, Any], manifest: str | Path, output: str | Path,
               *, validation: bool = False,
               metric_callback: Callable | None = None) -> dict[str, Any]:
    """Run independent seeds; optional validation never reads test features.

    A metric callback has signature ``callback(model, validation_loader,
    device) -> float``. It must perform genuine downstream evaluation. The
    runner records its identity instead of substituting a fake judge score.
    """
    from .model import NuGAT

    settings = TrainingSettings.from_mapping(config.get("training", {}))
    dataset = ManifestDataset(manifest, "train")
    if len(dataset.schema.modality_dims) < 2:
        raise ValueError(
            "Eq. (14) normalizes across modalities and has a constant loss for one modality. "
            "Nash training requires at least two modalities; no extra supervised loss is assumed."
        )
    val_dataset = ManifestDataset(manifest, "validation", dataset.schema) if validation else None
    if metric_callback is not None and not validation:
        raise ValueError("A selection metric requires validation=True")
    kwargs = _model_kwargs(config, dataset)
    device = resolve_device(settings.device)
    output = Path(output)
    _new_run_directory(output)
    manifest_sha = hashlib.sha256(Path(manifest).read_bytes()).hexdigest()
    resolved_config = {**dict(config), "model": kwargs, "training": asdict(settings)}
    _write_json(output / "config.json", resolved_config)
    runs: list[dict[str, Any]] = []
    for seed in settings.seeds:
        seed_everything(seed)
        run_dir = output / f"seed_{seed}"
        _new_run_directory(run_dir)
        model = NuGAT(**kwargs)
        loader = make_loader(dataset, settings.batch_size, shuffle=True, seed=seed)
        history, optimizer = fit(model, loader, settings, device,
                                 log_path=run_dir / "training.jsonl")
        result: dict[str, Any] = {"seed": seed, "epochs": settings.epochs,
                                  "training_nash_loss": history[-1]["training_nash_loss"]}
        if val_dataset is not None:
            val_loader = make_loader(val_dataset, settings.batch_size, shuffle=False, seed=seed)
            if metric_callback is None:
                value = validation_nash_loss(model, val_loader, device)
                result["validation_nash_loss"] = value
            else:
                model.eval()
                with torch.no_grad():
                    value = float(metric_callback(model, val_loader, device))
                if not math.isfinite(value):
                    raise ValueError("Validation callback must return a finite scalar")
                result["validation_metric"] = value
        checkpoint = {
            "format_version": 1, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "model_kwargs": kwargs,
            "config": resolved_config, "seed": seed, "epoch": settings.epochs,
            "manifest_sha256": manifest_sha, "torch_version": str(torch.__version__),
        }
        torch.save(checkpoint, run_dir / "checkpoint.pt")
        _write_json(run_dir / "result.json", result)
        runs.append(result)
    summary: dict[str, Any] = {
        "dataset": config.get("dataset"), "run_count": len(runs),
        "seeds": list(settings.seeds), "manifest_sha256": manifest_sha, "runs": runs,
        "note": "Training/validation losses are not M.J. scores or reproduced paper results.",
    }
    if validation:
        key = "validation_metric" if metric_callback else "validation_nash_loss"
        values = np.array([r[key] for r in runs], dtype=float)
        summary[key] = {"mean": float(values.mean()),
                        "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                        "std_ddof": 1}
    _write_json(output / "summary.json", summary)
    return summary


def load_checkpoint(path: str | Path, *, device: str | torch.device = "cpu"):
    """Load tensor/state dictionaries without enabling arbitrary pickle objects."""
    from .model import NuGAT

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict) or checkpoint.get("format_version") != 1:
        raise ValueError("Unsupported checkpoint format")
    model = NuGAT(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()
    return model, checkpoint


def load_metric_callback(spec: str) -> Callable:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("Metric callback must be 'module:function'")
    callback = getattr(importlib.import_module(module_name), attribute)
    if not callable(callback):
        raise ValueError(f"Not a callable: {spec}")
    return callback


def grid_search(config: Mapping[str, Any], manifest: str | Path,
                grid: Mapping[str, list[Any]], output: str | Path, *,
                metric_callback: Callable | None = None, metric_name: str | None = None,
                maximize: bool = False) -> dict[str, Any]:
    """Search an explicitly supplied grid using validation data only.

    A supplied grid is an experiment input, not a claim about the authors'
    historical search range. By default this minimizes validation Nash loss.
    Provide a genuine M.J. evaluation callback and maximize=True to select by
    validation M.J. instead. Test metrics are never a selection criterion.
    """
    if not grid or any(not isinstance(values, list) or not values for values in grid.values()):
        raise ValueError("Grid must map model parameter names to nonempty lists")
    tunable = {"qat_temperature", "utility_temperature", "residual_weight", "gate_temperature",
               "num_evidence_units", "evidence_dim", "iterations"}
    if set(grid) - tunable:
        raise ValueError(f"Unsupported grid parameters: {sorted(set(grid) - tunable)}")
    if metric_callback is None and maximize:
        raise ValueError("Validation Nash loss must be minimized")
    if metric_callback is None and len(set(grid.get("utility_temperature", []))) > 1:
        raise ValueError(
            "Comparing Nash losses across utility temperatures changes the loss scale. "
            "Supply a genuine validation metric callback (for example M.J.) to search utility_temperature."
        )
    if metric_callback is not None and not metric_name:
        raise ValueError("A custom validation callback requires an explicit metric name")
    output = Path(output)
    _new_run_directory(output)
    keys = list(grid)
    candidates = []
    for index, values in enumerate(itertools.product(*(grid[key] for key in keys))):
        parameters = dict(zip(keys, values))
        candidate = {**dict(config), "model": {**dict(config.get("model", {})), **parameters}}
        result = train_runs(candidate, manifest, output / f"candidate_{index:04d}",
                            validation=True, metric_callback=metric_callback)
        result_key = "validation_metric" if metric_callback else "validation_nash_loss"
        candidates.append({"parameters": parameters, "value": result[result_key]["mean"],
                           "directory": f"candidate_{index:04d}"})
    best = (max if maximize else min)(candidates, key=lambda result: result["value"])
    result = {
        "selection_split": "validation", "metric": metric_name or "validation_nash_loss",
        "maximize": maximize, "grid": dict(grid), "best": best, "candidates": candidates,
        "note": "This explicitly supplied grid is not inferred from the paper. Test data are not used for selection.",
    }
    _write_json(output / "selection.json", result)
    return result
