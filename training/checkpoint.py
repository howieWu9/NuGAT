from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch

from fotnuf.utils.io import ensure_dir


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    config: Dict[str, Any],
    metrics: Dict[str, float],
) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    payload: Dict[str, Any] = {
        "model": model.state_dict(),
        "epoch": epoch,
        "config": config,
        "metrics": metrics,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)

