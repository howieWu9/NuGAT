from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F


class CostNetwork(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        fused_dim: int,
        relation_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.networks = nn.ModuleDict()
        for modality, dim in modality_dims.items():
            input_dim = dim + fused_dim + relation_dim
            self.networks[modality] = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self,
        modality: str,
        x: torch.Tensor,
        fused: torch.Tensor,
        relation: torch.Tensor,
        epsilon: float,
    ) -> torch.Tensor:
        batch, source_units, dim = x.shape
        target_units = fused.shape[1]
        x_expanded = x[:, :, None, :].expand(batch, source_units, target_units, dim)
        fused_expanded = fused[:, None, :, :].expand(
            batch,
            source_units,
            target_units,
            fused.shape[-1],
        )
        relation_expanded = relation[:, None, None, :].expand(
            batch,
            source_units,
            target_units,
            relation.shape[-1],
        )
        inputs = torch.cat([x_expanded, fused_expanded, relation_expanded], dim=-1)
        costs = self.networks[modality](inputs).squeeze(-1)
        return F.softplus(costs) + epsilon


def sinkhorn_plan(
    cost: torch.Tensor,
    gamma: float,
    iterations: int,
    epsilon: float,
) -> torch.Tensor:
    if gamma <= 0:
        raise ValueError("gamma must be positive for entropic OT")
    batch, source_units, target_units = cost.shape
    log_kernel = -cost / gamma
    log_u = cost.new_zeros(batch, source_units)
    log_v = cost.new_zeros(batch, target_units)
    log_p = cost.new_full((batch, source_units), -math.log(float(source_units)))
    log_q = cost.new_full((batch, target_units), -math.log(float(target_units)))
    for _ in range(iterations):
        log_u = log_p - torch.logsumexp(log_kernel + log_v[:, None, :], dim=2)
        log_v = log_q - torch.logsumexp(log_kernel + log_u[:, :, None], dim=1)
    log_plan = log_kernel + log_u[:, :, None] + log_v[:, None, :]
    return torch.exp(log_plan).clamp_min(epsilon)


def uniform_transport_plan(
    batch_size: int,
    source_units: int,
    target_units: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    value = 1.0 / float(source_units * target_units)
    return torch.full(
        (batch_size, source_units, target_units),
        value,
        device=device,
        dtype=dtype,
    )


def target_normalized_plan(plan: torch.Tensor) -> torch.Tensor:
    target_units = plan.shape[-1]
    return plan * float(target_units)


def project_with_plan(plan: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    normalized = target_normalized_plan(plan)
    return torch.einsum("ban,bad->bnd", normalized, x)


def entropy_regularizer(plan: torch.Tensor, epsilon: float) -> torch.Tensor:
    safe = plan.clamp_min(epsilon)
    return (safe * (safe.log() - 1.0)).sum(dim=(1, 2))
