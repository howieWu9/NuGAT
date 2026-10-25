"""NuGAT: recurrent QAT grounding and Nash Utility Fusion.

Each forward pass performs exactly ``iterations`` grounding/fusion updates.
Inputs are frozen, real modality/query features; no ID-derived pseudo-features
or knowledge-graph scoring objective is part of this model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .qat import QATOutput, QueryConditionedAttentionTransport
from .fusion import FusionOutput, NashUtilityFusion


@dataclass
class IterationOutput:
    grounding: QATOutput
    fusion: FusionOutput


@dataclass
class NuGATOutput:
    fused: Tensor
    loss: Tensor
    grounded: dict[str, Tensor]
    transport: dict[str, Tensor]
    utilities: Tensor
    log_utilities: Tensor
    retention: Tensor
    next_retention: Tensor
    history: tuple[IterationOutput, ...]


class NuGAT(nn.Module):
    """Equations (3)--(17), with the revision's learnable Xavier evidence.

    The modality state for the next QAT update is its latest grounded evidence.
    Shared Y^0 is the mean of the learnable initial modality evidence, and the
    first alpha is uniform. These are explicit implementation conventions
    because the original manuscript leaves the initial states underspecified.
    """

    def __init__(
        self,
        modality_dims: Mapping[str, int],
        query_dim: int,
        num_evidence_units: int = 16,
        evidence_dim: int = 256,
        iterations: int = 3,
        qat_temperature: float = 0.10,
        utility_temperature: float = 0.15,
        residual_weight: float = 0.20,
        gate_temperature: float = 0.05,
        epsilon: float = 1e-8,
        qat_variant: str = "qat",
        fusion_variant: str = "nuf",
        objective: str = "nash",
        sinkhorn_iterations: int = 1000,
        compatibility_chunk_size: int = 32,
        subspace_rank: int | None = None,
        stochastic_eval: bool = False,
    ) -> None:
        super().__init__()
        self.modality_dims = dict(modality_dims)
        self.modalities = tuple(self.modality_dims)
        self.query_dim = query_dim
        self.num_evidence_units = num_evidence_units
        self.evidence_dim = evidence_dim
        self.iterations = iterations
        self.qat = QueryConditionedAttentionTransport(
            modality_dims=self.modality_dims, query_dim=query_dim,
            evidence_dim=evidence_dim, num_evidence_units=num_evidence_units,
            temperature=qat_temperature, variant=qat_variant,
            sinkhorn_iterations=sinkhorn_iterations,
            compatibility_chunk_size=compatibility_chunk_size,
        )
        self.fusion = NashUtilityFusion(
            modalities=list(self.modalities), num_evidence_units=num_evidence_units,
            evidence_dim=evidence_dim, iterations=iterations,
            utility_temperature=utility_temperature, residual_weight=residual_weight,
            gate_temperature=gate_temperature, epsilon=epsilon,
            variant=fusion_variant, objective=objective,
            subspace_rank=subspace_rank, stochastic_eval=stochastic_eval,
        )
        self.initial_evidence = nn.ParameterDict({
            name: nn.Parameter(torch.empty(num_evidence_units, evidence_dim))
            for name in self.modalities
        })
        for value in self.initial_evidence.values():
            nn.init.xavier_uniform_(value)

    def forward(
        self, features: Mapping[str, Tensor], query: Tensor,
        masks: Mapping[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> NuGATOutput:
        if query.ndim != 2 or query.shape[0] == 0:
            raise ValueError("query must have nonempty shape [batch, query_dim]")
        batch = query.shape[0]
        states = {m: y.unsqueeze(0).expand(batch, -1, -1)
                  for m, y in self.initial_evidence.items()}
        shared = torch.stack(tuple(states.values()), dim=1).mean(dim=1)
        retention = shared.new_full((batch, len(self.modalities)), 1 / len(self.modalities))
        history = []
        for iteration in range(self.iterations):
            grounding = self.qat(features, query, states, masks)
            current_retention = retention
            fusion = self.fusion(grounding.grounded, shared, retention, iteration, generator)
            states = grounding.grounded
            shared, retention = fusion.fused, fusion.next_retention
            history.append(IterationOutput(grounding=grounding, fusion=fusion))
        return NuGATOutput(
            fused=shared, loss=fusion.loss, grounded=grounding.grounded,
            transport=grounding.transport, utilities=fusion.utilities,
            log_utilities=fusion.log_utilities, retention=current_retention,
            next_retention=retention,
            history=tuple(history),
        )
