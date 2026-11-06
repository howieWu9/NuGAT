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


    def __init__(
        self,
        modality_dims: Mapping[str, int],
        modality_sizes: Mapping[str, int],
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
        if set(modality_sizes) != set(self.modalities) or any(
            isinstance(size, bool) or not isinstance(size, int) or size < 1
            for size in modality_sizes.values()
        ):
            raise ValueError("modality_sizes must give a positive source capacity for every modality")
        self.modality_sizes = {m: modality_sizes[m] for m in self.modalities}
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
        self.initial_mixing = nn.ParameterDict({
            name: nn.Parameter(torch.empty(num_evidence_units, self.modality_sizes[name]))
            for name in self.modalities
        })
        for value in self.initial_mixing.values():
            nn.init.xavier_uniform_(value)

    def _initialize_evidence(
        self, features: Mapping[str, Tensor], query: Tensor,
        masks: Mapping[str, Tensor] | None,
    ) -> dict[str, Tensor]:
        if set(features) != set(self.modalities):
            raise ValueError("Feature keys must match the configured modalities")
        if masks is not None and set(masks) - set(self.modalities):
            raise ValueError("Masks contain an unknown modality")
        states = {}
        for modality, dim in self.modality_dims.items():
            x = features[modality]
            if x.ndim != 3 or x.shape[0] != query.shape[0] or x.shape[2] != dim:
                raise ValueError(f"{modality}: features must have shape [batch, samples, {dim}]")
            if not 1 <= x.shape[1] <= self.modality_sizes[modality]:
                raise ValueError(f"{modality}: sample count exceeds capacity or is empty")
            if x.device != query.device or not x.is_floating_point():
                raise ValueError("Features must be floating-point values on the query device")
            valid = masks.get(modality) if masks is not None else None
            if valid is not None:
                if valid.shape != x.shape[:2] or valid.dtype != torch.bool or valid.device != x.device:
                    raise ValueError(f"{modality}: sample mask must be boolean [batch, samples] on the feature device")
                if not bool(valid.any(dim=1).all()):
                    raise ValueError(f"{modality}: every example requires at least one valid sample")
                x = x.masked_fill(~valid.unsqueeze(-1), 0.0)
            if not bool(torch.isfinite(x).all()):
                raise ValueError(f"{modality}: valid samples must be finite")
            values = self.qat.value_projections[modality](x)
            states[modality] = self.initial_mixing[modality][:, :x.shape[1]] @ values
        return states

    def forward(
        self, features: Mapping[str, Tensor], query: Tensor,
        masks: Mapping[str, Tensor] | None = None,
        generator: torch.Generator | None = None,
    ) -> NuGATOutput:
        if query.ndim != 2 or query.shape[0] == 0 or query.shape[1] != self.query_dim:
            raise ValueError("query must have nonempty shape [batch, query_dim]")
        batch = query.shape[0]
        states = self._initialize_evidence(features, query, masks)
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
