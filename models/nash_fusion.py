from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from fotnuf.models.functional_ot import (
    CostNetwork,
    entropy_regularizer,
    project_with_plan,
    sinkhorn_plan,
    uniform_transport_plan,
)


@dataclass
class FusionOutput:
    fused: torch.Tensor
    nash_loss: torch.Tensor
    weights: torch.Tensor
    mean_utility: torch.Tensor


class NashUtilityFusion(nn.Module):
    def __init__(
        self,
        modality_dims: Dict[str, int],
        modality_order: List[str],
        relation_dim: int,
        support_units: int,
        cost_hidden_dim: int,
        dropout: float,
        eta: float,
        gamma: float,
        epsilon: float,
        sinkhorn_iterations: int,
        fusion_iterations: int,
    ) -> None:
        super().__init__()
        self.modality_dims = dict(modality_dims)
        self.modality_order = list(modality_order)
        self.relation_dim = relation_dim
        self.support_units = support_units
        self.eta = eta
        self.gamma = gamma
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        self.fusion_iterations = fusion_iterations
        self.fused_dim = sum(self.modality_dims[m] for m in self.modality_order)
        self.cost = CostNetwork(
            modality_dims=self.modality_dims,
            fused_dim=self.fused_dim,
            relation_dim=relation_dim,
            hidden_dim=cost_hidden_dim,
            dropout=dropout,
        )

    def _split_blocks(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        blocks: Dict[str, torch.Tensor] = {}
        start = 0
        for modality in self.modality_order:
            dim = self.modality_dims[modality]
            blocks[modality] = fused[..., start : start + dim]
            start += dim
        return blocks

    def _transport_consensus_cost(
        self,
        projection: torch.Tensor,
        previous_block: torch.Tensor,
        plan: torch.Tensor,
        cost_matrix: torch.Tensor,
    ) -> torch.Tensor:
        agreement = 0.5 * (projection - previous_block).pow(2).mean(dim=(1, 2))
        transport = self.eta * (plan * cost_matrix).sum(dim=(1, 2))
        entropy = self.gamma * entropy_regularizer(plan, self.epsilon)
        return agreement + transport + entropy

    def _initial_state(
        self,
        features: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        projections: Dict[str, torch.Tensor] = {}
        plans: Dict[str, torch.Tensor] = {}
        batch_size = next(iter(features.values())).shape[0]
        device = next(iter(features.values())).device
        dtype = next(iter(features.values())).dtype
        for modality in self.modality_order:
            x = features[modality]
            plan = uniform_transport_plan(
                batch_size=batch_size,
                source_units=x.shape[1],
                target_units=self.support_units,
                device=device,
                dtype=dtype,
            )
            plans[modality] = plan
            projections[modality] = project_with_plan(plan, x)
        scale = 1.0 / float(len(self.modality_order))
        fused = torch.cat([scale * projections[m] for m in self.modality_order], dim=-1)
        return fused, plans, projections

    def forward(self, features: Dict[str, torch.Tensor], relation: torch.Tensor) -> FusionOutput:
        fused, plans, projections = self._initial_state(features)
        nash_terms: List[torch.Tensor] = []
        weights_history: List[torch.Tensor] = []
        utility_history: List[torch.Tensor] = []
        for _ in range(self.fusion_iterations):
            previous_blocks = self._split_blocks(fused)
            next_plans: Dict[str, torch.Tensor] = {}
            next_projections: Dict[str, torch.Tensor] = {}
            gains: List[torch.Tensor] = []
            for modality in self.modality_order:
                x = features[modality]
                cost_matrix = self.cost(modality, x, fused, relation, epsilon=self.epsilon)
                new_plan = sinkhorn_plan(
                    cost=cost_matrix,
                    gamma=self.gamma,
                    iterations=self.sinkhorn_iterations,
                    epsilon=self.epsilon,
                )
                new_projection = project_with_plan(new_plan, x)
                previous_cost = self._transport_consensus_cost(
                    projection=projections[modality],
                    previous_block=previous_blocks[modality],
                    plan=plans[modality],
                    cost_matrix=cost_matrix,
                )
                new_cost = self._transport_consensus_cost(
                    projection=new_projection,
                    previous_block=previous_blocks[modality],
                    plan=new_plan,
                    cost_matrix=cost_matrix,
                )
                raw_gain = previous_cost - new_cost
                positive_gain = F.softplus(raw_gain) + self.epsilon
                gains.append(positive_gain)
                nash_terms.append(-torch.log(positive_gain).mean())
                next_plans[modality] = new_plan
                next_projections[modality] = new_projection
            gain_matrix = torch.stack(gains, dim=1)
            inverse = 1.0 / (gain_matrix + self.epsilon)
            weights = inverse / inverse.sum(dim=1, keepdim=True)
            weighted_blocks = []
            for index, modality in enumerate(self.modality_order):
                weight = weights[:, index].view(-1, 1, 1)
                weighted_blocks.append(weight * next_projections[modality])
            fused = torch.cat(weighted_blocks, dim=-1)
            plans = next_plans
            projections = next_projections
            weights_history.append(weights)
            utility_history.append(gain_matrix)
        if nash_terms:
            nash_loss = torch.stack(nash_terms).mean()
            weights_out = torch.stack(weights_history).mean(dim=0)
            utility_out = torch.stack(utility_history).mean(dim=0)
        else:
            nash_loss = fused.new_tensor(0.0)
            batch_size = fused.shape[0]
            weights_out = fused.new_full(
                (batch_size, len(self.modality_order)),
                1.0 / float(len(self.modality_order)),
            )
            utility_out = fused.new_zeros(batch_size, len(self.modality_order))
        return FusionOutput(
            fused=fused,
            nash_loss=nash_loss,
            weights=weights_out,
            mean_utility=utility_out,
        )

