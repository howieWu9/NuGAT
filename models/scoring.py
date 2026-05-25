from __future__ import annotations

import torch
from torch import nn


class DistMultScorer(nn.Module):
    def __init__(self, input_dim: int, num_relations: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.entity_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.relation = nn.Embedding(num_relations, hidden_dim)
        self.bias = nn.Parameter(torch.zeros(1))
        nn.init.xavier_uniform_(self.relation.weight)

    def forward(
        self,
        head_representation: torch.Tensor,
        relation_ids: torch.LongTensor,
        tail_representation: torch.Tensor,
    ) -> torch.Tensor:
        head = self.entity_projection(head_representation)
        tail = self.entity_projection(tail_representation)
        relation = self.relation(relation_ids)
        return (head * relation * tail).sum(dim=-1) / head.shape[-1] ** 0.5 + self.bias

