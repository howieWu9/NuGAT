from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn

from fotnuf.data.features import FeatureStore
from fotnuf.models.nash_fusion import FusionOutput, NashUtilityFusion
from fotnuf.models.scoring import DistMultScorer


class NuGATModel(nn.Module):
    """NuGAT: evidence-grounded multimodal fusion with Attention Transport and Nash Utility Fusion."""
    def __init__(
        self,
        num_relations: int,
        modality_dims: Dict[str, int],
        modality_order: List[str],
        relation_dim: int = 256,
        support_units: int = 4,
        cost_hidden_dim: int = 256,
        scorer_hidden_dim: int = 256,
        dropout: float = 0.1,
        eta: float = 1.0e-1,
        gamma: float = 1.5e-1,
        epsilon: float = 1.0e-8,
        sinkhorn_iterations: int = 20,
        fusion_iterations: int = 2,
        residual_weight: float = 2.0e-1,
        gate_temperature: float = 5.0e-2,
    ) -> None:
        super().__init__()
        self.modality_dims = dict(modality_dims)
        self.modality_order = list(modality_order)
        self.fused_dim = sum(self.modality_dims[m] for m in self.modality_order)
        self.relation_context = nn.Embedding(num_relations, relation_dim)
        self.fusion = NashUtilityFusion(
            modality_dims=self.modality_dims,
            modality_order=self.modality_order,
            relation_dim=relation_dim,
            support_units=support_units,
            cost_hidden_dim=cost_hidden_dim,
            dropout=dropout,
            eta=eta,
            gamma=gamma,
            epsilon=epsilon,
            sinkhorn_iterations=sinkhorn_iterations,
            fusion_iterations=fusion_iterations,
            residual_weight=residual_weight,
            gate_temperature=gate_temperature,
        )
        self.scorer = DistMultScorer(
            input_dim=self.fused_dim,
            num_relations=num_relations,
            hidden_dim=scorer_hidden_dim,
            dropout=dropout,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.relation_context.weight)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _batch_features(
        self,
        feature_store: FeatureStore,
        entity_ids: torch.LongTensor,
    ) -> Dict[str, torch.Tensor]:
        device = self.relation_context.weight.device
        return {
            modality: tensor[entity_ids.detach().cpu()].to(device)
            for modality, tensor in feature_store.features.items()
            if modality in self.modality_order
        }

    def encode_entities(
        self,
        entity_ids: torch.LongTensor,
        relation_ids: torch.LongTensor,
        feature_store: FeatureStore,
    ) -> FusionOutput:
        device = self.relation_context.weight.device
        relation_ids = relation_ids.to(device)
        features = self._batch_features(feature_store, entity_ids)
        relation = self.relation_context(relation_ids)
        return self.fusion(features, relation)

    def score_triples(
        self,
        triples: torch.LongTensor,
        feature_store: FeatureStore,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.relation_context.weight.device
        triples = triples.to(device)
        heads = triples[:, 0]
        relations = triples[:, 1]
        tails = triples[:, 2]
        head_fusion = self.encode_entities(heads, relations, feature_store)
        tail_fusion = self.encode_entities(tails, relations, feature_store)
        head_rep = head_fusion.fused.mean(dim=1)
        tail_rep = tail_fusion.fused.mean(dim=1)
        scores = self.scorer(head_rep, relations, tail_rep)
        nash_loss = 0.5 * (head_fusion.nash_loss + tail_fusion.nash_loss)
        return scores, nash_loss

    def forward(
        self,
        positive_triples: torch.LongTensor,
        negative_triples: torch.LongTensor,
        feature_store: FeatureStore,
    ) -> Dict[str, torch.Tensor]:
        pos_scores, pos_nash = self.score_triples(positive_triples, feature_store)
        flat_neg = negative_triples.reshape(-1, 3)
        neg_scores, neg_nash = self.score_triples(flat_neg, feature_store)
        neg_scores = neg_scores.view(negative_triples.shape[0], negative_triples.shape[1])
        return {
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
            "nash_loss": 0.5 * (pos_nash + neg_nash),
        }


FoTNuFModel = NuGATModel

