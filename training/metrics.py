from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import torch
from tqdm import tqdm

from fotnuf.config import EvaluationConfig
from fotnuf.data.features import FeatureStore
from fotnuf.data.mmkg import MMKGData
from fotnuf.models.fotnuf import FoTNuFModel

Triple = Tuple[int, int, int]


@dataclass
class RankingMetrics:
    mrr: float
    hits: Dict[int, float]
    mean_rank: float
    evaluated: int

    def as_dict(self) -> Dict[str, float]:
        result = {"mrr": self.mrr, "mean_rank": self.mean_rank, "evaluated": float(self.evaluated)}
        for key, value in self.hits.items():
            result[f"hit@{key}"] = value
        return result


def _sample_candidates(
    num_entities: int,
    true_entity: int,
    count: int,
    generator: torch.Generator,
) -> torch.LongTensor:
    if count >= num_entities:
        return torch.arange(num_entities, dtype=torch.long)
    candidates = {int(true_entity)}
    while len(candidates) < count:
        draw = int(torch.randint(num_entities, size=(1,), generator=generator).item())
        candidates.add(draw)
    return torch.tensor(sorted(candidates), dtype=torch.long)


def _filter_candidates(
    candidates: torch.LongTensor,
    triple: Triple,
    true_triples: Set[Triple],
    side: str,
) -> torch.LongTensor:
    head, relation, tail = triple
    kept: List[int] = []
    for candidate in candidates.tolist():
        if side == "tail":
            candidate_triple = (head, relation, int(candidate))
            if candidate != tail and candidate_triple in true_triples:
                continue
        else:
            candidate_triple = (int(candidate), relation, tail)
            if candidate != head and candidate_triple in true_triples:
                continue
        kept.append(int(candidate))
    return torch.tensor(kept, dtype=torch.long)


def _candidate_triples(triple: Triple, candidates: torch.LongTensor, side: str) -> torch.LongTensor:
    head, relation, tail = triple
    triples = torch.empty(candidates.shape[0], 3, dtype=torch.long)
    if side == "tail":
        triples[:, 0] = head
        triples[:, 1] = relation
        triples[:, 2] = candidates
    else:
        triples[:, 0] = candidates
        triples[:, 1] = relation
        triples[:, 2] = tail
    return triples


@torch.no_grad()
def evaluate_ranking(
    model: FoTNuFModel,
    data: MMKGData,
    feature_store: FeatureStore,
    triples: torch.LongTensor,
    config: EvaluationConfig,
    batch_size: int = 128,
    seed: int = 2026,
    show_progress: bool = False,
) -> RankingMetrics:
    model.eval()
    generator = torch.Generator()
    generator.manual_seed(seed)
    if config.eval_max_triples is not None:
        triples = triples[: int(config.eval_max_triples)]
    true_triples = data.true_triples if config.filtered else set()
    ranks: List[int] = []
    sides: Sequence[str]
    if config.rank_side == "head":
        sides = ["head"]
    elif config.rank_side == "tail":
        sides = ["tail"]
    else:
        sides = ["head", "tail"]
    iterator: Iterable[List[int]] = triples.tolist()
    if show_progress:
        iterator = tqdm(list(iterator), desc="evaluate", leave=False)
    for row in iterator:
        triple = (int(row[0]), int(row[1]), int(row[2]))
        for side in sides:
            true_entity = triple[2] if side == "tail" else triple[0]
            if config.candidate_mode == "full":
                candidates = torch.arange(data.num_entities, dtype=torch.long)
            else:
                candidates = _sample_candidates(
                    data.num_entities,
                    true_entity,
                    int(config.sampled_candidates),
                    generator,
                )
            candidates = _filter_candidates(candidates, triple, true_triples, side)
            candidate_triples = _candidate_triples(triple, candidates, side)
            scores = []
            for start in range(0, candidate_triples.shape[0], batch_size):
                batch = candidate_triples[start : start + batch_size]
                batch_scores, _ = model.score_triples(batch, feature_store)
                scores.append(batch_scores.detach().cpu())
            score_tensor = torch.cat(scores, dim=0)
            true_index = (candidates == true_entity).nonzero(as_tuple=False)
            if true_index.numel() == 0:
                continue
            true_score = score_tensor[int(true_index[0].item())]
            rank = int((score_tensor > true_score).sum().item()) + 1
            ranks.append(rank)
    if not ranks:
        return RankingMetrics(mrr=0.0, hits={k: 0.0 for k in config.hits_at}, mean_rank=0.0, evaluated=0)
    ranks_tensor = torch.tensor(ranks, dtype=torch.float32)
    hits = {int(k): float((ranks_tensor <= int(k)).float().mean().item()) for k in config.hits_at}
    return RankingMetrics(
        mrr=float((1.0 / ranks_tensor).mean().item()),
        hits=hits,
        mean_rank=float(ranks_tensor.mean().item()),
        evaluated=len(ranks),
    )

