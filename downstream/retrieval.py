from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch

from fotnuf.data.features import FeatureStore
from fotnuf.data.mmkg import MMKGData, Triple
from fotnuf.models.fotnuf import NuGATModel


TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    return set(TOKEN_RE.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


@dataclass
class RankedTriple:
    triple: Triple
    score: float
    relevance: float
    plausibility: float


def ground_entities(question: str, entity_names: Dict[int, str], top_k: int = 16) -> List[int]:
    question_tokens = tokenize(question)
    scores = []
    for entity_id, name in entity_names.items():
        score = jaccard(question_tokens, tokenize(name))
        if score > 0:
            scores.append((score, entity_id))
    scores.sort(reverse=True)
    return [entity_id for _, entity_id in scores[:top_k]]


def retrieve_candidate_triples(
    question: str,
    data: MMKGData,
    max_triples: int = 128,
) -> List[Triple]:
    grounded = set(ground_entities(question, data.entity_names, top_k=32))
    candidates: List[Triple] = []
    if grounded:
        for triple in data.iter_all_triples():
            if triple[0] in grounded or triple[2] in grounded:
                candidates.append(triple)
                if len(candidates) >= max_triples:
                    break
    if not candidates:
        candidates = list(data.iter_all_triples())[:max_triples]
    return candidates


def triple_text(triple: Triple, data: MMKGData) -> str:
    head, relation, tail = triple
    return " ".join(
        [
            data.entity_names.get(head, str(head)),
            data.relation_names.get(relation, str(relation)),
            data.entity_names.get(tail, str(tail)),
        ]
    )


@torch.no_grad()
def rerank_triples_for_question(
    model: NuGATModel,
    data: MMKGData,
    feature_store: FeatureStore,
    question: str,
    candidates: Sequence[Triple] | None = None,
    top_k: int = 8,
    relevance_weight: float = 0.35,
    plausibility_weight: float = 0.65,
    batch_size: int = 128,
) -> List[RankedTriple]:
    model.eval()
    question_tokens = tokenize(question)
    candidates = list(candidates or retrieve_candidate_triples(question, data))
    if not candidates:
        return []
    triples = torch.tensor(candidates, dtype=torch.long)
    scores = []
    for start in range(0, triples.shape[0], batch_size):
        batch_scores, _ = model.score_triples(triples[start : start + batch_size], feature_store)
        scores.append(batch_scores.detach().cpu())
    plausibility = torch.cat(scores, dim=0)
    plausibility = torch.sigmoid(plausibility)
    ranked: List[RankedTriple] = []
    for idx, triple in enumerate(candidates):
        relevance = jaccard(question_tokens, tokenize(triple_text(triple, data)))
        kg_score = float(plausibility[idx].item())
        total = relevance_weight * relevance + plausibility_weight * kg_score
        ranked.append(
            RankedTriple(
                triple=triple,
                score=total,
                relevance=relevance,
                plausibility=kg_score,
            )
        )
    ranked.sort(key=lambda item: item.score, reverse=True)
    return ranked[:top_k]

