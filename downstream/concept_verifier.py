from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch

from fotnuf.data.features import FeatureStore
from fotnuf.data.mmkg import MMKGData, Triple
from fotnuf.downstream.retrieval import jaccard, retrieve_candidate_triples, tokenize, triple_text
from fotnuf.models.fotnuf import FoTNuFModel


@dataclass
class ConceptDecision:
    concept: str
    probability: float
    label: int
    evidence: List[Triple]


@torch.no_grad()
def score_candidate_concept(
    model: FoTNuFModel,
    data: MMKGData,
    feature_store: FeatureStore,
    concept: str,
    evidence_triples: Sequence[Triple] | None = None,
    threshold: float = 0.5,
    batch_size: int = 128,
) -> ConceptDecision:
    model.eval()
    candidates = list(evidence_triples or retrieve_candidate_triples(concept, data, max_triples=64))
    if not candidates:
        return ConceptDecision(concept=concept, probability=0.0, label=0, evidence=[])
    triples = torch.tensor(candidates, dtype=torch.long)
    scores = []
    for start in range(0, triples.shape[0], batch_size):
        batch_scores, _ = model.score_triples(triples[start : start + batch_size], feature_store)
        scores.append(batch_scores.detach().cpu())
    plausibility = torch.sigmoid(torch.cat(scores)).mean().item()
    concept_tokens = tokenize(concept)
    evidence_relevance = [
        jaccard(concept_tokens, tokenize(triple_text(triple, data))) for triple in candidates
    ]
    relevance = sum(evidence_relevance) / max(len(evidence_relevance), 1)
    probability = 0.7 * float(plausibility) + 0.3 * float(relevance)
    return ConceptDecision(
        concept=concept,
        probability=probability,
        label=int(probability >= threshold),
        evidence=candidates[:8],
    )


def evaluate_concepts(
    model: FoTNuFModel,
    data: MMKGData,
    feature_store: FeatureStore,
    rows: Iterable[Dict[str, object]],
    threshold: float = 0.5,
) -> List[Dict[str, object]]:
    results: List[Dict[str, object]] = []
    for row in rows:
        concept = str(row.get("concept", ""))
        evidence = row.get("evidence_triples")
        triples = None
        if isinstance(evidence, list):
            triples = [tuple(map(int, item)) for item in evidence]  # type: ignore[arg-type]
        decision = score_candidate_concept(
            model=model,
            data=data,
            feature_store=feature_store,
            concept=concept,
            evidence_triples=triples,
            threshold=threshold,
        )
        results.append(
            {
                "concept": decision.concept,
                "probability": decision.probability,
                "label": decision.label,
                "evidence": [list(triple) for triple in decision.evidence],
            }
        )
    return results

