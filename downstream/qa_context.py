from __future__ import annotations

from typing import Iterable

from fotnuf.data.mmkg import MMKGData
from fotnuf.downstream.retrieval import RankedTriple


def serialize_ranked_triples(data: MMKGData, ranked: Iterable[RankedTriple]) -> str:
    lines = []
    for item in ranked:
        head, relation, tail = item.triple
        lines.append(
            " | ".join(
                [
                    f"head={data.entity_names.get(head, str(head))}",
                    f"relation={data.relation_names.get(relation, str(relation))}",
                    f"tail={data.entity_names.get(tail, str(tail))}",
                    f"score={item.score:.6f}",
                    f"relevance={item.relevance:.6f}",
                    f"plausibility={item.plausibility:.6f}",
                ]
            )
        )
    return "\n".join(lines)

