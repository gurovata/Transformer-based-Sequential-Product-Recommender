from __future__ import annotations

from math import log2
from typing import Iterable, Mapping, Sequence


def _relevant_set(relevant_items: int | Iterable[int]) -> set[int]:
    if isinstance(relevant_items, int):
        return {relevant_items}
    return {int(item) for item in relevant_items}


def hit_rate_at_k(
    recommendations: Sequence[int], relevant_items: int | Iterable[int], k: int
) -> float:
    relevant = _relevant_set(relevant_items)
    if not relevant:
        return 0.0
    return float(any(item in relevant for item in recommendations[:k]))


def recall_at_k(
    recommendations: Sequence[int], relevant_items: int | Iterable[int], k: int
) -> float:
    relevant = _relevant_set(relevant_items)
    if not relevant:
        return 0.0
    hits = sum(1 for item in recommendations[:k] if item in relevant)
    return hits / len(relevant)


def ndcg_at_k(
    recommendations: Sequence[int], relevant_items: int | Iterable[int], k: int
) -> float:
    relevant = _relevant_set(relevant_items)
    if not relevant:
        return 0.0

    dcg = 0.0
    for rank, item in enumerate(recommendations[:k], start=1):
        if item in relevant:
            dcg += 1.0 / log2(rank + 1)

    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(
    recommendations: Sequence[int], relevant_items: int | Iterable[int], k: int
) -> float:
    relevant = _relevant_set(relevant_items)
    if not relevant:
        return 0.0

    for rank, item in enumerate(recommendations[:k], start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def ranking_metrics_at_k(
    recommendations: Mapping[int, Sequence[int]],
    targets: Mapping[int, int | Iterable[int]],
    ks: Sequence[int] = (5, 10, 20),
) -> dict[str, float]:
    if not targets:
        raise ValueError("Targets mapping is empty.")

    result: dict[str, float] = {}
    user_ids = list(targets.keys())
    for k in ks:
        hits = []
        recalls = []
        ndcgs = []
        mrrs = []
        for user_id in user_ids:
            recs = recommendations.get(user_id, [])
            target = targets[user_id]
            hits.append(hit_rate_at_k(recs, target, k))
            recalls.append(recall_at_k(recs, target, k))
            ndcgs.append(ndcg_at_k(recs, target, k))
            mrrs.append(mrr_at_k(recs, target, k))

        result[f"HitRate@{k}"] = sum(hits) / len(hits)
        result[f"Recall@{k}"] = sum(recalls) / len(recalls)
        result[f"NDCG@{k}"] = sum(ndcgs) / len(ndcgs)
        result[f"MRR@{k}"] = sum(mrrs) / len(mrrs)

    return result
