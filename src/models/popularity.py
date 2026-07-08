from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass
class PopularityRecommender:
    num_items: int
    scores: np.ndarray | None = None

    def fit(
        self,
        interactions: pd.DataFrame,
        item_col: str = "item_idx",
        weight_col: str = "event_weight",
    ) -> "PopularityRecommender":
        scores = np.zeros(self.num_items, dtype=np.float32)
        if interactions.empty:
            self.scores = scores
            return self

        weights = (
            interactions.groupby(item_col)[weight_col]
            .sum()
            .reindex(range(self.num_items), fill_value=0.0)
        )
        scores[: len(weights)] = weights.to_numpy(dtype=np.float32)
        scores[:2] = -np.inf
        self.scores = scores
        return self

    def recommend(
        self,
        user_idx: int | None = None,
        history: Sequence[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        if self.scores is None:
            raise RuntimeError("PopularityRecommender must be fitted before recommend().")

        scores = self.scores.copy()
        scores[:2] = -np.inf
        if exclude_seen and history:
            valid_seen = [item for item in history if 0 <= int(item) < len(scores)]
            scores[valid_seen] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []

        k = min(top_k, int(finite.sum()))
        top_indices = np.argpartition(-scores, kth=np.arange(k))[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(item), float(scores[item])) for item in top_indices]

