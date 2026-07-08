from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import normalize

from src.models.popularity import PopularityRecommender


@dataclass
class ItemKNNRecommender:
    num_users: int
    num_items: int
    top_neighbors: int = 80
    similarity: sparse.csr_matrix | None = None
    popularity: PopularityRecommender | None = None

    def fit(self, interactions: pd.DataFrame) -> "ItemKNNRecommender":
        user_item = sparse.csr_matrix(
            (
                interactions["event_weight"].astype(float),
                (interactions["user_idx"].astype(int), interactions["item_idx"].astype(int)),
            ),
            shape=(self.num_users, self.num_items),
        )
        item_user = normalize(user_item.T, norm="l2", axis=1)
        similarity = (item_user @ item_user.T).tocsr()
        similarity.setdiag(0.0)
        similarity.eliminate_zeros()
        self.similarity = self._keep_top_neighbors(similarity, self.top_neighbors)
        self.popularity = PopularityRecommender(self.num_items).fit(interactions)
        return self

    @staticmethod
    def _keep_top_neighbors(matrix: sparse.csr_matrix, k: int) -> sparse.csr_matrix:
        if k <= 0:
            return matrix

        matrix = matrix.tolil()
        for row_idx in range(matrix.shape[0]):
            values = np.asarray(matrix.data[row_idx], dtype=float)
            if len(values) <= k:
                continue
            cols = np.asarray(matrix.rows[row_idx], dtype=int)
            keep = np.argpartition(-values, kth=k - 1)[:k]
            matrix.rows[row_idx] = cols[keep].tolist()
            matrix.data[row_idx] = values[keep].tolist()
        return matrix.tocsr()

    def recommend(
        self,
        user_idx: int | None = None,
        history: Sequence[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        if self.similarity is None or self.popularity is None:
            raise RuntimeError("ItemKNNRecommender must be fitted before recommend().")

        history = [int(item) for item in (history or []) if 2 <= int(item) < self.num_items]
        if not history:
            return self.popularity.recommend(user_idx, history, top_k, exclude_seen)

        scores = np.asarray(self.similarity[history].sum(axis=0)).ravel().astype(np.float32)
        if self.popularity.scores is not None:
            pop = self.popularity.scores.copy()
            pop[~np.isfinite(pop)] = 0.0
            if pop.max() > 0:
                scores += 0.01 * (pop / pop.max())

        scores[:2] = -np.inf
        if exclude_seen:
            scores[history] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []

        k = min(top_k, int(finite.sum()))
        top_indices = np.argpartition(-scores, kth=np.arange(k))[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(item), float(scores[item])) for item in top_indices]

