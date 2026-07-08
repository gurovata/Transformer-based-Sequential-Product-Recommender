from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

from src.models.popularity import PopularityRecommender


logger = logging.getLogger(__name__)


@dataclass
class ALSRecommender:
    num_users: int
    num_items: int
    factors: int = 32
    iterations: int = 12
    regularization: float = 0.05
    alpha: float = 20.0
    mode: str = "svd_fallback"
    model: object | None = None
    user_item: sparse.csr_matrix | None = None
    user_factors: np.ndarray | None = None
    item_factors: np.ndarray | None = None
    popularity: PopularityRecommender | None = None

    def fit(self, interactions: pd.DataFrame) -> "ALSRecommender":
        self.user_item = sparse.csr_matrix(
            (
                interactions["event_weight"].astype(float),
                (interactions["user_idx"].astype(int), interactions["item_idx"].astype(int)),
            ),
            shape=(self.num_users, self.num_items),
        )
        self.popularity = PopularityRecommender(self.num_items).fit(interactions)

        try:
            from implicit.als import AlternatingLeastSquares  # type: ignore

            self.mode = "implicit_als"
            self.model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                iterations=self.iterations,
                random_state=42,
            )
            self.model.fit((self.user_item.T * self.alpha).astype("double"))
            logger.info("Fitted implicit ALS baseline.")
            return self
        except ModuleNotFoundError:
            logger.info("implicit is not installed; using TruncatedSVD fallback for ALS baseline.")

        max_components = min(self.user_item.shape) - 1
        n_components = max(2, min(self.factors, max_components))
        svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=self.iterations)
        self.user_factors = svd.fit_transform(self.user_item)
        self.item_factors = svd.components_.T
        self.model = svd
        self.mode = "svd_fallback"
        return self

    def recommend(
        self,
        user_idx: int | None = None,
        history: Sequence[int] | None = None,
        top_k: int = 10,
        exclude_seen: bool = True,
    ) -> list[tuple[int, float]]:
        if self.user_item is None or self.popularity is None:
            raise RuntimeError("ALSRecommender must be fitted before recommend().")

        history = [int(item) for item in (history or []) if 2 <= int(item) < self.num_items]
        if self.mode == "implicit_als" and self.model is not None and user_idx is not None:
            filter_items = [0, 1] + (history if exclude_seen else [])
            ids, scores = self.model.recommend(
                int(user_idx),
                self.user_item[int(user_idx)],
                N=top_k,
                filter_items=filter_items,
            )
            return [(int(item), float(score)) for item, score in zip(ids, scores)]

        if self.user_factors is None or self.item_factors is None:
            return self.popularity.recommend(user_idx, history, top_k, exclude_seen)

        if user_idx is not None and 0 <= int(user_idx) < len(self.user_factors):
            user_vector = self.user_factors[int(user_idx)]
        elif history:
            user_vector = self.item_factors[history].mean(axis=0)
        else:
            return self.popularity.recommend(user_idx, history, top_k, exclude_seen)

        scores = (user_vector @ self.item_factors.T).astype(np.float32)
        if self.popularity.scores is not None:
            pop = self.popularity.scores.copy()
            pop[~np.isfinite(pop)] = 0.0
            if pop.max() > 0:
                scores += 0.01 * (pop / pop.max())

        scores[:2] = -np.inf
        if exclude_seen and history:
            scores[history] = -np.inf

        finite = np.isfinite(scores)
        if not finite.any():
            return []

        k = min(top_k, int(finite.sum()))
        top_indices = np.argpartition(-scores, kth=np.arange(k))[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]
        return [(int(item), float(scores[item])) for item in top_indices]

