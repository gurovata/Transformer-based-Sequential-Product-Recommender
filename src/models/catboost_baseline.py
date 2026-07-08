from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CatBoostNextItemBaseline:
    iterations: int = 100
    depth: int = 6
    learning_rate: float = 0.05

    def __post_init__(self) -> None:
        try:
            from catboost import CatBoostClassifier  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "CatBoost is optional. Install it with `pip install catboost` "
                "to train this baseline."
            ) from exc

        self.model = CatBoostClassifier(
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            loss_function="Logloss",
            verbose=False,
            random_seed=42,
        )

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "CatBoostNextItemBaseline":
        self.model.fit(features, target)
        return self

    def predict_scores(self, features: pd.DataFrame) -> pd.Series:
        return pd.Series(self.model.predict_proba(features)[:, 1], index=features.index)
