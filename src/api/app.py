from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.models.transformer import load_transformer_recommender
from src.utils.config import load_config
from src.utils.logging import setup_logging


setup_logging()
logger = logging.getLogger(__name__)


class RecommendRequest(BaseModel):
    user_id: int | None = None
    history: list[int] = Field(default_factory=list)
    top_k: int = Field(default=10, ge=1, le=100)


class RecommendationItem(BaseModel):
    item_id: int
    score: float


class RecommendResponse(BaseModel):
    recommendations: list[RecommendationItem]
    model: str


class RecommendationService:
    def __init__(self) -> None:
        self.config = load_config()
        self.processed_dir = Path(self.config["paths"]["processed_dir"])
        self.models_dir = Path(self.config["paths"]["models_dir"])
        self.raw_to_idx: dict[int, int] = {}
        self.idx_to_raw: dict[int, int] = {}
        self.model: Any | None = None
        self.model_name = "unavailable"

        self._load_item_mapping()
        self._load_model()

    def _load_item_mapping(self) -> None:
        mapping_path = self.processed_dir / "item_mapping.csv"
        if not mapping_path.exists():
            logger.warning("Item mapping does not exist yet: %s", mapping_path)
            return

        mapping = pd.read_csv(mapping_path)
        self.raw_to_idx = {
            int(row.item_id): int(row.item_idx) for row in mapping.itertuples(index=False)
        }
        self.idx_to_raw = {
            int(row.item_idx): int(row.item_id) for row in mapping.itertuples(index=False)
        }

    def _load_model(self) -> None:
        transformer_path = self.models_dir / "transformer.pt"
        if transformer_path.exists():
            self.model = load_transformer_recommender(transformer_path, self.config)
            self.model_name = "transformer"
            logger.info("Loaded Transformer recommender from %s", transformer_path)
            return

        popularity_path = self.models_dir / "popularity.pkl"
        if popularity_path.exists():
            with popularity_path.open("rb") as file:
                self.model = pickle.load(file)
            self.model_name = "popularity"
            logger.info("Loaded popularity fallback from %s", popularity_path)
            return

        logger.warning("No trained model found in %s", self.models_dir)

    def recommend(self, request: RecommendRequest) -> RecommendResponse:
        if self.model is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No trained model is available. Run data preparation and training first: "
                    "`python -m src.training.train_baselines` or "
                    "`python -m src.training.train_transformer`."
                ),
            )

        internal_history = [
            self.raw_to_idx[item_id]
            for item_id in request.history
            if item_id in self.raw_to_idx
        ]
        recs = self.model.recommend(
            user_idx=None,
            history=internal_history,
            top_k=request.top_k,
            exclude_seen=True,
        )
        response_items = [
            RecommendationItem(
                item_id=int(self.idx_to_raw.get(int(item_idx), int(item_idx))),
                score=float(score),
            )
            for item_idx, score in recs
        ]
        return RecommendResponse(recommendations=response_items, model=self.model_name)


app = FastAPI(
    title="Sequential Product Recommender",
    version="0.1.0",
    description="Small Transformer-based next-item recommendation API.",
)
service = RecommendationService()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": service.model_name}


@app.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest) -> RecommendResponse:
    return service.recommend(request)


def main() -> None:
    import uvicorn

    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()

