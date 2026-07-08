from __future__ import annotations

import math
import unittest

from src.evaluation.metrics import (
    hit_rate_at_k,
    mrr_at_k,
    ndcg_at_k,
    ranking_metrics_at_k,
    recall_at_k,
)


class RankingMetricsTest(unittest.TestCase):
    def test_single_target_metrics(self) -> None:
        recommendations = [10, 20, 30]
        target = 20

        self.assertEqual(hit_rate_at_k(recommendations, target, 1), 0.0)
        self.assertEqual(hit_rate_at_k(recommendations, target, 2), 1.0)
        self.assertEqual(recall_at_k(recommendations, target, 2), 1.0)
        self.assertAlmostEqual(mrr_at_k(recommendations, target, 3), 0.5)
        self.assertAlmostEqual(ndcg_at_k(recommendations, target, 3), 1 / math.log2(3))

    def test_average_metrics(self) -> None:
        recommendations = {
            1: [10, 20, 30],
            2: [40, 50, 60],
            3: [70, 80, 90],
        }
        targets = {1: 20, 2: 99, 3: 70}
        metrics = ranking_metrics_at_k(recommendations, targets, ks=(1, 2))

        self.assertAlmostEqual(metrics["HitRate@1"], 1 / 3)
        self.assertAlmostEqual(metrics["Recall@1"], 1 / 3)
        self.assertAlmostEqual(metrics["MRR@2"], (0.5 + 0.0 + 1.0) / 3)
        self.assertAlmostEqual(metrics["NDCG@2"], ((1 / math.log2(3)) + 0.0 + 1.0) / 3)


if __name__ == "__main__":
    unittest.main()

