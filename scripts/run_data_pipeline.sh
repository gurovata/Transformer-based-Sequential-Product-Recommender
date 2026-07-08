#!/usr/bin/env bash
set -euo pipefail

python -m src.data.generate_synthetic
python -m src.data.prepare_sequences
python -m src.features.item_features
python -m src.evaluation.eda
python -m src.data.load_sqlite

