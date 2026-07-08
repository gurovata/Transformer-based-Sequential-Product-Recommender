#!/usr/bin/env bash
set -euo pipefail

python -m src.evaluation.evaluate --models popularity itemknn als transformer --split test

