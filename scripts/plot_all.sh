#!/usr/bin/env bash
# Generate all plots from sweep results.
set -euo pipefail

python scripts/plot_results.py runs/results/results.csv --out runs/plots
echo "✓ Plots saved to runs/plots/"
