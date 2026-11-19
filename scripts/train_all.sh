#!/usr/bin/env bash
# Train one diffusion model per environment.
set -euo pipefail

ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "antmaze-umaze-v2")
HORIZON=32

for ENV in "${ENVS[@]}"; do
    echo "=== Training: ${ENV}  H=${HORIZON} ==="
    python -m diffplan.train \
        --config configs/train.yaml \
        --env "${ENV}" \
        --H "${HORIZON}"
    echo ""
done

echo "✓ All training runs complete."
