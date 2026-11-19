#!/usr/bin/env bash
# Evaluate all checkpoints in both modes.
set -euo pipefail

CKPT_DIR="runs/checkpoints"
ENVS=("halfcheetah-medium-v2" "hopper-medium-v2" "antmaze-umaze-v2")

for ENV in "${ENVS[@]}"; do
    CKPT_NAME="${ENV//-/_}_H32.pt"
    CKPT="${CKPT_DIR}/${CKPT_NAME}"

    if [ ! -f "${CKPT}" ]; then
        echo "⚠ Checkpoint not found: ${CKPT}, skipping ${ENV}"
        continue
    fi

    echo "=== Policy mode: ${ENV} ==="
    python -m diffplan.eval \
        --config configs/eval_policy.yaml \
        --env "${ENV}" \
        --ckpt "${CKPT}"

    echo "=== Planner mode: ${ENV} (K=16) ==="
    python -m diffplan.eval \
        --config configs/eval_planner.yaml \
        --env "${ENV}" \
        --ckpt "${CKPT}" \
        --K 16

    echo ""
done

echo "✓ All evaluations complete."
