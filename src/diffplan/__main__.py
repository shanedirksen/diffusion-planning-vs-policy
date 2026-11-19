"""Top-level entry: python -m diffplan

Dispatches to train or eval based on the first positional argument,
but the intended usage is:
    python -m diffplan.train --config ...
    python -m diffplan.eval  --config ...
"""

import sys


def main():
    print("Usage:")
    print("  python -m diffplan.train --config configs/train.yaml --env halfcheetah-medium-v2 --H 32")
    print("  python -m diffplan.eval  --config configs/eval_policy.yaml --ckpt runs/checkpoints/model.pt")
    sys.exit(0)


if __name__ == "__main__":
    main()
