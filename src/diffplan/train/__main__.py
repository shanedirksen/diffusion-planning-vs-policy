"""Entry point: python -m diffplan.train

Usage:
    python -m diffplan.train --config configs/train.yaml --env halfcheetah-medium-v2 --H 32
"""

from diffplan.utils.config import build_config
from diffplan.algorithms.train_diffusion import train


def main():
    cfg = build_config()
    print(f"[diffplan.train] env={cfg['env']['name']}  H={cfg['env'].get('horizon', 32)}")
    train(cfg)


if __name__ == "__main__":
    main()
