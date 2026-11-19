#!/usr/bin/env python
"""Pre-download Minari datasets so training starts instantly.

Usage:
    python scripts/download_datasets.py [--envs halfcheetah-medium-v2 hopper-medium-v2 ...]
"""

from __future__ import annotations

import argparse

import minari

from diffplan.envs.make_env import get_minari_id

DEFAULT_ENVS = [
    "halfcheetah-medium-v2",
    "hopper-medium-v2",
    "invertedpendulum-medium-v0",
    "inverteddoublependulum-medium-v0",
]


def main():
    parser = argparse.ArgumentParser(description="Pre-download Minari datasets")
    parser.add_argument("--envs", nargs="+", default=DEFAULT_ENVS)
    args = parser.parse_args()

    for env_name in args.envs:
        minari_id = get_minari_id(env_name)
        print(f"Downloading: {env_name} -> {minari_id}")
        try:
            minari.load_dataset(minari_id)
            print(f"  Already cached.")
        except Exception:
            minari.download_dataset(minari_id)
            ds = minari.load_dataset(minari_id)
            print(f"  Downloaded ({ds.total_steps:,} steps)")

    print("\nAll datasets ready.")


if __name__ == "__main__":
    main()
