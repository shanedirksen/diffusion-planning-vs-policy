#!/usr/bin/env python
"""Run the full sweep grid: policy (H sweep) + planner (H × K sweep).

Usage:
    python scripts/run_sweep.py --config configs/sweeps.yaml
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sweeps.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sweep = cfg["sweep"]
    envs = sweep["envs"]
    seeds = sweep.get("seeds", [0])
    n_ep = sweep.get("n_episodes", 20)

    ckpt_dir = Path("runs/checkpoints")

    for env_name in envs:
        # Find checkpoint (try common naming patterns)
        ckpt_stem = env_name.replace("-", "_")
        ckpt_candidates = list(ckpt_dir.glob(f"{ckpt_stem}*.pt"))
        if not ckpt_candidates:
            print(f"⚠ No checkpoint found for {env_name}, skipping")
            continue
        ckpt = str(ckpt_candidates[0])

        for seed in seeds:
            # ── Policy mode sweep ──
            for H in sweep["policy"]["horizons"]:
                print(f"\n{'='*60}")
                print(f"POLICY | env={env_name} H={H} seed={seed}")
                print(f"{'='*60}")
                cmd = [
                    sys.executable, "-m", "diffplan.eval",
                    "--config", "configs/eval_policy.yaml",
                    "--env", env_name,
                    "--ckpt", ckpt,
                    "--H", str(H),
                    "--seed", str(seed),
                ]
                subprocess.run(cmd, check=True)

            # ── Planner mode sweep ──
            for H in sweep["planner"]["horizons"]:
                for K in sweep["planner"]["K_values"]:
                    print(f"\n{'='*60}")
                    print(f"PLANNER | env={env_name} H={H} K={K} seed={seed}")
                    print(f"{'='*60}")
                    cmd = [
                        sys.executable, "-m", "diffplan.eval",
                        "--config", "configs/eval_planner.yaml",
                        "--env", env_name,
                        "--ckpt", ckpt,
                        "--H", str(H),
                        "--K", str(K),
                        "--seed", str(seed),
                    ]
                    subprocess.run(cmd, check=True)

    print("\n✓ Full sweep complete. Results in runs/results/")


if __name__ == "__main__":
    main()
