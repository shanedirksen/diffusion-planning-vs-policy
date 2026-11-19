"""Run all evaluation benchmarks: GPU & CPU × Policy & Planner(K=4,K=16) × N episodes.

Supports all 4 model types:
    python run_benchmarks.py                                         # action (default)
    python run_benchmarks.py --model transformer --ckpt runs/checkpoints/transformer.pt
    python run_benchmarks.py --model trajectory  --ckpt runs/checkpoints/trajectory.pt
    python run_benchmarks.py --model decision    --ckpt runs/checkpoints/decision.pt
    python run_benchmarks.py --all               # run all models that have checkpoints
"""

import sys
import time
import argparse
import subprocess
from pathlib import Path

PYTHON = str(Path(sys.executable))
N_EPISODES = 5

# Default checkpoint paths per model type
DEFAULT_CKPTS = {
    "action":      "runs/checkpoints/halfcheetah_medium_v2_H32.pt",
    "transformer": "runs/checkpoints/halfcheetah_medium_v2_transformer_H32.pt",
    "trajectory":  "runs/checkpoints/halfcheetah_medium_v2_trajectory_H32.pt",
    "decision":    "runs/checkpoints/halfcheetah_medium_v2_decision_H32.pt",
}


def get_configs(model_type: str):
    """Return list of (label, device, mode, K) configs for a model type."""
    return [
        (f"{model_type} GPU Policy",       "cuda", "policy",  0),
        (f"{model_type} GPU Planner K=4",  "cuda", "planner", 4),
        (f"{model_type} GPU Planner K=16", "cuda", "planner", 16),
        (f"{model_type} CPU Policy",       "cpu",  "policy",  0),
        (f"{model_type} CPU Planner K=4",  "cpu",  "planner", 4),
        (f"{model_type} CPU Planner K=16", "cpu",  "planner", 16),
    ]


def run_eval(label, device, mode, K, ckpt):
    config_file = f"configs/eval_{mode}.yaml"
    cmd = [
        PYTHON, "-u", "-m", "diffplan.eval",
        "--config", config_file,
        "--ckpt", ckpt,
        "--H", "32",
        f"eval.n_episodes={N_EPISODES}",
        f"device={device}",
    ]
    if mode == "planner":
        cmd.extend(["--K", str(K)])

    print(f"\n{'='*70}", flush=True)
    print(f"  {label}  |  {mode} K={K}  |  device={device}  |  {N_EPISODES} episodes", flush=True)
    print(f"{'='*70}", flush=True)

    start = time.time()
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    elapsed = time.time() - start

    mins = elapsed / 60
    status = "OK" if result.returncode == 0 else f"FAILED (code {result.returncode})"
    print(f"\n  >> {label}: {status} in {mins:.1f} min", flush=True)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Run evaluation benchmarks")
    parser.add_argument("--model", type=str, default="action",
                        choices=["action", "transformer", "trajectory", "decision"],
                        help="Model type to benchmark")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Override checkpoint path (default: per-model convention)")
    parser.add_argument("--all", action="store_true",
                        help="Run benchmarks for all model types that have checkpoints")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Number of episodes per config")
    args = parser.parse_args()

    global N_EPISODES
    N_EPISODES = args.episodes

    # Determine which models to benchmark
    if args.all:
        model_types = []
        for mt, default_ckpt in DEFAULT_CKPTS.items():
            if Path(default_ckpt).exists():
                model_types.append((mt, default_ckpt))
            else:
                print(f"  Skipping {mt}: {default_ckpt} not found", flush=True)
    else:
        ckpt = args.ckpt or DEFAULT_CKPTS[args.model]
        model_types = [(args.model, ckpt)]

    # Clear old results
    results_csv = Path("runs/results/results.csv")
    if results_csv.exists():
        results_csv.unlink()
        print("Cleared old results.csv", flush=True)

    total_start = time.time()
    results = []

    for model_type, ckpt in model_types:
        configs = get_configs(model_type)
        print(f"\n{'#'*70}", flush=True)
        print(f"  MODEL: {model_type}  |  CKPT: {ckpt}", flush=True)
        print(f"{'#'*70}", flush=True)

        for label, device, mode, K in configs:
            ok = run_eval(label, device, mode, K, ckpt)
            results.append((label, ok))

    # Summary
    total_mins = (time.time() - total_start) / 60
    print(f"\n{'='*70}", flush=True)
    print(f"  ALL BENCHMARKS COMPLETE  ({total_mins:.1f} min total)", flush=True)
    print(f"{'='*70}", flush=True)
    for label, ok in results:
        print(f"  {'[OK]' if ok else '[FAIL]'}  {label}", flush=True)

    print(f"\nResults saved to: {results_csv}", flush=True)


if __name__ == "__main__":
    main()
