#!/usr/bin/env python
"""Generate publication-quality plots from sweep results.

Usage:
    python scripts/plot_results.py runs/results/results.csv

Generates:
    1. Return vs Compute Per Step (all envs, both modes)
    2. Success Rate vs Compute Per Step (AntMaze only)
    3. Horizon Sensitivity Curves (return vs H for each mode)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

COLORS = {
    "policy": "#2196F3",
    "planner": "#FF5722",
}


def load_results(csv_paths: list[str]) -> pd.DataFrame:
    dfs = [pd.read_csv(p) for p in csv_paths]
    return pd.concat(dfs, ignore_index=True)


def plot_return_vs_compute(df: pd.DataFrame, save_dir: Path):
    """Plot 1: Return vs per-step compute for all envs."""
    envs = df["env"].unique()
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)

    for i, env_name in enumerate(envs):
        ax = axes[0, i]
        env_df = df[df["env"] == env_name]

        for mode in ["policy", "planner"]:
            mode_df = env_df[env_df["mode"] == mode]
            if mode_df.empty:
                continue

            # Group by (H, K) and average over seeds
            if mode == "policy":
                grouped = mode_df.groupby("H").agg(
                    return_mean=("return_mean", "mean"),
                    return_std=("return_std", "mean"),
                    time_mean=("time_mean", "mean"),
                ).reset_index()
                label_fn = lambda row: f"H={int(row['H'])}"
            else:
                grouped = mode_df.groupby(["H", "K"]).agg(
                    return_mean=("return_mean", "mean"),
                    return_std=("return_std", "mean"),
                    time_mean=("time_mean", "mean"),
                ).reset_index()
                label_fn = lambda row: f"H={int(row['H'])},K={int(row['K'])}"

            ax.scatter(
                grouped["time_mean"] * 1000,  # ms
                grouped["return_mean"],
                c=COLORS[mode],
                label=mode.capitalize(),
                alpha=0.8,
                s=60,
                edgecolors="white",
                linewidth=0.5,
            )
            # Error bars
            ax.errorbar(
                grouped["time_mean"] * 1000,
                grouped["return_mean"],
                yerr=grouped["return_std"],
                fmt="none",
                ecolor=COLORS[mode],
                alpha=0.4,
            )

        ax.set_xlabel("Per-step compute (ms)")
        ax.set_ylabel("Return")
        ax.set_title(env_name)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "return_vs_compute.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✓ {out}")
    plt.close(fig)


def plot_success_vs_compute(df: pd.DataFrame, save_dir: Path):
    """Plot 2: Success rate vs compute (maze envs only)."""
    maze_df = df[df["env"].str.contains("maze", case=False)]
    if maze_df.empty:
        print("  ⚠ No maze env results, skipping success plot")
        return

    envs = maze_df["env"].unique()
    n_envs = len(envs)
    fig, axes = plt.subplots(1, max(n_envs, 1), figsize=(6 * max(n_envs, 1), 5), squeeze=False)

    for i, env_name in enumerate(envs):
        ax = axes[0, i]
        env_df = maze_df[maze_df["env"] == env_name]

        for mode in ["policy", "planner"]:
            mode_df = env_df[env_df["mode"] == mode]
            if mode_df.empty:
                continue

            if mode == "policy":
                grouped = mode_df.groupby("H").agg(
                    success_rate=("success_rate", "mean"),
                    time_mean=("time_mean", "mean"),
                ).reset_index()
            else:
                grouped = mode_df.groupby(["H", "K"]).agg(
                    success_rate=("success_rate", "mean"),
                    time_mean=("time_mean", "mean"),
                ).reset_index()

            ax.scatter(
                grouped["time_mean"] * 1000,
                grouped["success_rate"] * 100,
                c=COLORS[mode],
                label=mode.capitalize(),
                alpha=0.8,
                s=60,
                edgecolors="white",
                linewidth=0.5,
            )

        ax.set_xlabel("Per-step compute (ms)")
        ax.set_ylabel("Success rate (%)")
        ax.set_title(env_name)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-5, 105)

    plt.tight_layout()
    out = save_dir / "success_vs_compute.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✓ {out}")
    plt.close(fig)


def plot_horizon_sensitivity(df: pd.DataFrame, save_dir: Path):
    """Plot 3: Return vs H for policy and planner modes."""
    envs = df["env"].unique()
    n_envs = len(envs)
    fig, axes = plt.subplots(1, n_envs, figsize=(6 * n_envs, 5), squeeze=False)

    for i, env_name in enumerate(envs):
        ax = axes[0, i]
        env_df = df[df["env"] == env_name]

        # Policy: return vs H
        policy_df = env_df[env_df["mode"] == "policy"]
        if not policy_df.empty:
            grouped = policy_df.groupby("H")["return_mean"].agg(["mean", "std"]).reset_index()
            ax.plot(grouped["H"], grouped["mean"], "-o", color=COLORS["policy"],
                    label="Policy", linewidth=2, markersize=6)
            ax.fill_between(grouped["H"], grouped["mean"] - grouped["std"],
                            grouped["mean"] + grouped["std"],
                            color=COLORS["policy"], alpha=0.15)

        # Planner: one line per K value, vs H
        planner_df = env_df[env_df["mode"] == "planner"]
        if not planner_df.empty:
            K_values = sorted(planner_df["K"].unique())
            cmap = plt.cm.Oranges(np.linspace(0.4, 0.9, len(K_values)))
            for j, K in enumerate(K_values):
                k_df = planner_df[planner_df["K"] == K]
                grouped = k_df.groupby("H")["return_mean"].agg(["mean", "std"]).reset_index()
                ax.plot(grouped["H"], grouped["mean"], "-s", color=cmap[j],
                        label=f"Planner K={int(K)}", linewidth=1.5, markersize=5)

        ax.set_xlabel("Horizon H")
        ax.set_ylabel("Return")
        ax.set_title(env_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = save_dir / "horizon_sensitivity.png"
    fig.savefig(out, bbox_inches="tight")
    print(f"  ✓ {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_files", nargs="+", help="Path(s) to results CSV files")
    parser.add_argument("--out", type=str, default="runs/plots", help="Output directory for plots")
    args = parser.parse_args()

    save_dir = Path(args.out)
    save_dir.mkdir(parents=True, exist_ok=True)

    df = load_results(args.csv_files)
    print(f"Loaded {len(df)} rows from {len(args.csv_files)} file(s)")
    print(f"Envs: {df['env'].unique().tolist()}")
    print(f"Modes: {df['mode'].unique().tolist()}")
    print()

    plot_return_vs_compute(df, save_dir)
    plot_success_vs_compute(df, save_dir)
    plot_horizon_sensitivity(df, save_dir)

    print(f"\n✓ All plots saved to {save_dir}/")


if __name__ == "__main__":
    main()
