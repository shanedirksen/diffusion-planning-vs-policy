"""Evaluation metrics: return, success rate, trap-like failures."""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def compute_return_stats(returns: List[float]) -> dict:
    """Aggregate episode returns."""
    arr = np.array(returns)
    return {
        "return_mean": float(arr.mean()),
        "return_std": float(arr.std()),
        "return_min": float(arr.min()),
        "return_max": float(arr.max()),
        "return_median": float(np.median(arr)),
    }


def compute_success_rate(successes: List[bool]) -> dict:
    """Fraction of episodes that succeeded (e.g., reached goal in AntMaze)."""
    arr = np.array(successes, dtype=float)
    return {
        "success_rate": float(arr.mean()),
        "n_success": int(arr.sum()),
        "n_episodes": len(arr),
    }


# ── Trap-like failure detection ──────────────────────────────────────────────

def detect_trap_failures(
    trajectories: List[dict],
    progress_threshold: float = 0.1,
    stall_window: int = 50,
) -> dict:
    """Heuristic "trap-like failure" detector for maze environments.

    A trajectory is flagged as a "trap failure" if:
      1. The agent fails to reach the goal (not successful).
      2. The agent's progress (distance decrease toward goal) stalls for
         ``stall_window`` consecutive steps.

    Each trajectory dict should have:
        - ``"success"``: bool
        - ``"positions"``: np.ndarray of shape (T, 2+) - XY positions
        - ``"goal_pos"``: np.ndarray of shape (2,) - goal XY

    Returns summary statistics.
    """
    n_total = len(trajectories)
    n_trap = 0
    trap_details = []

    for traj in trajectories:
        if traj.get("success", False):
            continue  # not a failure at all

        positions = traj.get("positions")
        goal = traj.get("goal_pos")
        if positions is None or goal is None:
            continue

        # Distance to goal over time
        dists = np.linalg.norm(positions[:, :2] - goal[:2], axis=1)

        # Check for prolonged stall
        is_trap = False
        for start in range(len(dists) - stall_window):
            window = dists[start : start + stall_window]
            progress = dists[start] - dists[start + stall_window - 1]
            if progress < progress_threshold:
                is_trap = True
                break

        if is_trap:
            n_trap += 1
            trap_details.append({
                "final_dist": float(dists[-1]),
                "min_dist": float(dists.min()),
            })

    return {
        "n_trap_failures": n_trap,
        "trap_failure_rate": n_trap / max(n_total, 1),
        "n_total_episodes": n_total,
        "trap_details": trap_details,
    }


def time_to_first_progress(
    positions: np.ndarray,
    goal_pos: np.ndarray,
    threshold_fraction: float = 0.1,
) -> Optional[int]:
    """Number of steps before the agent reduces its distance to goal by
    ``threshold_fraction`` of the initial distance. Returns ``None`` if
    the agent never makes progress."""
    dists = np.linalg.norm(positions[:, :2] - goal_pos[:2], axis=1)
    initial = dists[0]
    target = initial * (1 - threshold_fraction)
    indices = np.where(dists <= target)[0]
    if len(indices) == 0:
        return None
    return int(indices[0])
