"""Episode rollout runner with metrics and timing collection.

Unified interface for both policy and planner modes.
"""

from __future__ import annotations

import csv
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from diffplan.algorithms.policy_mode import PolicyMode
from diffplan.algorithms.planner_mode import PlannerMode
from diffplan.eval.metrics import (
    compute_return_stats,
    compute_success_rate,
    detect_trap_failures,
    time_to_first_progress,
)
from diffplan.eval.timing import StepTimer


def run_episode_policy(
    env,
    agent: PolicyMode,
    max_steps: int = 1000,
    record_positions: bool = False,
    seed: int | None = None,
) -> dict:
    """Run one episode using PolicyMode.

    Returns a dict with ``return``, ``steps``, ``success``, timing, etc.
    """
    timer = StepTimer()
    agent.reset()
    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)
    total_reward = 0.0
    done = False
    step = 0
    info = {}
    positions = []

    while not done and step < max_steps:
        if record_positions:
            pos = _get_position(env)
            if pos is not None:
                positions.append(pos)

        timer.start()
        action = agent.act(obs)
        timer.stop()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    result = {
        "return": total_reward,
        "steps": step,
        "success": _check_success(info if done else {}, env),
    }
    result.update(timer.summary())

    if record_positions and positions:
        result["positions"] = np.array(positions)
        goal = _get_goal(env)
        if goal is not None:
            result["goal_pos"] = goal

    return result


def run_episode_planner(
    env,
    agent: PlannerMode,
    max_steps: int = 1000,
    record_positions: bool = False,
    seed: int | None = None,
) -> dict:
    """Run one episode using PlannerMode.

    Returns a dict with ``return``, ``steps``, ``success``, timing, etc.
    """
    timer = StepTimer()
    agent.reset()
    reset_kwargs = {"seed": seed} if seed is not None else {}
    obs, _ = env.reset(**reset_kwargs)
    total_reward = 0.0
    done = False
    step = 0
    info = {}
    positions = []

    while not done and step < max_steps:
        if record_positions:
            pos = _get_position(env)
            if pos is not None:
                positions.append(pos)

        timer.start()
        action = agent.act(obs, env)
        timer.stop()

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

    result = {
        "return": total_reward,
        "steps": step,
        "success": _check_success(info if done else {}, env),
    }
    result.update(timer.summary())

    if record_positions and positions:
        result["positions"] = np.array(positions)
        goal = _get_goal(env)
        if goal is not None:
            result["goal_pos"] = goal

    return result


def evaluate(
    env,
    agent,
    mode: str,
    n_episodes: int = 20,
    max_steps: int = 1000,
    record_positions: bool = False,
    seed: int = 0,
) -> dict:
    """Run N episodes and aggregate metrics.

    Args:
        mode: ``"policy"`` or ``"planner"``
        seed: base seed - episode i uses seed + i for reproducibility.
    """
    results: List[dict] = []
    run_fn = run_episode_policy if mode == "policy" else run_episode_planner

    for ep in range(n_episodes):
        ep_seed = seed + ep
        result = run_fn(
            env, agent, max_steps=max_steps,
            record_positions=record_positions, seed=ep_seed,
        )
        results.append(result)
        print(f"  Episode {ep+1:>3d}/{n_episodes}  "
              f"return={result['return']:.2f}  "
              f"steps={result['steps']}  "
              f"success={result['success']}  "
              f"time_mean={result['time_mean']*1000:.1f}ms")

    # ── Aggregate ──
    returns = [r["return"] for r in results]
    successes = [r["success"] for r in results]
    time_means = [r["time_mean"] for r in results]
    time_p95s = [r["time_p95"] for r in results]

    summary = compute_return_stats(returns)
    summary.update(compute_success_rate(successes))
    summary["time_mean"] = float(np.mean(time_means))
    summary["time_p95"] = float(np.mean(time_p95s))

    # Trap failure analysis (if position data is available)
    if record_positions:
        trap_info = detect_trap_failures(results)
        summary.update(trap_info)

    summary["mode"] = mode
    summary["episode_results"] = results

    # Inference timing from agent (if available)
    if hasattr(agent, "inference_times") and agent.inference_times:
        summary["time_inference_mean"] = float(np.mean(agent.inference_times))
        summary["time_inference_std"] = float(np.std(agent.inference_times))
    else:
        summary["time_inference_mean"] = summary.get("time_mean", 0)
        summary["time_inference_std"] = 0.0

    return summary


def save_results_csv(
    results: dict,
    env_name: str,
    mode: str,
    H: int,
    K: int = 0,
    seed: int = 0,
    device: str = "cuda",
    results_dir: str = "runs/results",
    model_type: str = "action",
    obs_horizon: int = 1,
    action_horizon: int = 1,
):
    """Append one row to a CSV results file."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / "results.csv"

    row = {
        "env": env_name,
        "model_type": model_type,
        "mode": mode,
        "device": device,
        "Tp": H,
        "To": obs_horizon,
        "Ta": action_horizon,
        "K": K,
        "seed": seed,
        "return_mean": results.get("return_mean", 0),
        "return_std": results.get("return_std", 0),
        "success_rate": results.get("success_rate", 0),
        "time_step_mean_ms": results.get("time_mean", 0) * 1000,
        "time_step_p95_ms": results.get("time_p95", 0) * 1000,
        "time_inference_mean_ms": results.get("time_inference_mean", 0) * 1000,
        "n_episodes": results.get("n_episodes", 0),
        "n_trap_failures": results.get("n_trap_failures", 0),
        "trap_failure_rate": results.get("trap_failure_rate", 0),
    }

    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print(f"  [saved] results -> {csv_path}")


# ── Private helpers ──────────────────────────────────────────────────────────

def _get_position(env) -> Optional[np.ndarray]:
    """Try to extract XY position from the environment (for maze envs)."""
    try:
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "data"):
            return unwrapped.data.qpos[:2].copy()
    except Exception:
        pass
    return None


def _get_goal(env) -> Optional[np.ndarray]:
    """Try to extract goal position from AntMaze / maze2d envs."""
    try:
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "target_goal"):
            return np.array(unwrapped.target_goal[:2])
        if hasattr(unwrapped, "goal"):
            return np.array(unwrapped.goal[:2])
        if hasattr(unwrapped, "_target"):
            return np.array(unwrapped._target[:2])
    except Exception:
        pass
    return None


def _check_success(info: dict, env) -> bool:
    """Determine if an episode was successful."""
    # D4RL AntMaze and maze2d use 'success' in info
    if "success" in info:
        return bool(info["success"])
    # Some versions use goal_achieved
    if "goal_achieved" in info:
        return bool(info["goal_achieved"])
    return False
