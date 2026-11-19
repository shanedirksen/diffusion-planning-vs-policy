"""Save / restore environment state for the planner's look-ahead rollouts.

Works with Gymnasium MuJoCo envs (mujoco >= 3.x).
Uses ``env.unwrapped.data.qpos / qvel`` and ``mujoco.mj_forward()``.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np


def _unwrapped_mujoco(env):
    """Return the unwrapped Gymnasium MuJoCo env that exposes .data/.model."""
    unwrapped = getattr(env, "unwrapped", env)
    if hasattr(unwrapped, "data") and hasattr(unwrapped, "model"):
        return unwrapped
    raise NotImplementedError(
        "save_state / restore_state only supports Gymnasium MuJoCo envs. "
        f"Got env type: {type(env)}, unwrapped: {type(unwrapped)}"
    )


def save_state(env) -> dict:
    """Capture a snapshot of the environment state.

    Returns a dict that can be passed back to :func:`restore_state`.
    """
    state: dict[str, Any] = {}
    unwrapped = _unwrapped_mujoco(env)

    state["qpos"] = unwrapped.data.qpos.copy()
    state["qvel"] = unwrapped.data.qvel.copy()

    # Elapsed steps counter (for TimeLimit wrappers)
    if hasattr(env, "_elapsed_steps"):
        state["elapsed_steps"] = env._elapsed_steps

    return state


def restore_state(env, state: dict):
    """Restore a previously saved snapshot."""
    import mujoco

    unwrapped = _unwrapped_mujoco(env)

    unwrapped.data.qpos[:] = state["qpos"]
    unwrapped.data.qvel[:] = state["qvel"]
    mujoco.mj_forward(unwrapped.model, unwrapped.data)

    if "elapsed_steps" in state:
        env._elapsed_steps = state["elapsed_steps"]


def clone_state(state: dict) -> dict:
    """Deep-copy a state snapshot."""
    return copy.deepcopy(state)
