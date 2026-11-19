"""Environment creation + wrappers for Gymnasium / MuJoCo tasks.

Uses Minari for offline datasets (replaces D4RL on Python 3.12+).
Environments are created via Gymnasium directly.
"""

from __future__ import annotations

from typing import Optional

import gymnasium as gym
import numpy as np


# ── Mapping: user-facing env name → (gymnasium env id, minari dataset id) ────

ENV_REGISTRY: dict[str, dict] = {
    # Locomotion (MuJoCo)
    "halfcheetah-medium-v2": {
        "gym_id": "HalfCheetah-v4",
        "minari_id": "mujoco/halfcheetah/medium-v0",
    },
    "hopper-medium-v2": {
        "gym_id": "Hopper-v4",
        "minari_id": "mujoco/hopper/medium-v0",
    },
    "invertedpendulum-medium-v0": {
        "gym_id": "InvertedPendulum-v5",
        "minari_id": "mujoco/invertedpendulum/medium-v0",
    },
    "inverteddoublependulum-medium-v0": {
        "gym_id": "InvertedDoublePendulum-v5",
        "minari_id": "mujoco/inverteddoublependulum/medium-v0",
    },
    # Point Maze
    "pointmaze-umaze-v2": {
        "gym_id": "PointMaze_UMaze-v3",
        "minari_id": "D4RL/pointmaze/umaze-v2",
    },
    "maze2d-umaze-v1": {
        "gym_id": "PointMaze_UMaze-v3",
        "minari_id": "D4RL/pointmaze/umaze-v2",
    },
    # AntMaze
    "antmaze-umaze-v2": {
        "gym_id": "AntMaze_UMaze-v4",
        "minari_id": "D4RL/antmaze/umaze-v1",
    },
    "antmaze-medium-play-v2": {
        "gym_id": "AntMaze_Medium-v4",
        "minari_id": "D4RL/antmaze/medium-play-v1",
    },
    "antmaze-large-play-v2": {
        "gym_id": "AntMaze_Large-v4",
        "minari_id": "D4RL/antmaze/large-play-v1",
    },
}


def get_gym_id(env_name: str) -> str:
    """Resolve a user-facing env name to a Gymnasium env ID."""
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name]["gym_id"]
    return env_name  # pass-through for raw Gymnasium IDs


def get_minari_id(env_name: str) -> str:
    """Resolve a user-facing env name to a Minari dataset ID."""
    if env_name in ENV_REGISTRY:
        return ENV_REGISTRY[env_name]["minari_id"]
    raise KeyError(f"No Minari dataset mapping for env '{env_name}'. "
                   f"Known envs: {list(ENV_REGISTRY.keys())}")


def make_env(
    env_name: str,
    seed: int = 0,
    max_episode_steps: Optional[int] = None,
) -> gym.Env:
    """Create a Gymnasium environment.

    Handles the env-name mapping and returns a standard Gymnasium Env
    with the new (obs, info) reset API and (obs, reward, terminated, truncated, info) step API.
    """
    gym_id = get_gym_id(env_name)

    kwargs = {}
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    env = gym.make(gym_id, **kwargs)
    env.reset(seed=seed)
    return env


def get_env_info(env) -> dict:
    """Extract useful shapes from an env."""
    obs_space = env.observation_space
    act_space = env.action_space
    return {
        "obs_dim": int(np.prod(obs_space.shape)),
        "act_dim": int(np.prod(act_space.shape)),
        "obs_shape": obs_space.shape,
        "act_shape": act_space.shape,
        "act_low": act_space.low.tolist(),
        "act_high": act_space.high.tolist(),
        "max_episode_steps": getattr(env, "_max_episode_steps",
                                     getattr(env.spec, "max_episode_steps", 1000) if env.spec else 1000),
    }


# ── Quick smoke test ─────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test an environment")
    parser.add_argument("--env", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    env = make_env(args.env, seed=args.seed)
    info = get_env_info(env)
    print(f"Env:      {args.env}")
    print(f"Gym ID:   {get_gym_id(args.env)}")
    print(f"Obs dim:  {info['obs_dim']}")
    print(f"Act dim:  {info['act_dim']}")

    obs, _ = env.reset(seed=args.seed)
    total_reward = 0.0
    steps = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, step_info = env.step(action)
        total_reward += reward
        steps += 1

    print(f"Random episode: {steps} steps, return = {total_reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
