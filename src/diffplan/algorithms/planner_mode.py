"""Planner mode: sample K sequences → rollout + score → execute best Ta actions.

This is our **experimental extension** of the Chi et al. (2023) Diffusion Policy,
adding K-candidate planning via environment rollouts (inspired by Janner et al.
2022).  It uses the same receding-horizon action-chunking scheme as PolicyMode
(To observation window, Ta action buffer) so that comparisons between policy
mode and planner mode are fair.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from diffplan.data.normalizer import Normalizer
from diffplan.envs.state_io import save_state, restore_state
from diffplan.models.sampler import DiffusionSampler


class PlannerMode:
    """Look-ahead planner with receding-horizon action chunking.

    Samples K candidate action sequences, scores them by simulated rollout
    in the environment, then commits to the first Ta actions of the best
    candidate.

    Parameters
    ----------
    sampler : DiffusionSampler
        Trained diffusion model wrapped in a sampling helper.
    normalizer : Normalizer
        Dataset statistics for obs/action normalization.
    horizon : int
        Prediction horizon Tp.
    act_dim : int
        Dimensionality of the action space.
    obs_dim : int
        Per-timestep observation dimensionality.
    obs_horizon : int
        Number of past observations to condition on (To). Default 1.
    action_horizon : int
        Number of actions to execute before replanning (Ta). Default 1.
    K : int
        Number of candidate plans to sample and score.
    discount : float
        Discount factor γ for scoring rollouts.
    device : torch.device
        Device for model inference.
    ddim_steps : int
        Number of DDIM denoising steps.
    """

    def __init__(
        self,
        sampler: DiffusionSampler,
        normalizer: Normalizer,
        horizon: int,
        act_dim: int,
        obs_dim: int,
        obs_horizon: int = 1,
        action_horizon: int = 1,
        K: int = 16,
        discount: float = 0.99,
        device: torch.device = torch.device("cpu"),
        ddim_steps: int = 20,
    ):
        self.sampler = sampler
        self.normalizer = normalizer
        self.horizon = horizon
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.K = K
        self.discount = discount
        self.device = device
        self.ddim_steps = ddim_steps

        # State buffers
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=obs_horizon)
        self._action_buffer: list[np.ndarray] = []
        self._buffer_idx: int = 0

        # Inference timing
        self.inference_times: list[float] = []

    def reset(self) -> None:
        """Clear all buffers. Must be called between episodes."""
        self._obs_buffer.clear()
        self._action_buffer = []
        self._buffer_idx = 0

    def _build_obs_window(self) -> np.ndarray:
        """Build flattened, normalized observation window.

        Returns (To * obs_dim,) numpy array.
        """
        while len(self._obs_buffer) < self.obs_horizon:
            self._obs_buffer.appendleft(self._obs_buffer[0])
        norm_obs = [self.normalizer.normalize_obs(o) for o in self._obs_buffer]
        return np.concatenate(norm_obs)

    @torch.no_grad()
    def act(self, obs: np.ndarray, env) -> np.ndarray:
        """Select an action by K-candidate planning with action chunking.

        Args:
            obs: current (unnormalized) observation.
            env: the live environment (must support save_state / restore_state).

        Returns:
            action: (act_dim,) numpy array in original action space.
        """
        self._obs_buffer.append(obs)

        # Return buffered action if available
        if self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            return action

        # --- Fresh planning ---
        state_snapshot = save_state(env)

        obs_window = self._build_obs_window()
        obs_t = torch.as_tensor(
            obs_window, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, To * obs_dim)

        t0 = time.perf_counter()
        # Sample K candidates: (K, Tp, act_dim) in normalized space
        action_seqs = self.sampler.ddim_sample(
            obs=obs_t,
            horizon=self.horizon,
            act_dim=self.act_dim,
            n_samples=self.K,
            ddim_steps=self.ddim_steps,
        )
        t1 = time.perf_counter()
        self.inference_times.append(t1 - t0)

        action_seqs_raw = self.normalizer.unnormalize_act(
            action_seqs.cpu().numpy()
        )  # (K, Tp, act_dim)

        # ── Score each candidate via rollout (all Tp steps) ──
        returns = np.zeros(self.K)
        for k in range(self.K):
            restore_state(env, state_snapshot)
            total_reward = 0.0
            for h in range(self.horizon):
                action = np.clip(
                    action_seqs_raw[k, h],
                    env.action_space.low,
                    env.action_space.high,
                )
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += (self.discount ** h) * reward
                if terminated or truncated:
                    break
            returns[k] = total_reward

        best_idx = int(np.argmax(returns))
        restore_state(env, state_snapshot)

        # Buffer the first Ta actions of the best sequence
        self._action_buffer = [
            np.clip(
                action_seqs_raw[best_idx, h],
                env.action_space.low,
                env.action_space.high,
            )
            for h in range(self.action_horizon)
        ]
        self._buffer_idx = 1  # returning index 0 now
        return self._action_buffer[0]

    @torch.no_grad()
    def act_with_info(self, obs: np.ndarray, env) -> tuple[np.ndarray, dict]:
        """Like ``act`` but also returns debug information."""
        self._obs_buffer.append(obs)

        if self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            return action, {"buffered": True}

        state_snapshot = save_state(env)

        obs_window = self._build_obs_window()
        obs_t = torch.as_tensor(
            obs_window, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        t0 = time.perf_counter()
        action_seqs = self.sampler.ddim_sample(
            obs=obs_t,
            horizon=self.horizon,
            act_dim=self.act_dim,
            n_samples=self.K,
            ddim_steps=self.ddim_steps,
        )
        t1 = time.perf_counter()
        self.inference_times.append(t1 - t0)

        action_seqs_raw = self.normalizer.unnormalize_act(
            action_seqs.cpu().numpy()
        )

        returns = np.zeros(self.K)
        for k in range(self.K):
            restore_state(env, state_snapshot)
            total_reward = 0.0
            for h in range(self.horizon):
                action = np.clip(
                    action_seqs_raw[k, h],
                    env.action_space.low,
                    env.action_space.high,
                )
                _, reward, terminated, truncated, _ = env.step(action)
                total_reward += (self.discount ** h) * reward
                if terminated or truncated:
                    break
            returns[k] = total_reward

        best_idx = int(np.argmax(returns))
        restore_state(env, state_snapshot)

        self._action_buffer = [
            np.clip(
                action_seqs_raw[best_idx, h],
                env.action_space.low,
                env.action_space.high,
            )
            for h in range(self.action_horizon)
        ]
        self._buffer_idx = 1

        info = {
            "returns": returns,
            "best_idx": best_idx,
            "best_return": returns[best_idx],
            "mean_return": returns.mean(),
            "std_return": returns.std(),
            "buffered": False,
        }
        return self._action_buffer[0], info
