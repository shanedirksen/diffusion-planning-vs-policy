"""Policy mode: sample action sequence → execute first Ta actions (receding horizon).

Implements the receding horizon control scheme from Chi et al. (2023):
  - Maintain a sliding window of the last To observations
  - When action buffer is empty, run diffusion to generate Tp actions
  - Buffer the first Ta actions and execute them one-by-one
  - After Ta steps, replan with fresh diffusion inference

This amortizes the cost of diffusion inference over Ta environment steps and
provides temporal consistency in the executed actions.
"""

from __future__ import annotations

import time
from collections import deque

import numpy as np
import torch

from diffplan.data.normalizer import Normalizer
from diffplan.models.sampler import DiffusionSampler


class PolicyMode:
    """Stateful policy with receding-horizon action chunking.

    Parameters
    ----------
    sampler : DiffusionSampler
        Trained diffusion model wrapped in a sampling helper.
    normalizer : Normalizer
        Dataset statistics for obs/action normalization.
    horizon : int
        Prediction horizon Tp (number of actions the model generates).
    act_dim : int
        Dimensionality of the action space.
    obs_dim : int
        Per-timestep observation dimensionality.
    obs_horizon : int
        Number of past observations to condition on (To). Default 1.
    action_horizon : int
        Number of actions to execute before replanning (Ta). Default 1.
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
        """Build flattened, normalized observation window from buffer.

        Returns (To * obs_dim,) numpy array.  If fewer than To observations
        have been collected, the buffer is left-padded by repeating the
        earliest available observation.
        """
        # Pad if not enough observations yet (beginning of episode)
        while len(self._obs_buffer) < self.obs_horizon:
            self._obs_buffer.appendleft(self._obs_buffer[0])

        # Normalize each obs individually, then concatenate
        norm_obs = [self.normalizer.normalize_obs(o) for o in self._obs_buffer]
        return np.concatenate(norm_obs)  # (To * obs_dim,)

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Select an action given a raw (unnormalized) observation.

        Returns:
            action: (act_dim,) numpy array in original action space.
        """
        # Always add the latest observation to the sliding window
        self._obs_buffer.append(obs)

        # If we still have buffered actions, return the next one
        if self._buffer_idx < len(self._action_buffer):
            action = self._action_buffer[self._buffer_idx]
            self._buffer_idx += 1
            return action

        # --- Need fresh diffusion inference ---
        obs_window = self._build_obs_window()
        obs_t = torch.as_tensor(
            obs_window, dtype=torch.float32, device=self.device
        ).unsqueeze(0)  # (1, To * obs_dim)

        t0 = time.perf_counter()
        action_seq = self.sampler.ddim_sample(
            obs=obs_t,
            horizon=self.horizon,
            act_dim=self.act_dim,
            n_samples=1,
            ddim_steps=self.ddim_steps,
        )  # (1, Tp, act_dim)
        t1 = time.perf_counter()
        self.inference_times.append(t1 - t0)

        # Unnormalize the first Ta actions and buffer them
        chunk = action_seq[0, : self.action_horizon].cpu().numpy()  # (Ta, act_dim)
        chunk_raw = self.normalizer.unnormalize_act(chunk)           # (Ta, act_dim)
        self._action_buffer = list(chunk_raw)
        self._buffer_idx = 1  # We'll return index 0 now
        return self._action_buffer[0]

    @torch.no_grad()
    def act_sequence(self, obs: np.ndarray) -> np.ndarray:
        """Return the full planned action sequence (useful for visualization).

        Returns:
            actions: (H, act_dim) numpy array in original action space.
        """
        self._obs_buffer.append(obs)
        obs_window = self._build_obs_window()
        obs_t = torch.as_tensor(
            obs_window, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        action_seq = self.sampler.sample(
            obs=obs_t,
            horizon=self.horizon,
            act_dim=self.act_dim,
            n_samples=1,
        )  # (1, H, act_dim)

        seq_np = action_seq[0].cpu().numpy()  # (H, act_dim)
        return self.normalizer.unnormalize_act(seq_np)
