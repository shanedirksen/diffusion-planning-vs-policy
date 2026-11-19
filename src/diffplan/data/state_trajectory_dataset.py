"""State-only trajectory dataset for Decision Diffuser (Ajay et al., ICLR 2023).

Each item is a state-only sliding window of shape (H, obs_dim) plus the
normalized return for that window.  Also provides (s_t, a_t, s_{t+1}) transition
tuples for training the inverse dynamics model.

Uses the same Normalizer as the action-only and trajectory datasets so all
model types share consistent normalization.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffplan.data.d4rl_dataset import segment_into_episodes
from diffplan.data.normalizer import Normalizer


class StateTrajectoryDataset(Dataset):
    """Sliding-window dataset: state-only sequences + returns + transitions.

    Each ``__getitem__`` returns ``(state_seq, normalized_return)`` where:
      - ``state_seq`` has shape ``(H, obs_dim)``
      - ``normalized_return`` is a scalar (float)

    Additionally stores transition tuples for inverse dynamics training,
    accessible via ``sample_transitions(batch_size)``.
    """

    def __init__(
        self,
        env_name: str,
        horizon: int = 32,
        normalizer: Optional[Normalizer] = None,
        max_episodes: Optional[int] = None,
        discount: float = 0.99,
    ):
        episodes = segment_into_episodes(env_name)
        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        # Compute or reuse normalizer
        all_obs = np.concatenate([e["observations"] for e in episodes], axis=0)
        all_act = np.concatenate([e["actions"] for e in episodes], axis=0)
        if normalizer is None:
            normalizer = Normalizer.from_dataset(all_obs, all_act)
        self.normalizer = normalizer
        self.horizon = horizon
        self.obs_dim = all_obs.shape[-1]
        self.act_dim = all_act.shape[-1]

        # Build sliding windows of state-only sequences + returns
        state_windows: list[np.ndarray] = []
        return_windows: list[float] = []

        # Also collect transitions for inverse dynamics
        all_s_t: list[np.ndarray] = []
        all_a_t: list[np.ndarray] = []
        all_s_tp1: list[np.ndarray] = []

        for ep in episodes:
            T = len(ep["observations"])
            obs_norm = self.normalizer.normalize_obs(ep["observations"])
            act_norm = self.normalizer.normalize_act(ep["actions"])
            rewards = ep["rewards"]

            # Precompute discounted returns from each timestep
            disc_returns = np.zeros(T, dtype=np.float32)
            running = 0.0
            for t in reversed(range(T)):
                running = rewards[t] + discount * running
                disc_returns[t] = running

            # State-only windows
            for t in range(T - horizon + 1):
                state_seq = obs_norm[t: t + horizon]  # (H, obs_dim)
                state_windows.append(state_seq)
                return_windows.append(disc_returns[t])

            # Transition tuples: (s_t, a_t, s_{t+1})
            for t in range(T - 1):
                all_s_t.append(obs_norm[t])
                all_a_t.append(act_norm[t])
                all_s_tp1.append(obs_norm[t + 1])

        states = np.stack(state_windows)                        # (N, H, obs_dim)
        returns = np.array(return_windows, dtype=np.float32)    # (N,)

        # Normalize returns to [0, 1] for classifier-free guidance
        self._return_min = returns.min()
        self._return_max = max(returns.max(), self._return_min + 1e-6)
        returns_norm = (returns - self._return_min) / (self._return_max - self._return_min)

        # Store state windows + returns
        self.states = torch.as_tensor(
            np.ascontiguousarray(states), dtype=torch.float32
        )
        self.returns = torch.as_tensor(
            np.ascontiguousarray(returns_norm), dtype=torch.float32
        )

        # Store transitions for inverse dynamics
        self.trans_s_t = torch.as_tensor(
            np.ascontiguousarray(np.stack(all_s_t)), dtype=torch.float32
        )
        self.trans_a_t = torch.as_tensor(
            np.ascontiguousarray(np.stack(all_a_t)), dtype=torch.float32
        )
        self.trans_s_tp1 = torch.as_tensor(
            np.ascontiguousarray(np.stack(all_s_tp1)), dtype=torch.float32
        )

        print(f"[StateTrajectoryDataset] {len(self)} state windows  |  H={horizon}  |  "
              f"states {self.states.shape}  |  "
              f"return range [{returns.min():.1f}, {returns.max():.1f}] → [0, 1]  |  "
              f"{len(self.trans_s_t)} transitions for inverse dynamics")

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int):
        return self.states[idx], self.returns[idx]

    @property
    def return_min(self) -> float:
        return float(self._return_min)

    @property
    def return_max(self) -> float:
        return float(self._return_max)

    def to(self, device: torch.device):
        """Move entire dataset to device. Returns self."""
        self.states = self.states.to(device)
        self.returns = self.returns.to(device)
        self.trans_s_t = self.trans_s_t.to(device)
        self.trans_a_t = self.trans_a_t.to(device)
        self.trans_s_tp1 = self.trans_s_tp1.to(device)
        return self

    def sample_batch(self, batch_size: int, device: torch.device | None = None):
        """Random batch of (state_seq, returns) for diffusion training."""
        storage = self.states.device
        idx = torch.randint(0, len(self), (batch_size,), device=storage)
        s = self.states[idx]
        r = self.returns[idx]
        if device is not None and device != storage:
            s = s.to(device, non_blocking=True)
            r = r.to(device, non_blocking=True)
        return s, r

    def sample_transitions(self, batch_size: int, device: torch.device | None = None):
        """Random batch of (s_t, a_t, s_{t+1}) for inverse dynamics training."""
        storage = self.trans_s_t.device
        idx = torch.randint(0, len(self.trans_s_t), (batch_size,), device=storage)
        s = self.trans_s_t[idx]
        a = self.trans_a_t[idx]
        sp = self.trans_s_tp1[idx]
        if device is not None and device != storage:
            s = s.to(device, non_blocking=True)
            a = a.to(device, non_blocking=True)
            sp = sp.to(device, non_blocking=True)
        return s, a, sp
