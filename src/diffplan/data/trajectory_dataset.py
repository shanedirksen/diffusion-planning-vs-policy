"""Trajectory-level dataset for the Diffuser-paper-style model.

Each item is a full [obs | act] trajectory window of shape (H, obs_dim+act_dim)
plus discounted cumulative returns (used for value function training).

Observations and actions are normalized via the same Normalizer used by the
action-only dataset, so the two models are interchangeable at eval time.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from diffplan.data.d4rl_dataset import segment_into_episodes
from diffplan.data.normalizer import Normalizer


class TrajectoryDataset(Dataset):
    """Sliding-window dataset: full ``[obs | act]`` trajectories + returns.

    Each item is a tuple ``(trajectory, discounted_return)`` where
    - ``trajectory`` has shape ``(H, obs_dim + act_dim)``
    - ``discounted_return`` is a scalar (float) representing the
      discounted cumulative reward from the start of this window.
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
        self.transition_dim = self.obs_dim + self.act_dim

        # Build sliding windows of [obs | act] trajectories + returns
        traj_windows: list[np.ndarray] = []
        return_windows: list[float] = []

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

            for t in range(T - horizon + 1):
                # Concatenate normalized obs and act along feature dimension
                traj = np.concatenate(
                    [obs_norm[t: t + horizon], act_norm[t: t + horizon]], axis=-1
                )  # (H, obs_dim + act_dim)
                traj_windows.append(traj)
                return_windows.append(disc_returns[t])

        trajectories = np.stack(traj_windows)                       # (N, H, transition_dim)
        returns = np.array(return_windows, dtype=np.float32)        # (N,)

        # Normalize returns to roughly [-1, 1] for stable value function training
        self._return_mean = returns.mean()
        self._return_std = max(returns.std(), 1e-6)
        returns_norm = (returns - self._return_mean) / self._return_std

        # Store as contiguous tensors
        self.trajectories = torch.as_tensor(
            np.ascontiguousarray(trajectories), dtype=torch.float32
        )
        self.returns = torch.as_tensor(
            np.ascontiguousarray(returns_norm), dtype=torch.float32
        )

        print(f"[TrajectoryDataset] {len(self)} windows  |  H={horizon}  |  "
              f"traj {self.trajectories.shape}  |  "
              f"return range [{returns.min():.1f}, {returns.max():.1f}]  "
              f"(norm μ={self._return_mean:.1f} σ={self._return_std:.1f})")

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int):
        return self.trajectories[idx], self.returns[idx]

    def to(self, device: torch.device):
        """Move entire dataset to device. Returns self."""
        self.trajectories = self.trajectories.to(device)
        self.returns = self.returns.to(device)
        self._device = device
        return self

    def sample_batch(self, batch_size: int, device: torch.device | None = None):
        """Fast random batch - zero-copy if data is already on device."""
        storage_device = self.trajectories.device
        idx = torch.randint(0, len(self), (batch_size,), device=storage_device)
        traj = self.trajectories[idx]
        ret = self.returns[idx]
        if device is not None and device != storage_device:
            traj = traj.to(device, non_blocking=True)
            ret = ret.to(device, non_blocking=True)
        return traj, ret

    def return_stats(self) -> dict:
        """Return mean/std of raw returns (useful for unnormalization later)."""
        return {"mean": self._return_mean, "std": self._return_std}


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = TrajectoryDataset("halfcheetah-medium-v2", horizon=16, max_episodes=10)
    traj, ret = ds[0]
    print(f"Sample 0 → trajectory {traj.shape}, return {ret.item():.3f}")
    print(f"  traj range: [{traj.min():.2f}, {traj.max():.2f}]")
