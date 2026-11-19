"""Offline dataset → sliding windows of (s_t, a_{t:t+H-1}).

Uses **Minari** to load datasets (modern replacement for D4RL, works on Python 3.12+).
Provides a PyTorch ``Dataset`` that can be fed directly to a ``DataLoader``.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

import minari

from diffplan.data.normalizer import Normalizer
from diffplan.envs.make_env import get_minari_id


def load_minari_dataset(env_name: str) -> dict:
    """Download (if needed) and load a Minari dataset.

    Returns a dict with keys: observations, actions, rewards, terminations, truncations.
    Each value is a flat numpy array spanning all episodes.
    """
    minari_id = get_minari_id(env_name)

    # Download if not cached locally
    try:
        dataset = minari.load_dataset(minari_id)
    except Exception:
        minari.download_dataset(minari_id)
        dataset = minari.load_dataset(minari_id)

    all_obs, all_act, all_rew, all_term, all_trunc = [], [], [], [], []

    for ep in dataset.iterate_episodes():
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        rew = np.asarray(ep.rewards, dtype=np.float32)
        term = np.asarray(ep.terminations, dtype=bool)
        trunc = np.asarray(ep.truncations, dtype=bool)

        # obs has T+1 entries (includes terminal obs), act/rew/term have T
        T = len(act)
        all_obs.append(obs[:T])      # align with actions
        all_act.append(act)
        all_rew.append(rew)
        all_term.append(term)
        all_trunc.append(trunc)

    return {
        "observations": np.concatenate(all_obs, axis=0),
        "actions": np.concatenate(all_act, axis=0),
        "rewards": np.concatenate(all_rew, axis=0),
        "terminations": np.concatenate(all_term, axis=0),
        "truncations": np.concatenate(all_trunc, axis=0),
    }


def segment_into_episodes(env_name: str) -> list[dict]:
    """Load a Minari dataset and return per-episode dicts."""
    minari_id = get_minari_id(env_name)

    try:
        dataset = minari.load_dataset(minari_id)
    except Exception:
        minari.download_dataset(minari_id)
        dataset = minari.load_dataset(minari_id)

    episodes = []
    for ep in dataset.iterate_episodes():
        obs = np.asarray(ep.observations, dtype=np.float32)
        act = np.asarray(ep.actions, dtype=np.float32)
        rew = np.asarray(ep.rewards, dtype=np.float32)
        term = np.asarray(ep.terminations, dtype=bool)
        trunc = np.asarray(ep.truncations, dtype=bool)
        T = len(act)
        episodes.append({
            "observations": obs[:T],
            "actions": act,
            "rewards": rew,
            "terminations": term,
            "truncations": trunc,
        })
    return episodes


class ActionSequenceDataset(Dataset):
    """Sliding-window dataset: ``(obs_window, a_{t:t+H-1})``.

    Each item is a tuple ``(obs, action_seq)`` where
    - ``obs`` has shape ``(To * obs_dim,)`` - flattened observation window
    - ``action_seq`` has shape ``(H, act_dim)``

    When ``obs_horizon > 1``, the dataset stacks the last ``To`` observations
    at each timestep and flattens them.  Each observation is normalized
    individually *before* concatenation so the normalizer statistics (which
    have shape ``(obs_dim,)``) are applied correctly.

    Observations and actions are normalized.
    Data is stored as contiguous float32 tensors for fast GPU transfer.
    """

    def __init__(
        self,
        env_name: str,
        horizon: int = 32,
        obs_horizon: int = 1,
        normalizer: Optional[Normalizer] = None,
        max_episodes: Optional[int] = None,
    ):
        episodes = segment_into_episodes(env_name)
        if max_episodes is not None:
            episodes = episodes[:max_episodes]

        # Compute or reuse normalizer (stats are per single obs)
        all_obs = np.concatenate([e["observations"] for e in episodes], axis=0)
        all_act = np.concatenate([e["actions"] for e in episodes], axis=0)
        if normalizer is None:
            normalizer = Normalizer.from_dataset(all_obs, all_act)
        self.normalizer = normalizer
        self.horizon = horizon
        self.obs_horizon = obs_horizon

        # Build sliding windows
        obs_windows = []
        act_windows = []
        for ep in episodes:
            T = len(ep["observations"])
            for t in range(T - horizon + 1):
                # Build observation window: stack last To observations
                # Pad with first available obs if t < obs_horizon - 1
                obs_stack = []
                for i in range(obs_horizon):
                    idx = max(0, t - obs_horizon + 1 + i)
                    obs_stack.append(ep["observations"][idx])
                obs_windows.append(np.stack(obs_stack))  # (To, obs_dim)
                act_windows.append(ep["actions"][t : t + horizon])

        observations = np.stack(obs_windows)         # (N, To, obs_dim)
        action_seqs = np.stack(act_windows)           # (N, H, act_dim)

        # Normalize each observation individually, then flatten
        observations = self.normalizer.normalize_obs(observations)  # broadcasts on last dim
        observations = observations.reshape(len(observations), -1)  # (N, To * obs_dim)
        action_seqs = self.normalizer.normalize_act(action_seqs)

        # Store as contiguous tensors (much faster than numpy→tensor per item)
        self.observations = torch.as_tensor(
            np.ascontiguousarray(observations), dtype=torch.float32
        )
        self.action_seqs = torch.as_tensor(
            np.ascontiguousarray(action_seqs), dtype=torch.float32
        )

        print(f"[Dataset] {len(self)} windows  |  H={horizon}  To={obs_horizon}  |  "
              f"obs {self.observations.shape}  |  act_seq {self.action_seqs.shape}")

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, idx: int):
        return self.observations[idx], self.action_seqs[idx]

    def to(self, device: torch.device):
        """Move entire dataset to device (e.g. GPU). Returns self."""
        self.observations = self.observations.to(device)
        self.action_seqs = self.action_seqs.to(device)
        self._device = device
        return self

    def sample_batch(self, batch_size: int, device: torch.device | None = None):
        """Fast random batch - zero-copy if data is already on `device`."""
        storage_device = self.observations.device
        idx = torch.randint(0, len(self), (batch_size,), device=storage_device)
        obs = self.observations[idx]
        act = self.action_seqs[idx]
        if device is not None and device != storage_device:
            obs = obs.to(device, non_blocking=True)
            act = act.to(device, non_blocking=True)
        return obs, act


# ── Quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    ds = ActionSequenceDataset("halfcheetah-medium-v2", horizon=16, max_episodes=10)
    obs, act_seq = ds[0]
    print(f"Sample 0 → obs {obs.shape}, act_seq {act_seq.shape}")
    print(f"  obs range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"  act range: [{act_seq.min():.2f}, {act_seq.max():.2f}]")
