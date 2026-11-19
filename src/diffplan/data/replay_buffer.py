"""Optional replay buffer wrapping ActionSequenceDataset for uniform sampling."""

from __future__ import annotations

import numpy as np
import torch


class ReplayBuffer:
    """Simple buffer that stores precomputed (obs, action_seq) pairs
    and allows random batch sampling."""

    def __init__(self, dataset):
        """
        Args:
            dataset: an ``ActionSequenceDataset`` (or anything with
                     ``.observations`` and ``.action_seqs`` numpy arrays).
        """
        self.observations = dataset.observations   # (N, obs_dim)
        self.action_seqs = dataset.action_seqs     # (N, H, act_dim)
        self.size = len(self.observations)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample a random batch."""
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.observations[idxs], dtype=torch.float32)
        act = torch.as_tensor(self.action_seqs[idxs], dtype=torch.float32)
        return obs, act
