"""Observation and action normalization for D4RL datasets."""

from __future__ import annotations

import numpy as np
import torch


class Normalizer:
    """Running-statistics normalizer (obs → zero-mean unit-var, actions → [-1, 1])."""

    def __init__(
        self,
        obs_mean: np.ndarray,
        obs_std: np.ndarray,
        act_min: np.ndarray,
        act_max: np.ndarray,
        eps: float = 1e-6,
    ):
        self.obs_mean = obs_mean.astype(np.float32)
        self.obs_std = np.clip(obs_std.astype(np.float32), eps, None)
        self.act_min = act_min.astype(np.float32)
        self.act_max = act_max.astype(np.float32)
        self.act_range = np.clip(self.act_max - self.act_min, eps, None)
        self.eps = eps

    # ── observation ──────────────────────────────────────────────────────────

    def normalize_obs(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(obs, torch.Tensor):
            m = torch.as_tensor(self.obs_mean, device=obs.device)
            s = torch.as_tensor(self.obs_std, device=obs.device)
            return (obs - m) / s
        return (obs - self.obs_mean) / self.obs_std

    def unnormalize_obs(self, obs: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        if isinstance(obs, torch.Tensor):
            m = torch.as_tensor(self.obs_mean, device=obs.device)
            s = torch.as_tensor(self.obs_std, device=obs.device)
            return obs * s + m
        return obs * self.obs_std + self.obs_mean

    # ── action ───────────────────────────────────────────────────────────────

    def normalize_act(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map actions to [-1, 1]."""
        if isinstance(act, torch.Tensor):
            lo = torch.as_tensor(self.act_min, device=act.device)
            r = torch.as_tensor(self.act_range, device=act.device)
            return 2.0 * (act - lo) / r - 1.0
        return 2.0 * (act - self.act_min) / self.act_range - 1.0

    def unnormalize_act(self, act: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map actions from [-1, 1] back to original range."""
        if isinstance(act, torch.Tensor):
            lo = torch.as_tensor(self.act_min, device=act.device)
            r = torch.as_tensor(self.act_range, device=act.device)
            return (act + 1.0) / 2.0 * r + lo
        return (act + 1.0) / 2.0 * self.act_range + self.act_min

    # ── serialization ────────────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_min": self.act_min,
            "act_max": self.act_max,
        }

    @classmethod
    def from_state_dict(cls, d: dict) -> "Normalizer":
        return cls(d["obs_mean"], d["obs_std"], d["act_min"], d["act_max"])

    @classmethod
    def from_dataset(cls, observations: np.ndarray, actions: np.ndarray) -> "Normalizer":
        """Compute statistics from a dataset of observations and actions."""
        return cls(
            obs_mean=observations.mean(axis=0),
            obs_std=observations.std(axis=0),
            act_min=actions.min(axis=0),
            act_max=actions.max(axis=0),
        )
