"""State encoder, FiLM conditioning, and sinusoidal time embedding."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


# ── Sinusoidal positional / diffusion-time embedding ─────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Maps scalar diffusion timestep → vector embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) int or float tensor of diffusion timesteps.
        Returns:
            (B, dim) embedding.
        """
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / half
        )
        # t can be (B,) or (B,1)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


# ── FiLM (Feature-wise Linear Modulation) ───────────────────────────────────

class FiLM(nn.Module):
    """Applies feature-wise affine transform: ``gamma * x + beta``."""

    def __init__(self, cond_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, hidden_dim) or (B, T, hidden_dim)
            cond: (B, cond_dim)
        """
        gamma_beta = self.net(cond)                        # (B, 2*hidden_dim)
        gamma, beta = gamma_beta.chunk(2, dim=-1)          # each (B, hidden_dim)
        if x.dim() == 3:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x + beta


# ── State encoder ────────────────────────────────────────────────────────────

class StateEncoder(nn.Module):
    """Simple MLP that encodes observation → conditioning vector."""

    def __init__(self, obs_dim: int, cond_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Linear(256, cond_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ── Combined conditioning ────────────────────────────────────────────────────

class ConditioningModule(nn.Module):
    """Combines diffusion-time embedding and state embedding into a single
    conditioning vector.

    ``c = MLP(concat(time_emb, state_emb))``
    """

    def __init__(self, obs_dim: int, time_dim: int = 64, cond_dim: int = 128):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.state_enc = StateEncoder(obs_dim, cond_dim)
        self.merge = nn.Sequential(
            nn.Linear(time_dim + cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

    def forward(self, obs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim)
            t:   (B,) diffusion timestep
        Returns:
            (B, cond_dim)
        """
        te = self.time_emb(t)          # (B, time_dim)
        se = self.state_enc(obs)       # (B, cond_dim)
        return self.merge(torch.cat([te, se], dim=-1))
