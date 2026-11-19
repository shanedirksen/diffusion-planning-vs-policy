"""Value function for trajectory-guided diffusion (Diffuser paper).

Predicts cumulative return from a (possibly noisy) trajectory segment,
conditioned on the diffusion timestep.  Used to provide classifier-free
guidance gradients during the reverse diffusion process.

V(tau, t) ≈ E[sum of discounted rewards | trajectory tau at noise level t]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10_000) * torch.arange(half, device=device) / half
        )
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ValueFunction(nn.Module):
    """MLP-based value function over trajectory segments.

    Takes a (possibly noisy) trajectory and diffusion timestep,
    outputs a scalar value estimate.
    """

    def __init__(
        self,
        transition_dim: int,
        horizon: int,
        hidden_dim: int = 256,
        time_dim: int = 64,
        n_layers: int = 4,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.horizon = horizon

        # Time embedding
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Trajectory encoder: flatten trajectory + time embedding -> hidden
        input_dim = transition_dim * horizon + time_dim

        layers = [nn.Linear(input_dim, hidden_dim), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])
        layers.append(nn.Linear(hidden_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        trajectory: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            trajectory: (B, H, transition_dim) - possibly noisy trajectory
            t:          (B,) - diffusion timestep

        Returns:
            (B, 1) - predicted value
        """
        B = trajectory.shape[0]
        t_emb = self.time_emb(t)                              # (B, time_dim)
        traj_flat = trajectory.reshape(B, -1)                  # (B, H * transition_dim)
        x = torch.cat([traj_flat, t_emb], dim=-1)             # (B, H*D + time_dim)
        return self.net(x)                                     # (B, 1)
