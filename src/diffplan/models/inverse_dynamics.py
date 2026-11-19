"""Inverse dynamics model for Decision Diffuser (Ajay et al., ICLR 2023).

Given two consecutive states (s_t, s_{t+1}), predicts the action a_t that
caused the transition.  Trained separately from the diffusion model on
the same offline dataset.

    a_t = f_phi(s_t, s_{t+1})

Architecture: 2-layer MLP with 512 hidden units and ReLU (matching the paper).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class InverseDynamics(nn.Module):
    """MLP inverse dynamics: (s_t, s_{t+1}) → a_t."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 512,
        n_layers: int = 2,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        input_dim = obs_dim * 2  # concatenation of s_t and s_{t+1}
        layers = []
        prev = input_dim
        for _ in range(n_layers):
            layers.extend([nn.Linear(prev, hidden_dim), nn.ReLU()])
            prev = hidden_dim
        layers.append(nn.Linear(prev, act_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        s_t: torch.Tensor,
        s_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            s_t:   (B, obs_dim) - current state
            s_tp1: (B, obs_dim) - next state

        Returns:
            (B, act_dim) - predicted action
        """
        x = torch.cat([s_t, s_tp1], dim=-1)  # (B, 2 * obs_dim)
        return self.net(x)

    def predict_actions(
        self,
        state_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """Extract actions from a full state sequence via inverse dynamics.

        Args:
            state_sequence: (B, H, obs_dim) - predicted state trajectory

        Returns:
            (B, H-1, act_dim) - predicted actions for each consecutive pair.
            Note: returns H-1 actions (one fewer than states).
            In practice we only need the first action a_0.
        """
        s_t = state_sequence[:, :-1]   # (B, H-1, obs_dim)
        s_tp1 = state_sequence[:, 1:]  # (B, H-1, obs_dim)

        B, L, D = s_t.shape
        s_t_flat = s_t.reshape(B * L, D)
        s_tp1_flat = s_tp1.reshape(B * L, D)

        a_flat = self.forward(s_t_flat, s_tp1_flat)  # (B*L, act_dim)
        return a_flat.reshape(B, L, -1)  # (B, H-1, act_dim)
