"""DDPM / DDIM sampler for action-sequence diffusion.

Handles:
    - beta schedule (linear / cosine)
    - forward diffusion  q(x_t | x_0)
    - training loss (simplified MSE on predicted noise)
    - reverse sampling (DDPM ancestral, DDIM deterministic)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np


# ── Beta schedules ───────────────────────────────────────────────────────────

def linear_beta_schedule(n_timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02):
    return torch.linspace(beta_start, beta_end, n_timesteps)


def cosine_beta_schedule(n_timesteps: int, s: float = 0.008):
    steps = torch.arange(n_timesteps + 1, dtype=torch.float64)
    f = torch.cos((steps / n_timesteps + s) / (1 + s) * (math.pi / 2)) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return torch.clamp(betas, 0.0, 0.999).float()


# ── Diffusion Sampler ────────────────────────────────────────────────────────

class DiffusionSampler(nn.Module):
    """Wraps a noise-prediction model with DDPM forward process + reverse sampling."""

    def __init__(
        self,
        model: nn.Module,
        n_timesteps: int = 100,
        beta_schedule: str = "cosine",
        clip_denoised: bool = True,
    ):
        super().__init__()
        self.model = model
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised

        # Compute schedule quantities
        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        # Register as buffers (move to device with model)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )

    # ── Forward diffusion (for training) ─────────────────────────────────────

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.sqrt_one_minus_alphas_cumprod[t]
        # Broadcast to match x_0 shape
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    def training_loss(
        self,
        x_0: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """Simplified DDPM loss: MSE between true noise and predicted noise.

        Args:
            x_0: (B, H, act_dim) - clean action sequences
            obs: (B, obs_dim)    - conditioning observations
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        noise_pred = self.model(x_t, t, obs)
        return nn.functional.mse_loss(noise_pred, noise)

    # ── Reverse sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def p_sample_step(
        self,
        x_t: torch.Tensor,
        t_idx: int,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """One DDPM reverse step: x_{t} → x_{t-1}."""
        B = x_t.shape[0]
        t = torch.full((B,), t_idx, device=x_t.device, dtype=torch.long)
        noise_pred = self.model(x_t, t, obs)

        # Predict x_0
        x_0_pred = (
            x_t - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
        ) / self.sqrt_alphas_cumprod[t_idx]

        if self.clip_denoised:
            x_0_pred = x_0_pred.clamp(-1.0, 1.0)

        # Posterior mean
        mean = (
            self.posterior_mean_coef1[t_idx] * x_0_pred
            + self.posterior_mean_coef2[t_idx] * x_t
        )

        if t_idx > 0:
            noise = torch.randn_like(x_t)
            variance = self.posterior_variance[t_idx]
            return mean + torch.sqrt(variance) * noise
        return mean

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        horizon: int,
        act_dim: int,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """Full reverse process: sample action sequences from noise.

        Args:
            obs:       (B, obs_dim) or (obs_dim,) - conditioning observations.
            horizon:   int - planning horizon H.
            act_dim:   int - action dimensionality.
            n_samples: int - how many sequences per observation.

        Returns:
            (B * n_samples, H, act_dim)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]

        # Repeat obs for n_samples
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)  # (B*n_samples, obs_dim)
        total = B * n_samples

        # Start from pure noise
        x = torch.randn(total, horizon, act_dim, device=obs.device)

        for t_idx in reversed(range(self.n_timesteps)):
            x = self.p_sample_step(x, t_idx, obs_expanded)

        return x  # (B*n_samples, H, act_dim)

    @torch.no_grad()
    def ddim_sample(
        self,
        obs: torch.Tensor,
        horizon: int,
        act_dim: int,
        n_samples: int = 1,
        ddim_steps: int = 20,
        eta: float = 0.0,
    ) -> torch.Tensor:
        """DDIM sampling (faster, deterministic when eta=0).

        Args:
            ddim_steps: number of denoising steps (< n_timesteps).
            eta:        0 → deterministic, 1 → DDPM-like stochastic.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)
        total = B * n_samples

        # Subsequence of timesteps
        step_size = self.n_timesteps // ddim_steps
        timesteps = list(range(0, self.n_timesteps, step_size))
        timesteps = list(reversed(timesteps))

        x = torch.randn(total, horizon, act_dim, device=obs.device)

        for i, t_idx in enumerate(timesteps):
            t = torch.full((total,), t_idx, device=obs.device, dtype=torch.long)
            noise_pred = self.model(x, t, obs_expanded)

            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = self.alphas_cumprod[timesteps[i + 1]] if i + 1 < len(timesteps) else torch.tensor(1.0)

            x_0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            if self.clip_denoised:
                x_0_pred = x_0_pred.clamp(-1.0, 1.0)

            sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev))
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise

        return x
