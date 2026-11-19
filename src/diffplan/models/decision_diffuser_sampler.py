"""Decision Diffuser sampler (Ajay et al., ICLR 2023).

Generates state-only sequences via classifier-free guidance on returns,
inpaints history of observed states, then extracts actions via inverse
dynamics.

Key features:
  - Classifier-free guidance: ε̂ = ε_θ(x, Ø, k) + ω (ε_θ(x, R, k) − ε_θ(x, Ø, k))
  - Low-temperature sampling: variance scaled by α ∈ [0, 1)
  - Inpainting: overwrites first C states with observed state history
  - Inverse dynamics: f_φ(s_t, s_{t+1}) → a_t for action extraction

The ``ddim_sample`` and ``sample`` methods return action sequences
(B*n_samples, H, act_dim) to match the DiffusionSampler interface, so
PolicyMode / PlannerMode work unchanged.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from diffplan.models.sampler import cosine_beta_schedule, linear_beta_schedule


class DecisionDiffuserSampler(nn.Module):
    """Decision Diffuser: state-only diffusion + CFG + inverse dynamics.

    Drop-in replacement for DiffusionSampler - returns actions.
    """

    def __init__(
        self,
        model: nn.Module,
        inverse_dynamics: nn.Module,
        obs_dim: int,
        act_dim: int,
        n_timesteps: int = 100,
        beta_schedule: str = "cosine",
        clip_denoised: bool = True,
        guidance_scale: float = 1.2,
        temperature: float = 0.5,
        context_len: int = 1,
    ):
        super().__init__()
        self.model = model
        self.inverse_dynamics = inverse_dynamics
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.guidance_scale = guidance_scale
        self.temperature = temperature       # α in the paper - scale variance
        self.context_len = context_len       # C - how many past states to inpaint

        # Compute beta schedule
        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod",
                             torch.sqrt(1.0 - alphas_cumprod))
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

    def q_sample(self, x_0, t, noise=None):
        """Sample x_t from q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_a = self.sqrt_alphas_cumprod[t]
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t]
        while sqrt_a.dim() < x_0.dim():
            sqrt_a = sqrt_a.unsqueeze(-1)
            sqrt_1ma = sqrt_1ma.unsqueeze(-1)
        return sqrt_a * x_0 + sqrt_1ma * noise

    def training_loss(
        self,
        x_0: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """DDPM loss on state-only sequences with return conditioning.

        Following Ajay et al. (2023): the known observation at t=0 is
        inpainted into the noisy input during training, and the noise
        target at t=0 is zeroed, so the model learns to use the state
        anchor.

        Args:
            x_0:     (B, H, obs_dim) - clean state sequences
            returns: (B,)            - normalized returns [0, 1]
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Inpaint: overwrite state at t=0 with clean values
        x_t[:, 0, :] = x_0[:, 0, :].clone()

        noise_pred = self.model(x_t, t, returns)

        # Zero noise target at conditioned position
        noise[:, 0, :] = 0.0

        return nn.functional.mse_loss(noise_pred, noise)

    def inverse_dynamics_loss(
        self,
        s_t: torch.Tensor,
        a_t: torch.Tensor,
        s_tp1: torch.Tensor,
    ) -> torch.Tensor:
        """MSE loss for the inverse dynamics model.

        Args:
            s_t:   (B, obs_dim)
            a_t:   (B, act_dim) - ground truth action
            s_tp1: (B, obs_dim)
        """
        a_pred = self.inverse_dynamics(s_t, s_tp1)
        return nn.functional.mse_loss(a_pred, a_t)

    # ── Inpainting ───────────────────────────────────────────────────────────

    def _apply_inpainting(
        self,
        x: torch.Tensor,
        obs_history: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """Overwrite the first C states with clean observed history.

        Matching Ajay et al.: conditioning is done with clean values
        at every diffusion step, same as during training.

        Args:
            x:           (B, H, obs_dim) - current noisy state sequence
            obs_history: (B, C, obs_dim) - last C observed states
            t_idx:       current diffusion timestep (unused, kept for API compat)
        """
        C = obs_history.shape[1]
        x = x.clone()
        x[:, :C, :] = obs_history
        return x

    def _apply_inpainting_clean(
        self,
        x: torch.Tensor,
        obs_history: torch.Tensor,
    ) -> torch.Tensor:
        """Final inpainting with clean observed history."""
        C = obs_history.shape[1]
        x = x.clone()
        x[:, :C, :] = obs_history
        return x

    # ── Classifier-free guidance ─────────────────────────────────────────────

    def _guided_noise_pred(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: torch.Tensor,
        omega: float,
    ) -> torch.Tensor:
        """Classifier-free guided noise prediction.

        ε̂ = ε_θ(x, Ø, k) + ω (ε_θ(x, R, k) − ε_θ(x, Ø, k))
        """
        # Conditional prediction
        eps_cond = self.model(x, t, returns)

        # Unconditional prediction (returns=None zeroes out the return embedding)
        eps_uncond = self.model(x, t, None)

        # Guided combination
        return eps_uncond + omega * (eps_cond - eps_uncond)

    # ── Reverse sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        horizon: int,
        act_dim: int,
        n_samples: int = 1,
        target_return: float = 1.0,
    ) -> torch.Tensor:
        """DDPM reverse sampling with CFG + inpainting + inverse dynamics.

        Returns action sequences (B*n_samples, H, act_dim).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)  # (total, obs_dim)
        total = B * n_samples

        # Context: just the current obs as a single-step history
        obs_history = obs_expanded.unsqueeze(1)  # (total, 1, obs_dim)

        # Target return for guidance
        returns = torch.full((total,), target_return, device=obs.device)

        x = torch.randn(total, horizon, self.obs_dim, device=obs.device)

        for t_idx in reversed(range(self.n_timesteps)):
            # Inpainting
            x = self._apply_inpainting(x, obs_history, t_idx)

            # Guided noise prediction
            t = torch.full((total,), t_idx, device=obs.device, dtype=torch.long)
            noise_pred = self._guided_noise_pred(x, t, returns, self.guidance_scale)

            # Predict x_0
            x_0_pred = (
                x - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
            ) / self.sqrt_alphas_cumprod[t_idx]
            # NOTE: Do NOT clip state predictions to [-1, 1]. States are
            # normalized to zero-mean unit-variance (range ~[-6, 9]) so
            # clamping destroys ~34% of values and ruins trajectory quality.

            # Posterior mean
            mean = (
                self.posterior_mean_coef1[t_idx] * x_0_pred
                + self.posterior_mean_coef2[t_idx] * x
            )

            # Low-temperature sampling
            if t_idx > 0:
                noise = torch.randn_like(x)
                variance = self.posterior_variance[t_idx]
                x = mean + torch.sqrt(self.temperature * variance) * noise
            else:
                x = mean

        # Final clean inpainting
        x = self._apply_inpainting_clean(x, obs_history)

        # Extract actions via inverse dynamics
        actions = self.inverse_dynamics.predict_actions(x)  # (total, H-1, act_dim)

        # Pad to full horizon H (last action = repeat second-to-last)
        last_action = actions[:, -1:, :]
        actions = torch.cat([actions, last_action], dim=1)  # (total, H, act_dim)

        return actions

    @torch.no_grad()
    def ddim_sample(
        self,
        obs: torch.Tensor,
        horizon: int,
        act_dim: int,
        n_samples: int = 1,
        ddim_steps: int = 20,
        eta: float = 0.0,
        target_return: float = 1.0,
    ) -> torch.Tensor:
        """DDIM sampling with CFG + inpainting + inverse dynamics.

        Returns action sequences (B*n_samples, H, act_dim).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)
        total = B * n_samples

        obs_history = obs_expanded.unsqueeze(1)  # (total, 1, obs_dim)
        returns = torch.full((total,), target_return, device=obs.device)

        step_size = self.n_timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.n_timesteps, step_size)))

        x = torch.randn(total, horizon, self.obs_dim, device=obs.device)

        for i, t_idx in enumerate(timesteps):
            # Inpainting
            x = self._apply_inpainting(x, obs_history, t_idx)

            # Guided noise prediction
            t = torch.full((total,), t_idx, device=obs.device, dtype=torch.long)
            noise_pred = self._guided_noise_pred(x, t, returns, self.guidance_scale)

            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0)
            )

            x_0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            # NOTE: Do NOT clip state predictions - see comment in sample()

            # DDIM update with low-temperature scaling of the stochastic part
            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            )
            sigma = sigma * self.temperature  # low-temperature scaling
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise

        # Final clean inpainting
        x = self._apply_inpainting_clean(x, obs_history)

        # Extract actions via inverse dynamics
        actions = self.inverse_dynamics.predict_actions(x)  # (total, H-1, act_dim)

        # Pad to full horizon
        last_action = actions[:, -1:, :]
        actions = torch.cat([actions, last_action], dim=1)  # (total, H, act_dim)

        return actions
