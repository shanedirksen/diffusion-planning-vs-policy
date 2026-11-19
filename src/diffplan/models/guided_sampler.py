"""Guided diffusion sampler for trajectory-level models (Diffuser paper style).

Key differences from the action-only DiffusionSampler:
  - Generates full trajectories (obs + act) instead of just actions
  - Uses INPAINTING to fix known states (e.g. current obs at t=0)
  - Uses VALUE FUNCTION GUIDANCE to steer toward high-reward trajectories
  - Returns only the action portion to callers (same interface as DiffusionSampler)

The sampling interface (ddim_sample, sample) is compatible with PolicyMode /
PlannerMode so it works as a drop-in replacement.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import numpy as np

from diffplan.models.sampler import cosine_beta_schedule, linear_beta_schedule


class GuidedDiffusionSampler(nn.Module):
    """Trajectory diffusion sampler with inpainting + value guidance.

    Drop-in replacement for DiffusionSampler - the ``ddim_sample`` and
    ``sample`` methods return action sequences (B*n_samples, H, act_dim)
    so PolicyMode / PlannerMode work unchanged.
    """

    def __init__(
        self,
        model: nn.Module,
        obs_dim: int,
        act_dim: int,
        n_timesteps: int = 100,
        beta_schedule: str = "cosine",
        clip_denoised: bool = True,
        value_function: Optional[nn.Module] = None,
        guidance_scale: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.transition_dim = obs_dim + act_dim
        self.n_timesteps = n_timesteps
        self.clip_denoised = clip_denoised
        self.value_function = value_function
        self.guidance_scale = guidance_scale

        # Compute schedule (same as DiffusionSampler)
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

    # ── Forward diffusion ────────────────────────────────────────────────────

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

    def training_loss(self, x_0: torch.Tensor) -> torch.Tensor:
        """DDPM loss on full trajectories with inpainting conditioning.

        Following Janner et al. (2022): at each training step, the known
        observation at t=0 is inpainted into the noisy input *and* the
        noise target, so the model learns to use the observation anchor.

        Args:
            x_0: (B, H, transition_dim) - clean trajectories [obs | act]
        """
        B = x_0.shape[0]
        t = torch.randint(0, self.n_timesteps, (B,), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)

        # Inpaint: overwrite obs channels at t=0 with clean values
        # (same as apply_conditioning in Janner's code)
        cond_obs = x_0[:, 0, :self.obs_dim].clone()  # (B, obs_dim)
        x_t[:, 0, :self.obs_dim] = cond_obs

        noise_pred = self.model(x_t, t)

        # Also zero out the noise target at the conditioned positions,
        # since those positions are replaced by clean values (no noise to predict)
        noise[:, 0, :self.obs_dim] = 0.0

        return nn.functional.mse_loss(noise_pred, noise)

    # ── Inpainting helper ────────────────────────────────────────────────────

    def _apply_inpainting(
        self,
        x: torch.Tensor,
        obs_cond: torch.Tensor,
        t_idx: int,
    ) -> torch.Tensor:
        """Overwrite the first timestep's obs portion with the known state.

        Following Janner et al. (2022): the conditioning observation is
        applied *clean* (not noised) at every denoising step, exactly as
        during training.  This is the "replacement" inpainting method.
        """
        x = x.clone()
        x[:, 0, :self.obs_dim] = obs_cond
        return x

    def _apply_inpainting_clean(
        self,
        x: torch.Tensor,
        obs_cond: torch.Tensor,
    ) -> torch.Tensor:
        """Final inpainting with clean (un-noised) observation."""
        x = x.clone()
        x[:, 0, :self.obs_dim] = obs_cond
        return x

    # ── Value guidance helper ────────────────────────────────────────────────

    def _apply_guidance(
        self,
        x: torch.Tensor,
        t_idx: int,
        scale: float,
    ) -> torch.Tensor:
        """Nudge the trajectory toward higher value using gradient of V."""
        if self.value_function is None or scale == 0.0:
            return x

        # We need gradients here even though the caller is in torch.no_grad().
        with torch.enable_grad():
            x_grad = x.detach().requires_grad_(True)
            B = x_grad.shape[0]
            t = torch.full((B,), t_idx, device=x.device, dtype=torch.long)

            value = self.value_function(x_grad, t)
            grad = torch.autograd.grad(value.sum(), x_grad)[0]

        # Scale gradient by posterior variance (as in the paper)
        variance = self.posterior_variance[t_idx]
        return x + scale * variance * grad

    # ── Reverse sampling ─────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        obs: torch.Tensor,
        horizon: int,
        act_dim: int,
        n_samples: int = 1,
    ) -> torch.Tensor:
        """DDPM reverse sampling with inpainting + guidance.

        Returns action sequences only (same interface as DiffusionSampler).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)
        total = B * n_samples

        x = torch.randn(total, horizon, self.transition_dim, device=obs.device)

        for t_idx in reversed(range(self.n_timesteps)):
            # Inpainting
            x = self._apply_inpainting(x, obs_expanded, t_idx)

            # Denoise
            t = torch.full((total,), t_idx, device=obs.device, dtype=torch.long)
            noise_pred = self.model(x, t)

            x_0_pred = (
                x - self.sqrt_one_minus_alphas_cumprod[t_idx] * noise_pred
            ) / self.sqrt_alphas_cumprod[t_idx]
            if self.clip_denoised:
                # Only clip action channels (last act_dim dims) to [-1, 1].
                # State channels are normalized to zero-mean unit-variance and
                # extend well beyond [-1, 1] - clamping them destroys ~34% of
                # state predictions and ruins trajectory quality.
                x_0_pred[:, :, self.obs_dim:] = x_0_pred[:, :, self.obs_dim:].clamp(-1.0, 1.0)

            mean = (
                self.posterior_mean_coef1[t_idx] * x_0_pred
                + self.posterior_mean_coef2[t_idx] * x
            )

            if t_idx > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(self.posterior_variance[t_idx]) * noise
            else:
                x = mean

            # Value guidance
            x = self._apply_guidance(x, t_idx, self.guidance_scale)

        # Final clean inpainting
        x = self._apply_inpainting_clean(x, obs_expanded)

        # Extract actions only: obs portion is [:, :, :obs_dim], act is [:, :, obs_dim:]
        return x[:, :, self.obs_dim:]  # (total, H, act_dim)

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
        """DDIM sampling with inpainting + guidance.

        Returns action sequences only (same interface as DiffusionSampler).
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        B = obs.shape[0]
        obs_expanded = obs.repeat_interleave(n_samples, dim=0)
        total = B * n_samples

        step_size = self.n_timesteps // ddim_steps
        timesteps = list(reversed(range(0, self.n_timesteps, step_size)))

        x = torch.randn(total, horizon, self.transition_dim, device=obs.device)

        for i, t_idx in enumerate(timesteps):
            # Inpainting
            x = self._apply_inpainting(x, obs_expanded, t_idx)

            # Denoise
            t = torch.full((total,), t_idx, device=obs.device, dtype=torch.long)
            noise_pred = self.model(x, t)

            alpha = self.alphas_cumprod[t_idx]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0)
            )

            x_0_pred = (x - torch.sqrt(1 - alpha) * noise_pred) / torch.sqrt(alpha)
            if self.clip_denoised:
                # Only clip action channels - see comment in sample() above
                x_0_pred[:, :, self.obs_dim:] = x_0_pred[:, :, self.obs_dim:].clamp(-1.0, 1.0)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha) * (1 - alpha / alpha_prev)
            )
            dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * noise_pred
            noise = torch.randn_like(x) if t_idx > 0 else torch.zeros_like(x)
            x = torch.sqrt(alpha_prev) * x_0_pred + dir_xt + sigma * noise

            # Value guidance
            x = self._apply_guidance(x, t_idx, self.guidance_scale)

        # Final clean inpainting
        x = self._apply_inpainting_clean(x, obs_expanded)

        # Return actions only
        return x[:, :, self.obs_dim:]  # (total, H, act_dim)
