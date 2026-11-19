"""1-D Temporal U-Net for denoising state-only sequences (Decision Diffuser).

This is the Decision Diffuser (Ajay et al., ICLR 2023) architecture where the
model denoises *state* sequences only - no actions.  Actions are recovered via
a separate inverse dynamics model.

Conditioning:
  - Diffusion timestep k  (always)
  - Return R(τ)           (classifier-free guidance: dropped with prob p during training)

Architecture:
    Input:  noisy state sequence  (B, H, obs_dim)
    Cond:   diffusion time k      (B,)
            return conditioning y  (B,) or None
    Output: predicted noise        (B, H, obs_dim)
"""

from __future__ import annotations

from typing import List, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding ────────────────────────────────────────────────

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


# ── Building blocks ──────────────────────────────────────────────────────────

class ResidualBlock1d(nn.Module):
    """Conv1d -> GroupNorm -> Mish -> Conv1d -> GroupNorm -> Mish + cond + skip."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act = nn.Mish()

        # Conditioning: project cond_dim -> out_ch, add to activations
        self.cond_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_ch),
        )

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, L)
            cond: (B, cond_dim)  - concatenation of time + return embeddings
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Additive conditioning (same as Diffuser / Decision Diffuser paper)
        c = self.cond_mlp(cond)  # (B, out_ch)
        h = h + c.unsqueeze(-1)  # broadcast over length

        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class Downsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


# ── State U-Net (Decision Diffuser) ─────────────────────────────────────────

class StateUNet1d(nn.Module):
    """Decision Diffuser: 1-D U-Net over state-only sequences.

    Conditioned on:
      - Diffusion timestep k  (always present)
      - Return R(τ)           (dropped with prob ``cond_drop_prob`` for CFG)

    The return embedding is zeroed out when ``returns=None`` or during
    classifier-free guidance training dropout.
    """

    def __init__(
        self,
        obs_dim: int,
        horizon: int,
        dim: int = 128,
        dim_mults: List[int] = (1, 2, 4),
        time_dim: int = 128,
        return_dim: int = 128,
        dropout: float = 0.1,
        cond_drop_prob: float = 0.25,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.cond_drop_prob = cond_drop_prob

        # Time embedding: scalar k -> time_dim
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.Mish(),
            nn.Linear(time_dim, time_dim),
        )

        # Return embedding: scalar R -> return_dim
        self.return_emb = nn.Sequential(
            nn.Linear(1, return_dim),
            nn.Mish(),
            nn.Linear(return_dim, return_dim),
        )

        # Combined conditioning dimension
        cond_dim = time_dim + return_dim

        # Input projection: obs_dim -> dim
        self.input_proj = nn.Conv1d(obs_dim, dim, kernel_size=1)

        # Encoder
        dims = [dim * m for m in dim_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_dim = dim
        for d in dims:
            self.down_blocks.append(ResidualBlock1d(prev_dim, d, cond_dim))
            self.downsamples.append(Downsample1d(d))
            prev_dim = d

        # Bottleneck
        self.mid_block = ResidualBlock1d(prev_dim, prev_dim, cond_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for d in reversed(dims):
            self.upsamples.append(Upsample1d(prev_dim))
            self.up_blocks.append(ResidualBlock1d(prev_dim + d, d, cond_dim))
            prev_dim = d

        # Output projection: dim -> obs_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, prev_dim), prev_dim),
            nn.Mish(),
            nn.Conv1d(prev_dim, obs_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        returns: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:       (B, H, obs_dim) - noisy state sequence
            t:       (B,)            - diffusion timestep
            returns: (B,) or None    - normalized return conditioning

        Returns:
            (B, H, obs_dim) - predicted noise
        """
        B = x.shape[0]

        # Time embedding (always present)
        t_emb = self.time_emb(t)  # (B, time_dim)

        # Return embedding (zeroed when None or during CFG dropout)
        if returns is not None:
            r = returns.float().unsqueeze(-1)  # (B, 1)

            # Classifier-free guidance: drop conditioning with probability p
            if self.training:
                mask = torch.rand(B, 1, device=x.device) > self.cond_drop_prob
                r = r * mask.float()

            r_emb = self.return_emb(r)  # (B, return_dim)
        else:
            # Unconditional: zero out return embedding
            r_emb = torch.zeros(B, self.return_emb[-1].out_features, device=x.device)

        # Concatenate time + return embeddings as the conditioning vector
        cond = torch.cat([t_emb, r_emb], dim=-1)  # (B, cond_dim)

        # (B, H, D) -> (B, D, H)
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)

        # Encoder
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block(h, cond)

        # Decoder
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        out = self.output_proj(h)
        return out.permute(0, 2, 1)  # (B, H, obs_dim)
