"""1-D Temporal U-Net for denoising full trajectories (state + action).

This is the Diffuser-paper-style architecture where the model generates
complete trajectories (observations + actions jointly) instead of just
action sequences.  Conditioning is via diffusion timestep only - the
current state is injected via *inpainting* during sampling, not FiLM.

Architecture:
    Input:  noisy trajectory  (B, H, transition_dim)   [transition_dim = obs_dim + act_dim]
    Cond:   diffusion time    (B,)
    Output: predicted noise   (B, H, transition_dim)
"""

from __future__ import annotations

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Sinusoidal time embedding (self-contained) ──────────────────────────────

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
    """Conv1d -> GroupNorm -> SiLU -> Conv1d -> GroupNorm -> SiLU + time-MLP + skip."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act = nn.SiLU()

        # Time conditioning: project time_dim -> out_ch, add to features
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch),
        )

        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:     (B, C, L)
            t_emb: (B, time_dim)
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Add time embedding (broadcast over length dimension)
        t_out = self.time_mlp(t_emb)          # (B, out_ch)
        h = h + t_out.unsqueeze(-1)           # (B, out_ch, L)

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


# ── Trajectory U-Net ─────────────────────────────────────────────────────────

class TrajectoryUNet1d(nn.Module):
    """Diffuser-paper-style 1-D U-Net over full trajectories.

    Key differences from DiffusionUNet1d:
      - Input channels = obs_dim + act_dim  (full trajectory)
      - No FiLM / state encoder - conditioning is timestep only
      - State info enters via inpainting in the sampler, not here
    """

    def __init__(
        self,
        transition_dim: int,
        horizon: int,
        dim: int = 128,
        dim_mults: List[int] = (1, 2, 4),
        time_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.transition_dim = transition_dim
        self.horizon = horizon

        # Time embedding: scalar t -> time_dim vector
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(dim),
            nn.Linear(dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        # Input projection: transition_dim -> dim
        self.input_proj = nn.Conv1d(transition_dim, dim, kernel_size=1)

        # Encoder
        dims = [dim * m for m in dim_mults]
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev_dim = dim
        for d in dims:
            self.down_blocks.append(ResidualBlock1d(prev_dim, d, time_dim))
            self.downsamples.append(Downsample1d(d))
            prev_dim = d

        # Bottleneck
        self.mid_block = ResidualBlock1d(prev_dim, prev_dim, time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for d in reversed(dims):
            self.upsamples.append(Upsample1d(prev_dim))
            self.up_blocks.append(ResidualBlock1d(prev_dim + d, d, time_dim))
            prev_dim = d

        # Output projection: dim -> transition_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, prev_dim), prev_dim),
            nn.SiLU(),
            nn.Conv1d(prev_dim, transition_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, H, transition_dim) - noisy trajectory
            t: (B,) - diffusion timestep

        Returns:
            (B, H, transition_dim) - predicted noise
        """
        # Time embedding
        t_emb = self.time_emb(t)               # (B, time_dim)

        # (B, H, D) -> (B, D, H) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)

        # Encoder
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        # Bottleneck
        h = self.mid_block(h, t_emb)

        # Decoder
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = block(h, t_emb)

        out = self.output_proj(h)
        return out.permute(0, 2, 1)            # (B, H, transition_dim)
