"""1-D Temporal U-Net for denoising action sequences.

Architecture:
    Input:  noisy action sequence  (B, H, act_dim)
    Cond:   conditioning vector    (B, cond_dim)   [from obs + diffusion time]
    Output: predicted noise        (B, H, act_dim)

The "temporal" axis is the planning horizon H.
Conv1d operates with channels = hidden_dim, length = H.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffplan.models.conditioning import ConditioningModule, FiLM


# ── Building blocks ──────────────────────────────────────────────────────────

class ResidualBlock1d(nn.Module):
    """Conv1d → GroupNorm → SiLU → FiLM → Conv1d → GroupNorm → SiLU + skip.

    FiLM conditioning is applied *between* the two conv layers, matching
    the ConditionalResidualBlock1D in Chi et al. (2023).
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, num_groups: int = 8):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.norm2 = nn.GroupNorm(min(num_groups, out_ch), out_ch)
        self.act = nn.SiLU()

        # FiLM conditioning applied between conv1 and conv2 (paper-faithful)
        self.film = FiLM(cond_dim, out_ch)

        # Skip projection if channel sizes differ
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:    (B, C, L)  - feature map
            cond: (B, cond_dim)
        """
        h = self.act(self.norm1(self.conv1(x)))

        # Apply FiLM conditioning between conv1 and conv2 (Chi et al. 2023)
        h = h.permute(0, 2, 1)                        # (B, L, C)
        h = self.film(h, cond)
        h = h.permute(0, 2, 1)                        # (B, C, L)

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


# ── U-Net ────────────────────────────────────────────────────────────────────

class DiffusionUNet1d(nn.Module):
    """1-D U-Net conditioned on (obs, diffusion_time).

    Parameters match the ``model`` section of ``base.yaml``.
    """

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        horizon: int,
        dim: int = 128,
        dim_mults: List[int] = (1, 2, 4),
        time_dim: int = 64,
        cond_dim: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.horizon = horizon

        # Conditioning: obs + diffusion time → cond vector
        self.conditioning = ConditioningModule(obs_dim, time_dim, cond_dim)

        # Input projection: act_dim → dim
        self.input_proj = nn.Conv1d(act_dim, dim, kernel_size=1)

        # Encoder (downsampling path)
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

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for d in reversed(dims):
            self.upsamples.append(Upsample1d(prev_dim))
            # After concat with skip connection: prev_dim + d channels
            self.up_blocks.append(ResidualBlock1d(prev_dim + d, d, cond_dim))
            prev_dim = d

        # Output projection: dim → act_dim (predict noise)
        self.output_proj = nn.Sequential(
            nn.GroupNorm(min(8, prev_dim), prev_dim),
            nn.SiLU(),
            nn.Conv1d(prev_dim, act_dim, kernel_size=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, H, act_dim) - noisy action sequence
            t:   (B,)            - diffusion timestep
            obs: (B, obs_dim)    - current state

        Returns:
            (B, H, act_dim) - predicted noise ε_θ
        """
        # Conditioning vector
        cond = self.conditioning(obs, t)       # (B, cond_dim)

        # (B, H, act_dim) → (B, act_dim, H) for Conv1d
        h = x.permute(0, 2, 1)
        h = self.input_proj(h)                 # (B, dim, H)

        # ── Encoder ──
        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, cond)
            skips.append(h)
            h = down(h)

        # ── Bottleneck ──
        h = self.mid_block(h, cond)

        # ── Decoder ──
        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            # Match temporal dimension after upsampling
            if h.shape[-1] != skip.shape[-1]:
                h = F.pad(h, (0, skip.shape[-1] - h.shape[-1]))
            h = torch.cat([h, skip], dim=1)
            h = block(h, cond)

        # (B, dim, H) → (B, act_dim, H) → (B, H, act_dim)
        out = self.output_proj(h)
        return out.permute(0, 2, 1)
