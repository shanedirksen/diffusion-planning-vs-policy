"""Time-series Diffusion Transformer for action-sequence denoising.

Implements the Transformer-based Diffusion Policy (DiffusionPolicy-T) from:
  "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
  (Chi et al., 2023 - arXiv:2303.04137)

Architecture (Fig 2c of the paper):
    Input:  noisy action sequence  A^k_t  (B, Tp, act_dim)
    Cond:   observation embedding  O_t    (B, To, obs_emb_dim)  via cross-attention
            diffusion timestep     k      (B,)  prepended as first token
    Output: predicted noise        ε_θ    (B, Tp, act_dim)

Key design choices from the paper:
  - minGPT-style transformer decoder blocks
  - Observation enters via cross-attention (not FiLM)
  - Causal self-attention mask on action tokens
  - Sinusoidal embedding for diffusion iteration k prepended as first token
  - Each output token predicts the noise for the corresponding action step

This is a drop-in replacement for DiffusionUNet1d - same forward() signature:
    forward(x, t, obs) → noise
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffplan.models.conditioning import SinusoidalTimeEmbedding


# ── Observation encoder (shared MLP, maps raw state → embeddings) ────────────

class ObservationEncoder(nn.Module):
    """MLP that maps raw obs → observation embedding sequence.

    For state-based observations with To=1, the output is (B, 1, emb_dim).
    The paper uses a shared MLP across timesteps; here we support To=1.
    """

    def __init__(self, obs_dim: int, emb_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: (B, obs_dim) or (B, To, obs_dim)
        Returns:
            (B, To, emb_dim)  - To=1 if obs is 2D
        """
        if obs.dim() == 2:
            obs = obs.unsqueeze(1)  # (B, 1, obs_dim)
        return self.net(obs)


# ── Transformer decoder block with cross-attention ───────────────────────────

class TransformerDecoderBlock(nn.Module):
    """Single transformer decoder block:
    causal self-attention → cross-attention on obs → FFN.
    """

    def __init__(
        self,
        emb_dim: int,
        n_heads: int = 4,
        ffn_mult: int = 4,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        # Causal self-attention
        self.ln1 = nn.LayerNorm(emb_dim)
        self.self_attn = nn.MultiheadAttention(
            emb_dim, n_heads,
            dropout=attn_dropout, batch_first=True,
        )

        # Cross-attention on observation
        self.ln2 = nn.LayerNorm(emb_dim)
        self.cross_attn = nn.MultiheadAttention(
            emb_dim, n_heads,
            dropout=attn_dropout, batch_first=True,
        )
        self.ln_cross_kv = nn.LayerNorm(emb_dim)

        # Feed-forward network
        self.ln3 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * ffn_mult),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(emb_dim * ffn_mult, emb_dim),
            nn.Dropout(ffn_dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        obs_emb: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:        (B, S, emb_dim)  - action tokens + timestep token
            obs_emb:  (B, To, emb_dim) - observation embeddings
            attn_mask: (S, S) causal mask for self-attention
        """
        # Causal self-attention
        h = self.ln1(x)
        h, _ = self.self_attn(h, h, h, attn_mask=attn_mask)
        x = x + h

        # Cross-attention on observation
        q = self.ln2(x)
        kv = self.ln_cross_kv(obs_emb)
        h, _ = self.cross_attn(q, kv, kv)
        x = x + h

        # FFN
        x = x + self.ffn(self.ln3(x))
        return x


# ── Full Diffusion Transformer ───────────────────────────────────────────────

class DiffusionTransformer(nn.Module):
    """Time-series Diffusion Transformer (DiffusionPolicy-T).

    Drop-in replacement for DiffusionUNet1d in the sampler -
    same ``forward(x, t, obs) → noise`` signature.

    Parameters:
        act_dim:      action dimensionality
        obs_dim:      observation (state) dimensionality (per timestep)
        horizon:      action prediction horizon Tp
        obs_horizon:  observation context window To (default 1)
        emb_dim:      transformer embedding dimension (default 256)
        n_heads:      number of attention heads (default 4)
        n_layers:     number of transformer decoder blocks (default 8)
        time_dim:     sinusoidal timestep embedding dimension (default 64)
        attn_dropout: attention dropout probability (default 0.1)
        ffn_dropout:  FFN dropout probability (default 0.1)
    """

    def __init__(
        self,
        act_dim: int,
        obs_dim: int,
        horizon: int,
        obs_horizon: int = 1,
        emb_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 8,
        time_dim: int = 64,
        attn_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
    ):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.obs_horizon = obs_horizon
        self.horizon = horizon
        self.emb_dim = emb_dim

        # ── Timestep embedding: sinusoidal → linear → emb_dim ──
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, emb_dim),
            nn.SiLU(),
            nn.Linear(emb_dim, emb_dim),
        )

        # ── Observation encoder (shared MLP) ──
        self.obs_encoder = ObservationEncoder(obs_dim, emb_dim)

        # ── Action token embedding ──
        self.action_proj = nn.Linear(act_dim, emb_dim)

        # ── Learnable positional embedding for action tokens ──
        # +1 for the prepended timestep token
        self.pos_emb = nn.Parameter(
            torch.randn(1, horizon + 1, emb_dim) * 0.02
        )

        # ── Transformer decoder stack ──
        self.blocks = nn.ModuleList([
            TransformerDecoderBlock(
                emb_dim=emb_dim,
                n_heads=n_heads,
                attn_dropout=attn_dropout,
                ffn_dropout=ffn_dropout,
            )
            for _ in range(n_layers)
        ])

        # ── Output head: project back to act_dim ──
        self.ln_out = nn.LayerNorm(emb_dim)
        self.output_proj = nn.Linear(emb_dim, act_dim)

        # ── Causal attention mask (registered as buffer) ──
        # Shape (S, S) where S = horizon + 1 (timestep token + action tokens)
        S = horizon + 1
        mask = torch.triu(
            torch.ones(S, S, dtype=torch.bool), diagonal=1
        )
        self.register_buffer("causal_mask", mask)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following minGPT conventions."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        obs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x:   (B, Tp, act_dim) - noisy action sequence A^k
            t:   (B,)             - diffusion timestep k
            obs: (B, To*obs_dim)  - flattened observation window

        Returns:
            (B, Tp, act_dim) - predicted noise ε_θ
        """
        B = x.shape[0]

        # Reshape flattened obs to (B, To, obs_dim) for cross-attention
        obs = obs.view(B, self.obs_horizon, self.obs_dim)

        # Timestep token: (B, emb_dim) → (B, 1, emb_dim)
        t_emb = self.time_proj(self.time_emb(t)).unsqueeze(1)

        # Action tokens: (B, Tp, act_dim) → (B, Tp, emb_dim)
        act_tokens = self.action_proj(x)

        # Prepend timestep token → (B, 1+Tp, emb_dim)
        tokens = torch.cat([t_emb, act_tokens], dim=1)

        # Add positional embedding
        tokens = tokens + self.pos_emb[:, :tokens.shape[1], :]

        # Observation embeddings for cross-attention (To tokens)
        obs_emb = self.obs_encoder(obs)  # (B, To, emb_dim)

        # Pass through transformer blocks
        for block in self.blocks:
            tokens = block(tokens, obs_emb, attn_mask=self.causal_mask)

        # Take only action tokens (skip timestep token at index 0)
        action_output = tokens[:, 1:, :]  # (B, Tp, emb_dim)

        # Project to action space
        noise_pred = self.output_proj(self.ln_out(action_output))

        return noise_pred  # (B, Tp, act_dim)
