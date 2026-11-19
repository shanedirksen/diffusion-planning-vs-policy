"""Diffusion model training loop.

Usage:
    python -m diffplan.train --config configs/train.yaml --env halfcheetah-medium-v2 --H 32
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path
from typing import Optional

import torch

from diffplan.data.d4rl_dataset import ActionSequenceDataset
from diffplan.data.trajectory_dataset import TrajectoryDataset
from diffplan.data.state_trajectory_dataset import StateTrajectoryDataset
from diffplan.data.normalizer import Normalizer
from diffplan.envs.make_env import make_env, get_env_info
from diffplan.models.diffusion_unet1d import DiffusionUNet1d
from diffplan.models.diffusion_transformer import DiffusionTransformer
from diffplan.models.trajectory_unet1d import TrajectoryUNet1d
from diffplan.models.state_unet1d import StateUNet1d
from diffplan.models.inverse_dynamics import InverseDynamics
from diffplan.models.sampler import DiffusionSampler
from diffplan.models.guided_sampler import GuidedDiffusionSampler
from diffplan.models.decision_diffuser_sampler import DecisionDiffuserSampler
from diffplan.utils.logger import Logger
from diffplan.utils.seed import set_seed
from diffplan.utils.device import get_device


# ── EMA helper ───────────────────────────────────────────────────────────────

class EMA:
    """Exponential moving average of model parameters (vectorized)."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.995):
        self.decay = decay
        self.shadow = copy.deepcopy(model)
        self.shadow.eval()
        # Cache param lists for fast foreach ops
        self._shadow_params = list(self.shadow.parameters())
        self._model_params_ref: list | None = None

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        if self._model_params_ref is None:
            self._model_params_ref = list(model.parameters())
        # in-place: shadow = decay * shadow + (1 - decay) * model
        torch._foreach_lerp_(self._shadow_params, self._model_params_ref, 1.0 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


# ── Training function ────────────────────────────────────────────────────────

def train(cfg: dict):
    """Main training entry point."""

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    print(f"[Train] device={device}  seed={seed}")

    # ── Environment + dataset ──
    env_cfg = cfg.get("env", {})
    env_name = env_cfg["name"]
    horizon = env_cfg.get("horizon", 32)
    env = make_env(env_name, seed=seed)
    info = get_env_info(env)
    env.close()

    model_cfg = cfg.get("model", {})
    model_type = model_cfg.get("type", "action")  # "action" | "transformer" | "trajectory" | "decision"
    diff_cfg = cfg.get("diffusion", {})
    batch_size = int(cfg["train"]["batch_size"])

    # Chi et al. observation window (only applies to action/transformer models)
    obs_horizon = env_cfg.get("obs_horizon", 1)
    if model_type not in ("action", "transformer"):
        obs_horizon = 1

    if model_type == "decision":
        # ── Decision Diffuser: state-only U-Net + CFG + inverse dynamics ──
        dd_cfg = cfg.get("decision_diffuser", {})
        dataset = StateTrajectoryDataset(env_name, horizon=horizon)
        normalizer = dataset.normalizer

        if device.type == "cuda":
            dataset.to(device)
            mem_mb = dataset.states.nbytes / 1e6
            print(f"[Train] state dataset on GPU ({mem_mb:.0f} MB)")

        state_unet = StateUNet1d(
            obs_dim=info["obs_dim"],
            horizon=horizon,
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=dd_cfg.get("time_dim", 128),
            return_dim=dd_cfg.get("return_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
            cond_drop_prob=dd_cfg.get("cond_drop_prob", 0.25),
        ).to(device)

        inv_dyn = InverseDynamics(
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            hidden_dim=dd_cfg.get("inv_dyn_hidden_dim", 512),
            n_layers=dd_cfg.get("inv_dyn_n_layers", 2),
        ).to(device)

        sampler = DecisionDiffuserSampler(
            model=state_unet,
            inverse_dynamics=inv_dyn,
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
            guidance_scale=dd_cfg.get("guidance_scale", 1.2),
            temperature=dd_cfg.get("temperature", 0.5),
        ).to(device)
        print(f"[Train] model_type=decision  obs_dim={info['obs_dim']}  "
              f"guidance_scale={dd_cfg.get('guidance_scale', 1.2)}")
    elif model_type == "trajectory":
        # ── Diffuser-paper-style: trajectory U-Net + GuidedSampler ──
        dataset = TrajectoryDataset(env_name, horizon=horizon)
        normalizer = dataset.normalizer

        if device.type == "cuda":
            dataset.to(device)
            mem_mb = dataset.trajectories.nbytes / 1e6
            print(f"[Train] trajectory dataset on GPU ({mem_mb:.0f} MB)")

        transition_dim = info["obs_dim"] + info["act_dim"]
        unet = TrajectoryUNet1d(
            transition_dim=transition_dim,
            horizon=horizon,
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=model_cfg.get("time_dim", 64),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)

        sampler = GuidedDiffusionSampler(
            model=unet,
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
            value_function=None,   # guidance only at eval time
            guidance_scale=0.0,
        ).to(device)
        print(f"[Train] model_type=trajectory  transition_dim={transition_dim}")
    elif model_type == "transformer":
        # ── Diffusion Policy Transformer (Chi et al. 2023) ──
        dataset = ActionSequenceDataset(
            env_name, horizon=horizon, obs_horizon=obs_horizon,
        )
        normalizer = dataset.normalizer

        if device.type == "cuda":
            dataset.to(device)
            mem_mb = (dataset.observations.nbytes + dataset.action_seqs.nbytes) / 1e6
            print(f"[Train] dataset on GPU ({mem_mb:.0f} MB)")

        net = DiffusionTransformer(
            act_dim=info["act_dim"],
            obs_dim=info["obs_dim"],
            horizon=horizon,
            obs_horizon=obs_horizon,
            emb_dim=model_cfg.get("emb_dim", 256),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 8),
            time_dim=model_cfg.get("time_dim", 64),
            attn_dropout=model_cfg.get("attn_dropout", 0.1),
            ffn_dropout=model_cfg.get("dropout", 0.1),
        ).to(device)

        sampler = DiffusionSampler(
            model=net,
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
        ).to(device)
        print(f"[Train] model_type=transformer  To={obs_horizon}  emb_dim={model_cfg.get('emb_dim', 256)}")
    else:
        # ── Default: action-only U-Net (Chi et al. 2023 CNN variant) ──
        dataset = ActionSequenceDataset(
            env_name, horizon=horizon, obs_horizon=obs_horizon,
        )
        normalizer = dataset.normalizer

        if device.type == "cuda":
            dataset.to(device)
            mem_mb = (dataset.observations.nbytes + dataset.action_seqs.nbytes) / 1e6
            print(f"[Train] dataset on GPU ({mem_mb:.0f} MB)")

        # CNN takes flattened obs window: obs_dim * obs_horizon
        net = DiffusionUNet1d(
            act_dim=info["act_dim"],
            obs_dim=info["obs_dim"] * obs_horizon,
            horizon=horizon,
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=model_cfg.get("time_dim", 64),
            cond_dim=model_cfg.get("cond_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
        ).to(device)

        sampler = DiffusionSampler(
            model=net,
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
        ).to(device)
        print(f"[Train] model_type=action  To={obs_horizon}")

    # ── torch.compile for faster kernels ──
    use_compile = cfg.get("train", {}).get("compile", True)
    if use_compile and device.type == "cuda":
        try:
            sampler.model = torch.compile(sampler.model)
            print("[Train] torch.compile enabled")
        except Exception as e:
            print(f"[Train] torch.compile failed ({e}), continuing without it")

    # ── Optimizer ──
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        sampler.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    ema = EMA(sampler, decay=float(train_cfg.get("ema_decay", 0.995)))
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # ── AMP (automatic mixed precision) ──
    use_amp = cfg.get("train", {}).get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"[Train] AMP enabled ({amp_dtype})")

    # ── Logging ──
    log_dir = cfg.get("log_dir", "runs/logs")
    logger = Logger(
        log_dir=log_dir,
        config=cfg,
        name=f"{env_name}_H{horizon}",
    )

    # ── Checkpoint dir ──
    ckpt_dir = Path(cfg.get("checkpoint_dir", "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    n_steps = int(train_cfg["n_train_steps"])
    save_every = int(train_cfg.get("save_every", 50_000))
    log_every = int(train_cfg.get("log_every", 1_000))

    t0 = time.perf_counter()

    for step in range(1, n_steps + 1):
        # Fast random batch (no DataLoader overhead)
        if model_type == "decision":
            states, returns = dataset.sample_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Diffusion loss on state sequences (CFG dropout handled inside model)
                diff_loss = sampler.training_loss(states, returns)
                # Inverse dynamics loss (joint training)
                s_t, a_t, s_tp1 = dataset.sample_transitions(batch_size, device=device)
                inv_loss = sampler.inverse_dynamics_loss(s_t, a_t, s_tp1)
                loss = diff_loss + inv_loss
        elif model_type == "trajectory":
            traj, _returns = dataset.sample_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = sampler.training_loss(traj)
        else:
            obs, act_seq = dataset.sample_batch(batch_size, device=device)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = sampler.training_loss(act_seq, obs)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(sampler.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(sampler)

        # ── Logging ──
        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            steps_per_sec = step / elapsed
            eta_min = (n_steps - step) / steps_per_sec / 60
            metrics = {"loss": loss.item(), "step": step, "steps_per_sec": steps_per_sec}
            logger.log(metrics, step=step)
            print(f"  step {step:>7d}/{n_steps}  loss={loss.item():.5f}  "
                  f"{steps_per_sec:.0f} it/s  ETA {eta_min:.1f}min")

        # ── Checkpointing ──
        if step % save_every == 0 or step == n_steps:
            type_tag = f"_{model_type}" if model_type != "action" else ""
            ckpt_name = f"{env_name.replace('-', '_')}{type_tag}_H{horizon}_step{step}.pt"
            ckpt_path = ckpt_dir / ckpt_name
            torch.save(
                {
                    "step": step,
                    "model_state": sampler.state_dict(),
                    "ema_state": ema.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  [saved] {ckpt_path}")

    # Final checkpoint with simple name
    type_tag = f"_{model_type}" if model_type != "action" else ""
    final_name = f"{env_name.replace('-', '_')}{type_tag}_H{horizon}.pt"
    final_path = ckpt_dir / final_name
    torch.save(
        {
            "step": n_steps,
            "model_state": sampler.state_dict(),
            "ema_state": ema.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "normalizer": normalizer.state_dict(),
            "config": cfg,
        },
        final_path,
    )
    total_time = time.perf_counter() - t0
    print(f"  [saved] final checkpoint: {final_path}")
    print(f"  [done] total time: {total_time/60:.1f}min  ({n_steps/total_time:.0f} it/s)")

    logger.finish()
