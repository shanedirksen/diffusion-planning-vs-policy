"""Value function training loop (for trajectory-level guidance).

Trains a ValueFunction network to predict discounted cumulative returns
from (possibly noisy) trajectory segments - used during guided diffusion
sampling to steer toward high-reward trajectories.

Usage:
    python -m diffplan.algorithms.train_value --config configs/base.yaml
"""

from __future__ import annotations

import copy
import os
import time
from pathlib import Path

import torch

from diffplan.data.trajectory_dataset import TrajectoryDataset
from diffplan.envs.make_env import make_env, get_env_info
from diffplan.models.value_function import ValueFunction
from diffplan.models.sampler import cosine_beta_schedule, linear_beta_schedule
from diffplan.utils.logger import Logger
from diffplan.utils.seed import set_seed
from diffplan.utils.device import get_device


def train_value(cfg: dict):
    """Train a value function on noisy trajectory segments."""

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    print(f"[TrainValue] device={device}  seed={seed}")

    # ── Environment info ──
    env_cfg = cfg.get("env", {})
    env_name = env_cfg["name"]
    horizon = env_cfg.get("horizon", 32)
    env = make_env(env_name, seed=seed)
    info = get_env_info(env)
    env.close()

    obs_dim = info["obs_dim"]
    act_dim = info["act_dim"]
    transition_dim = obs_dim + act_dim

    # ── Dataset ──
    dataset = TrajectoryDataset(env_name, horizon=horizon)
    normalizer = dataset.normalizer
    batch_size = int(cfg["train"]["batch_size"])

    if device.type == "cuda":
        dataset.to(device)
        mem_mb = dataset.trajectories.nbytes / 1e6
        print(f"[TrainValue] dataset on GPU ({mem_mb:.0f} MB)")

    # ── Diffusion schedule (needed to noise trajectories at random timesteps) ──
    diff_cfg = cfg.get("diffusion", {})
    n_timesteps = diff_cfg.get("n_timesteps", 100)
    beta_sched = diff_cfg.get("beta_schedule", "cosine")
    if beta_sched == "cosine":
        betas = cosine_beta_schedule(n_timesteps)
    else:
        betas = linear_beta_schedule(n_timesteps)

    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_ac = torch.sqrt(1.0 - alphas_cumprod)

    # ── Value function model ──
    val_cfg = cfg.get("guidance", {})
    value_fn = ValueFunction(
        transition_dim=transition_dim,
        horizon=horizon,
        hidden_dim=val_cfg.get("hidden_dim", 256),
        time_dim=val_cfg.get("time_dim", 64),
    ).to(device)

    # ── Optimizer ──
    train_cfg = cfg["train"]
    lr = float(train_cfg.get("value_lr", train_cfg["lr"]))
    optimizer = torch.optim.AdamW(
        value_fn.parameters(), lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # ── AMP ──
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"[TrainValue] AMP enabled ({amp_dtype})")

    # ── Logging ──
    log_cfg = cfg.get("logging", {})
    log_dir = cfg.get("log_dir", "runs/logs")
    logger = Logger(
        log_dir=log_dir,
        backend=log_cfg.get("backend", "csv"),
        project=log_cfg.get("project", "diffusion-planning"),
        config=cfg,
        name=f"value_{env_name}_H{horizon}",
    )

    # ── Checkpoint dir ──
    ckpt_dir = Path(cfg.get("checkpoint_dir", "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    n_steps = int(train_cfg.get("value_train_steps", 100_000))
    log_every = int(train_cfg.get("log_every", 1_000))
    save_every = int(train_cfg.get("save_every", 50_000))

    t0 = time.perf_counter()
    value_fn.train()

    for step in range(1, n_steps + 1):
        # Sample batch
        traj, returns = dataset.sample_batch(batch_size, device=device)

        # Noise trajectories at random timesteps (the value function must
        # handle noisy inputs because it's applied during diffusion sampling)
        B = traj.shape[0]
        t = torch.randint(0, n_timesteps, (B,), device=device)
        noise = torch.randn_like(traj)
        sqrt_a = sqrt_alphas_cumprod[t].unsqueeze(-1).unsqueeze(-1)
        sqrt_1ma = sqrt_one_minus_ac[t].unsqueeze(-1).unsqueeze(-1)
        traj_noisy = sqrt_a * traj + sqrt_1ma * noise

        # Forward + loss
        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            pred = value_fn(traj_noisy, t)  # (B, 1)
            loss = torch.nn.functional.mse_loss(pred.squeeze(-1), returns)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(value_fn.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        # ── Logging ──
        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            eta_min = (n_steps - step) / sps / 60
            metrics = {"value_loss": loss.item(), "step": step, "steps_per_sec": sps}
            logger.log(metrics, step=step)
            print(f"  step {step:>7d}/{n_steps}  value_loss={loss.item():.5f}  "
                  f"{sps:.0f} it/s  ETA {eta_min:.1f}min")

        # ── Checkpointing ──
        if step % save_every == 0 or step == n_steps:
            ckpt_name = f"value_{env_name.replace('-', '_')}_H{horizon}.pt"
            ckpt_path = ckpt_dir / ckpt_name
            torch.save(
                {
                    "step": step,
                    "value_state": value_fn.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "return_stats": dataset.return_stats(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  [saved] {ckpt_path}")

    total_time = time.perf_counter() - t0
    print(f"  [done] value fn training: {total_time/60:.1f}min  ({n_steps/total_time:.0f} it/s)")
    logger.finish()


# ── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    from diffplan.utils.config import build_config

    # Use dot-notation for overrides: train.value_train_steps=50000
    cfg = build_config()
    train_value(cfg)
