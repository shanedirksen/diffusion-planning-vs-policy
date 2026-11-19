"""Inverse dynamics training loop for Decision Diffuser.

Trains the InverseDynamics MLP to predict actions from consecutive state
pairs (s_t, s_{t+1}) → a_t.  Uses the same offline dataset as the
diffusion model.

Usage:
    python -m diffplan.algorithms.train_inverse_dynamics --config configs/base.yaml
"""

from __future__ import annotations

import time
from pathlib import Path

import torch

from diffplan.data.state_trajectory_dataset import StateTrajectoryDataset
from diffplan.envs.make_env import make_env, get_env_info
from diffplan.models.inverse_dynamics import InverseDynamics
from diffplan.utils.logger import Logger
from diffplan.utils.seed import set_seed
from diffplan.utils.device import get_device


def train_inverse_dynamics(cfg: dict):
    """Train an inverse dynamics model on the offline dataset."""

    seed = cfg.get("seed", 42)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))
    print(f"[TrainInvDyn] device={device}  seed={seed}")

    # ── Environment info ──
    env_cfg = cfg.get("env", {})
    env_name = env_cfg["name"]
    horizon = env_cfg.get("horizon", 32)
    env = make_env(env_name, seed=seed)
    info = get_env_info(env)
    env.close()

    obs_dim = info["obs_dim"]
    act_dim = info["act_dim"]

    # ── Dataset ──
    dataset = StateTrajectoryDataset(env_name, horizon=horizon)
    normalizer = dataset.normalizer
    batch_size = int(cfg["train"]["batch_size"])

    if device.type == "cuda":
        dataset.to(device)
        print(f"[TrainInvDyn] dataset on GPU  |  {len(dataset.trans_s_t)} transitions")

    # ── Model ──
    dd_cfg = cfg.get("decision_diffuser", {})
    inv_dyn = InverseDynamics(
        obs_dim=obs_dim,
        act_dim=act_dim,
        hidden_dim=dd_cfg.get("inv_dyn_hidden_dim", 512),
        n_layers=dd_cfg.get("inv_dyn_n_layers", 2),
    ).to(device)
    n_params = sum(p.numel() for p in inv_dyn.parameters())
    print(f"[TrainInvDyn] InverseDynamics: {n_params:,} params")

    # ── Optimizer ──
    train_cfg = cfg["train"]
    lr = float(dd_cfg.get("inv_dyn_lr", train_cfg.get("lr", 2e-4)))
    optimizer = torch.optim.AdamW(
        inv_dyn.parameters(), lr=lr,
        weight_decay=float(train_cfg.get("weight_decay", 1e-6)),
    )
    grad_clip = float(train_cfg.get("grad_clip", 1.0))

    # ── AMP ──
    use_amp = train_cfg.get("amp", True) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
    if use_amp:
        print(f"[TrainInvDyn] AMP enabled ({amp_dtype})")

    # ── Logging ──
    log_cfg = cfg.get("logging", {})
    log_dir = cfg.get("log_dir", "runs/logs")
    logger = Logger(
        log_dir=log_dir,
        backend=log_cfg.get("backend", "csv"),
        project=log_cfg.get("project", "diffusion-planning"),
        config=cfg,
        name=f"inv_dyn_{env_name}",
    )

    # ── Checkpoint dir ──
    ckpt_dir = Path(cfg.get("checkpoint_dir", "runs/checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Training loop ──
    n_steps = int(dd_cfg.get("inv_dyn_train_steps", 100_000))
    log_every = int(train_cfg.get("log_every", 1_000))
    save_every = int(train_cfg.get("save_every", 50_000))

    t0 = time.perf_counter()
    inv_dyn.train()

    for step in range(1, n_steps + 1):
        s_t, a_t, s_tp1 = dataset.sample_transitions(batch_size, device=device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            a_pred = inv_dyn(s_t, s_tp1)
            loss = torch.nn.functional.mse_loss(a_pred, a_t)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(inv_dyn.parameters(), grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if step % log_every == 0:
            elapsed = time.perf_counter() - t0
            sps = step / elapsed
            eta_min = (n_steps - step) / sps / 60
            metrics = {"inv_dyn_loss": loss.item(), "step": step}
            logger.log(metrics, step=step)
            print(f"  step {step:>7d}/{n_steps}  loss={loss.item():.6f}  "
                  f"{sps:.0f} it/s  ETA {eta_min:.1f}min")

        if step % save_every == 0 or step == n_steps:
            ckpt_name = f"inv_dyn_{env_name.replace('-', '_')}.pt"
            ckpt_path = ckpt_dir / ckpt_name
            torch.save(
                {
                    "step": step,
                    "inv_dyn_state": inv_dyn.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  [saved] {ckpt_path}")

    total_time = time.perf_counter() - t0
    print(f"  [done] total time: {total_time/60:.1f}min  ({n_steps/total_time:.0f} it/s)")
    logger.finish()


if __name__ == "__main__":
    from diffplan.utils.config import build_config
    cfg = build_config()
    train_inverse_dynamics(cfg)
