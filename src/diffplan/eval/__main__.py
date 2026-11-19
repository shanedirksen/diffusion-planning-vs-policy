"""Entry point: python -m diffplan.eval

Usage:
    python -m diffplan.eval --config configs/eval_policy.yaml --ckpt runs/checkpoints/model.pt
    python -m diffplan.eval --config configs/eval_planner.yaml --ckpt runs/checkpoints/model.pt --K 16
"""

from __future__ import annotations

import torch
import numpy as np

from diffplan.utils.config import build_config
from diffplan.utils.device import get_device
from diffplan.utils.seed import set_seed
from diffplan.envs.make_env import make_env, get_env_info
from diffplan.data.normalizer import Normalizer
from diffplan.models.diffusion_unet1d import DiffusionUNet1d
from diffplan.models.diffusion_transformer import DiffusionTransformer
from diffplan.models.trajectory_unet1d import TrajectoryUNet1d
from diffplan.models.state_unet1d import StateUNet1d
from diffplan.models.inverse_dynamics import InverseDynamics
from diffplan.models.sampler import DiffusionSampler
from diffplan.models.guided_sampler import GuidedDiffusionSampler
from diffplan.models.decision_diffuser_sampler import DecisionDiffuserSampler
from diffplan.models.value_function import ValueFunction
from diffplan.algorithms.policy_mode import PolicyMode
from diffplan.algorithms.planner_mode import PlannerMode
from diffplan.eval.rollout import evaluate, save_results_csv


def load_model(ckpt_path: str, device: torch.device, guidance_cfg: dict | None = None):
    """Load a trained diffusion model from checkpoint.

    Supports model types: 'action', 'transformer', 'trajectory', 'decision'.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    train_cfg = ckpt["config"]

    # Reconstruct normalizer
    normalizer = Normalizer.from_state_dict(ckpt["normalizer"])

    # Reconstruct model
    env_cfg = train_cfg.get("env", {})
    model_cfg = train_cfg.get("model", {})
    diff_cfg = train_cfg.get("diffusion", {})
    model_type = model_cfg.get("type", "action")

    env = make_env(env_cfg["name"])
    info = get_env_info(env)
    env.close()

    if model_type == "decision":
        # ── Decision Diffuser: state U-Net + inverse dynamics + CFG ──
        dd_cfg = train_cfg.get("decision_diffuser", {})

        state_unet = StateUNet1d(
            obs_dim=info["obs_dim"],
            horizon=env_cfg.get("horizon", 32),
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=dd_cfg.get("time_dim", 128),
            return_dim=dd_cfg.get("return_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
            cond_drop_prob=dd_cfg.get("cond_drop_prob", 0.25),
        )

        inv_dyn = InverseDynamics(
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            hidden_dim=dd_cfg.get("inv_dyn_hidden_dim", 512),
            n_layers=dd_cfg.get("inv_dyn_n_layers", 2),
        )

        # Optionally load a separately trained inverse dynamics checkpoint
        inv_dyn_ckpt_path = (guidance_cfg or {}).get("inv_dyn_ckpt") or dd_cfg.get("inv_dyn_ckpt")
        if inv_dyn_ckpt_path:
            inv_ckpt = torch.load(inv_dyn_ckpt_path, map_location=device, weights_only=False)
            inv_dyn.load_state_dict(inv_ckpt["inv_dyn_state"])
            print(f"[Eval] Loaded inverse dynamics from {inv_dyn_ckpt_path}")

        # Merge CLI overrides (guidance_cfg) with checkpoint defaults (dd_cfg)
        g_cfg = guidance_cfg or {}
        gs = g_cfg.get("guidance_scale", dd_cfg.get("guidance_scale", 1.2))
        temp = g_cfg.get("temperature", dd_cfg.get("temperature", 0.5))

        sampler = DecisionDiffuserSampler(
            model=state_unet,
            inverse_dynamics=inv_dyn,
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
            guidance_scale=gs,
            temperature=temp,
        )
        print(f"[Eval] model_type=decision  guidance_scale={gs}  temperature={temp}")
    elif model_type == "trajectory":
        # ── Diffuser-paper-style model ──
        transition_dim = info["obs_dim"] + info["act_dim"]
        unet = TrajectoryUNet1d(
            transition_dim=transition_dim,
            horizon=env_cfg.get("horizon", 32),
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=model_cfg.get("time_dim", 64),
            dropout=model_cfg.get("dropout", 0.1),
        )

        # Optionally load value function for guided sampling
        value_fn = None
        g_cfg = guidance_cfg or train_cfg.get("guidance", {})
        value_ckpt_path = g_cfg.get("value_ckpt")
        guidance_scale = g_cfg.get("scale", 0.1)

        if value_ckpt_path:
            v_ckpt = torch.load(value_ckpt_path, map_location=device, weights_only=False)
            value_fn = ValueFunction(
                transition_dim=transition_dim,
                horizon=env_cfg.get("horizon", 32),
                hidden_dim=g_cfg.get("hidden_dim", 256),
                time_dim=g_cfg.get("time_dim", 64),
            )
            value_fn.load_state_dict(v_ckpt["value_state"])
            value_fn.eval()
            print(f"[Eval] Loaded value function from {value_ckpt_path}")

        sampler = GuidedDiffusionSampler(
            model=unet,
            obs_dim=info["obs_dim"],
            act_dim=info["act_dim"],
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
            value_function=value_fn,
            guidance_scale=guidance_scale,
        )
        print(f"[Eval] model_type=trajectory  guidance_scale={guidance_scale}")
    elif model_type == "transformer":
        # ── Diffusion Policy Transformer (Chi et al. 2023) ──
        obs_horizon = env_cfg.get("obs_horizon", 1)
        net = DiffusionTransformer(
            act_dim=info["act_dim"],
            obs_dim=info["obs_dim"],
            horizon=env_cfg.get("horizon", 32),
            obs_horizon=obs_horizon,
            emb_dim=model_cfg.get("emb_dim", 256),
            n_heads=model_cfg.get("n_heads", 4),
            n_layers=model_cfg.get("n_layers", 8),
            time_dim=model_cfg.get("time_dim", 64),
            attn_dropout=model_cfg.get("attn_dropout", 0.1),
            ffn_dropout=model_cfg.get("dropout", 0.1),
        )

        sampler = DiffusionSampler(
            model=net,
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
        )
        print(f"[Eval] model_type=transformer")
    else:
        # ── Default: action-only model (Chi et al. 2023 CNN variant) ──
        obs_horizon = env_cfg.get("obs_horizon", 1)
        unet = DiffusionUNet1d(
            act_dim=info["act_dim"],
            obs_dim=info["obs_dim"] * obs_horizon,
            horizon=env_cfg.get("horizon", 32),
            dim=model_cfg.get("dim", 128),
            dim_mults=model_cfg.get("dim_mults", [1, 2, 4]),
            time_dim=model_cfg.get("time_dim", 64),
            cond_dim=model_cfg.get("cond_dim", 128),
            dropout=model_cfg.get("dropout", 0.1),
        )

        sampler = DiffusionSampler(
            model=unet,
            n_timesteps=diff_cfg.get("n_timesteps", 100),
            beta_schedule=diff_cfg.get("beta_schedule", "cosine"),
            clip_denoised=diff_cfg.get("clip_denoised", True),
        )
        print(f"[Eval] model_type=action")

    # Load EMA weights (preferred) or model weights.
    # Use strict=False so that externally-loaded sub-modules (e.g. a value
    # function added at eval time) don't cause missing-key errors.
    state_key = "ema_state" if "ema_state" in ckpt else "model_state"
    missing, unexpected = sampler.load_state_dict(ckpt[state_key], strict=False)
    # Only warn about genuinely unexpected keys (missing value_function keys are fine)
    if unexpected:
        print(f"[Eval] WARNING unexpected keys in checkpoint: {unexpected}")
    if missing:
        # Filter out value_function keys - they were loaded from a separate ckpt
        non_vf = [k for k in missing if not k.startswith("value_function.")]
        if non_vf:
            print(f"[Eval] WARNING missing non-value-function keys: {non_vf}")

    sampler = sampler.to(device)
    sampler.eval()

    return sampler, normalizer, info, train_cfg


def main():
    cfg = build_config()

    seed = cfg.get("seed", 0)
    set_seed(seed)
    device = get_device(cfg.get("device", "auto"))

    ckpt_path = cfg["ckpt"]
    mode = cfg.get("mode", "policy")
    env_name = cfg.get("env", {}).get("name")

    print(f"[diffplan.eval] mode={mode}  ckpt={ckpt_path}  device={device}")

    # Load model - pass both guidance (for trajectory) and decision_diffuser overrides
    guidance_cfg = cfg.get("guidance") or cfg.get("decision_diffuser")
    sampler, normalizer, info, train_cfg = load_model(ckpt_path, device, guidance_cfg=guidance_cfg)

    # Override env name from CLI if provided, else use training env
    if env_name is None:
        env_name = train_cfg["env"]["name"]
    horizon = cfg.get("env", {}).get("horizon") or train_cfg["env"].get("horizon", 32)

    # Chi et al. horizons: read from training config, override for non-Chi models
    model_type = train_cfg.get("model", {}).get("type", "action")
    if model_type in ("action", "transformer"):
        obs_horizon = train_cfg["env"].get("obs_horizon", 1)
        action_horizon = train_cfg["env"].get("action_horizon", 1)
    else:
        obs_horizon = 1
        action_horizon = 1

    eval_cfg = cfg.get("eval", {})
    n_episodes = eval_cfg.get("n_episodes", 20)
    is_maze = "maze" in env_name.lower()

    # Determine what horizons and K values to sweep
    horizons = eval_cfg.get("horizons", [horizon])
    if isinstance(horizons, int):
        horizons = [horizons]

    K_values = eval_cfg.get("K_values", [cfg.get("K", 16)])
    if isinstance(K_values, int):
        K_values = [K_values]

    # If a specific --H was passed, only use that
    if cfg.get("env", {}).get("horizon") is not None:
        horizons = [cfg["env"]["horizon"]]
    if cfg.get("K") is not None:
        K_values = [cfg["K"]]

    results_dir = cfg.get("results_dir", "runs/results")

    for H in horizons:
        env = make_env(env_name, seed=seed, max_episode_steps=info.get("max_episode_steps", 1000))

        if mode == "policy":
            print(f"\n{'='*60}")
            print(f"  POLICY MODE | env={env_name}  H={H}  To={obs_horizon}  Ta={action_horizon}")
            print(f"{'='*60}")

            agent = PolicyMode(
                sampler=sampler,
                normalizer=normalizer,
                horizon=H,
                act_dim=info["act_dim"],
                obs_dim=info["obs_dim"],
                obs_horizon=obs_horizon,
                action_horizon=action_horizon,
                device=device,
            )
            results = evaluate(
                env, agent, mode="policy",
                n_episodes=n_episodes,
                max_steps=info.get("max_episode_steps", 1000),
                record_positions=is_maze,
                seed=seed,
            )
            save_results_csv(results, env_name, "policy", H=H, K=0,
                             seed=seed, device=str(device), results_dir=results_dir,
                             model_type=model_type, obs_horizon=obs_horizon,
                             action_horizon=action_horizon)

        elif mode == "planner":
            for K in K_values:
                print(f"\n{'='*60}")
                print(f"  PLANNER MODE | env={env_name}  H={H}  K={K}  To={obs_horizon}  Ta={action_horizon}")
                print(f"{'='*60}")

                discount = eval_cfg.get("discount", 0.99)
                agent = PlannerMode(
                    sampler=sampler,
                    normalizer=normalizer,
                    horizon=H,
                    act_dim=info["act_dim"],
                    obs_dim=info["obs_dim"],
                    obs_horizon=obs_horizon,
                    action_horizon=action_horizon,
                    K=K,
                    discount=discount,
                    device=device,
                )
                results = evaluate(
                    env, agent, mode="planner",
                    n_episodes=n_episodes,
                    max_steps=info.get("max_episode_steps", 1000),
                    record_positions=is_maze,
                    seed=seed,
                )
                save_results_csv(results, env_name, "planner", H=H, K=K,
                                 seed=seed, device=str(device), results_dir=results_dir,
                                 model_type=model_type, obs_horizon=obs_horizon,
                                 action_horizon=action_horizon)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        env.close()

    print("\n[done] Evaluation complete.")


if __name__ == "__main__":
    main()
