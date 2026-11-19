"""Render policy & planner episodes as GIFs so you can watch the cheetah run.

Supports all 4 model types:
    python render_episodes.py                          # action (default)
    python render_episodes.py --model transformer --ckpt runs/checkpoints/transformer.pt
    python render_episodes.py --model trajectory   --ckpt runs/checkpoints/trajectory.pt
    python render_episodes.py --model decision     --ckpt runs/checkpoints/decision.pt
"""

import sys
import os
import argparse
import numpy as np
import torch
import gymnasium as gym
from pathlib import Path

# We need imageio for GIF writing
try:
    import imageio
except ImportError:
    print("Installing imageio...", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "imageio[ffmpeg]"])
    import imageio

import mujoco

from diffplan.eval.__main__ import load_model
from diffplan.envs.make_env import get_gym_id, get_env_info
from diffplan.algorithms.policy_mode import PolicyMode
from diffplan.algorithms.planner_mode import PlannerMode
from diffplan.data.normalizer import Normalizer


def extend_floor(env, size=500):
    """Extend the checkerboard floor so the cheetah never runs off it.
    
    Scales geom_size AND texrepeat proportionally so the tiles stay
    the same size - just more of them.
    """
    m = env.unwrapped.model
    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    if floor_id < 0:
        return

    old_size = m.geom_size[floor_id][0]  # original half-extent (40)
    scale = size / old_size               # how much bigger

    # Extend the visible plane
    m.geom_size[floor_id] = [size, size, size]

    # Scale texture repeat so tiles stay the same density
    mat_id = m.geom_matid[floor_id]
    if mat_id >= 0:
        m.mat_texrepeat[mat_id] *= scale


def render_episode(env, agent, mode, max_steps=1000):
    """Run one episode, collecting frames."""
    frames = []
    obs, _ = env.reset()
    total_reward = 0.0
    done = False
    step = 0

    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        if mode == "planner":
            action = agent.act(obs, env)
        else:
            action = agent.act(obs)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += reward
        step += 1

        if step % 100 == 0:
            print(f"    step {step}/{max_steps}  reward_so_far={total_reward:.0f}", flush=True)

    # Capture final frame
    frame = env.render()
    if frame is not None:
        frames.append(frame)

    return frames, total_reward, step


def save_gif(frames, path, fps=20, skip=1):
    """Save frames as a GIF. MuJoCo control runs at 20Hz so fps=20 is real-time."""
    frames = frames[::skip]
    imageio.mimsave(str(path), frames, fps=fps, loop=0)
    size_mb = os.path.getsize(path) / 1024 / 1024
    print(f"  Saved {path.name} ({size_mb:.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Render diffusion RL episodes as GIFs")
    parser.add_argument("--model", type=str, default="action",
                        choices=["action", "transformer", "trajectory", "decision"],
                        help="Model type to render")
    parser.add_argument("--ckpt", type=str, default="runs/checkpoints/halfcheetah_medium_v2_H32.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--env", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--K", type=int, default=16, help="K samples for planner mode")
    parser.add_argument("--max-steps-policy", type=int, default=500)
    parser.add_argument("--max-steps-planner", type=int, default=300)
    parser.add_argument("--no-random", action="store_true", help="Skip random agent rendering")
    args = parser.parse_args()

    device = torch.device("cuda")
    gym_id = get_gym_id(args.env)
    out_dir = Path("runs/renders")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_type = args.model
    prefix = f"halfcheetah_{model_type}"

    print(f"Loading model (type={model_type})...", flush=True)
    sampler, normalizer, info, train_cfg = load_model(args.ckpt, device)

    # ── Policy mode episode ──
    print(f"\n=== {model_type.upper()} - POLICY MODE ===", flush=True)
    env = gym.make(gym_id, render_mode="rgb_array", max_episode_steps=1000)
    env.reset(seed=42)
    extend_floor(env)

    agent_policy = PolicyMode(
        sampler=sampler,
        normalizer=normalizer,
        horizon=32,
        act_dim=info["act_dim"],
        obs_dim=info["obs_dim"],
        device=device,
    )

    frames, ret, steps = render_episode(env, agent_policy, "policy", max_steps=args.max_steps_policy)
    print(f"  Policy: return={ret:.0f}, steps={steps}", flush=True)
    save_gif(frames, out_dir / f"{prefix}_policy.gif")
    env.close()

    # ── Planner mode episode ──
    print(f"\n=== {model_type.upper()} - PLANNER MODE (K={args.K}) ===", flush=True)
    env = gym.make(gym_id, render_mode="rgb_array", max_episode_steps=1000)
    env.reset(seed=42)
    extend_floor(env)

    agent_planner = PlannerMode(
        sampler=sampler,
        normalizer=normalizer,
        horizon=32,
        act_dim=info["act_dim"],
        obs_dim=info["obs_dim"],
        K=args.K,
        discount=0.99,
        device=device,
    )

    frames, ret, steps = render_episode(env, agent_planner, "planner", max_steps=args.max_steps_planner)
    print(f"  Planner: return={ret:.0f}, steps={steps}", flush=True)
    save_gif(frames, out_dir / f"{prefix}_planner_k{args.K}.gif")
    env.close()

    # ── Random agent (once is enough; skip with --no-random) ──
    random_gif = out_dir / "halfcheetah_random.gif"
    if not args.no_random and not random_gif.exists():
        print("\n=== RANDOM AGENT ===", flush=True)
        env = gym.make(gym_id, render_mode="rgb_array", max_episode_steps=1000)
        env.reset(seed=42)
        extend_floor(env)

        frames = []
        obs, _ = env.reset()
        total_reward = 0.0
        for step in range(500):
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break

        print(f"  Random: return={total_reward:.0f}, steps={step+1}", flush=True)
        save_gif(frames, random_gif)
        env.close()
    elif random_gif.exists():
        print(f"\n  Random GIF already exists, skipping (use --no-random to skip explicitly)")

    # ── Summary ──
    print(f"\nGIFs saved to {out_dir}/", flush=True)
    for f in sorted(out_dir.glob(f"{prefix}*.gif")):
        print(f"  - {f.name}", flush=True)


if __name__ == "__main__":
    main()
