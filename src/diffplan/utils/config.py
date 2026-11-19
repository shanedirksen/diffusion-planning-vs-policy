"""YAML configuration loader with CLI overrides."""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


# ── helpers ──────────────────────────────────────────────────────────────────

def _deep_update(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates base)."""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _resolve_defaults(cfg: dict, config_dir: Path) -> dict:
    """If cfg has a ``defaults`` list, load each file and merge in order."""
    defaults = cfg.pop("defaults", [])
    merged: dict = {}
    for name in defaults:
        path = config_dir / f"{name}.yaml"
        if path.exists():
            with open(path) as f:
                _deep_update(merged, yaml.safe_load(f) or {})
    _deep_update(merged, cfg)
    return merged


# ── public API ───────────────────────────────────────────────────────────────

def load_config(path: str | Path, cli_overrides: Optional[Dict[str, Any]] = None) -> dict:
    """Load a YAML config file, resolve ``defaults``, apply CLI overrides.

    CLI overrides use dot-notation: ``train.lr=3e-4`` → ``cfg["train"]["lr"] = 3e-4``.
    """
    path = Path(path)
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    cfg = _resolve_defaults(cfg, path.parent)

    for key, val in (cli_overrides or {}).items():
        parts = key.split(".")
        d = cfg
        for p in parts[:-1]:
            d = d.setdefault(p, {})
        # Attempt YAML-style conversion (handles lists, bools, numbers)
        try:
            val = yaml.safe_load(val)
        except Exception:
            pass
        d[parts[-1]] = val

    return cfg


def parse_cli_overrides(argv: list[str] | None = None) -> tuple[argparse.Namespace, dict]:
    """Minimal CLI parser.

    Known args: ``--config``, ``--env``, ``--ckpt``, ``--H``, ``--K``, ``--seed``.
    Everything else of the form ``key=value`` is treated as a dot-notation override.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--H", type=int, default=None, dest="horizon")
    parser.add_argument("--K", type=int, default=None, dest="K")
    parser.add_argument("--seed", type=int, default=None)
    args, unknown = parser.parse_known_args(argv)

    overrides: Dict[str, Any] = {}
    for token in unknown:
        if "=" in token:
            k, v = token.split("=", 1)
            overrides[k] = v

    return args, overrides


def build_config(argv: list[str] | None = None) -> dict:
    """One-call helper: parse CLI → load config → merge overrides → return dict."""
    args, overrides = parse_cli_overrides(argv)
    cfg = load_config(args.config, overrides)

    # Apply named CLI args as top-level overrides
    if args.env is not None:
        cfg.setdefault("env", {})["name"] = args.env
    if args.ckpt is not None:
        cfg["ckpt"] = args.ckpt
    if args.horizon is not None:
        cfg.setdefault("env", {})["horizon"] = args.horizon
    if args.K is not None:
        cfg["K"] = args.K
    if args.seed is not None:
        cfg["seed"] = args.seed

    return cfg
