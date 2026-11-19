"""Reproducible seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set seed for Python, NumPy, and PyTorch (CPU + GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Deterministic cuDNN (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
