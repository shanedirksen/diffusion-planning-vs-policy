"""Device selection utility."""

from __future__ import annotations

import torch


def get_device(preference: str = "auto") -> torch.device:
    """Return a torch device.

    Args:
        preference: ``"cuda"``, ``"cpu"``, or ``"auto"`` (pick GPU if available).
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
