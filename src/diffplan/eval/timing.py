"""Per-step inference timing utilities."""

from __future__ import annotations

import time
from typing import List

import numpy as np


class StepTimer:
    """Accumulates per-step wall-clock times and computes aggregates."""

    def __init__(self):
        self._times: List[float] = []
        self._start: float = 0.0

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        elapsed = time.perf_counter() - self._start
        self._times.append(elapsed)
        return elapsed

    def reset(self):
        self._times.clear()

    @property
    def times(self) -> np.ndarray:
        return np.array(self._times)

    @property
    def mean(self) -> float:
        return float(np.mean(self._times)) if self._times else 0.0

    @property
    def p95(self) -> float:
        return float(np.percentile(self._times, 95)) if self._times else 0.0

    @property
    def p99(self) -> float:
        return float(np.percentile(self._times, 99)) if self._times else 0.0

    @property
    def total(self) -> float:
        return float(np.sum(self._times))

    def summary(self) -> dict:
        return {
            "time_mean": self.mean,
            "time_p95": self.p95,
            "time_p99": self.p99,
            "time_total": self.total,
            "n_steps": len(self._times),
        }
