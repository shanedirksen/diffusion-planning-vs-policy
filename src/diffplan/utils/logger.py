"""Logging utilities - CSV backend."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Optional


class Logger:
    """Simple CSV logger for training metrics."""

    def __init__(
        self,
        log_dir: str | Path = "runs/logs",
        **kwargs,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0

        # ── CSV ──
        self._csv_path = self.log_dir / "metrics.csv"
        self._csv_file = None
        self._csv_writer = None

    # ── public interface ─────────────────────────────────────────────────────

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if step is not None:
            self._step = step
        else:
            self._step += 1

        metrics_with_step = {"step": self._step, **metrics}
        self._write_csv(metrics_with_step)

    def finish(self):
        if self._csv_file is not None:
            self._csv_file.close()

    # ── CSV helpers ──────────────────────────────────────────────────────────

    def _write_csv(self, row: dict):
        if self._csv_writer is None:
            self._csv_file = open(self._csv_path, "w", newline="")
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=list(row.keys()))
            self._csv_writer.writeheader()
        self._csv_writer.writerow(row)
        self._csv_file.flush()
