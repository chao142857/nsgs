"""Uniform random task sampler baseline."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import TaskPool, TaskSampler


class UniformSampler(TaskSampler):
    """Samples every task in the pool with equal probability 1 / N."""

    method_name = "uniform"

    def __init__(self, pool: TaskPool, seed: int) -> None:
        super().__init__(pool, seed)

    def sample(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        ids = self.rng.integers(0, self.n, size=count, dtype=np.int64)
        np.add.at(self.sample_counts, ids, 1)
        self.total_sampled += int(count)
        return ids

    def record(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
        *,
        update: int,
        scores: np.ndarray | None = None,
    ) -> None:
        self._record_base_counts(task_ids, successes)

    def step_update(self, update: int) -> dict[str, Any]:
        self.current_update = int(update)
        self.last_update_wall_s = 0.0
        return self.diagnostics()
