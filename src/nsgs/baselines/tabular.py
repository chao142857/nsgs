"""Per-configuration tabular Success-Guided Sampling baseline."""

from __future__ import annotations

from time import monotonic
from typing import Any

import numpy as np

from .base import TaskPool, TaskSampler
from .kernels import EMPTY_HISTORY_PRIOR, SGSKernel, kernel_probabilities, resolve_kernel


class TabularSGSSampler(TaskSampler):
    """Per-task circular buffer of recent binary outcomes + SGS kernel.

    Matches the original SGS formulation (A Balanced Data Diet, H=100,
    empty-history prior = 0.5). Used to demonstrate label sparsity collapse
    on large task pools where most tasks receive only 0-2 outcomes.
    """

    method_name = "tabular_sgs"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        kernel: str | SGSKernel = "sharp",
        history: int = 100,
        empty_prior: float = EMPTY_HISTORY_PRIOR,
    ) -> None:
        super().__init__(pool, seed)
        if history <= 0:
            raise ValueError(f"history must be > 0, got {history}")
        self.kernel = resolve_kernel(kernel)
        self.history = int(history)
        self.empty_prior = float(empty_prior)
        self._ring = np.zeros((self.n, self.history), dtype=np.uint8)
        self._ring_sum = np.zeros(self.n, dtype=np.uint16)
        self._seen = np.zeros(self.n, dtype=np.uint32)
        self._rates = np.full(self.n, self.empty_prior, dtype=np.float64)
        self._dirty = True
        self._refresh()

    def _refresh(self) -> None:
        self._set_probabilities(kernel_probabilities(self._rates, self.kernel))
        self._dirty = False

    def predicted_rates(self) -> np.ndarray:
        return self._rates.copy()

    def record(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
        *,
        update: int,
        scores: np.ndarray | None = None,
    ) -> None:
        ids, succ = self._record_base_counts(task_ids, successes)
        if ids.size == 0:
            return
        # Sequential write preserves exact duplicate-ID ordering within a tick.
        for task, won in zip(ids.tolist(), succ.tolist()):
            slot = int(self._seen[task]) % self.history
            val = 1 if won else 0
            old = int(self._ring[task, slot])
            self._ring[task, slot] = val
            self._ring_sum[task] = int(self._ring_sum[task]) - old + val
            count = int(self._seen[task]) + 1
            self._seen[task] = count
            denom = count if count < self.history else self.history
            self._rates[task] = self._ring_sum[task] / denom
        self._dirty = True

    def step_update(self, update: int) -> dict[str, Any]:
        t0 = monotonic()
        self.current_update = int(update)
        if self._dirty:
            self._refresh()
        self.last_update_wall_s = monotonic() - t0
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "kernel": self.kernel.to_dict(),
                "history": self.history,
                "labeled_tasks": int(np.count_nonzero(self._seen)),
                "full_history_tasks": int(np.count_nonzero(self._seen >= self.history)),
                "mean_rate": float(self._rates.mean()),
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "ring": self._ring.copy(),
                "ring_sum": self._ring_sum.copy(),
                "seen": self._seen.copy(),
                "rates": self._rates.copy(),
                "dirty": self._dirty,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        self._ring = np.asarray(state["ring"], dtype=np.uint8).copy()
        self._ring_sum = np.asarray(state["ring_sum"], dtype=np.uint16).copy()
        self._seen = np.asarray(state["seen"], dtype=np.uint32).copy()
        self._rates = np.asarray(state["rates"], dtype=np.float64).copy()
        self._dirty = bool(state["dirty"])
