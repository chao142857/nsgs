"""Sparse axis-aligned spatial grid SGS (GSGS) baseline."""

from __future__ import annotations

import hashlib
import math
from time import monotonic
from typing import Any, Sequence

import numpy as np

from .base import TaskPool, TaskSampler
from .kernels import EMPTY_HISTORY_PRIOR, SGSKernel, kernel_probabilities, resolve_kernel


def _bin_axis(values: np.ndarray, lo: float, hi: float, count: int, *, periodic: bool) -> np.ndarray:
    span = hi - lo
    if not (math.isfinite(span) and span > 0.0):
        raise ValueError(f"Invalid bin bounds ({lo}, {hi})")
    scaled = (values - lo) / span
    if periodic:
        scaled = np.mod(scaled, 1.0)
        idx = np.floor(scaled * count).astype(np.int64)
        return np.mod(idx, count)
    idx = np.floor(scaled * count).astype(np.int64)
    return np.clip(idx, 0, count - 1)


def compute_sparse_cell_ids(
    pool: TaskPool,
    dims: Sequence[int],
    bins: int | Sequence[int],
) -> tuple[np.ndarray, np.ndarray, int, str]:
    """Map N tasks into compact active cell indices [0, n_active) without allocating
    empty cells (supports high-dimensional grids whose product exceeds memory).
    """
    dim_tuple = tuple(int(d) for d in dims)
    if not dim_tuple:
        raise ValueError("dims must be non-empty")
    if isinstance(bins, int):
        bin_counts = (int(bins),) * len(dim_tuple)
    else:
        bin_counts = tuple(int(b) for b in bins)
    if len(bin_counts) != len(dim_tuple) or any(b <= 0 for b in bin_counts):
        raise ValueError(f"Invalid bins {bins!r} for dims {dim_tuple}")

    periodic_set = set(pool.periodic_dims)
    coords = []
    for dim, count in zip(dim_tuple, bin_counts):
        if not (0 <= dim < pool.raw_dim):
            raise ValueError(f"dim {dim} out of range for raw_dim={pool.raw_dim}")
        lo, hi = pool.bounds[dim]
        coords.append(_bin_axis(pool.raw_features[:, dim], lo, hi, count, periodic=(dim in periodic_set)))

    # Hashable tuple-row encoding supports arbitrarily high product(bins) without int64 overflow.
    coord_matrix = np.ascontiguousarray(np.column_stack(coords), dtype=np.int32)
    row_view = coord_matrix.view(np.dtype((np.void, coord_matrix.dtype.itemsize * coord_matrix.shape[1]))).ravel()
    _, inverse, occupancy = np.unique(row_view, return_inverse=True, return_counts=True)
    cell_ids = inverse.astype(np.int32, copy=False)
    total_grid_cells = math.prod(bin_counts)
    digest = hashlib.sha256(coord_matrix.tobytes()).hexdigest()
    return cell_ids, occupancy.astype(np.int64, copy=False), int(total_grid_cells), digest


class GridSGSSampler(TaskSampler):
    """Sparse-cell Grid Success-Guided Sampling (GSGS).

    Pools recent binary outcomes (ring buffer of length `history`) within each
    occupied axis-aligned cell over `dims`, applies the SGS kernel per task,
    and normalizes across the full task pool. Supports both low-D projection
    grids (e.g. 3D 16^3) and high-D grids (where occupied cells approach N).
    """

    method_name = "grid_sgs"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        dims: Sequence[int] | None = None,
        bins: int | Sequence[int] = 16,
        kernel: str | SGSKernel = "sharp",
        history: int = 100,
        empty_prior: float = EMPTY_HISTORY_PRIOR,
    ) -> None:
        super().__init__(pool, seed)
        if history <= 0:
            raise ValueError(f"history must be > 0, got {history}")
        self.dims = tuple(range(pool.raw_dim)) if dims is None else tuple(int(d) for d in dims)
        self.bins = (int(bins),) * len(self.dims) if isinstance(bins, int) else tuple(int(b) for b in bins)
        self.kernel = resolve_kernel(kernel)
        self.history = int(history)
        self.empty_prior = float(empty_prior)

        self.cell_ids, self.occupancy, self.total_grid_cells, self.map_sha256 = compute_sparse_cell_ids(
            pool, self.dims, self.bins
        )
        self.n_active = int(self.occupancy.size)
        self._ring = np.zeros((self.n_active, self.history), dtype=np.uint8)
        self._ring_sum = np.zeros(self.n_active, dtype=np.uint16)
        self._cell_seen = np.zeros(self.n_active, dtype=np.uint32)
        self._cell_rates = np.full(self.n_active, self.empty_prior, dtype=np.float64)
        self._dirty = True
        self._refresh()

    def _refresh(self) -> None:
        per_task_rates = self._cell_rates[self.cell_ids]
        self._set_probabilities(kernel_probabilities(per_task_rates, self.kernel))
        self._dirty = False

    def predicted_rates(self) -> np.ndarray:
        return self._cell_rates[self.cell_ids].copy()

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
        cells = self.cell_ids[ids]
        for cell, won in zip(cells.tolist(), succ.tolist()):
            slot = int(self._cell_seen[cell]) % self.history
            val = 1 if won else 0
            old = int(self._ring[cell, slot])
            self._ring[cell, slot] = val
            self._ring_sum[cell] = int(self._ring_sum[cell]) - old + val
            count = int(self._cell_seen[cell]) + 1
            self._cell_seen[cell] = count
            denom = count if count < self.history else self.history
            self._cell_rates[cell] = self._ring_sum[cell] / denom
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
        labeled = int(np.count_nonzero(self._cell_seen))
        info.update(
            {
                "dims": list(self.dims),
                "bins": list(self.bins),
                "kernel": self.kernel.to_dict(),
                "occupied_cells": self.n_active,
                "total_grid_cells": self.total_grid_cells,
                "labeled_cells": labeled,
                "unlabeled_cells": self.n_active - labeled,
                "mean_tasks_per_occupied_cell": float(self.n / self.n_active),
                "map_sha256": self.map_sha256,
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "ring": self._ring.copy(),
                "ring_sum": self._ring_sum.copy(),
                "cell_seen": self._cell_seen.copy(),
                "cell_rates": self._cell_rates.copy(),
                "dirty": self._dirty,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        self._ring = np.asarray(state["ring"], dtype=np.uint8).copy()
        self._ring_sum = np.asarray(state["ring_sum"], dtype=np.uint16).copy()
        self._cell_seen = np.asarray(state["cell_seen"], dtype=np.uint32).copy()
        self._cell_rates = np.asarray(state["cell_rates"], dtype=np.float64).copy()
        self._dirty = bool(state["dirty"])
