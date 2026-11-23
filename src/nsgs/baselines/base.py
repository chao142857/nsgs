"""Common TaskPool container and abstract TaskSampler contract for all baselines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Sequence

import numpy as np


def build_fourier_features(
    raw: np.ndarray,
    *,
    periodic_dims: Sequence[int] = (),
    freqs: Sequence[float] = (1.0, 2.0),
) -> tuple[np.ndarray, dict[str, Any]]:
    """Standardize raw task coordinates and append sinusoidal Fourier features.

    Periodic dimensions (e.g. azimuth in radians) are expanded to (sin, cos)
    before standardization so wrap-around discontinuities never appear.
    """
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError(f"Expected non-empty 2D raw features, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("Raw task features contain non-finite values")

    periodic_set = set(int(d) for d in periodic_dims)
    cols = []
    for d in range(arr.shape[1]):
        if d in periodic_set:
            cols.append(np.sin(arr[:, d]))
            cols.append(np.cos(arr[:, d]))
        else:
            cols.append(arr[:, d])
    expanded = np.column_stack(cols)
    mean = expanded.mean(axis=0)
    std = expanded.std(axis=0)
    valid = std > 1e-12
    norm = (expanded - mean) / np.where(valid, std, 1.0)

    parts = [norm]
    if freqs and np.any(valid):
        base = norm[:, valid]
        for freq in freqs:
            phase = math.pi * float(freq) * base
            parts.append(np.sin(phase))
            parts.append(np.cos(phase))
    matrix = np.ascontiguousarray(np.column_stack(parts), dtype=np.float32)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Encoded Fourier features contain non-finite values")
    meta = {
        "raw_dim": int(arr.shape[1]),
        "encoded_dim": int(matrix.shape[1]),
        "periodic_dims": sorted(periodic_set),
        "freqs": [float(f) for f in freqs],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "nonconstant": valid.tolist(),
        "sha256": hashlib.sha256(matrix.tobytes()).hexdigest(),
    }
    return matrix, meta


@dataclass(frozen=True)
class TaskPool:
    """Immutable descriptor of a discrete candidate task pool of size N."""

    raw_features: np.ndarray
    features: np.ndarray
    bounds: tuple[tuple[float, float], ...]
    periodic_dims: tuple[int, ...] = ()
    feature_names: tuple[str, ...] = ()
    metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        raw = np.asarray(self.raw_features, dtype=np.float64)
        feat = np.asarray(self.features, dtype=np.float32)
        if raw.ndim != 2 or feat.ndim != 2 or raw.shape[0] != feat.shape[0] or raw.shape[0] == 0:
            raise ValueError(f"Mismatched pool shapes: raw={raw.shape}, features={feat.shape}")
        if len(self.bounds) != raw.shape[1]:
            raise ValueError(f"Expected {raw.shape[1]} bounds, got {len(self.bounds)}")
        if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(feat)):
            raise ValueError("TaskPool features must be finite")

    @property
    def num_tasks(self) -> int:
        return int(self.raw_features.shape[0])

    @property
    def raw_dim(self) -> int:
        return int(self.raw_features.shape[1])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    @classmethod
    def from_raw(
        cls,
        raw_features: np.ndarray,
        *,
        bounds: Sequence[tuple[float, float]] | None = None,
        periodic_dims: Sequence[int] = (),
        feature_names: Sequence[str] = (),
        freqs: Sequence[float] = (1.0, 2.0),
    ) -> "TaskPool":
        raw = np.ascontiguousarray(np.asarray(raw_features, dtype=np.float64))
        if bounds is None:
            lo = raw.min(axis=0)
            hi = raw.max(axis=0)
            pad = np.where(hi > lo, 0.0, 1.0)
            resolved_bounds = tuple((float(a), float(b + p)) for a, b, p in zip(lo, hi, pad))
        else:
            resolved_bounds = tuple((float(a), float(b)) for a, b in bounds)
        encoded, meta = build_fourier_features(raw, periodic_dims=periodic_dims, freqs=freqs)
        names = tuple(feature_names) if feature_names else tuple(f"x{i}" for i in range(raw.shape[1]))
        return cls(
            raw_features=raw,
            features=encoded,
            bounds=resolved_bounds,
            periodic_dims=tuple(sorted(set(int(d) for d in periodic_dims))),
            feature_names=names,
            metadata=meta,
        )


def distribution_stats(probs: np.ndarray, completed_counts: np.ndarray) -> dict[str, float]:
    """Shared distribution diagnostics computed identically across all baselines."""
    p = np.asarray(probs, dtype=np.float64)
    n = p.size
    nz = p > 0.0
    entropy = float(-np.sum(p[nz] * np.log(p[nz])))
    norm_entropy = entropy / math.log(n) if n > 1 else 0.0
    ess = float(1.0 / np.sum(p * p)) if n > 0 else 0.0
    order = np.sort(p)[::-1]
    k1 = max(1, int(round(0.01 * n)))
    k10 = max(1, int(round(0.10 * n)))
    unvisited = completed_counts == 0
    return {
        "entropy": entropy,
        "normalized_entropy": norm_entropy,
        "effective_sample_size": ess,
        "ess_fraction": ess / n if n > 0 else 0.0,
        "top_1pct_mass": float(order[:k1].sum()),
        "top_10pct_mass": float(order[:k10].sum()),
        "unvisited_fraction": float(unvisited.mean()),
        "unvisited_prob_mass": float(p[unvisited].sum()),
        "max_prob": float(order[0]),
        "min_prob": float(order[-1]),
    }


class TaskSampler(ABC):
    """Unified interface for all curriculum and task-sampling baselines.

    Contract:
      - Owns an isolated NumPy RNG (`self.rng`) so sampling never mutates
        external PyTorch/Python/NumPy global RNG state.
      - Accepts natural episode completions via `record(task_ids, successes, update=..., scores=...)`.
      - Refreshes sampling probabilities once per PPO update via `step_update(update)`.
      - Supports bit-exact checkpointing via `state_dict()` and `load_state_dict()`.
    """

    method_name: str = "base"

    def __init__(self, pool: TaskPool, seed: int) -> None:
        self.pool = pool
        self.n = pool.num_tasks
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.current_update = 0
        self.sample_counts = np.zeros(self.n, dtype=np.uint32)
        self.completed_counts = np.zeros(self.n, dtype=np.uint32)
        self.success_counts = np.zeros(self.n, dtype=np.uint32)
        self.total_sampled = 0
        self.total_completed = 0
        self.last_update_wall_s = 0.0
        self._probs = np.full(self.n, 1.0 / self.n, dtype=np.float64)
        self._cdf: np.ndarray | None = None

    def _set_probabilities(self, probs: np.ndarray) -> None:
        p = np.asarray(probs, dtype=np.float64)
        if p.shape != (self.n,) or not np.all(np.isfinite(p)) or np.any(p < 0.0):
            raise ValueError("Invalid probability vector")
        total = float(p.sum())
        if not (math.isfinite(total) and total > 0.0):
            raise ValueError("Probability vector must have positive sum")
        self._probs = p / total
        self._cdf = None

    def probabilities(self) -> np.ndarray:
        return self._probs.copy()

    def predicted_rates(self) -> np.ndarray | None:
        return None

    def sample(self, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        if count == 0:
            return np.empty(0, dtype=np.int64)
        if self._cdf is None:
            cdf = np.cumsum(self._probs, dtype=np.float64)
            cdf[-1] = 1.0
            self._cdf = cdf
        u = self.rng.random(count)
        ids = np.searchsorted(self._cdf, u, side="right").astype(np.int64)
        np.clip(ids, 0, self.n - 1, out=ids)
        np.add.at(self.sample_counts, ids, 1)
        self.total_sampled += int(count)
        return ids

    def _record_base_counts(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray(task_ids, dtype=np.int64).reshape(-1)
        succ = np.asarray(successes, dtype=np.bool_).reshape(-1)
        if ids.shape != succ.shape:
            raise ValueError(f"task_ids shape {ids.shape} != successes shape {succ.shape}")
        if ids.size > 0 and (int(ids.min()) < 0 or int(ids.max()) >= self.n):
            raise ValueError("task_ids out of range")
        if ids.size > 0:
            np.add.at(self.completed_counts, ids, 1)
            np.add.at(self.success_counts, ids, succ.astype(np.uint32))
            self.total_completed += int(ids.size)
        return ids, succ

    @abstractmethod
    def record(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
        *,
        update: int,
        scores: np.ndarray | None = None,
    ) -> None:
        """Ingest natural episode completion outcomes."""

    @abstractmethod
    def step_update(self, update: int) -> dict[str, Any]:
        """Refresh sampling distribution after PPO update `update`."""

    def diagnostics(self) -> dict[str, Any]:
        stats = distribution_stats(self._probs, self.completed_counts)
        stats.update(
            {
                "method": self.method_name,
                "update": int(self.current_update),
                "total_sampled": int(self.total_sampled),
                "total_completed": int(self.total_completed),
                "last_update_wall_s": float(self.last_update_wall_s),
            }
        )
        return stats

    def _base_state_dict(self) -> dict[str, Any]:
        return {
            "method_name": self.method_name,
            "seed": self.seed,
            "current_update": self.current_update,
            "rng_state": self.rng.bit_generator.state,
            "sample_counts": self.sample_counts.copy(),
            "completed_counts": self.completed_counts.copy(),
            "success_counts": self.success_counts.copy(),
            "total_sampled": self.total_sampled,
            "total_completed": self.total_completed,
            "last_update_wall_s": self.last_update_wall_s,
            "probs": self._probs.copy(),
        }

    def _load_base_state_dict(self, state: dict[str, Any]) -> None:
        self.current_update = int(state["current_update"])
        self.rng.bit_generator.state = state["rng_state"]
        self.sample_counts = np.asarray(state["sample_counts"], dtype=np.uint32).copy()
        self.completed_counts = np.asarray(state["completed_counts"], dtype=np.uint32).copy()
        self.success_counts = np.asarray(state["success_counts"], dtype=np.uint32).copy()
        self.total_sampled = int(state["total_sampled"])
        self.total_completed = int(state["total_completed"])
        self.last_update_wall_s = float(state["last_update_wall_s"])
        self._set_probabilities(np.asarray(state["probs"], dtype=np.float64))

    def state_dict(self) -> dict[str, Any]:
        return self._base_state_dict()

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
