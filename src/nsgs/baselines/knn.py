"""Non-parametric time-weighted k-Nearest Neighbors SGS (kNN-SGS) baseline."""

from __future__ import annotations

import math
from time import monotonic
from typing import Any

import numpy as np
import torch

from .base import TaskPool, TaskSampler
from .kernels import EMPTY_HISTORY_PRIOR, SGSKernel, kernel_probabilities, resolve_kernel


def _standardized_metric_coords(pool: TaskPool) -> np.ndarray:
    """Build standardized Euclidean coordinates (expanding periodic dims to sin/cos)."""
    raw = pool.raw_features
    periodic_set = set(pool.periodic_dims)
    cols = []
    for d in range(raw.shape[1]):
        if d in periodic_set:
            cols.append(np.sin(raw[:, d]))
            cols.append(np.cos(raw[:, d]))
        else:
            cols.append(raw[:, d])
    mat = np.column_stack(cols).astype(np.float64)
    mean = mat.mean(axis=0)
    std = mat.std(axis=0)
    norm = (mat - mean) / np.where(std > 1e-12, std, 1.0)
    return np.ascontiguousarray(norm, dtype=np.float32)


class KNNSGSSampler(TaskSampler):
    """Distance- and recency-weighted k-Nearest Neighbors success estimator + SGS kernel.

    Maintains a circular buffer of recent completed episodes (task_id, success, update).
    At each update, estimates each pool task's success rate from its k nearest
    recent observations in standardized task space:

        w_j(x) = exp(-(u - u_j) / tau) * K_dist(||x - x_j||_2)
        p_hat(x) = (prior_weight * empty_prior + sum_j w_j y_j) / (prior_weight + sum_j w_j)

    Uses chunked PyTorch matrix multiplication so it runs on CPU or CUDA with
    zero external ANN dependencies and reports exact query wall-time and
    neighbor-distance diagnostics (capturing high-dimensional distance concentration).
    """

    method_name = "knn_sgs"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        k: int = 32,
        kernel: str | SGSKernel = "sharp",
        tau: float = 50.0,
        history_limit: int = 100_000,
        max_anchors: int = 16_384,
        distance_weighting: str = "inv_dist",
        distance_eps: float = 0.05,
        bandwidth: float = 1.0,
        prior_weight: float = 1.0,
        empty_prior: float = EMPTY_HISTORY_PRIOR,
        cold_updates: int = 1,
        query_chunk_size: int = 4096,
        device: str = "cpu",
    ) -> None:
        super().__init__(pool, seed)
        if k <= 0:
            raise ValueError(f"k must be > 0, got {k}")
        if not (math.isfinite(tau) and tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if distance_weighting not in ("inv_dist", "rbf"):
            raise ValueError(f"Unknown distance_weighting {distance_weighting!r}")
        if history_limit <= 0 or max_anchors <= 0:
            raise ValueError("history_limit and max_anchors must be > 0")

        self.k = int(k)
        self.kernel = resolve_kernel(kernel)
        self.tau = float(tau)
        self.history_limit = int(history_limit)
        self.max_anchors = int(max_anchors)
        self.distance_weighting = distance_weighting
        self.distance_eps = float(distance_eps)
        self.bandwidth = float(bandwidth)
        self.prior_weight = float(prior_weight)
        self.empty_prior = float(empty_prior)
        self.cold_updates = int(cold_updates)
        self.query_chunk_size = int(query_chunk_size)
        self.device = torch.device(device)

        coords_np = _standardized_metric_coords(pool)
        self._coords = torch.as_tensor(coords_np, dtype=torch.float32, device=self.device)
        self._coord_sq = (self._coords * self._coords).sum(dim=1, keepdim=True)  # (N, 1)

        self._buf_ids = np.zeros(self.history_limit, dtype=np.int64)
        self._buf_y = np.zeros(self.history_limit, dtype=np.float32)
        self._buf_u = np.zeros(self.history_limit, dtype=np.int32)
        self._buf_size = 0
        self._buf_ptr = 0

        self._rates = np.full(self.n, self.empty_prior, dtype=np.float64)
        self._last_median_kth_dist = float("nan")
        self._last_mean_support = 0.0
        self._last_anchor_count = 0
        self._set_probabilities(kernel_probabilities(self._rates, self.kernel))

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
        m = ids.size
        if m == 0:
            return
        y = succ.astype(np.float32)
        u = int(update)
        if m >= self.history_limit:
            self._buf_ids[:] = ids[-self.history_limit :]
            self._buf_y[:] = y[-self.history_limit :]
            self._buf_u[:] = u
            self._buf_size = self.history_limit
            self._buf_ptr = 0
            return
        end = self._buf_ptr + m
        if end <= self.history_limit:
            self._buf_ids[self._buf_ptr : end] = ids
            self._buf_y[self._buf_ptr : end] = y
            self._buf_u[self._buf_ptr : end] = u
        else:
            first = self.history_limit - self._buf_ptr
            self._buf_ids[self._buf_ptr :] = ids[:first]
            self._buf_y[self._buf_ptr :] = y[:first]
            self._buf_u[self._buf_ptr :] = u
            rem = m - first
            self._buf_ids[:rem] = ids[first:]
            self._buf_y[:rem] = y[first:]
            self._buf_u[:rem] = u
        self._buf_ptr = end % self.history_limit
        self._buf_size = min(self.history_limit, self._buf_size + m)

    def _select_anchors(self, update: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n_obs = self._buf_size
        ids = self._buf_ids[:n_obs]
        y = self._buf_y[:n_obs]
        u = self._buf_u[:n_obs]
        age = np.maximum(0, int(update) - u).astype(np.float64)
        recency = np.exp(-age / self.tau)
        if n_obs <= self.max_anchors:
            return ids, y, recency.astype(np.float32)
        # Keep the most recent anchor observations (ties broken deterministically by RNG).
        probs = recency / float(recency.sum())
        chosen = self.rng.choice(n_obs, size=self.max_anchors, replace=False, p=probs)
        return ids[chosen], y[chosen], recency[chosen].astype(np.float32)

    @torch.no_grad()
    def step_update(self, update: int) -> dict[str, Any]:
        t0 = monotonic()
        self.current_update = int(update)
        if self._buf_size == 0 or self.current_update < self.cold_updates:
            self.last_update_wall_s = monotonic() - t0
            return self.diagnostics()

        anchor_ids_np, anchor_y_np, anchor_rec_np = self._select_anchors(self.current_update)
        m = int(anchor_ids_np.size)
        self._last_anchor_count = m
        k_eff = min(self.k, m)

        anchor_ids = torch.as_tensor(anchor_ids_np, dtype=torch.long, device=self.device)
        anchor_x = self._coords.index_select(0, anchor_ids)  # (M, D)
        anchor_sq = (anchor_x * anchor_x).sum(dim=1).unsqueeze(0)  # (1, M)
        anchor_y = torch.as_tensor(anchor_y_np, dtype=torch.float32, device=self.device)
        anchor_rec = torch.as_tensor(anchor_rec_np, dtype=torch.float32, device=self.device)

        rates_out = np.empty(self.n, dtype=np.float64)
        kth_dists = np.empty(self.n, dtype=np.float32)
        support_sum = 0.0

        for start in range(0, self.n, self.query_chunk_size):
            end = min(self.n, start + self.query_chunk_size)
            q_x = self._coords[start:end]  # (C, D)
            q_sq = self._coord_sq[start:end]  # (C, 1)
            dist_sq = (q_sq + anchor_sq - 2.0 * (q_x @ anchor_x.T)).clamp_min_(0.0)
            top_d2, top_idx = torch.topk(dist_sq, k=k_eff, dim=1, largest=False, sorted=True)
            top_d = torch.sqrt(top_d2)
            kth_dists[start:end] = top_d[:, -1].cpu().numpy()

            if self.distance_weighting == "inv_dist":
                w_dist = 1.0 / (top_d + self.distance_eps)
            else:
                w_dist = torch.exp(-0.5 * top_d2 / (self.bandwidth * self.bandwidth))

            w = w_dist * anchor_rec[top_idx]
            y_nbr = anchor_y[top_idx]
            w_sum = w.sum(dim=1)
            num = self.prior_weight * self.empty_prior + (w * y_nbr).sum(dim=1)
            den = self.prior_weight + w_sum
            p_chunk = (num / den).clamp(0.0, 1.0)
            rates_out[start:end] = p_chunk.cpu().numpy().astype(np.float64)
            support_sum += float(w_sum.sum().item())

        self._rates = rates_out
        self._last_median_kth_dist = float(np.median(kth_dists))
        self._last_mean_support = support_sum / self.n
        self._set_probabilities(kernel_probabilities(self._rates, self.kernel))
        self.last_update_wall_s = monotonic() - t0
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "k": self.k,
                "tau": self.tau,
                "kernel": self.kernel.to_dict(),
                "buffer_size": self._buf_size,
                "anchor_count": self._last_anchor_count,
                "median_kth_distance": self._last_median_kth_dist,
                "mean_neighbor_support": self._last_mean_support,
                "mean_rate": float(self._rates.mean()),
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "buf_ids": self._buf_ids[: self._buf_size].copy(),
                "buf_y": self._buf_y[: self._buf_size].copy(),
                "buf_u": self._buf_u[: self._buf_size].copy(),
                "buf_size": self._buf_size,
                "buf_ptr": self._buf_ptr,
                "rates": self._rates.copy(),
                "last_median_kth_dist": self._last_median_kth_dist,
                "last_mean_support": self._last_mean_support,
                "last_anchor_count": self._last_anchor_count,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        size = int(state["buf_size"])
        self._buf_ids.fill(0)
        self._buf_y.fill(0.0)
        self._buf_u.fill(0)
        if size > 0:
            self._buf_ids[:size] = np.asarray(state["buf_ids"], dtype=np.int64)
            self._buf_y[:size] = np.asarray(state["buf_y"], dtype=np.float32)
            self._buf_u[:size] = np.asarray(state["buf_u"], dtype=np.int32)
        self._buf_size = size
        self._buf_ptr = int(state["buf_ptr"])
        self._rates = np.asarray(state["rates"], dtype=np.float64).copy()
        self._last_median_kth_dist = float(state["last_median_kth_dist"])
        self._last_mean_support = float(state["last_mean_support"])
        self._last_anchor_count = int(state["last_anchor_count"])
