"""Neural Success-Guided Sampling: NSGS (v1) and Epistemic-Optimism NSGS (v2)."""

from __future__ import annotations

from dataclasses import replace
import math
from time import monotonic
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import TaskPool, TaskSampler
from .kernels import (
    EMPTY_HISTORY_PRIOR,
    AdaptiveTargetTracker,
    SGSKernel,
    apply_one_sided_optimism,
    kernel_probabilities,
    resolve_kernel,
)


def _build_mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.ReLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class NSGSSampler(TaskSampler):
    """Ensemble Neural Success-Guided Sampler supporting both NSGS (v1) and NSGS v2.

    - NSGS v1 (`beta_optimism=0.0, history_limit=1_000_000, tau=100.0, kernel='sharp'`):
      Exact ensemble-mean neural success estimator from the 3D/5D study.
    - NSGS v2 (`beta_optimism=1.0, history_limit=200_000, tau=30.0, kernel='sharp_safe'`):
      Addresses both root causes diagnosed in the million-pool audit without adding
      any spatial grids:
        1. One-sided ensemble epistemic optimism: lifts pessimistic predictions
           (mu < target) to min(target, mu + beta * sigma) while leaving mastered
           tasks (mu >= target) untouched.
        2. Faster non-stationary tracking via shorter ring buffer and smaller tau.
        3. Anti-starvation safety via sharp_safe kernel (epsilon=1e-4, 5% uniform floor).
        4. Optional empirical learning-progress adaptive target tracker.
    """

    method_name = "nsgs"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        kernel: str | SGSKernel = "sharp",
        num_members: int = 5,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        history_limit: int = 1_000_000,
        tau: float = 100.0,
        lr: float = 3e-4,
        adam_eps: float = 1e-8,
        optimizer_steps: int = 20,
        batch_size: int = 4096,
        cold_updates: int = 20,
        beta_optimism: float = 0.0,
        adaptive_target: bool = False,
        device: str = "cpu",
        method_label: str = "nsgs",
    ) -> None:
        super().__init__(pool, seed)
        if num_members <= 0:
            raise ValueError(f"num_members must be > 0, got {num_members}")
        if history_limit <= 0 or optimizer_steps <= 0 or batch_size <= 0:
            raise ValueError("history_limit, optimizer_steps, batch_size must be > 0")
        if not (math.isfinite(tau) and tau > 0.0):
            raise ValueError(f"tau must be > 0, got {tau}")
        if beta_optimism < 0.0 or not math.isfinite(beta_optimism):
            raise ValueError(f"beta_optimism must be >= 0, got {beta_optimism}")

        self.method_name = method_label
        self.kernel = resolve_kernel(kernel)
        self.num_members = int(num_members)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.history_limit = int(history_limit)
        self.tau = float(tau)
        self.lr = float(lr)
        self.adam_eps = float(adam_eps)
        self.optimizer_steps = int(optimizer_steps)
        self.batch_size = int(batch_size)
        self.cold_updates = int(cold_updates)
        self.beta_optimism = float(beta_optimism)
        self.device = torch.device(device)
        self.target_tracker = (
            AdaptiveTargetTracker(initial_target=self.kernel.target) if adaptive_target else None
        )

        self._features = torch.as_tensor(pool.features, dtype=torch.float32, device=self.device)
        self.models: list[nn.Module] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        with torch.random.fork_rng(devices=[] if self.device.type == "cpu" else [self.device]):
            for m in range(self.num_members):
                torch.manual_seed(self.seed + 7919 * (m + 1))
                net = _build_mlp(pool.feature_dim, self.hidden_dims).to(self.device)
                self.models.append(net)
                self.optimizers.append(torch.optim.Adam(net.parameters(), lr=self.lr, eps=self.adam_eps))

        self._buf_ids = np.zeros(self.history_limit, dtype=np.int64)
        self._buf_y = np.zeros(self.history_limit, dtype=np.float32)
        self._buf_u = np.zeros(self.history_limit, dtype=np.int32)
        self._buf_size = 0
        self._buf_ptr = 0

        self._pending_ids: list[np.ndarray] = []
        self._pending_y: list[np.ndarray] = []

        self.mean_rates = np.full(self.n, EMPTY_HISTORY_PRIOR, dtype=np.float64)
        self.std_rates = np.zeros(self.n, dtype=np.float64)
        self.effective_rates = self.mean_rates.copy()
        self._last_bce = float("nan")
        self._active_target = float(self.kernel.target)

    def predicted_rates(self) -> np.ndarray:
        return self.mean_rates.copy()

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
        if self.target_tracker is not None:
            self._pending_ids.append(ids)
            self._pending_y.append(y)
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

    def _fit_ensemble(self) -> float:
        ids_all = self._buf_ids[: self._buf_size]
        y_all = self._buf_y[: self._buf_size]
        age = np.maximum(0, self.current_update - self._buf_u[: self._buf_size]).astype(np.float64)
        rec_w = np.exp(-age / self.tau).astype(np.float32)
        bs = min(self.batch_size, self._buf_size)

        losses = []
        for model, opt in zip(self.models, self.optimizers):
            model.train()
            for _ in range(self.optimizer_steps):
                idx = self.rng.integers(0, self._buf_size, size=bs)
                xb = self._features.index_select(0, torch.as_tensor(ids_all[idx], dtype=torch.long, device=self.device))
                yb = torch.as_tensor(y_all[idx], dtype=torch.float32, device=self.device)
                wb = torch.as_tensor(rec_w[idx], dtype=torch.float32, device=self.device)
                wb = wb / wb.sum().clamp_min(1e-8)
                logits = model(xb).squeeze(-1)
                per_sample = nn.functional.binary_cross_entropy_with_logits(logits, yb, reduction="none")
                loss = (wb * per_sample).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")

    @torch.no_grad()
    def _predict_ensemble(self) -> tuple[np.ndarray, np.ndarray]:
        members = np.empty((self.num_members, self.n), dtype=np.float64)
        for m_idx, model in enumerate(self.models):
            model.eval()
            for start in range(0, self.n, self.batch_size):
                end = min(self.n, start + self.batch_size)
                probs = torch.sigmoid(model(self._features[start:end]).squeeze(-1))
                members[m_idx, start:end] = probs.cpu().numpy()
        return members.mean(axis=0), members.std(axis=0)

    def step_update(self, update: int) -> dict[str, Any]:
        t0 = monotonic()
        self.current_update = int(update)
        if self._buf_size == 0:
            self.last_update_wall_s = monotonic() - t0
            return self.diagnostics()

        if self.target_tracker is not None and self._pending_ids:
            p_ids = np.concatenate(self._pending_ids)
            p_y = np.concatenate(self._pending_y)
            self._pending_ids.clear()
            self._pending_y.clear()
            if self.current_update >= self.cold_updates:
                self._active_target = self.target_tracker.update(self.mean_rates[p_ids], p_y)

        self._last_bce = self._fit_ensemble()

        if self.current_update >= self.cold_updates:
            self.mean_rates, self.std_rates = self._predict_ensemble()
            active_kernel = (
                replace(self.kernel, target=self._active_target)
                if self.target_tracker is not None
                else self.kernel
            )
            self.effective_rates = apply_one_sided_optimism(
                self.mean_rates, self.std_rates, active_kernel.target, self.beta_optimism
            )
            self._set_probabilities(kernel_probabilities(self.effective_rates, active_kernel))

        self.last_update_wall_s = monotonic() - t0
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "kernel": self.kernel.to_dict(),
                "active_target": self._active_target,
                "tau": self.tau,
                "history_limit": self.history_limit,
                "beta_optimism": self.beta_optimism,
                "buffer_size": self._buf_size,
                "last_bce": self._last_bce,
                "mean_pred_rate": float(self.mean_rates.mean()),
                "mean_epistemic_std": float(self.std_rates.mean()),
                "p90_epistemic_std": float(np.quantile(self.std_rates, 0.90)),
                "mean_effective_rate": float(self.effective_rates.mean()),
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "models": [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.models],
                "optimizers": [o.state_dict() for o in self.optimizers],
                "buf_ids": self._buf_ids[: self._buf_size].copy(),
                "buf_y": self._buf_y[: self._buf_size].copy(),
                "buf_u": self._buf_u[: self._buf_size].copy(),
                "buf_size": self._buf_size,
                "buf_ptr": self._buf_ptr,
                "mean_rates": self.mean_rates.copy(),
                "std_rates": self.std_rates.copy(),
                "effective_rates": self.effective_rates.copy(),
                "last_bce": self._last_bce,
                "active_target": self._active_target,
                "target_tracker": None if self.target_tracker is None else self.target_tracker.state_dict(),
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        for model, sd in zip(self.models, state["models"]):
            model.load_state_dict(sd)
        for opt, sd in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(sd)
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
        self.mean_rates = np.asarray(state["mean_rates"], dtype=np.float64).copy()
        self.std_rates = np.asarray(state["std_rates"], dtype=np.float64).copy()
        self.effective_rates = np.asarray(state["effective_rates"], dtype=np.float64).copy()
        self._last_bce = float(state["last_bce"])
        self._active_target = float(state["active_target"])
        if self.target_tracker is not None and state.get("target_tracker") is not None:
            self.target_tracker.load_state_dict(state["target_tracker"])
