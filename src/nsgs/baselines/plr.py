"""Prioritized Level Replay (PLR) and Neural-PLR ablation baselines.

Implements standard PLR (Jiang et al., ICML 2021):
  P_replay(i) = (1 - rho) * P_S(i) + rho * P_C(i)
with rank-based or proportional score prioritization P_S over seen tasks,
staleness prioritization P_C proportional to steps since last sampled, and
an unseen-task exploration gate d_replay.

Also provides NeuralPLRSampler to decouple neural generalization across unseen
tasks from the choice of learning signal (value-loss / PLR score vs SGS
success-rate frontier).
"""

from __future__ import annotations

import copy
import math
from time import monotonic
from typing import Any

import numpy as np
import torch
from torch import nn

from .base import TaskPool, TaskSampler


def _score_distribution(scores: np.ndarray, *, temperature: float, strategy: str) -> np.ndarray:
    s = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
    m = s.size
    if m == 0:
        return np.empty(0, dtype=np.float64)
    if strategy == "rank":
        # Stable descending rank: rank 1 is highest score.
        order = np.argsort(-s, kind="mergesort")
        ranks = np.empty(m, dtype=np.float64)
        ranks[order] = np.arange(1, m + 1, dtype=np.float64)
        weights = np.power(1.0 / ranks, 1.0 / temperature)
    elif strategy == "proportional":
        if float(s.max()) <= 0.0:
            weights = np.ones(m, dtype=np.float64)
        else:
            weights = np.power(s + 1e-8, 1.0 / temperature)
    else:
        raise ValueError(f"Unknown PLR score strategy {strategy!r}")
    total = float(weights.sum())
    if not (math.isfinite(total) and total > 0.0):
        return np.full(m, 1.0 / m, dtype=np.float64)
    return weights / total


class PLRSampler(TaskSampler):
    """Tabular Prioritized Level Replay (Jiang et al., ICML 2021).

    Maintains per-task score estimates S_i and last-sampled timestamps C_i.
    When explicit per-episode `scores` (e.g., positive value loss or GAE L1)
    are provided to `record(...)`, S_i tracks an EMA of those scores. When
    only binary `successes` are provided, S_i falls back to the Bernoulli
    outcome variance proxy `4 * p_i * (1 - p_i)` (peaking at p = 0.5).
    """

    method_name = "plr"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        rho: float = 0.3,
        temperature: float = 1.0,
        strategy: str = "rank",
        score_ema: float = 0.25,
        replay_prob: float = 0.8,
        scale_replay_by_seen: bool = True,
    ) -> None:
        super().__init__(pool, seed)
        if not (0.0 <= rho <= 1.0):
            raise ValueError(f"rho must be in [0, 1], got {rho}")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if strategy not in ("rank", "proportional"):
            raise ValueError(f"strategy must be 'rank' or 'proportional', got {strategy!r}")
        if not (0.0 < score_ema <= 1.0):
            raise ValueError(f"score_ema must be in (0, 1], got {score_ema}")
        if not (0.0 <= replay_prob <= 1.0):
            raise ValueError(f"replay_prob must be in [0, 1], got {replay_prob}")

        self.rho = float(rho)
        self.temperature = float(temperature)
        self.strategy = strategy
        self.score_ema = float(score_ema)
        self.replay_prob = float(replay_prob)
        self.scale_replay_by_seen = bool(scale_replay_by_seen)

        self._seen = np.zeros(self.n, dtype=np.bool_)
        self._scores = np.zeros(self.n, dtype=np.float64)
        self._ema_success = np.full(self.n, 0.5, dtype=np.float64)
        self._last_sampled_step = np.zeros(self.n, dtype=np.int64)
        self._step_clock = 0
        self._dirty = True
        self._refresh()

    def _effective_replay_prob(self, n_seen: int) -> float:
        if n_seen == 0:
            return 0.0
        if n_seen == self.n:
            return 1.0
        if self.scale_replay_by_seen:
            return float(self.replay_prob * (n_seen / self.n))
        return float(self.replay_prob)

    def _refresh(self) -> None:
        seen_idx = np.flatnonzero(self._seen)
        n_seen = int(seen_idx.size)
        if n_seen == 0:
            self._set_probabilities(np.full(self.n, 1.0 / self.n, dtype=np.float64))
            self._dirty = False
            return

        p_s = _score_distribution(
            self._scores[seen_idx], temperature=self.temperature, strategy=self.strategy
        )
        staleness = np.maximum(0, self._step_clock - self._last_sampled_step[seen_idx]).astype(np.float64)
        stale_sum = float(staleness.sum())
        if stale_sum > 0.0:
            p_c = staleness / stale_sum
        else:
            p_c = np.full(n_seen, 1.0 / n_seen, dtype=np.float64)

        p_seen = (1.0 - self.rho) * p_s + self.rho * p_c
        d_replay = self._effective_replay_prob(n_seen)

        probs = np.zeros(self.n, dtype=np.float64)
        probs[seen_idx] = d_replay * p_seen
        n_unseen = self.n - n_seen
        if n_unseen > 0 and d_replay < 1.0:
            probs[~self._seen] = (1.0 - d_replay) / n_unseen
        self._set_probabilities(probs)
        self._dirty = False

    def sample(self, count: int) -> np.ndarray:
        ids = super().sample(count)
        if ids.size > 0:
            self._step_clock += int(ids.size)
            self._last_sampled_step[ids] = self._step_clock
        return ids

    def record(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
        *,
        update: int,
        scores: np.ndarray | None = None,
    ) -> None:
        ids_check = np.asarray(task_ids, dtype=np.int64).reshape(-1)
        sc: np.ndarray | None = None
        if scores is not None:
            sc_raw = np.asarray(scores, dtype=np.float64)
            if sc_raw.ndim != 1 or sc_raw.shape != ids_check.shape or not np.all(np.isfinite(sc_raw)):
                raise ValueError(
                    f"PLR scores must be a finite 1D array matching task_ids shape {ids_check.shape}, "
                    f"got shape {sc_raw.shape}"
                )
            sc = np.maximum(sc_raw, 0.0)
        ids, succ = self._record_base_counts(task_ids, successes)
        if ids.size == 0:
            return
        if sc is not None:
            for task, val in zip(ids.tolist(), sc.tolist()):
                if not self._seen[task]:
                    self._scores[task] = val
                    self._seen[task] = True
                else:
                    self._scores[task] = (1.0 - self.score_ema) * self._scores[task] + self.score_ema * val
        else:
            for task, won in zip(ids.tolist(), succ.tolist()):
                y = 1.0 if won else 0.0
                if not self._seen[task]:
                    self._ema_success[task] = 0.5 * (0.5 + y)
                    self._seen[task] = True
                else:
                    self._ema_success[task] = (1.0 - self.score_ema) * self._ema_success[task] + self.score_ema * y
                p = self._ema_success[task]
                self._scores[task] = 4.0 * p * (1.0 - p)
        self._dirty = True

    def step_update(self, update: int) -> dict[str, Any]:
        t0 = monotonic()
        self.current_update = int(update)
        if self._dirty or np.any(self._seen):
            self._refresh()
        self.last_update_wall_s = monotonic() - t0
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        n_seen = int(self._seen.sum())
        info.update(
            {
                "rho": self.rho,
                "temperature": self.temperature,
                "strategy": self.strategy,
                "seen_tasks": n_seen,
                "effective_replay_prob": self._effective_replay_prob(n_seen),
                "mean_seen_score": float(self._scores[self._seen].mean()) if n_seen > 0 else 0.0,
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "seen": self._seen.copy(),
                "scores": self._scores.copy(),
                "ema_success": self._ema_success.copy(),
                "last_sampled_step": self._last_sampled_step.copy(),
                "step_clock": self._step_clock,
                "dirty": self._dirty,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        self._seen = np.asarray(state["seen"], dtype=np.bool_).copy()
        self._scores = np.asarray(state["scores"], dtype=np.float64).copy()
        self._ema_success = np.asarray(state["ema_success"], dtype=np.float64).copy()
        self._last_sampled_step = np.asarray(state["last_sampled_step"], dtype=np.int64).copy()
        self._step_clock = int(state["step_clock"])
        self._dirty = bool(state["dirty"])


def _build_score_mlp(in_dim: int, hidden_dims: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = []
    prev = in_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev, h))
        layers.append(nn.SiLU())
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


class NeuralPLRSampler(TaskSampler):
    """Neural-generalized PLR ablation: predicts per-task PLR score across unseen tasks.

    Uses the same time-weighted MLP ensemble architecture as NSGS, trained with
    time-weighted Huber loss to regress positive episode learning scores (e.g.
    positive value loss / GAE L1), then converts predicted scores into PLR's
    rank- or proportional-prioritized distribution mixed with staleness.
    """

    method_name = "neural_plr"

    def __init__(
        self,
        pool: TaskPool,
        seed: int,
        *,
        rho: float = 0.2,
        temperature: float = 1.0,
        strategy: str = "rank",
        num_members: int = 5,
        hidden_dims: tuple[int, ...] = (256, 256, 256),
        history_limit: int = 200_000,
        tau: float = 50.0,
        lr: float = 3e-4,
        optimizer_steps: int = 20,
        batch_size: int = 4096,
        cold_updates: int = 10,
        device: str = "cpu",
    ) -> None:
        super().__init__(pool, seed)
        self.rho = float(rho)
        self.temperature = float(temperature)
        self.strategy = strategy
        self.num_members = int(num_members)
        self.hidden_dims = tuple(int(h) for h in hidden_dims)
        self.history_limit = int(history_limit)
        self.tau = float(tau)
        self.lr = float(lr)
        self.optimizer_steps = int(optimizer_steps)
        self.batch_size = int(batch_size)
        self.cold_updates = int(cold_updates)
        self.device = torch.device(device)

        self._features = torch.as_tensor(pool.features, dtype=torch.float32, device=self.device)
        self.models: list[nn.Module] = []
        self.optimizers: list[torch.optim.Optimizer] = []
        # Seed CPU default_generator only inside fork_rng(devices=[]) so CUDA RNGs are never touched.
        with torch.random.fork_rng(devices=[]):
            for m in range(self.num_members):
                torch.default_generator.manual_seed(self.seed + 1009 * (m + 1))
                net = _build_score_mlp(pool.feature_dim, self.hidden_dims).to(self.device)
                self.models.append(net)
                self.optimizers.append(torch.optim.Adam(net.parameters(), lr=self.lr))

        self._buf_ids = np.zeros(self.history_limit, dtype=np.int64)
        self._buf_s = np.zeros(self.history_limit, dtype=np.float32)
        self._buf_u = np.zeros(self.history_limit, dtype=np.int32)
        self._buf_size = 0
        self._buf_ptr = 0
        self._last_sampled_step = np.zeros(self.n, dtype=np.int64)
        self._step_clock = 0
        self._pred_scores = np.ones(self.n, dtype=np.float64)
        self._last_loss = float("nan")

    def sample(self, count: int) -> np.ndarray:
        ids = super().sample(count)
        if ids.size > 0:
            self._step_clock += int(ids.size)
            self._last_sampled_step[ids] = self._step_clock
        return ids

    def record(
        self,
        task_ids: np.ndarray,
        successes: np.ndarray,
        *,
        update: int,
        scores: np.ndarray | None = None,
    ) -> None:
        ids_check = np.asarray(task_ids, dtype=np.int64).reshape(-1)
        sc_validated: np.ndarray | None = None
        if scores is not None:
            sc_raw = np.asarray(scores, dtype=np.float64)
            if sc_raw.ndim != 1 or sc_raw.shape != ids_check.shape or not np.all(np.isfinite(sc_raw)):
                raise ValueError(
                    f"NeuralPLR scores must be a finite 1D array matching task_ids shape {ids_check.shape}, "
                    f"got shape {sc_raw.shape}"
                )
            sc_validated = np.maximum(sc_raw, 0.0).astype(np.float32)
        ids, succ = self._record_base_counts(task_ids, successes)
        m = ids.size
        if m == 0:
            return
        if sc_validated is not None:
            sc = sc_validated
        else:
            # Binary fallback: surprise against running batch mean.
            mean_y = float(succ.mean())
            sc = np.abs(succ.astype(np.float32) - mean_y)
        u = int(update)
        if m >= self.history_limit:
            self._buf_ids[:] = ids[-self.history_limit :]
            self._buf_s[:] = sc[-self.history_limit :]
            self._buf_u[:] = u
            self._buf_size = self.history_limit
            self._buf_ptr = 0
            return
        end = self._buf_ptr + m
        if end <= self.history_limit:
            self._buf_ids[self._buf_ptr : end] = ids
            self._buf_s[self._buf_ptr : end] = sc
            self._buf_u[self._buf_ptr : end] = u
        else:
            first = self.history_limit - self._buf_ptr
            self._buf_ids[self._buf_ptr :] = ids[:first]
            self._buf_s[self._buf_ptr :] = sc[:first]
            self._buf_u[self._buf_ptr :] = u
            rem = m - first
            self._buf_ids[:rem] = ids[first:]
            self._buf_s[:rem] = sc[first:]
            self._buf_u[:rem] = u
        self._buf_ptr = end % self.history_limit
        self._buf_size = min(self.history_limit, self._buf_size + m)

    def step_update(self, update: int) -> dict[str, Any]:
        t0 = monotonic()
        self.current_update = int(update)
        if self._buf_size == 0:
            self.last_update_wall_s = monotonic() - t0
            return self.diagnostics()

        ids_all = self._buf_ids[: self._buf_size]
        s_all = self._buf_s[: self._buf_size]
        age = np.maximum(0, self.current_update - self._buf_u[: self._buf_size]).astype(np.float64)
        rec_w = np.exp(-age / self.tau).astype(np.float32)

        losses = []
        for model, opt in zip(self.models, self.optimizers):
            model.train()
            for _ in range(self.optimizer_steps):
                idx = self.rng.integers(0, self._buf_size, size=min(self.batch_size, self._buf_size))
                xb = self._features.index_select(0, torch.as_tensor(ids_all[idx], dtype=torch.long, device=self.device))
                yb = torch.as_tensor(s_all[idx], dtype=torch.float32, device=self.device)
                wb = torch.as_tensor(rec_w[idx], dtype=torch.float32, device=self.device)
                wb = wb / wb.sum().clamp_min(1e-8)
                pred = nn.functional.softplus(model(xb).squeeze(-1))
                per_sample = nn.functional.smooth_l1_loss(pred, yb, reduction="none")
                loss = (wb * per_sample).sum()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                losses.append(float(loss.item()))
        self._last_loss = float(np.mean(losses)) if losses else float("nan")

        if self.current_update >= self.cold_updates:
            with torch.no_grad():
                preds = np.zeros((self.num_members, self.n), dtype=np.float64)
                for m_idx, model in enumerate(self.models):
                    model.eval()
                    for start in range(0, self.n, self.batch_size):
                        end = min(self.n, start + self.batch_size)
                        out = nn.functional.softplus(model(self._features[start:end]).squeeze(-1))
                        preds[m_idx, start:end] = out.cpu().numpy()
                self._pred_scores = preds.mean(axis=0)
            p_s = _score_distribution(self._pred_scores, temperature=self.temperature, strategy=self.strategy)
            staleness = np.maximum(0, self._step_clock - self._last_sampled_step).astype(np.float64)
            stale_sum = float(staleness.sum())
            p_c = staleness / stale_sum if stale_sum > 0.0 else np.full(self.n, 1.0 / self.n, dtype=np.float64)
            self._set_probabilities((1.0 - self.rho) * p_s + self.rho * p_c)

        self.last_update_wall_s = monotonic() - t0
        return self.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        info = super().diagnostics()
        info.update(
            {
                "rho": self.rho,
                "temperature": self.temperature,
                "strategy": self.strategy,
                "buffer_size": self._buf_size,
                "last_loss": self._last_loss,
                "mean_pred_score": float(self._pred_scores.mean()),
            }
        )
        return info

    def state_dict(self) -> dict[str, Any]:
        state = self._base_state_dict()
        state.update(
            {
                "models": [{k: v.detach().cpu().clone() for k, v in m.state_dict().items()} for m in self.models],
                "optimizers": [copy.deepcopy(o.state_dict()) for o in self.optimizers],
                "buf_ids": self._buf_ids[: self._buf_size].copy(),
                "buf_s": self._buf_s[: self._buf_size].copy(),
                "buf_u": self._buf_u[: self._buf_size].copy(),
                "buf_size": self._buf_size,
                "buf_ptr": self._buf_ptr,
                "last_sampled_step": self._last_sampled_step.copy(),
                "step_clock": self._step_clock,
                "pred_scores": self._pred_scores.copy(),
                "last_loss": self._last_loss,
            }
        )
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self._load_base_state_dict(state)
        for model, sd in zip(self.models, state["models"]):
            model.load_state_dict({k: v.clone() for k, v in sd.items()})
        for opt, sd in zip(self.optimizers, state["optimizers"]):
            opt.load_state_dict(copy.deepcopy(sd))
        size = int(state["buf_size"])
        self._buf_ids.fill(0)
        self._buf_s.fill(0.0)
        self._buf_u.fill(0)
        if size > 0:
            self._buf_ids[:size] = np.asarray(state["buf_ids"], dtype=np.int64)
            self._buf_s[:size] = np.asarray(state["buf_s"], dtype=np.float32)
            self._buf_u[:size] = np.asarray(state["buf_u"], dtype=np.int32)
        self._buf_size = size
        self._buf_ptr = int(state["buf_ptr"])
        self._last_sampled_step = np.asarray(state["last_sampled_step"], dtype=np.int64).copy()
        self._step_clock = int(state["step_clock"])
        self._pred_scores = np.asarray(state["pred_scores"], dtype=np.float64).copy()
        self._last_loss = float(state["last_loss"])
