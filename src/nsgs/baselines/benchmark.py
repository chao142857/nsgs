"""Fast synthetic multi-dimensional curriculum benchmark for verifying all samplers.

Simulates a non-stationary learner whose capability frontier c_t advances fastest
when trained on tasks near its current Zone of Proximal Development (ZPD).
Supports both low-D (e.g. 3D) and high-D (e.g. 12D with nonlinear coupled active
dimensions plus irrelevant distractors) task pools to verify scaling behavior
across Uniform, Tabular SGS, Grid-SGS, kNN-SGS, PLR, Neural-PLR, NSGS, and NSGS v2.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from time import monotonic
from typing import Any, Sequence

import numpy as np

from .base import TaskPool
from .registry import AVAILABLE_METHODS, build_sampler


@dataclass(frozen=True)
class SyntheticDomain:
    pool: TaskPool
    difficulty: np.ndarray
    active_dims: int
    total_dims: int


def make_synthetic_domain(
    *,
    num_tasks: int = 8192,
    total_dims: int = 12,
    active_dims: int = 4,
    seed: int = 0,
) -> SyntheticDomain:
    """Create a task pool with nonlinear coupled difficulty over `active_dims`
    and `total_dims - active_dims` irrelevant distractor coordinates.
    """
    if not (1 <= active_dims <= total_dims):
        raise ValueError(f"Expected 1 <= active_dims <= total_dims, got {active_dims}, {total_dims}")
    rng = np.random.default_rng(seed)
    raw = rng.uniform(-1.0, 1.0, size=(num_tasks, total_dims))
    # Nonlinear coupled difficulty in [0, 1] depending ONLY on active_dims.
    act = raw[:, :active_dims]
    radial = np.linalg.norm(act, axis=1) / np.sqrt(active_dims)
    if active_dims >= 2:
        coupling = 0.25 * np.abs(act[:, 0] * act[:, 1])
    else:
        coupling = 0.0
    diff = radial + coupling
    # Rank-normalize difficulty to uniform [0, 1] so frontier progress is directly comparable across dims.
    order = np.argsort(diff, kind="mergesort")
    ranks = np.empty(num_tasks, dtype=np.float64)
    ranks[order] = np.linspace(0.0, 1.0, num_tasks)
    bounds = [(-1.0, 1.0)] * total_dims
    names = [f"sig_{i}" if i < active_dims else f"dist_{i - active_dims}" for i in range(total_dims)]
    pool = TaskPool.from_raw(raw, bounds=bounds, feature_names=names, freqs=(1.0, 2.0))
    return SyntheticDomain(pool=pool, difficulty=ranks, active_dims=active_dims, total_dims=total_dims)


def run_single_benchmark(
    method: str,
    domain: SyntheticDomain,
    *,
    seed: int = 43,
    updates: int = 80,
    batch_per_update: int = 256,
    initial_capability: float = 0.12,
    sharpness: float = 16.0,
    learning_gain: float = 0.055,
    device: str = "cpu",
    sampler_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one sampler on the non-stationary synthetic curriculum oracle."""
    kwargs = dict(sampler_kwargs or {})
    # Scale down neural network size for quick synthetic verification while keeping fast convergence.
    if method in ("nsgs", "nsgs_v2", "neural_plr"):
        kwargs.setdefault("hidden_dims", (128, 128))
        kwargs.setdefault("optimizer_steps", 16)
        kwargs.setdefault("batch_size", 512)
        kwargs.setdefault("cold_updates", 5)
        kwargs.setdefault("lr", 1e-3)
        kwargs.setdefault("num_members", 5)
    elif method == "grid_sgs":
        kwargs.setdefault("bins", 8 if domain.total_dims <= 3 else 4)
    elif method == "knn_sgs":
        kwargs.setdefault("k", 24)
        kwargs.setdefault("max_anchors", 4096)

    sampler = build_sampler(method, domain.pool, seed=seed, device=device, **kwargs)
    env_rng = np.random.default_rng(seed + 99991)
    cap = float(initial_capability)
    coverages: list[float] = []
    t_start = monotonic()

    for u in range(1, updates + 1):
        ids = sampler.sample(batch_per_update)
        d = domain.difficulty[ids]
        p_true = 1.0 / (1.0 + np.exp(-sharpness * (cap - d)))
        succ = env_rng.random(batch_per_update) < p_true
        # Realistic PLR score signal: TD-error / surprise proxy peaking on uncertain frontier tasks.
        scores = np.abs(succ.astype(np.float64) - p_true)
        sampler.record(ids, succ, update=u, scores=scores)
        sampler.step_update(u)

        # Learner improves proportional to ZPD signal 4 * p * (1 - p) on sampled tasks.
        zpd = 4.0 * p_true * (1.0 - p_true)
        cap = min(1.05, cap + learning_gain * float(zpd.mean()))
        full_p = 1.0 / (1.0 + np.exp(-sharpness * (cap - domain.difficulty)))
        coverages.append(float(full_p.mean()))

    wall_s = monotonic() - t_start
    return {
        "method": method,
        "total_dims": domain.total_dims,
        "active_dims": domain.active_dims,
        "num_tasks": domain.pool.num_tasks,
        "seed": seed,
        "updates": updates,
        "final_coverage": coverages[-1],
        "coverage_auc": float(np.mean(coverages)),
        "final_capability": cap,
        "wall_s": wall_s,
        "diagnostics": sampler.diagnostics(),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--methods", nargs="+", default=list(AVAILABLE_METHODS), choices=AVAILABLE_METHODS)
    parser.add_argument("--num-tasks", type=int, default=8192)
    parser.add_argument("--total-dims", type=int, default=12)
    parser.add_argument("--active-dims", type=int, default=4)
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    domain = make_synthetic_domain(
        num_tasks=args.num_tasks,
        total_dims=args.total_dims,
        active_dims=args.active_dims,
        seed=args.seed,
    )
    rows = [
        run_single_benchmark(
            method,
            domain,
            seed=args.seed,
            updates=args.updates,
            batch_per_update=args.batch,
            device=args.device,
        )
        for method in args.methods
    ]
    if args.json:
        print(json.dumps(rows, indent=2))
    else:
        print(
            f"Synthetic benchmark: N={args.num_tasks}, D={args.total_dims} "
            f"(active={args.active_dims}), updates={args.updates}, seed={args.seed}"
        )
        print(f"{'method':14s} | {'final_cov':>9s} | {'cov_auc':>8s} | {'ess_frac':>8s} | {'wall_s':>7s}")
        print("-" * 58)
        for r in rows:
            print(
                f"{r['method']:14s} | {r['final_coverage']:9.4f} | {r['coverage_auc']:8.4f} | "
                f"{r['diagnostics']['ess_fraction']:8.4f} | {r['wall_s']:7.2f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
