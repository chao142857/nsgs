"""Unified baseline sampler registry and factory."""

from __future__ import annotations

from typing import Any

from .base import TaskPool, TaskSampler
from .grid import GridSGSSampler
from .knn import KNNSGSSampler
from .nsgs import NSGSSampler
from .plr import NeuralPLRSampler, PLRSampler
from .tabular import TabularSGSSampler
from .uniform import UniformSampler

AVAILABLE_METHODS: tuple[str, ...] = (
    "uniform",
    "tabular_sgs",
    "grid_sgs",
    "knn_sgs",
    "plr",
    "neural_plr",
    "nsgs",
    "nsgs_v2",
)


def build_sampler(
    method: str,
    pool: TaskPool,
    *,
    seed: int,
    device: str = "cpu",
    **kwargs: Any,
) -> TaskSampler:
    """Instantiate any registered curriculum / sampling baseline with a single API."""
    if method == "uniform":
        return UniformSampler(pool, seed=seed, **kwargs)
    if method == "tabular_sgs":
        return TabularSGSSampler(pool, seed=seed, **kwargs)
    if method == "grid_sgs":
        return GridSGSSampler(pool, seed=seed, **kwargs)
    if method == "knn_sgs":
        return KNNSGSSampler(pool, seed=seed, device=device, **kwargs)
    if method == "plr":
        return PLRSampler(pool, seed=seed, **kwargs)
    if method == "neural_plr":
        return NeuralPLRSampler(pool, seed=seed, device=device, **kwargs)
    if method == "nsgs":
        defaults: dict[str, Any] = dict(
            kernel="sharp",
            history_limit=1_000_000,
            tau=100.0,
            beta_optimism=0.0,
            adaptive_target=False,
            method_label="nsgs",
        )
        defaults.update(kwargs)
        return NSGSSampler(pool, seed=seed, device=device, **defaults)
    if method == "nsgs_v2":
        defaults = dict(
            kernel="sharp_safe",
            history_limit=200_000,
            tau=30.0,
            beta_optimism=1.0,
            adaptive_target=False,
            method_label="nsgs_v2",
        )
        defaults.update(kwargs)
        return NSGSSampler(pool, seed=seed, device=device, **defaults)
    raise ValueError(f"Unknown sampler method {method!r}; expected one of {AVAILABLE_METHODS}")
