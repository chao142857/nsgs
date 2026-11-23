"""Unified task-sampling baseline framework for NSGS comparative studies.

Provides a single TaskSampler contract shared across:
  - uniform      : UniformSampler
  - tabular_sgs  : TabularSGSSampler (per-configuration SGS)
  - grid_sgs     : GridSGSSampler (sparse spatial GSGS)
  - knn_sgs      : KNNSGSSampler (distance + recency weighted kNN-SGS)
  - plr          : PLRSampler (Jiang et al., ICML 2021)
  - neural_plr   : NeuralPLRSampler (neural-generalized PLR ablation)
  - nsgs         : NSGSSampler (v1 ensemble mean)
  - nsgs_v2      : NSGSSampler (v2 one-sided epistemic optimism + fast tracking)
"""

from .base import TaskPool, TaskSampler, build_fourier_features, distribution_stats
from .grid import GridSGSSampler, compute_sparse_cell_ids
from .kernels import (
    DEFAULT_KERNEL,
    EMPTY_HISTORY_PRIOR,
    KERNELS,
    SHARP_KERNEL,
    SHARP_SAFE_KERNEL,
    AdaptiveTargetTracker,
    SGSKernel,
    apply_one_sided_optimism,
    kernel_probabilities,
    kernel_unnormalized_weights,
    resolve_kernel,
)
from .knn import KNNSGSSampler
from .nsgs import NSGSSampler
from .plr import NeuralPLRSampler, PLRSampler
from .registry import AVAILABLE_METHODS, build_sampler
from .tabular import TabularSGSSampler
from .uniform import UniformSampler

__all__ = [
    "AVAILABLE_METHODS",
    "AdaptiveTargetTracker",
    "DEFAULT_KERNEL",
    "EMPTY_HISTORY_PRIOR",
    "GridSGSSampler",
    "KERNELS",
    "KNNSGSSampler",
    "NSGSSampler",
    "NeuralPLRSampler",
    "PLRSampler",
    "SGSKernel",
    "SHARP_KERNEL",
    "SHARP_SAFE_KERNEL",
    "TabularSGSSampler",
    "TaskPool",
    "TaskSampler",
    "UniformSampler",
    "apply_one_sided_optimism",
    "build_fourier_features",
    "build_sampler",
    "compute_sparse_cell_ids",
    "distribution_stats",
    "kernel_probabilities",
    "kernel_unnormalized_weights",
    "resolve_kernel",
]
