"""SGS Beta-shaped sampling kernels, epistemic optimism, and adaptive target tracking."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Mapping

import numpy as np

EMPTY_HISTORY_PRIOR = 0.5


@dataclass(frozen=True)
class SGSKernel:
    """Beta-shaped frontier sampling kernel from Success-Guided Sampling."""

    target: float = 0.66
    concentration: float = 5.0
    temperature: float = 2.0
    epsilon: float = 1e-8
    uniform_floor: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.target < 1.0):
            raise ValueError(f"target must be in (0, 1), got {self.target}")
        if not (math.isfinite(self.concentration) and self.concentration > 0.0):
            raise ValueError(f"concentration must be > 0, got {self.concentration}")
        if not (math.isfinite(self.temperature) and self.temperature > 0.0):
            raise ValueError(f"temperature must be > 0, got {self.temperature}")
        if not (math.isfinite(self.epsilon) and self.epsilon > 0.0):
            raise ValueError(f"epsilon must be > 0, got {self.epsilon}")
        if not (0.0 <= self.uniform_floor < 1.0):
            raise ValueError(f"uniform_floor must be in [0, 1), got {self.uniform_floor}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_KERNEL = SGSKernel(
    target=0.50, concentration=1.0, temperature=2.0, epsilon=1e-4, uniform_floor=0.0
)
SHARP_KERNEL = SGSKernel(
    target=0.66, concentration=5.0, temperature=2.0, epsilon=1e-8, uniform_floor=0.0
)
SHARP_SAFE_KERNEL = SGSKernel(
    target=0.66, concentration=5.0, temperature=2.0, epsilon=1e-4, uniform_floor=0.05
)

KERNELS: Mapping[str, SGSKernel] = {
    "default": DEFAULT_KERNEL,
    "sharp": SHARP_KERNEL,
    "sharp_safe": SHARP_SAFE_KERNEL,
}


def resolve_kernel(kernel: str | SGSKernel) -> SGSKernel:
    if isinstance(kernel, SGSKernel):
        return kernel
    if kernel not in KERNELS:
        raise ValueError(f"Unknown kernel {kernel!r}; expected one of {tuple(KERNELS)}")
    return KERNELS[kernel]


def kernel_unnormalized_weights(rates: np.ndarray, kernel: str | SGSKernel) -> np.ndarray:
    """Compute positive unnormalized SGS scores S_i = exp(log(w_i + eps) / T)."""
    spec = resolve_kernel(kernel)
    p = np.clip(np.asarray(rates, dtype=np.float64), 0.0, 1.0)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("rates must be a finite, non-empty 1D array")
    a = spec.concentration * spec.target
    b = spec.concentration * (1.0 - spec.target)
    log_w = a * np.log(p + spec.epsilon) + b * np.log(1.0 - p + spec.epsilon)
    # Use logaddexp(log_w, log(eps)) so log(w + eps) stays accurate even for tiny w.
    log_eps = math.log(spec.epsilon)
    logits = np.logaddexp(log_w, log_eps) / spec.temperature
    logits = logits - float(np.max(logits))
    return np.exp(logits)


def kernel_probabilities(rates: np.ndarray, kernel: str | SGSKernel) -> np.ndarray:
    """Map predicted success rates in [0, 1] to a normalized sampling distribution."""
    spec = resolve_kernel(kernel)
    weights = kernel_unnormalized_weights(rates, spec)
    total = float(weights.sum())
    if not (math.isfinite(total) and total > 0.0):
        probs = np.full(weights.shape, 1.0 / weights.size, dtype=np.float64)
    else:
        probs = weights / total
    if spec.uniform_floor > 0.0:
        probs = (1.0 - spec.uniform_floor) * probs + spec.uniform_floor / probs.size
        probs = probs / float(probs.sum())
    return probs


def apply_one_sided_optimism(
    mean: np.ndarray,
    std: np.ndarray,
    target: float,
    beta: float,
) -> np.ndarray:
    """Lift pessimistic predictions (mean < target) via epistemic uncertainty without
    pulling mastered configurations (mean >= target) back toward the frontier peak.
    """
    mu = np.clip(np.asarray(mean, dtype=np.float64), 0.0, 1.0)
    sigma = np.maximum(np.asarray(std, dtype=np.float64), 0.0)
    if mu.shape != sigma.shape:
        raise ValueError(f"mean shape {mu.shape} != std shape {sigma.shape}")
    if beta <= 0.0:
        return mu
    lifted = np.minimum(float(target), mu + float(beta) * sigma)
    return np.where(mu < target, lifted, mu)


class AdaptiveTargetTracker:
    """Lightweight empirical learning-progress tracker for adapting SGS target t.

    Tracks observed empirical success-rate gain across probability bins over a
    sliding window and shifts target t via EMA toward the highest-progress bin.
    """

    def __init__(
        self,
        initial_target: float = 0.66,
        *,
        num_bins: int = 5,
        min_target: float = 0.30,
        max_target: float = 0.75,
        ema: float = 0.15,
        min_bin_count: int = 32,
    ) -> None:
        self.target = float(initial_target)
        self.num_bins = int(num_bins)
        self.min_target = float(min_target)
        self.max_target = float(max_target)
        self.ema = float(ema)
        self.min_bin_count = int(min_bin_count)
        self._edges = np.linspace(0.0, 1.0, self.num_bins + 1)
        self._centers = 0.5 * (self._edges[:-1] + self._edges[1:])
        self._prev_bin_rates: np.ndarray | None = None
        self.last_bin_progress: np.ndarray = np.zeros(self.num_bins, dtype=np.float64)

    def update(self, predicted_rates: np.ndarray, successes: np.ndarray) -> float:
        p = np.clip(np.asarray(predicted_rates, dtype=np.float64), 0.0, 1.0)
        y = np.asarray(successes, dtype=np.float64)
        if p.size == 0:
            return self.target
        bin_idx = np.clip(np.floor(p * self.num_bins).astype(np.int64), 0, self.num_bins - 1)
        counts = np.bincount(bin_idx, minlength=self.num_bins)
        sums = np.bincount(bin_idx, weights=y, minlength=self.num_bins)
        valid = counts >= self.min_bin_count
        if not np.any(valid):
            return self.target
        rates = np.where(valid, sums / np.maximum(counts, 1), np.nan)
        if self._prev_bin_rates is not None:
            both = valid & np.isfinite(self._prev_bin_rates)
            if np.any(both):
                progress = np.where(both, rates - self._prev_bin_rates, -np.inf)
                self.last_bin_progress = np.where(both, progress, 0.0)
                best_bin = int(np.argmax(progress))
                if math.isfinite(float(progress[best_bin])) and progress[best_bin] > 0.0:
                    candidate = float(np.clip(self._centers[best_bin], self.min_target, self.max_target))
                    self.target = (1.0 - self.ema) * self.target + self.ema * candidate
        if self._prev_bin_rates is None:
            self._prev_bin_rates = rates
        else:
            self._prev_bin_rates = np.where(valid, rates, self._prev_bin_rates)
        return self.target

    def state_dict(self) -> dict:
        return {
            "target": self.target,
            "num_bins": self.num_bins,
            "min_target": self.min_target,
            "max_target": self.max_target,
            "ema": self.ema,
            "min_bin_count": self.min_bin_count,
            "prev_bin_rates": None if self._prev_bin_rates is None else self._prev_bin_rates.copy(),
            "last_bin_progress": self.last_bin_progress.copy(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.target = float(state["target"])
        prev = state["prev_bin_rates"]
        self._prev_bin_rates = None if prev is None else np.asarray(prev, dtype=np.float64).copy()
        self.last_bin_progress = np.asarray(state["last_bin_progress"], dtype=np.float64).copy()
